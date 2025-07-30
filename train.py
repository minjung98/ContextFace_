#!/usr/-bin/env python3
"""
ContextFace Training Script

🚀 Example Usage:

1. Basic SitGen + ExpGen training (with auto-resume):
   python train.py --use_sitgen_data --use_expgen_data --auto_resume

2. Full data training including VQA:
   python train.py --use_sitgen_data --use_expgen_data --use_cap_data --auto_resume

3. Adjust learning rate:
   python train.py --use_sitgen_data --use_expgen_data --lr 1e-4 --auto_resume

4. Change LoRA rank:
   python train.py --use_sitgen_data --use_expgen_data --lora_r 16 --auto_resume

5. Resume from a specific checkpoint:
   python train.py --resume "/path/to/your/checkpoints/your_experiment_name/checkpoint_latest.pt"

6. Set experiment name manually:
   python train.py --exp_name "my_experiment" --project_name "test_project" --use_sitgen_data --use_expgen_data

7. Full fine-tuning (without LoRA):
   python train.py --use_sitgen_data --use_expgen_data --lora_r 0 --lr 2e-5 --auto_resume

8. Adjust number of epochs:
   python train.py --use_sitgen_data --use_expgen_data --epochs 20 --auto_resume

9. Adjust batch size:
   python train.py --use_sitgen_data --use_expgen_data --batch_size 2 --grad_accumulation_steps 20 --auto_resume

10. Adjust dataset weights:
    python train.py --use_sitgen_data --use_expgen_data --use_cap_data --weight_sitgen 0.6 --weight_expgen 0.3 --weight_cap 0.1 --auto_resume

💡 Tips:
- --auto_resume: Automatically resumes training from the latest checkpoint.
- Experiment/project names are auto-generated based on settings by default.
- If you encounter GPU memory issues, decrease --batch_size and increase --grad_accumulation_steps.
"""

# Set Python path to resolve module import issues
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import tqdm
import random
import torch
import argparse
import deepspeed
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
import wandb
from model.FaceModel import FaceForCausalLM
from model.llava import conversation as conversation_lib
from dataset.dataset import custom_collate_fn, HybridSitGenDataset, HybridExpGenDataset, HybridCapDataset
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda)

def generate_experiment_names(args):
    """
    Automatically parses experiment conditions to generate exp_name and project_name.
    """
    # Identify which datasets are in use
    datasets_used = []
    if args.use_sitgen_data:
        datasets_used.append(f"SG{int(args.weight_sitgen*100):02d}")  # e.g., SG45 (SitGen 45%)
    if args.use_expgen_data:
        datasets_used.append(f"EG{int(args.weight_expgen*100):02d}")  # e.g., EG45 (ExpGen 45%)
    if args.use_cap_data:
        datasets_used.append(f"VQ{int(args.weight_cap*100):02d}")    # e.g., VQ10 (VQA 10%)
    
    dataset_str = "_".join(datasets_used) if datasets_used else "NoData"
    
    # Format the learning rate string
    lr_str = f"lr{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")  # e.g., lr5e-5
    
    # LoRA rank string
    lora_str = f"lora{args.lora_r}" if args.lora_r > 0 else "fullft"  # e.g., lora8 or fullft
    
    # Epochs string
    epoch_str = f"ep{args.epochs}"
    
    # Steps per epoch (simplified)
    steps_str = f"s{args.steps_per_epoch}"
    
    # Auto-generated experiment name
    auto_exp_name = f"face_{dataset_str}_{lr_str}_{lora_str}_{epoch_str}_{steps_str}"
    
    # Project name (more concise)
    project_components = []
    if args.use_sitgen_data or args.use_expgen_data:
        project_components.append("ContextFace")
    if args.use_cap_data:
        project_components.append("VQA")
    
    auto_project_name = "_".join(project_components) if project_components else "FaceModel_Experiments"
    
    return auto_exp_name, auto_project_name

def parse_args(args):
    parser = argparse.ArgumentParser(description="Face Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--mm_vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_llama_2", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=100, type=int)
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use Caption Dataset (VQA)")
    parser.add_argument("--use_sitgen_data", action="store_true", help="Use Situation Generation Dataset")
    parser.add_argument("--use_expgen_data", action="store_true", help="Use Expression Generation Dataset")
    parser.add_argument("--weight_sitgen", default=0.35, type=float, help="Sampling weight for Situation Generation Dataset")
    parser.add_argument("--weight_expgen", default=0.55, type=float, help="Sampling weight for Expression Generation Dataset")
    parser.add_argument("--weight_cap", default=0.1, type=float, help="Sampling weight for VQA Dataset")
    parser.add_argument("--dataset_dir", default="/path/to/your/datasets", type=str)
    parser.add_argument("--sitgen_sample_rates", default="1", type=str)
    parser.add_argument("--situation_generation_dataset", default="CAER_sitgen_dataset",
                        type=str, help="Choose from: SFEW_sitgen_dataset, CAER_sitgen_dataset")
    parser.add_argument("--expgen_sample_rates", default="1", type=str)
    parser.add_argument("--expression_generation_dataset", default="CAER_expgen_dataset", type=str,
                        help="Choose from: SFEW_expgen_dataset, CAER_expgen_dataset")
    parser.add_argument("--cap_sample_rates", default="1", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str, help="Path to a specific checkpoint to resume from")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from the latest checkpoint")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", default=1055, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank (set to 0 for full fine-tuning)")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--face_loss_weight", default=10.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    
    # Checkpoint and saving intervals
    parser.add_argument("--checkpoint_interval", default=20, type=int, help="Save a training checkpoint every N steps")
    parser.add_argument("--save_interval", default=5, type=int, help="Save a complete, inference-ready model every N steps")
    parser.add_argument("--val_interval", default=10, type=int, help="Run validation every N steps")
    
    # Evaluation settings
    parser.add_argument("--val_dataset", default="CAER_ExpGenVal", type=str,
                        help="Choose from: CAER_ExpGenVal, SFEW_ExpGenVal")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="/path/to/your/output/checkpoints", type=str)
    parser.add_argument("--exp_name", default="auto", type=str, help="Experiment name ('auto' for automatic generation)")
    parser.add_argument("--project_name", default="auto", type=str, help="Project name ('auto' for automatic generation)")
    parser.add_argument("--disable_auto_naming", action="store_true", help="Disable automatic experiment naming")
    
    args = parser.parse_args(args)
    
    # Auto-generate names if the user specified 'auto' or used the default
    if not args.disable_auto_naming and (args.exp_name == "auto" or args.project_name == "auto"):
        auto_exp_name, auto_project_name = generate_experiment_names(args)
        
        if args.exp_name == "auto":
            args.exp_name = auto_exp_name
            print(f"🚀 Auto-generated experiment name: {args.exp_name}")
        
        if args.project_name == "auto":
            args.project_name = auto_project_name
            print(f"📁 Auto-generated project name: {args.project_name}")
    
    return args


def initialize_environment(args):
    """ Set up logging directories and wandb. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)
    wandb.init(
        project=args.project_name,   
        name=args.exp_name,         
        dir=args.log_dir,           
        config=args,             
    )
    return wandb


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + f"---- Initialized tokenizer from: {args.version} ----" + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
       
        face_tokens = ['<FACE>']
        
        special_tokens = face_tokens 
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.face_token_idx = tokenizer("<FACE>", add_special_tokens=False).input_ids[0]
  
    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the FaceForCausalLM model. """
    model_args = {k: getattr(args, k) for k in
                  ["out_dim", "ce_loss_weight", "face_loss_weight",
                   "face_token_idx", "mm_vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end"]}

    model = FaceForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, **model_args
    )
    
    print('\033[92m' + f"---- Initialized model from: {args.version} ----" + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Ensure model is properly materialized (no meta tensors)
    if hasattr(model, '_modules'):
        for name, param in model.named_parameters():
            if param.is_meta:
                print(f"⚠️  Warning: Found meta tensor: {name}. Forcing materialization.")
                _ = param.data
    
    print("✅ Model materialization check completed.")

    return model


def prepare_model_for_training(model, tokenizer, args):
    """Prepares the model for training by setting up gradients, modules, and LoRA."""
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device='cuda')

    # Initialize Face model and adjust requires_grad
    model.get_model().initialize_face_model()
    
    # Set projection layer to trainable
    model.get_model().text_hidden_fcs.train()
    for param in model.get_model().text_hidden_fcs.parameters():
        param.requires_grad = True

    # Freeze vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Handle full fine-tuning case
    if args.lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if args.lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Make final specified modules trainable
    set_trainable_modules(model)


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    lora_config.save_pretrained(args.log_dir)
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable and print parameter counts. """
    trainable_modules = ["lm_head", "embed_tokens", "text_hidden_fcs"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + f"---- Total parameters: {total_params} ----" + '\033[0m')
        print('\033[92m' + f"---- Trainable parameters: {trainable_params} ----" + '\033[0m')

    count_parameters(model)

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, is_best=False):
    """
    Saves a complete checkpoint with all necessary information for resuming training.
    """
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
        'random_state': random.getstate(),
        'torch_random_state': torch.get_rng_state(),
    }
    
    checkpoint_path = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch}_step_{global_step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    latest_path = os.path.join(args.log_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    print(f"📌 Latest checkpoint saved: {latest_path}")
    
    if is_best:
        best_path = os.path.join(args.log_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        print(f"🏆 Best checkpoint saved: {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, args):
    """
    Loads a checkpoint and restores the training state.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, 0, 0
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model state loaded.")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✅ Optimizer state loaded.")
    
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("✅ Scheduler state loaded.")
    
    random.setstate(checkpoint['random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'])
    print("✅ Random states restored.")
    
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    
    print(f"🚀 Resumed from epoch {epoch}, global step {global_step}.")
    return checkpoint, epoch, global_step

def find_latest_checkpoint(log_dir):
    """
    Finds the latest checkpoint in the specified log directory.
    """
    latest_path = os.path.join(log_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    checkpoint_files = []
    for file in os.listdir(log_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
            try:
                parts = file.replace("checkpoint_epoch_", "").replace(".pt", "").split("_step_")
                epoch = int(parts[0])
                step = int(parts[1])
                checkpoint_files.append((epoch, step, os.path.join(log_dir, file)))
            except (ValueError, IndexError):
                continue
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: (x[0], x[1]))
        return checkpoint_files[-1][2]
    
    return None

def save_complete_model(model, merged_state_dict, args, step, tokenizer=None):
    """
    Saves a complete, inference-ready model that can be loaded independently.
    """
    complete_model_dir = os.path.join(args.log_dir, f"complete_model_step_{step}")
    os.makedirs(complete_model_dir, exist_ok=True)
    
    print(f"💾 Saving complete model to: {complete_model_dir}")
    
    # 1. Save the merged state dictionary
    torch.save(merged_state_dict, os.path.join(complete_model_dir, "pytorch_model.bin"))
    
    # 2. Save the model configuration
    model.config.save_pretrained(complete_model_dir)
    
    # 3. Save the generation configuration
    if hasattr(model, 'generation_config'):
        model.generation_config.save_pretrained(complete_model_dir)
    
    # 4. Copy tokenizer files from the main log directory
    import shutil
    import glob
    
    tokenizer_patterns = ["tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]
    tokenizer_files = []
    for pattern in tokenizer_patterns:
        tokenizer_files.extend(glob.glob(os.path.join(args.log_dir, pattern)))
    
    print(f"  🔍 Found {len(tokenizer_files)} tokenizer files to copy:")
    for file_path in tokenizer_files:
        if os.path.exists(file_path):
            shutil.copy2(file_path, complete_model_dir)
            print(f"    ✅ Copied: {os.path.basename(file_path)}")
        else:
            print(f"    ❌ Missing: {os.path.basename(file_path)}")
    
    # 5. Create a loading script for easy use
    loading_script = f'''
"""
Easy loading script for a complete ContextFace model.

Usage:
    from load_model import load_contextface_model
    model, tokenizer = load_contextface_model("{complete_model_dir}")
"""
import torch
from transformers import AutoTokenizer
from model.FaceModel import FaceForCausalLM

def load_contextface_model(model_dir):
    """Loads a complete ContextFace model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length={args.model_max_length}, padding_side="right", use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("{args.version}", model_max_length={args.model_max_length}, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    
    model_args = {{
        "out_dim": {args.out_dim}, "ce_loss_weight": {args.ce_loss_weight}, "face_loss_weight": {args.face_loss_weight},
        "face_token_idx": {args.face_token_idx}, "mm_vision_tower": "{args.mm_vision_tower}", "use_mm_start_end": {args.mm_use_im_start_end},
        "mm_vision_select_layer": {args.mm_vision_select_layer}, "pretrain_mm_mlp_adapter": "", "tune_mm_mlp_adapter": False,
        "freeze_mm_mlp_adapter": True, "mm_use_im_start_end": {args.mm_use_im_start_end}
    }}
    
    model = FaceForCausalLM.from_pretrained("{args.version}", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    state_dict = torch.load(f"{{model_dir}}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    
    print(f"✅ ContextFace model loaded successfully from {{model_dir}}!")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_contextface_model("{complete_model_dir}")
'''
    with open(os.path.join(complete_model_dir, "load_model.py"), "w") as f:
        f.write(loading_script)
    
    print(f"✅ Complete model saved! To use, run: python {complete_model_dir}/load_model.py")

def lora_merge(model, args):
    """
    Merges LoRA weights with the base model and returns the complete state dictionary.
    """
    print("🔄 Starting LoRA merge...")
    
    merged_state_dict = model.state_dict().copy()
    lora_pairs = {}
    
    for key in merged_state_dict.keys():
        if key.endswith('.lora_A.weight'):
            parent_key = key.rsplit('.lora_A.weight', 1)[0]
            if parent_key not in lora_pairs: lora_pairs[parent_key] = {}
            lora_pairs[parent_key]['lora_A'] = key
        elif key.endswith('.lora_B.weight'):
            parent_key = key.rsplit('.lora_B.weight', 1)[0]
            if parent_key not in lora_pairs: lora_pairs[parent_key] = {}
            lora_pairs[parent_key]['lora_B'] = key
    
    print(f"📊 Found {len(lora_pairs)} LoRA pairs to merge.")
    
    scaling = args.lora_alpha / args.lora_r
    print(f"🔧 LoRA scaling factor: {args.lora_alpha} / {args.lora_r} = {scaling:.4f}")
    
    lora_layers_count = 0
    for parent_key, lora_info in lora_pairs.items():
        if 'lora_A' in lora_info and 'lora_B' in lora_info:
            weight_key = f"{parent_key}.weight"
            if weight_key in merged_state_dict:
                base_weight = merged_state_dict[weight_key].clone()
                lora_A_weight = merged_state_dict[lora_info['lora_A']]
                lora_B_weight = merged_state_dict[lora_info['lora_B']]
                
                # Merge: base_weight + scaling * (lora_B @ lora_A)
                lora_delta = lora_B_weight @ lora_A_weight
                merged_state_dict[weight_key] = base_weight + (lora_delta * scaling)
                lora_layers_count += 1
                print(f"  ✅ Merged: {parent_key}")
               
    print(f"✅ LoRA merge completed: {lora_layers_count} layers merged.")
    print(f"📊 Total keys in final state dict: {len(merged_state_dict)}")
    return merged_state_dict

def initialize_datasets_and_loaders(args, tokenizer):
    """Initializes training and validation datasets."""
    print("🎯 Initializing datasets for single GPU mode...")

    base_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                    "clip_image_encoder": args.mm_vision_tower, "precision": args.precision}

    # Training datasets with specified epoch samples
    cap_train_dataset = HybridCapDataset(**base_ds_args, dataset=args.vqa_data, epoch_samples=8000, batch_size=args.batch_size, validation=False) if args.use_cap_data else None
    expgen_train_dataset = HybridExpGenDataset(**base_ds_args, dataset=args.expression_generation_dataset, epoch_samples=42196, batch_size=args.batch_size) if args.use_expgen_data else None
    sitgen_train_dataset = HybridSitGenDataset(**base_ds_args, dataset=args.situation_generation_dataset, epoch_samples=42196, batch_size=args.batch_size) if args.use_sitgen_data else None
    
    # Validation datasets
    val_base_args = {**base_ds_args, "validation": True}
    val_datasets = []
    
    if args.use_expgen_data:
        val_datasets.append(HybridExpGenDataset(**val_base_args, dataset=args.expression_generation_dataset, epoch_samples=200, batch_size=args.val_batch_size))
    if args.use_sitgen_data:
        val_datasets.append(HybridSitGenDataset(**val_base_args, dataset=args.situation_generation_dataset, epoch_samples=200, batch_size=args.val_batch_size))
    if args.use_cap_data:
        val_datasets.append(HybridCapDataset(**val_base_args, dataset=args.vqa_data, epoch_samples=100, batch_size=args.val_batch_size))
    
    return cap_train_dataset, expgen_train_dataset, sitgen_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, expgen_train_dataset, sitgen_train_dataset, val_datasets, tokenizer):
    """Sets up training and validation data loaders."""
    train_loader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": args.workers, "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers, "pin_memory": False}
    
    # Use training-mode collate function for both train and val to calculate loss
    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.mm_use_im_start_end, local_rank=0, inference=False)

    cap_train_loader = torch.utils.data.DataLoader(cap_train_dataset, collate_fn=collate_fn, **train_loader_args) if cap_train_dataset else None
    expgen_train_loader = torch.utils.data.DataLoader(expgen_train_dataset, collate_fn=collate_fn, **train_loader_args) if expgen_train_dataset else None
    sitgen_train_loader = torch.utils.data.DataLoader(sitgen_train_dataset, collate_fn=collate_fn, **train_loader_args) if sitgen_train_dataset else None
    
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(combined_val_datasets, **val_loader_args, collate_fn=collate_fn)

    return cap_train_loader, expgen_train_loader, sitgen_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    """Initializes the DeepSpeed engine."""
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0, "betas": (args.beta1, args.beta2)}},
        "scheduler": {"type": "WarmupDecayLR", "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0, "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 0},
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        collate_fn=partial(custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.mm_use_im_start_end, local_rank=0),
        config=ds_config
    )
    return model_engine, optimizer, scheduler


def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)

    # Save initial configs
    model.config.save_pretrained(args.log_dir)
    model.generation_config.save_pretrained(args.log_dir) 
    tokenizer.save_pretrained(args.log_dir)
    
    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    
    # Resume logic
    start_epoch = args.start_epoch
    start_global_step = 0
    if args.resume:
        checkpoint, start_epoch, start_global_step = load_checkpoint(args.resume, model_engine, optimizer, scheduler, args)
        if checkpoint is None: return
    elif args.auto_resume:
        latest_checkpoint = find_latest_checkpoint(args.log_dir)
        if latest_checkpoint:
            _, start_epoch, start_global_step = load_checkpoint(latest_checkpoint, model_engine, optimizer, scheduler, args)
            print(f"🔄 Auto-resumed from: {latest_checkpoint}")
        else:
            print("📝 No checkpoint found, starting fresh training.")
    else:
        print("🆕 Starting fresh training.")
    
    cap_train_dataset, expgen_train_dataset, sitgen_train_dataset, val_datasets = initialize_datasets_and_loaders(args, tokenizer)
    cap_train_loader, expgen_train_loader, sitgen_train_loader, val_loader = setup_data_loaders(args, cap_train_dataset, expgen_train_dataset, sitgen_train_dataset, val_datasets, tokenizer)
    
    # Determine active datasets and their weights
    active_dataloaders, weights = [], []
    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_sitgen_data:
        active_dataloaders.append(('sitgen', sitgen_train_loader))
        weights.append(args.weight_sitgen)
    if args.use_expgen_data:
        active_dataloaders.append(('expgen', expgen_train_loader))
        weights.append(args.weight_expgen)
        
    assert active_dataloaders, "Error: At least one dataset must be active."
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    dataset_iters = {name: iter(loader) for name, loader in active_dataloaders if loader}
    wandb_run = initialize_environment(args)

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    dataset_indices = range(len(active_dataloaders))

    for epoch in range(start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])
        step_choices = random.choices(dataset_indices, weights=weights, k=args.steps_per_epoch)

        dataset_iters = train(
            active_dataloaders, model_engine, epoch, scheduler, wandb_run, dataset_iters, args, step_choices, val_loader, start_global_step, tokenizer
        )
        
        current_global_step = (epoch + 1) * args.steps_per_epoch
        save_checkpoint(model_engine, optimizer, scheduler, epoch + 1, current_global_step, args)
        
        # Reset resume step counter after the first epoch
        start_global_step = 0        
    
def train(active_datasets, model, epoch, scheduler, wandb, dataset_iters, args, step_choices, val_loader, start_global_step=0, tokenizer=None):
    """Main training loop for one epoch."""

    def get_next_input(iterator, data_loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress(current_step):
        """Logs training progress to console and wandb."""
        if current_step % args.print_freq == 0:
            progress.display(current_step + 1)
            curr_lr = scheduler.get_last_lr()[0]
            log_dict = {
                "train/loss": trackers["loss"].avg,
                "train/ce_loss": trackers["ce_loss"].avg,
                "metrics/total_secs_per_batch": batch_time.avg,
                "metrics/data_secs_per_batch": data_time.avg,
                "train/lr": curr_lr,
            }
            if args.use_expgen_data:
                log_dict["train/face_loss"] = trackers["face_loss"].avg
            
            wandb.log(log_dict, step=actual_global_step)

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    
    loss_keys = ["loss", "ce_loss"]
    if args.use_expgen_data: loss_keys.append("face_loss")
    trackers = {key: AverageMeter(key.replace('_', ' ').title(), ":.4f") for key in loss_keys}

    progress = ProgressMeter(args.steps_per_epoch, [batch_time, data_time] + list(trackers.values()), prefix=f"Epoch: [{epoch}]")   
    
    model.train()
    end = time.time()
    
    step_offset = start_global_step % args.steps_per_epoch if start_global_step > 0 else 0
    
    for step_idx, step_in_epoch in enumerate(range(step_offset, args.steps_per_epoch)):
        for _ in range(args.grad_accumulation_steps):
            dataset_type, data_loader = active_datasets[step_choices[step_in_epoch]]
            data_batch, dataset_iters[dataset_type] = get_next_input(dataset_iters[dataset_type], data_loader)

            data_time.update(time.time() - end)
            data_batch = dict_to_cuda(data_batch)
            if data_batch.get("clip_enc_images") is not None:
                data_batch["clip_enc_images"] = data_batch["clip_enc_images"].bfloat16()

            output_dict = model(**data_batch)
            
            for key in trackers:
                if key in output_dict:
                    trackers[key].update(output_dict[key].item(), data_batch["clip_enc_images"].size(0))
            
            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        actual_global_step = start_global_step + step_idx if start_global_step > 0 else epoch * args.steps_per_epoch + step_in_epoch
        log_progress(step_in_epoch)
        
        if actual_global_step > 0 and actual_global_step % args.val_interval == 0:
            validate_model_performance(val_loader, model, actual_global_step, wandb, args, tokenizer)
        
        if actual_global_step > 0 and actual_global_step % args.save_interval == 0:
            unwrapped_model = model.module if hasattr(model, 'module') else model
            merged_state_dict = lora_merge(unwrapped_model, args)
            save_complete_model(unwrapped_model, merged_state_dict, args, actual_global_step, tokenizer)
            del merged_state_dict
            torch.cuda.empty_cache()
        
        if actual_global_step > 0 and actual_global_step % args.checkpoint_interval == 0:
            save_checkpoint(model, model.optimizer, model.lr_scheduler, epoch, actual_global_step, args)
    
    model.train()
    torch.cuda.empty_cache()
    return dataset_iters
    
def validate_model_performance(validation_loader, model, current_step, wandb, args, tokenizer):
    """
    Calculates and logs validation losses without performing inference.
    """
    trackers = {
        "loss": AverageMeter("Loss", ":.4f"),
        "ce_loss": AverageMeter("CeLoss", ":.4f"),
        "face_loss": AverageMeter("FaceLoss", ":.4f")
    }
    face_loss_count = 0
    total_samples = 0
    
    model.train()  # Keep training mode for consistent loss calculation
    
    print("\n🚀 Running validation for loss calculation...")
    for data_batch in tqdm.tqdm(validation_loader, desc="Validating"):
        data_batch = dict_to_cuda(data_batch)
        
        if data_batch.get("clip_enc_images") is not None:
            data_batch["clip_enc_images"] = data_batch["clip_enc_images"].bfloat16()
        
        with torch.no_grad():
            results = model(**data_batch)
        
        batch_size = data_batch["clip_enc_images"].size(0)
        total_samples += batch_size
        
        trackers["ce_loss"].update(results["ce_loss"].item(), batch_size)
        trackers["loss"].update(results["loss"].item(), batch_size)
        
        has_face_data = any(exp.numel() > 0 for exp in data_batch["gt_exps"])
        if has_face_data and "face_loss" in results:
            trackers["face_loss"].update(results["face_loss"].item(), batch_size)
            face_loss_count += batch_size

    torch.cuda.empty_cache()
    
    # Log metrics to wandb
    log_dict = {
        "global_step": current_step,
        "val/ce_loss": trackers["ce_loss"].avg,
        "val/loss": trackers["loss"].avg,
    }
    if face_loss_count > 0:
        log_dict["val/face_loss"] = trackers["face_loss"].avg
        print(f"Validation Face Loss: {trackers['face_loss'].avg:.4f} (from {face_loss_count}/{total_samples} samples)")
    
    wandb.log(log_dict, step=current_step)
    
    print(f"Validation CE Loss: {trackers['ce_loss'].avg:.4f}")
    print(f"Validation Total Loss: {trackers['loss'].avg:.4f}")
    print(f"Validation samples breakdown: ExpGen={face_loss_count}, Others={total_samples-face_loss_count}\n")
    
    return trackers["loss"].avg

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
    wandb.finish()
