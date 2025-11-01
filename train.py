import os
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
import wandb
from model.FaceModel import FaceForCausalLM
from model.llava import conversation as conversation_lib
import pandas as pd
from dataset.dataset import custom_collate_fn, HybridRecognitionDataset, HybridReconstructionDataset, HybridCapDataset
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda)
import torch.nn.functional as F

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
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_recon_data", action="store_true", help="Use expression reconstruction data")
    parser.add_argument("--use_recog_data", action="store_true", help="Use emotion recognition data")
    parser.add_argument("--weight_recog", default=0.3, type=float, help="Sampling weight for emotion recognition data")
    parser.add_argument("--weight_recon", default=0.4, type=float, help="Sampling weight for expression reconstruction data")
    parser.add_argument("--weight_cap", default=0.3, type=float, help="Sampling weight for caption data")
    parser.add_argument("--dataset_dir", default="/DATA/minjung", type=str)
    parser.add_argument("--recog_sample_rates", default="1", type=str)
    parser.add_argument("--emo_recog_dataset", default="CAER-S_recog_dataset",
                        type=str, help="Choose from: SFEW_recog_dataset, CAER-S_recog_datase")
    parser.add_argument("--recon_sample_rates", default="1", type=str)
    parser.add_argument("--exp_recon_dataset", default="CAER-S_recon_dataset", type=str,
                        help="Choose from: SFEW_recon_dataset, CAER-S_recon_dataset")
    parser.add_argument("--cap_sample_rates", default="1", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument("--batch_size", default=4, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--face_loss_weight", default=10.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--check_txt", action="store_true")

    parser.add_argument("--exp_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--project_name", type=str)
    return parser.parse_args(args)

def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.init(
            project=args.project_name,   
            name=args.exp_name,         
            dir=args.log_dir,           
            config=args,             
        )
        return wandb
    return None

def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
       
        face_tokens = ['<FACE>']
        
        special_tokens = face_tokens 
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.face_token_idx = tokenizer("<FACE>", add_special_tokens=False).input_ids[0]
  
    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the Face model. """
    model_args = {k: getattr(args, k) for k in
                  ["out_dim", "ce_loss_weight", "face_loss_weight",
                   "face_token_idx", "mm_vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end"]}

    model = FaceForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.mm_vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize Face model and adjust requires_grad
    model.get_model().initialize_face_model()
    
    # Projection layer
    model.get_model().text_hidden_fcs.train()
    for param in model.get_model().text_hidden_fcs.parameters():
        param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
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

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    lora_config.save_pretrained(args.log_dir)
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "text_hidden_fcs"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)

def manually_merge_lora(model):
   
    merged_state_dict = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'base_layer') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            print(f"LoRA 모듈 발견: {name}")
            
            base_weight = module.base_layer.weight.data.clone()
            
            for adapter_name in module.lora_A.keys():
                lora_A = module.lora_A[adapter_name].weight.data
                lora_B = module.lora_B[adapter_name].weight.data
                scaling = module.scaling[adapter_name] if hasattr(module, 'scaling') else module.lora_alpha / module.r
                
                lora_weight = (lora_B @ lora_A) * scaling
                base_weight += lora_weight
            
            merged_state_dict[f"{name}.weight"] = base_weight
    
    return merged_state_dict

def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "clip_image_encoder": args.mm_vision_tower,
                      "epoch_samples" : 12000,
                      "precision": args.precision}

    
       # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset=args.vqa_data,
        batch_size=args.batch_size, validation= False) if args.use_cap_data else None
    recon_train_dataset = HybridReconstructionDataset(
        **common_ds_args, dataset=args.exp_recon_dataset,
        batch_size=args.batch_size, ) if args.use_recon_data else None
    recog_train_dataset = HybridRecognitionDataset(
        **common_ds_args, dataset=args.emo_recog_dataset,
        batch_size=args.batch_size, ) if args.use_recog_data else None
    
    val_ds_args = {"dataset_dir": args.dataset_dir,"tokenizer": tokenizer,
                   "clip_image_encoder": args.mm_vision_tower,
                   "epoch_samples" : 100,
                   "precision": args.precision,
                   "validation" : True
                   }
    
    # Validation datasets
    val_datasets = []
    val_datasets.append(
        HybridReconstructionDataset(
        **val_ds_args, dataset=args.exp_recon_dataset, sample_rate=[float(x) for x in args.recon_sample_rates.split(",")],
        batch_size=args.val_batch_size, ) )
            
    return cap_train_dataset, recon_train_dataset, recog_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, recon_train_dataset, recog_train_dataset, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.exp_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )
    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            cap_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if cap_train_dataset is not None else None
    
    recon_train_loader = torch.utils.data.DataLoader(
        recon_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            recon_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if recon_train_dataset is not None else None

    recog_train_loader = torch.utils.data.DataLoader(
        recog_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            recog_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if recog_train_dataset is not None else None
    
    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return cap_train_loader, recon_train_loader, recog_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model_path")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)
    ################################################################
    print("\n=== Model Training Status ===")
    for name, param in model.named_parameters():
        trainable = param.requires_grad
        frozen = not trainable
        status = "Trainable" if trainable else "Frozen"
        print(f"{name}: {status} (Shape: {param.shape})")

    print("\n=== Component Status ===")
    if hasattr(model, 'text_hidden_fcs'):
        text_hidden_trainable = any(p.requires_grad for p in model.text_hidden_fcs.parameters())
        print(f"text_hidden_fcs: {'Trainable' if text_hidden_trainable else 'Frozen'}")
    
    if hasattr(model, 'mm_projector'):
        mm_proj_trainable = any(p.requires_grad for p in model.mm_projector.parameters())
        print(f"mm_projector: {'Trainable' if mm_proj_trainable else 'Frozen'}")
        
    if hasattr(model, 'lm_head'):
        lm_head_trainable = any(p.requires_grad for p in model.lm_head.parameters())
        print(f"lm_head: {'Trainable' if lm_head_trainable else 'Frozen'}")
    
    print("face loss weight:", args.face_loss_weight)
    ################################################################
    
    model.config.save_pretrained(args.log_dir)
    model.generation_config.save_pretrained(args.log_dir) 
    tokenizer.save_pretrained(args.log_dir)
    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    
    base_optimizer = model_engine.optimizer.optimizer if hasattr(model_engine.optimizer, 'optimizer') else model_engine.optimizer

    for i, group in enumerate(base_optimizer.param_groups):
        lr = group['lr']
        param_count = sum(p.numel() for p in group['params'] if p.requires_grad)
        
        sample_param = next(iter(group['params']))
        is_projection = False
        
        for name, param in model_engine.module.named_parameters():
            if param is sample_param and 'text_hidden_fcs' in name:
                is_projection = True
                break
        
        print(f"Group {i}: LR = {lr}, Params: {param_count}, Projection: {is_projection}")
    
    resume_training_from_checkpoint(model_engine, args)

    cap_train_dataset, recon_train_dataset, recog_train_dataset, val_datasets = (
        initialize_datasets_and_loaders(args, tokenizer))
    cap_train_loader, recon_train_loader, recog_train_loader, val_loader = (
        setup_data_loaders(args, cap_train_dataset, recon_train_dataset, recog_train_dataset, val_datasets, tokenizer))
    
    # Determine active datasets and their weights
    active_dataloaders = []
    weights = []
   
    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_recog_data:
        active_dataloaders.append(('recog', recog_train_loader))
        weights.append(args.weight_recog)
    if args.use_recon_data:
        active_dataloaders.append(('recon', recon_train_loader))
        weights.append(args.weight_recon)
        
    total = sum(weights)
    weights = [w/total for w in weights]
    
    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (recon, recog, or cap) must be active."
    

    dataset_iters = {
    'cap': iter(cap_train_loader) if (args.use_cap_data and cap_train_loader is not None) else None,
    'recog': iter(recog_train_loader) if (args.use_recog_data and recog_train_loader is not None) else None,
    'recon': iter(recon_train_loader) if (args.use_recon_data and recon_train_loader is not None) else None }
    wandb = initialize_environment(args)

    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, wandb, args)[0]
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    dataset_choices = [idx for idx, _ in enumerate(active_dataloaders)]

    best_face_loss, best_val_loss = np.inf, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        step_choices = random.choices(dataset_choices, weights=weights, k=args.steps_per_epoch)

        dataset_iters = train(
            active_dataloaders, val_loader, tokenizer, model_engine, epoch, scheduler, wandb, dataset_iters, args, step_choices
        )
        
        save_dir = os.path.join(os.path.join(args.log_base_dir, args.exp_name))
        tokenizer.save_pretrained(save_dir)
        model.config.save_pretrained(save_dir)
        model.generation_config.save_pretrained(save_dir)
        model.save_pretrained(os.path.join(save_dir, "adapter"))
        
        
def save_checkpoint(model_engine, args):
    """ Saves the model checkpoint. """
    save_dir_name = "ckpt_model_path"
    save_dir = os.path.join(args.log_dir, save_dir_name)

    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)
        
    
def train(active_datasets, validation_loader, tokenizer, model, epoch, scheduler, wandb, dataset_iters, args, step_choices):

    def get_next_input(iterator, data_loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                curr_lr = scheduler.get_last_lr()
                log_dict = {
                "train/loss": trackers["loss"].avg,
                "train/ce_loss": trackers["ce_loss"].avg,
                "metrics/total_secs_per_batch": batch_time.avg,
                "metrics/data_secs_per_batch": data_time.avg,
                "train/lr": curr_lr[0]}
                
                if args.use_recon_data:
                    log_dict["train/face_loss"] = trackers["face_loss"].avg
                
                global_step_count = epoch * args.steps_per_epoch + global_step
                wandb.log(log_dict, step=global_step_count)

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    if args.use_recon_data:
        trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "face_loss": AverageMeter("FaceLoss", ":.4f")}
    else:
        trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f")}
    progress = ProgressMeter(
    args.steps_per_epoch, 
    [batch_time, data_time] + list(trackers.values()), 
    prefix=f"Epoch: [{epoch}]")   
    
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            dataset_type, data_loader = active_datasets[step_choices[global_step]]
            data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
            dataset_iters[dataset_type] = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["clip_enc_images"]:
                if data_batch[key] is not None:
                    # data_batch[key] = data_batch[key].to(torch.float32)
                    data_batch[key] = data_batch[key].bfloat16()
            output_dict = model(**data_batch)
            # Update training metrics
            
            if args.use_recon_data:
                for key in ["ce_loss", "face_loss", "loss"]:
                    trackers[key].update(output_dict[key].item(), data_batch["clip_enc_images"].size(0))
            else:
                for key in ["ce_loss", "loss"]:
                    trackers[key].update(output_dict[key].item(), data_batch["clip_enc_images"].size(0))
                    
            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()
        
        global_step_count = epoch * args.steps_per_epoch + global_step
        
        
        
        if global_step_count >= 1400 and global_step_count % 100 == 0:
            unwrapped_model = model.module if hasattr(model, 'module') else model
            merged_weights = manually_merge_lora(unwrapped_model)
            print(f"Merged {len(merged_weights)} layers.")
            full_state_dict = unwrapped_model.state_dict()
            full_state_dict.update(merged_weights)
            torch.save(full_state_dict, os.path.join(args.log_dir, f"manually_merged_{global_step_count}.pt"))
            
    return dataset_iters
    
def validate_model_performance(validation_loader, model, current_step, wandb, args):
    if args.use_recon_data:
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"),
                   "face_loss": AverageMeter("FaceLoss", ":.4f")}
    else:
        trackers = {
            "ce_loss": AverageMeter("CeLoss", ":.4f"),
            "loss": AverageMeter("Loss", ":.4f")
        }
    model.train()
    
    for data_batch in tqdm.tqdm(validation_loader):
        data_batch = dict_to_cuda(data_batch)
        for key in ["clip_enc_images"]:
            if data_batch[key] is not None:
                data_batch[key] = data_batch[key].bfloat16()
                # data_batch[key] = data_batch[key].to(torch.float32)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            results = model(**data_batch)
                 
        if args.use_recon_data:
            for key in ["ce_loss", "face_loss", "loss"]:
                trackers[key].update(results[key].item(), data_batch["clip_enc_images"].size(0))
        else:
            for key in ["ce_loss", "loss"]:
                trackers[key].update(results[key].item(), data_batch["clip_enc_images"].size(0))
                
        for tracker in trackers.values():
            tracker.all_reduce()
        
        if args.local_rank == 0:
            log_dict = {
                "global_step": current_step,
                "val/ce_loss": trackers["ce_loss"].avg,
                "val/loss": trackers["loss"].avg,
            }
            
            if args.use_recon_data:
                log_dict["val/face_loss"] = trackers["face_loss"].avg
                print(f"Validation Face Loss: {trackers['face_loss'].avg:.4f}")
            
            wandb.log(log_dict, step=current_step)
            
            print(f"Validation CE Loss: {trackers['ce_loss'].avg:.4f}")
            print(f"Validation total Loss: {trackers['loss'].avg:.4f}")
            
        total_loss_avg = trackers["loss"].avg
        return total_loss_avg

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
    if args.local_rank == 0:
        wandb.finish()
