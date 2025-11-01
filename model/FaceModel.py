import torch
import torch.nn as nn
import torch.nn.functional as F

from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from tools.utils import IMAGE_TOKEN_INDEX

def face_loss_function(pred_exp: torch.Tensor, gt_exp: torch.Tensor, face_count: float):
    pred_exp = pred_exp.to(torch.bfloat16)
    gt_exp = gt_exp.to(torch.bfloat16)
    face_loss = F.mse_loss(pred_exp, gt_exp)
    face_loss = face_loss.sum() / (face_count + 1e-8)
    return face_loss


class FaceBaseModel:
    def __init__(self, config, **kwargs):
        super(FaceBaseModel, self).__init__(config)
        self.config = config
       
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 103))
        self.initialize_face_model()

    def initialize_face_model(self):
        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class FaceModel(FaceBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(FaceModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class FaceForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = FaceModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        self.face_token_idx = kwargs.pop("face_token_idx")
    
    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.face_loss_weight = kwargs.pop("face_loss_weight", 10.0)

    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward(self,  clip_enc_images: torch.FloatTensor,
                      input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, gt_exps: torch.FloatTensor, inference: bool = False, **kwargs, ):
        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, clip_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                clip_enc_images, input_ids, labels, attention_masks, offset
            )

        # Create segmentation token mask

        face_token_mask = self._create_face_token_mask(input_ids)
        
        # Process hidden states
        hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, face_token_mask, offset)
        
        pred_exps = torch.stack(pred_embeddings)
        gt_exps = torch.stack(gt_exps)
      
        if inference:
                return {"pred_exps": pred_exps, "gt_exps": gt_exps}
            
        return self._calculate_losses(pred_exps, gt_exps, output)

    def _create_face_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.face_token_idx
        mask = torch.cat([mask, torch.zeros((mask.shape[0], 1)).bool().cuda()], dim=1)
        image_embeds_len = 575 
        if IMAGE_TOKEN_INDEX in input_ids:
            new_face_token_mask = torch.zeros([mask.shape[0], mask.shape[1]+image_embeds_len]).bool().cuda()
            for i in range(input_ids.shape[0]):
                if IMAGE_TOKEN_INDEX in input_ids[i]:
                    new_face_token_mask[i, image_embeds_len:] = mask[i]
                else:
                    new_face_token_mask[i, :mask.shape[1]] = mask[i] 
            mask = new_face_token_mask
                
        return mask

    def _inference_path(self, input_ids, images_clip, attention_masks):
        length = input_ids.shape[0]
        images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=images_clip_extend[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, images_clip, input_ids, labels, attention_masks, offset):
        clip_images = self._prepare_clip_image(images_clip, offset)
        
        output = super().forward(
            images=clip_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True)
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_clip_image(self, images_clip, offset):
        clip_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = images_clip[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            clip_image_list.append(images_clip_i)
        return torch.cat(clip_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, face_token_mask, offset, infer=False):
        
        first_param = next(self.model.text_hidden_fcs[0].parameters())
        device = first_param.device
        dtype = first_param.dtype
        output_hidden_states = [hidden.to(device=device, dtype=dtype) for hidden in output_hidden_states]
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        face_token_mask = face_token_mask.to(device)
        pred_embeddings = last_hidden_state[face_token_mask]
       
        face_token_counts = face_token_mask.int().sum(-1)

        face_token_offset = face_token_counts.cumsum(-1)
        face_token_offset = torch.cat([torch.zeros(1, device = device).long(), face_token_offset], dim=0)
        
        if not infer:
            face_token_offset = face_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(face_token_offset) - 1):
            start_i, end_i = face_token_offset[i], face_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _calculate_losses(self, pred_exps, gt_exps, output):
        loss_components = self._compute_loss_components(pred_exps, gt_exps, output)
        return loss_components

    def _compute_loss_components(self, pred_exps, gt_exps, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        face_loss = torch.tensor(0.0, device=ce_loss.device)
        num_faces = 0

        if pred_exps is not None and len(pred_exps) > 0: 
            # Iterate over batch and compute mask-related losses
            for batch_idx, pred_exp in enumerate(pred_exps):
                if pred_exps.numel() > 0:  # Ensure pred_mask is not empty
                    gt_exp = gt_exps[batch_idx]
                    assert gt_exp.shape[0] == pred_exp.shape[
                        0], f"Shape mismatch: gt_exp {gt_exp.shape}, pred_mask {pred_exp.shape}"
                    face_loss += (face_loss_function(pred_exp, gt_exp, face_count=gt_exp.shape[0]) *
                                      gt_exp.shape[0])
                    num_faces += gt_exp.shape[0]    
        # Normalize the losses
        face_loss = self.face_loss_weight * face_loss / (num_faces + 1e-8)

        # Aggregate all loss components
        total_loss = ce_loss + face_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "face_loss": face_loss}

    def evaluate(self, input_ids, images, tokenizer, max_new_tokens=100,**kwargs):
        with torch.no_grad():
         
            generate_kwargs = {
                'input_ids': input_ids, 
                'images': images,
                'max_new_tokens': max_new_tokens,
                'num_beams': 1, 
                'output_hidden_states': True, 
                'return_dict_in_generate': True,
                'eos_token_id': [tokenizer.eos_token_id, self.face_token_idx],
                'pad_token_id': tokenizer.pad_token_id,
                'do_sample': False}
            
            generation_outputs = self.generate(**generate_kwargs)
            output_hidden_states = generation_outputs.hidden_states
                
            generated_output_ids = generation_outputs.sequences

            face_token_mask = generated_output_ids[:, 1:] == self.face_token_idx
            face_token_mask[:,:input_ids.shape[1]] = False    
           
            image_embeds_len = 575
            if IMAGE_TOKEN_INDEX in input_ids:
                new_face_token_mask = torch.zeros([face_token_mask.shape[0], face_token_mask.shape[1]+image_embeds_len]).bool().cuda()
                for i in range(input_ids.shape[0]):
                    if IMAGE_TOKEN_INDEX in input_ids[i]:
                        new_face_token_mask[i, image_embeds_len:] = face_token_mask[i]
                    else:
                        new_face_token_mask[i, :face_token_mask.shape[1]] = face_token_mask[i]
                face_token_mask = new_face_token_mask

            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, face_token_mask, None, infer=True
            )
           
        return generated_output_ids, predicted_embeddings
