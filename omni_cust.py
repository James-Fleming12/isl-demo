from copy import deepcopy
import os
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Any, Dict, Optional

from diffusers.loaders import PeftAdapterMixin
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from OmniGenCode.OmniGen.train_helper.loss import mean_flat, sample_timestep, sample_x0
from OmniGenCode.OmniGen.transformer import Phi3Config

from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image

from cust_phi import BlockPhi3Transformer

class JsonFolderDataset(Dataset):
    def __init__(self, folder_path, processor, image_transform=None, max_input_length=1024, small_subset=False):
        """
        folder_path: folder containing JSON files and PNGs
        processor: OmniGenProcessor
        image_transform: optional transforms to apply to the images
        """
        samples_loaded = 1
        max_samples = 1

        self.folder_path = folder_path
        self.processor = processor
        self.image_transform = image_transform
        self.max_input_length = max_input_length

        self.json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
        if not self.json_files:
            raise ValueError("No JSON files found in folder")

        self.data = []
        for jf in self.json_files:
            if small_subset and samples_loaded > max_samples:
                break
            with open(os.path.join(folder_path, jf), "r") as f:
                item = json.load(f)
                self.data.append(item)
            samples_loaded+=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item.get("TEXT") or item.get("caption")
        if text is None:
            raise ValueError(f"No text found for index {idx}")

        key = item["key"]
        image_path = os.path.join(self.folder_path, f"{key}.png")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512), resample=Image.BICUBIC)
        if self.image_transform:
            image = self.image_transform(image)

        model_input = self.processor.process_multi_modal_prompt(text, None)

        return model_input, image

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=torch.float32):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PatchEmbedMR(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 2,
            in_chans: int = 4,
            embed_dim: int = 768,
            bias: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x

class CustomOmniGen(nn.Module, PeftAdapterMixin):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        transformer_config: Phi3Config,
        patch_size=2,
        in_channels=4,
        pe_interpolation: float = 1.0,
        pos_embed_max_size: int = 192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.pos_embed_max_size = pos_embed_max_size

        hidden_size = transformer_config.hidden_size

        self.x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, bias=True)
        self.input_x_embedder = PatchEmbedMR(patch_size, in_channels, hidden_size, bias=True)

        self.time_token = TimestepEmbedder(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.pe_interpolation = pe_interpolation
        pos_embed = get_2d_sincos_pos_embed(hidden_size, pos_embed_max_size, interpolation_scale=self.pe_interpolation, base_size=64)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        self.llm = BlockPhi3Transformer(config=transformer_config)
        self.llm.config.use_cache = False

        self.num_layers = transformer_config.num_hidden_layers + 1
    
    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        config = Phi3Config.from_pretrained(model_name)
        model = cls(config)
        if os.path.exists(os.path.join(model_name, 'model.safetensors')):
            print("Loading safetensors")
            ckpt = load_file(os.path.join(model_name, 'model.safetensors'))
        else:
            ckpt = torch.load(os.path.join(model_name, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt)
        return model

    @staticmethod
    def _map_omni_to_custom_state_dict(omni_state_dict: Dict[str, Any], custom_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps OmniGen checkpoint (with Phi3Model.layers) to CustomOmniGen 
        (with QuarterBlockPhi3Transformer.block1/2/3/4).
        
        Args:
            omni_state_dict: State dict from original OmniGen checkpoint
            custom_state_dict: State dict from CustomOmniGen model (used for structure reference)
        
        Returns:
            Mapped state dict compatible with CustomOmniGen
        """
        mapped_state_dict = {}

        target_format = None
        has_blocks_modulelist = False
        has_blocks_direct = False
        
        for key in custom_state_dict.keys():
            if 'llm.blocks.' in key and not has_blocks_modulelist:
                parts = key.split('.')
                if len(parts) >= 3 and parts[1] == 'blocks' and parts[2].isdigit():
                    has_blocks_modulelist = True
            elif any(f'llm.block{i}.' in key for i in range(1, 5)):
                has_blocks_direct = True

        if has_blocks_modulelist:
            target_format = 'blocks_modulelist'
        elif has_blocks_direct:
            target_format = 'blocks_direct'
        
        if target_format is None:
            print("Warning: Could not determine target format from custom_state_dict")
            return omni_state_dict
        
        print(f"Detected target format: {target_format}")

        num_layers = None
        for key in omni_state_dict.keys():
            if key.startswith('llm.layers.'):
                layer_idx = int(key.split('.')[2])
                if num_layers is None or layer_idx >= num_layers:
                    num_layers = layer_idx + 1
        
        if num_layers is None:
            print("Warning: Could not determine number of layers, attempting direct mapping")
            return omni_state_dict

        print(f"Mapping {num_layers} layers (1 per block)")

        for key, value in omni_state_dict.items():
            if key.startswith('llm.layers.'):
                parts = key.split('.')
                layer_idx = int(parts[2])
                param_path = '.'.join(parts[3:])

                block_idx = layer_idx
                new_idx = 0

                if target_format == 'blocks_modulelist':
                    new_key = f'llm.blocks.{block_idx}.{new_idx}.{param_path}'
                    mapped_state_dict[new_key] = value
                else:
                    new_key = f'llm.blocks.{block_idx}.{new_idx}.{param_path}'
                    mapped_state_dict[new_key] = value
                    
            elif key.startswith('llm.'):
                mapped_state_dict[key] = value
            else:
                mapped_state_dict[key] = value

        mapped_layer_keys = [k for k in mapped_state_dict.keys() if 'llm.block' in k]

        missing_keys = set(custom_state_dict.keys()) - set(mapped_state_dict.keys())
        unexpected_keys = set(mapped_state_dict.keys()) - set(custom_state_dict.keys())
        
        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys in mapped state dict")
            print(f"First few missing keys: {list(missing_keys)[:5]}")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys in mapped state dict")
            print(f"First few unexpected keys: {list(unexpected_keys)[:5]}")
        
        return mapped_state_dict

    @classmethod
    def from_pretrained_other(cls, model_name: str, map_llm_params: bool = True, strict: bool = True):
        """
        Load OmniGen checkpoint into CustomOmniGen model.
        
        Args:
            model_name: Path or repo ID of OmniGen model
            map_llm_params: If True, map llm.layers to llm.blockN structure
            strict: Whether to strictly enforce state dict key matching
        
        Returns:
            CustomOmniGen model loaded with weights from OmniGen checkpoint
        """
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(
                repo_id=model_name, 
                cache_dir=cache_folder, 
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
        config = Phi3Config.from_pretrained(model_name)
        model = cls(config)

        if os.path.exists(os.path.join(model_name, 'model.safetensors')):
            print("Loading safetensors from OmniGen snapshot")
            ckpt = load_file(os.path.join(model_name, 'model.safetensors'))
        else:
            ckpt = torch.load(os.path.join(model_name, 'model.pt'), map_location='cpu')
        
        if not map_llm_params:
            try:
                model.load_state_dict(ckpt, strict=strict)
            except RuntimeError as e:
                print(f"Direct loading failed: {e}")
                print("Try with map_llm_params=True to map transformer weights")
                raise
        else:
            mapped_ckpt = cls._map_omni_to_custom_state_dict(ckpt, model.state_dict())
            model.load_state_dict(mapped_ckpt, strict=strict)
        
        return model

    def initialize_weights(self):
        assert not hasattr(self, "llama")

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.input_x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.input_x_embedder.proj.bias, 0)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.time_token.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_token.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels

        x = x.reshape(shape=(x.shape[0], h//self.patch_size, w//self.patch_size, self.patch_size, self.patch_size, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs


    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed


    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images:bool=False):
        if isinstance(latents, list):
            return_list = False
            if padding_latent is None:
                padding_latent = [None] * len(latents)
                return_list = True
            patched_latents, num_tokens, shapes = [], [], []
            for latent, padding in zip(latents, padding_latent):
                height, width = latent.shape[-2:]
                if is_input_images:
                    latent = self.input_x_embedder(latent)
                else:
                    latent = self.x_embedder(latent)
                pos_embed = self.cropped_pos_embed(height, width)    
                latent = latent + pos_embed
                if padding is not None:
                    latent = torch.cat([latent, padding], dim=-2)
                patched_latents.append(latent)

                num_tokens.append(pos_embed.size(1))
                shapes.append([height, width])
            if not return_list:
                latents = torch.cat(patched_latents, dim=0)
            else:
                latents = patched_latents
        else:
            height, width = latents.shape[-2:]
            if is_input_images:
                latents = self.input_x_embedder(latents)
            else:
                latents = self.x_embedder(latents)
            pos_embed = self.cropped_pos_embed(height, width)  
            latents = latents + pos_embed
            num_tokens = latents.size(1)
            shapes = [height, width]
        return latents, num_tokens, shapes

    
    def forward(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, padding_latent=None, past_key_values=None, return_past_key_values=True, offload_model:bool=False):
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        time_token = self.time_token(timestep, dtype=x[0].dtype).unsqueeze(1)   
        
        if input_img_latents is not None:
            input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
        if input_ids is not None:
            condition_embeds = self.llm.embed_tokens(input_ids).clone()
            input_img_inx = 0
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
                    input_img_inx += 1
            if input_img_latents is not None:
                assert input_img_inx == len(input_latents) 

            input_emb = torch.cat([condition_embeds, time_token, x], dim=1)
        else:
            input_emb = torch.cat([time_token, x], dim=1)

        # for index, i in enumerate(block_inputs):
        #     if i.dim() == 5 and i.shape[1] == 1: # problem with stacking
        #         block_inputs[index] = i.squeeze(1)

        # for index, i in enumerate(block_inputs):
        #     block_inputs[index] = self.x_embedder(i)

        batch_size = timestep.size(0)
        # num_blocks = len(self.llm.blocks)
        block_timesteps = []
        num_blocks = len(self.llm.blocks)
        for b in range(batch_size):
            t_schedule = torch.linspace(1.0, 0.0, num_blocks, device=timestep.device, dtype=timestep.dtype)
            t_schedule *= timestep[b]
            block_timesteps.append(t_schedule)

        block_timesteps = torch.stack(block_timesteps, dim=0)

        output = self.llm(inputs_embeds=input_emb, block_timesteps=block_timesteps, num_tokens=num_tokens, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model, output_hidden_states=True)
        hidden_states = output.hidden_states
        output, past_key_values = output.last_hidden_state, output.past_key_values
        if input_is_list:
            image_embedding = output[:, -max(num_tokens):]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = []
            for i in range(x.size(0)):
                latent = x[i:i+1, :num_tokens[i]]
                latent = self.unpatchify(latent, shapes[i][0], shapes[i][1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])

        layer_image_embeddings = []
        for layer_hidden_state in hidden_states:
            if input_is_list:
                layer_image_embedding = layer_hidden_state[:, -max(num_tokens):]
            else:
                layer_image_embedding = layer_hidden_state[:, -num_tokens:]
            layer_image_embeddings.append(layer_image_embedding)

        projected_hidden_states = []
        batch_size = timestep.size(0)
        hidden_timesteps = torch.zeros((batch_size, self.num_layers), device=timestep.device, dtype=timestep.dtype)
        for b in range(batch_size):
            hidden_timesteps[b] = torch.linspace(float(timestep[b]), 0, self.num_layers, device=timestep.device, dtype=timestep.dtype)

        time_embs = []
        for layer_idx in range(self.num_layers):
            layer_t = hidden_timesteps[:, layer_idx]
            time_emb = self.t_embedder(layer_t, dtype=x.dtype)
            time_embs.append(time_emb)

        for i, layer_image_embedding in enumerate(layer_image_embeddings):
            projected = self.final_layer(layer_image_embedding, time_embs[i])
            projected_hidden_states.append(projected)

        unpatched_hidden_states = []
        for i, projected in enumerate(projected_hidden_states):
            if input_is_list:
                latents_per_layer = []
                for j in range(projected.size(0)):
                    latent = projected[j:j+1, :num_tokens[j]]
                    latent_unpatched = self.unpatchify(latent, shapes[j][0], shapes[j][1])
                    latents_per_layer.append(latent_unpatched)
                unpatched_hidden_states.append(latents_per_layer)
            else:
                latent_unpatched = self.unpatchify(projected, shapes[0], shapes[1])
                unpatched_hidden_states.append(latent_unpatched)

        if return_past_key_values:
            return latents, past_key_values
        return latents, unpatched_hidden_states
    
    def inference(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, padding_latent=None, past_key_values=None, return_past_key_values=True, offload_model:bool=False):
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        time_token = self.time_token(timestep, dtype=x[0].dtype).unsqueeze(1)   
        
        if input_img_latents is not None:
            input_latents, _, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
        if input_ids is not None:
            condition_embeds = self.llm.embed_tokens(input_ids).clone()
            input_img_inx = 0
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    condition_embeds[b_inx, start_inx: end_inx] = input_latents[input_img_inx]
                    input_img_inx += 1
            if input_img_latents is not None:
                assert input_img_inx == len(input_latents) 

            input_emb = torch.cat([condition_embeds, time_token, x], dim=1)
        else:
            input_emb = torch.cat([time_token, x], dim=1)

        output = self.llm.inference(inputs_embeds=input_emb, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model, output_hidden_states=True)
        hidden_states = output.hidden_states
        output, past_key_values = output.last_hidden_state, output.past_key_values
        if input_is_list:
            image_embedding = output[:, -max(num_tokens):]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = []
            for i in range(x.size(0)):
                latent = x[i:i+1, :num_tokens[i]]
                latent = self.unpatchify(latent, shapes[i][0], shapes[i][1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])

        layer_image_embeddings = []
        for layer_hidden_state in hidden_states:
            if input_is_list:
                layer_image_embedding = layer_hidden_state[:, -max(num_tokens):]
            else:
                layer_image_embedding = layer_hidden_state[:, -num_tokens:]
            layer_image_embeddings.append(layer_image_embedding)

        projected_hidden_states = []
        batch_size = timestep.size(0)
        hidden_timesteps = torch.zeros((batch_size, self.num_layers), device=timestep.device, dtype=timestep.dtype)
        for b in range(batch_size):
            hidden_timesteps[b] = torch.linspace(float(timestep[b]), 0, self.num_layers, device=timestep.device, dtype=timestep.dtype)

        time_embs = []
        for layer_idx in range(self.num_layers):
            layer_t = hidden_timesteps[:, layer_idx]
            time_emb = self.t_embedder(layer_t, dtype=x.dtype)
            time_embs.append(time_emb)

        for i, layer_image_embedding in enumerate(layer_image_embeddings):
            projected = self.final_layer(layer_image_embedding, time_embs[i])
            projected_hidden_states.append(projected)

        unpatched_hidden_states = []
        for i, projected in enumerate(projected_hidden_states):
            if input_is_list:
                latents_per_layer = []
                for j in range(projected.size(0)):
                    latent = projected[j:j+1, :num_tokens[j]]
                    latent_unpatched = self.unpatchify(latent, shapes[j][0], shapes[j][1])
                    latents_per_layer.append(latent_unpatched)
                unpatched_hidden_states.append(latents_per_layer)
            else:
                latent_unpatched = self.unpatchify(projected, shapes[0], shapes[1])
                unpatched_hidden_states.append(latent_unpatched)

        if return_past_key_values:
            return latents, past_key_values
        return latents, unpatched_hidden_states

    @torch.no_grad()
    def forward_with_cfg(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values, use_kv_cache, offload_model):      
        self.llm.config.use_cache = use_kv_cache
        model_out, past_key_values = self.forward(x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, past_key_values=past_key_values, return_past_key_values=True, offload_model=offload_model)
        if use_img_cfg:
            cond, uncond, img_cond = torch.split(model_out, len(model_out) // 3, dim=0)
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        else:
            cond, uncond = torch.split(model_out, len(model_out) // 2, dim=0)
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
        
        return torch.cat(model_out, dim=0), past_key_values


    @torch.no_grad()
    def forward_with_separate_cfg(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, cfg_scale, use_img_cfg, img_cfg_scale, past_key_values, use_kv_cache, offload_model):
        self.llm.config.use_cache = use_kv_cache
        if past_key_values is None:
            past_key_values = [None] * len(attention_mask)

        x = torch.split(x, len(x) // len(attention_mask), dim=0)
        timestep = timestep.to(x[0].dtype)
        timestep = torch.split(timestep, len(timestep) // len(input_ids), dim=0)

        model_out, pask_key_values = [], []
        for i in range(len(input_ids)):
            temp_out, temp_pask_key_values = self.forward(x[i], timestep[i], input_ids[i], input_img_latents[i], input_image_sizes[i], attention_mask[i], position_ids[i], past_key_values=past_key_values[i], return_past_key_values=True, offload_model=offload_model)
            model_out.append(temp_out)
            pask_key_values.append(temp_pask_key_values)

        if len(model_out) == 3:
            cond, uncond, img_cond = model_out
            cond = uncond + img_cfg_scale * (img_cond - uncond) + cfg_scale * (cond - img_cond)
            model_out = [cond, cond, cond]
        elif len(model_out) == 2:
            cond, uncond = model_out
            cond = uncond + cfg_scale * (cond - uncond)
            model_out = [cond, cond]
        else:
            return model_out[0]
        
        return torch.cat(model_out, dim=0), pask_key_values
    
    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        input_img_latents: Optional[torch.Tensor],
        input_image_sizes: dict,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate image using single-pass progressive refinement.
        Each block progressively denoises the image.
        """
        B = x.shape[0] if not isinstance(x, list) else len(x)
        device = x.device if not isinstance(x, list) else x[0].device

        timestep = torch.ones((B,), device=device, dtype=torch.float32)

        final_pred, intermediate_preds = self.forward(
            x=x,
            timestep=timestep,
            input_ids=input_ids,
            input_img_latents=input_img_latents,
            input_image_sizes=input_image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_latent=None,
            past_key_values=None,
            return_past_key_values=False,
            offload_model=False
        )

        current = x
        num_blocks = self.num_layers - 1
        block_indices = list(range(num_blocks)) + [-1]
        
        intermediate_results = []

        for layer_idx in range(num_blocks):
            denoised_pred = intermediate_preds[layer_idx]
            
            if isinstance(current, list):
                current = [denoised_pred[i] for i in range(len(current))]
            else:
                current = denoised_pred

            intermediate_results.append(deepcopy(current))

        if isinstance(current, list):
            current = [final_pred[i] for i in range(len(current))]
        else:
            current = final_pred
        intermediate_results.append(deepcopy(current))

def isl_training_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None):
    """Loss for training the score model
    Args:
    - model: DeepSpeed Model Engine
    - x1: clean datapoint (can be list of tensors or tensor)
    - model_kwargs: additional arguments for torch model

    Trains the model to have each block predict a quarter of the movement
    """
    if model_kwargs == None:
        model_kwargs = {}
    
    if isinstance(x1, list):
        if x1[0].dim() == 4:
            x1 = torch.cat(x1, dim=0)
        else:
            x1 = torch.stack(x1, dim=0)

    device = x1.device
    model_dtype = next(model.parameters()).dtype

    B = x1.shape[0]
    x0 = sample_x0(x1)

    x0 = x0.to(model_dtype)
    x1 = x1.to(model_dtype)

    if isinstance(x0, list):
        if x0[0].dim() == 4:
            x0 = torch.cat(x0, dim=0)
        else:
            x0 = torch.stack(x0, dim=0)

    # t = sample_timestep(x1)
    t = torch.ones(B).to(device)
    t = t.to(model_dtype)

    xt = t.view(-1,1,1,1) * x0 + (1 - t.view(-1,1,1,1)) * x1
    xt = xt.to(model_dtype)

    num_layers = model.module.num_layers # changed for deepspeed
    num_transformer_layers = num_layers - 1 # exclude final layer
    intermediate_layer_indices = list(range(num_transformer_layers))
    intermediate_noise_levels = [1.0 - (i+1)/(num_transformer_layers+1) for i in range(num_transformer_layers)]
    model_output, hidden_states = model(xt, t, **model_kwargs)

    if isinstance(model_output, list):
        if model_output[0].dim() == 4:
            model_output = torch.cat(model_output, dim=0)
        else:
            model_output = torch.stack(model_output, dim=0)

    terms = {}
    total_loss = 0.0

    if patch_weight is not None:
        main_loss = torch.stack(
            [((x1[i] - model_output[i]) ** 2 * patch_weight[i]).mean() for i in range(B)],
            dim=0,
        )
    else:
        main_loss = torch.stack(
            [((x1[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        )

    intermediate_losses = []
    for index, layer_idx in enumerate(intermediate_layer_indices):
        hidden_state = hidden_states[layer_idx]
        effective_t = t.view(-1,1,1,1) * (1 - intermediate_noise_levels[index])
        target = effective_t * x0 + (1 - effective_t) * x1

        if isinstance(hidden_state, list):
            if hidden_state[0].dim() == 4:
                hidden_state = torch.cat(hidden_state, dim=0)
            else:
                hidden_state = torch.stack(hidden_state, dim=0)

        if patch_weight is not None:
            layer_loss = torch.stack(
                [((target[i] - hidden_state[i]) ** 2 * patch_weight[i]).mean() for i in range(B)],
                dim=0,
            )
        else:
            layer_loss = torch.stack(
                [((target[i] - hidden_state[i]) ** 2).mean() for i in range(B)],
                dim=0,
            )
        intermediate_losses.append(layer_loss)

    total_loss = main_loss + sum(intermediate_losses)
    terms["loss"] = total_loss.mean()
    terms["main_loss"] = main_loss.mean()
    terms["intermediate_loss"] = sum([loss.mean() for loss in intermediate_losses])
    
    return terms