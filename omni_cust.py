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
from torchvision import transforms
import json
from PIL import Image

from cust_phi import BlockPhi3Transformer
from diffusers import DDIMScheduler

class JsonFolderDataset(Dataset):
    def __init__(self, folder_path, processor, vae=None, device='cuda', 
                 image_transform=None, max_input_length=1024, small_subset=False,
                 use_preencoded=True, latents_folder=None):
        """
        folder_path: folder containing JSON files and PNGs
        processor: OmniGenProcessor
        vae: VAE model for encoding (only needed if latents don't exist)
        device: device for VAE encoding
        image_transform: optional transforms to apply to the images
        use_preencoded: if True, use pre-encoded latents; if False, encode on-the-fly
        latents_folder: folder to save/load latents (defaults to folder_path/latents)
        """
        samples_loaded = 1
        max_samples = 1000

        self.folder_path = folder_path
        self.processor = processor
        self.image_transform = image_transform
        self.max_input_length = max_input_length
        self.use_preencoded = use_preencoded

        if latents_folder is None:
            self.latents_folder = os.path.join(folder_path, "latents")
        else:
            self.latents_folder = latents_folder
        
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
            samples_loaded += 1

        if self.use_preencoded:
            self._setup_latents(vae, device)

    def _setup_latents(self, vae, device):
        """Pre-encode all images and save latents to disk"""
        os.makedirs(self.latents_folder, exist_ok=True)

        missing_latents = []
        for item in self.data:
            key = item["key"]
            latent_path = os.path.join(self.latents_folder, f"{key}.pt")
            if not os.path.exists(latent_path):
                missing_latents.append((key, item))
        
        if not missing_latents:
            print(f"All {len(self.data)} latents already exist in {self.latents_folder}")
            return
        
        if vae is None:
            raise ValueError(
                f"VAE is required to encode {len(missing_latents)} missing latents. "
                "Either provide a VAE model or set use_preencoded=False"
            )
        
        print(f"Encoding {len(missing_latents)} images to latents...")
        vae.eval()
        vae.to(device)

        with torch.no_grad():
            for i, (key, item) in enumerate(missing_latents):
                if i % 100 == 0:
                    print(f"Encoding {i}/{len(missing_latents)}...")

                image_path = os.path.join(self.folder_path, f"{key}.png")
                image = Image.open(image_path).convert("RGB")
                image = image.resize((512, 512), resample=Image.BICUBIC)
                
                if self.image_transform:
                    image = self.image_transform(image)
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])
                    image = transform(image)
                
                image = image.unsqueeze(0).to(device)

                latent = vae.encode(image).latent_dist.sample()
                latent_scaled = latent * vae.config.scaling_factor
                latent_scaled = latent_scaled.squeeze(0)

                latent_path = os.path.join(self.latents_folder, f"{key}.pt")
                torch.save(latent_scaled.cpu(), latent_path)
        
        print(f"Finished encoding all latents to {self.latents_folder}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item.get("TEXT") or item.get("caption")
        if text is None:
            raise ValueError(f"No text found for index {idx}")

        key = item["key"]
        
        if self.use_preencoded:
            latent_path = os.path.join(self.latents_folder, f"{key}.pt")
            image = torch.load(latent_path)

            original_height = image.shape[-2] * 8 # for the collator
            original_width = image.shape[-1] * 8
        else:
            image_path = os.path.join(self.folder_path, f"{key}.png")
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512), resample=Image.BICUBIC)
            if self.image_transform:
                image = self.image_transform(image)
            original_height = 512
            original_width = 512

        model_input = self.processor.process_multi_modal_prompt(text, None)

        return model_input, image, (original_height, original_width)

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

        self.llm = BlockPhi3Transformer(transformer_config)
        self.llm.config.use_cache = False

        self.num_layers = transformer_config.num_hidden_layers + 1
    
    @classmethod
    def from_pretrained(cls, model_name: str, cache_dir: str = None):
        """
        Load CustomOmniGen checkpoint with caching support.
        
        Args:
            model_name: Path or repo ID of the model
            cache_dir: Optional cache directory (defaults to HF_HUB_CACHE env var)
        
        Returns:
            CustomOmniGen model loaded with weights
        """
        if cache_dir is None:
            cache_dir = os.getenv('HF_HUB_CACHE')

        if not os.path.exists(model_name):
            print(f"Downloading model from {model_name}...")
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
            print(f"Model cached at: {model_name}")
        else:
            print(f"Loading model from local path: {model_name}")

        config = Phi3Config.from_pretrained(model_name)
        model = cls(config)

        safetensors_path = os.path.join(model_name, 'model.safetensors')
        pt_path = os.path.join(model_name, 'model.pt')
        
        if os.path.exists(safetensors_path):
            print("Loading safetensors checkpoint")
            ckpt = load_file(safetensors_path)
        elif os.path.exists(pt_path):
            print("Loading PyTorch checkpoint")
            ckpt = torch.load(pt_path, map_location='cpu')
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {model_name}. "
                f"Expected either 'model.safetensors' or 'model.pt'"
            )

        model.load_state_dict(ckpt)
        print("Model loaded successfully")
        
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
            
        batch_size = timestep.size(0)
        block_timesteps = []
        num_blocks = len(self.llm.layers)

        t_schedule = torch.zeros(num_blocks + 1, device=timestep.device, dtype=timestep.dtype)
        t_schedule[0] = 1.0
        for i in range(num_blocks):
            t_schedule[i+1] = 1.0 - (i+1)/(num_blocks+1)
        t_schedule = t_schedule[:-1]
        
        block_timesteps = t_schedule.unsqueeze(0).expand(batch_size, -1)

        output = self.llm(inputs_embeds=input_emb, t_embedder=self.t_embedder, block_timesteps=block_timesteps, num_tokens=num_tokens, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model, output_hidden_states=True)
        hidden_states = output.hidden_states
        output, past_key_values = output.last_hidden_state, output.past_key_values

        if input_is_list:
            max_tokens = max(num_tokens)
            image_embedding = output[:, -max_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)

            latents = []
            for i, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                latent = x[i:i+1, :nt]
                latent = self.unpatchify(latent, shape[0], shape[1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])

        if input_is_list:
            max_tokens = max(num_tokens)
            layer_image_embeddings = [layer_hidden_state[:, -max_tokens:] for layer_hidden_state in hidden_states]
        else:
            layer_image_embeddings = [layer_hidden_state[:, -num_tokens:] for layer_hidden_state in hidden_states]

        num_layers = self.num_layers
        layer_idx_tensor = torch.arange(num_layers, device=timestep.device, dtype=timestep.dtype)
        hidden_t_schedule = timestep.unsqueeze(1) * (1.0 - layer_idx_tensor[:-1].unsqueeze(0) / (num_blocks + 1))
        last_layer = torch.zeros(batch_size, 1, device=timestep.device, dtype=timestep.dtype)
        hidden_timesteps = torch.cat([hidden_t_schedule, last_layer], dim=1)

        all_times = hidden_timesteps.flatten()
        all_time_embs = self.t_embedder(all_times, dtype=x.dtype)
        time_embs = all_time_embs.view(batch_size, num_layers, -1)

        projected_hidden_states = []
        for i, (layer_embedding, layer_time_emb) in enumerate(zip(layer_image_embeddings, time_embs.unbind(dim=1))):
            projected = self.final_layer(layer_embedding, layer_time_emb)
            projected_hidden_states.append(projected)

        unpatched_hidden_states = []
        for i, projected in enumerate(projected_hidden_states):
            if input_is_list:
                latents_per_layer = []
                for j, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                    latent = projected[j:j+1, :nt]
                    latent_unpatched = self.unpatchify(latent, shape[0], shape[1])
                    latents_per_layer.append(latent_unpatched)
                unpatched_hidden_states.append(latents_per_layer)
            else:
                latent_unpatched = self.unpatchify(projected, shapes[0], shapes[1])
                unpatched_hidden_states.append(latent_unpatched)

        if return_past_key_values:
            return latents, past_key_values
        return latents, unpatched_hidden_states
    
    def scheduled(self, x, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, padding_latent=None, past_key_values=None, return_past_key_values=True, offload_model:bool=False):
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="sample", # for x0 prediction
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
        )
        scheduler.set_timesteps(self.num_layers)
        timestep = scheduler.timesteps[0].to(x.device)
        timestep = torch.full((x.shape[0],), timestep, device=x.device)

        time_token = self.time_token(timestep, dtype=x.dtype).unsqueeze(1)
        
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
            
        batch_size = timestep.size(0)

        output = self.llm.scheduled(inputs_embeds=input_emb, t_embedder=self.t_embedder, scheduler=scheduler, num_tokens=num_tokens, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, offload_model=offload_model, output_hidden_states=True)
        hidden_states = output.hidden_states
        output, past_key_values = output.last_hidden_state, output.past_key_values

        if input_is_list:
            max_tokens = max(num_tokens)
            image_embedding = output[:, -max_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)

            latents = []
            for i, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                latent = x[i:i+1, :nt]
                latent = self.unpatchify(latent, shape[0], shape[1])
                latents.append(latent)
        else:
            image_embedding = output[:, -num_tokens:]
            time_emb = self.t_embedder(timestep, dtype=x.dtype)
            x = self.final_layer(image_embedding, time_emb)
            latents = self.unpatchify(x, shapes[0], shapes[1])

        if input_is_list:
            max_tokens = max(num_tokens)
            layer_image_embeddings = [layer_hidden_state[:, -max_tokens:] for layer_hidden_state in hidden_states]
        else:
            layer_image_embeddings = [layer_hidden_state[:, -num_tokens:] for layer_hidden_state in hidden_states]

        num_layers = self.num_layers

        hidden_t_schedule = scheduler.timesteps.to(device=x.device)
        all_timesteps = hidden_t_schedule.unsqueeze(0).expand(batch_size, -1).flatten()
        all_time_embs = self.t_embedder(all_timesteps, dtype=x[0].dtype if input_is_list else x.dtype)
        time_embs = all_time_embs.view(batch_size, num_layers, -1)

        projected_hidden_states = []
        for i, (layer_embedding, layer_time_emb) in enumerate(zip(layer_image_embeddings, time_embs.unbind(dim=1))):
            projected = self.final_layer(layer_embedding, layer_time_emb)
            projected_hidden_states.append(projected)

        unpatched_hidden_states = []
        for i, projected in enumerate(projected_hidden_states):
            if input_is_list:
                latents_per_layer = []
                for j, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                    latent = projected[j:j+1, :nt]
                    latent_unpatched = self.unpatchify(latent, shape[0], shape[1])
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
    ):
        B = x.shape[0]
        device = x.device

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
            offload_model=False,
        )

        intermediate_results = [pred.clone() for pred in intermediate_preds]

        return final_pred, intermediate_results
    
    @torch.no_grad()
    def scheduled_generate(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        input_img_latents: Optional[torch.Tensor],
        input_image_sizes: dict,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ):
        B = x.shape[0]
        device = x.device

        final_pred, intermediate_preds = self.scheduled(
            x=x,
            input_ids=input_ids,
            input_img_latents=input_img_latents,
            input_image_sizes=input_image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_latent=None,
            past_key_values=None,
            return_past_key_values=False,
            offload_model=False,
        )

        intermediate_results = [pred.clone() for pred in intermediate_preds]

        return final_pred, intermediate_results

class EffISLOmniGen(nn.Module, PeftAdapterMixin):
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

        self.llm = BlockPhi3Transformer(transformer_config)
        self.llm.config.use_cache = False

        self.num_layers = transformer_config.num_hidden_layers + 1
    
    @classmethod
    def from_pretrained(cls, model_name: str, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.getenv('HF_HUB_CACHE')

        if not os.path.exists(model_name):
            print(f"Downloading model from {model_name}...")
            model_name = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
            print(f"Model cached at: {model_name}")
        else:
            print(f"Loading model from local path: {model_name}")

        config = Phi3Config.from_pretrained(model_name)
        model = cls(config)

        safetensors_path = os.path.join(model_name, 'model.safetensors')
        pt_path = os.path.join(model_name, 'model.pt')
        
        if os.path.exists(safetensors_path):
            print("Loading safetensors checkpoint")
            ckpt = load_file(safetensors_path)
        elif os.path.exists(pt_path):
            print("Loading PyTorch checkpoint")
            ckpt = torch.load(pt_path, map_location='cpu')
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {model_name}. "
                f"Expected either 'model.safetensors' or 'model.pt'"
            )

        model.load_state_dict(ckpt)
        print("Model loaded successfully")
        
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
    
    def _process_hidden_state_to_image(self, hidden_state, num_tokens, shapes, layer_time_emb, input_is_list):
        """
        Convert a single hidden state to image space
        """
        if input_is_list:
            max_tokens = max(num_tokens)
            layer_embedding = hidden_state[:, -max_tokens:]
        else:
            layer_embedding = hidden_state[:, -num_tokens:]

        projected = self.final_layer(layer_embedding, layer_time_emb)

        if input_is_list:
            latents_per_layer = []
            for j, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                latent = projected[j:j+1, :nt]
                latent_unpatched = self.unpatchify(latent, shape[0], shape[1])
                latents_per_layer.append(latent_unpatched)
            return latents_per_layer
        else:
            return self.unpatchify(projected, shapes[0], shapes[1])

    def forward(self, x, timestep, input_ids, input_img_latents, input_image_sizes, attention_mask, position_ids, padding_latent=None, past_key_values=None, return_past_key_values=True, offload_model:bool=False):
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        
        if input_is_list:
            time_token = self.time_token(timestep, dtype=x[0].dtype).unsqueeze(1)
        else:
            time_token = self.time_token(timestep, dtype=x.dtype).unsqueeze(1)

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

        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = input_emb.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).to(input_emb.dtype)
        
        batch_size = timestep.size(0)
        num_blocks = len(self.llm.layers)

        layer_idx_tensor = torch.arange(self.num_layers, device=timestep.device, dtype=timestep.dtype)
        hidden_t_schedule = timestep.unsqueeze(1) * (1.0 - layer_idx_tensor[:-1].unsqueeze(0) / (num_blocks + 1))
        last_layer = torch.zeros(batch_size, 1, device=timestep.device, dtype=timestep.dtype)
        hidden_timesteps = torch.cat([hidden_t_schedule, last_layer], dim=1)

        all_times = hidden_timesteps.flatten()
        all_time_embs = self.t_embedder(all_times, dtype=input_emb.dtype)
        time_embs = all_time_embs.view(batch_size, self.num_layers, -1)

        t_schedule = torch.zeros(num_blocks + 1, device=timestep.device, dtype=timestep.dtype)
        t_schedule[0] = 1.0
        for i in range(num_blocks):
            t_schedule[i+1] = 1.0 - (i+1)/(num_blocks+1)
        t_schedule = t_schedule[:-1]
        block_timesteps = t_schedule.unsqueeze(0).expand(batch_size, -1)

        unpatched_hidden_states = []
        hidden_states = input_emb

        # process each layer independently
        for index, layer in enumerate(self.llm.layers):
            if block_timesteps is not None:
                current_t = block_timesteps[:, index]
                time_emb = self.t_embedder(current_t, dtype=hidden_states.dtype)
                hidden_states = hidden_states + time_emb.unsqueeze(1)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
            )
            hidden_states = layer_outputs[0]

            layer_time_emb_proj = time_embs[:, index]

            layer_image = self._process_hidden_state_to_image(
                hidden_states, 
                num_tokens, 
                shapes, 
                layer_time_emb_proj,
                input_is_list
            )
            unpatched_hidden_states.append(layer_image)

        hidden_states = self.llm.norm(hidden_states)

        if input_is_list:
            max_tokens = max(num_tokens)
            image_embedding = hidden_states[:, -max_tokens:]
            final_time_emb = self.t_embedder(timestep, dtype=x[0].dtype)
            final_proj = self.final_layer(image_embedding, final_time_emb)
            
            latents = []
            for i, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                latent = final_proj[i:i+1, :nt]
                latent = self.unpatchify(latent, shape[0], shape[1])
                latents.append(latent)
        else:
            image_embedding = hidden_states[:, -num_tokens:]
            final_time_emb = self.t_embedder(timestep, dtype=x.dtype)
            final_proj = self.final_layer(image_embedding, final_time_emb)
            latents = self.unpatchify(final_proj, shapes[0], shapes[1])

        if input_is_list:
            final_latents = []
            for i in range(len(latents)):
                final_latents.append(latents[i])
            unpatched_hidden_states.append(final_latents)
        else:
            unpatched_hidden_states.append(latents)
        
        if return_past_key_values:
            return latents, None
        return latents, unpatched_hidden_states
    
    def forward_with_loss_callback(
        self, 
        x, 
        timestep, 
        input_ids, 
        input_img_latents, 
        input_image_sizes, 
        attention_mask, 
        position_ids,
        ground_truth_x1,  # ground truth for loss computation
        layer_weights,     # weights for each layer
        padding_latent=None, 
        past_key_values=None, 
        return_past_key_values=False,
        offload_model:bool=False
    ):
        input_is_list = isinstance(x, list)
        x, num_tokens, shapes = self.patch_multiple_resolutions(x, padding_latent)
        
        if input_is_list:
            time_token = self.time_token(timestep, dtype=x[0].dtype).unsqueeze(1)
        else:
            time_token = self.time_token(timestep, dtype=x.dtype).unsqueeze(1)

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

        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = input_emb.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).to(input_emb.dtype)
        
        batch_size = timestep.size(0)
        num_blocks = len(self.llm.layers)

        layer_idx_tensor = torch.arange(self.num_layers, device=timestep.device, dtype=timestep.dtype)
        hidden_t_schedule = timestep.unsqueeze(1) * (1.0 - layer_idx_tensor[:-1].unsqueeze(0) / (num_blocks + 1))
        last_layer = torch.zeros(batch_size, 1, device=timestep.device, dtype=timestep.dtype)
        hidden_timesteps = torch.cat([hidden_t_schedule, last_layer], dim=1)

        all_times = hidden_timesteps.flatten()
        all_time_embs = self.t_embedder(all_times, dtype=input_emb.dtype)
        time_embs = all_time_embs.view(batch_size, self.num_layers, -1)

        t_schedule = torch.zeros(num_blocks + 1, device=timestep.device, dtype=timestep.dtype)
        t_schedule[0] = 1.0
        for i in range(num_blocks):
            t_schedule[i+1] = 1.0 - (i+1)/(num_blocks+1)
        t_schedule = t_schedule[:-1]
        block_timesteps = t_schedule.unsqueeze(0).expand(batch_size, -1)

        total_loss = torch.tensor(0.0, device=x.device if not input_is_list else x[0].device, dtype=torch.float32)
        hidden_states = input_emb

        for index, layer in enumerate(self.llm.layers):
            if block_timesteps is not None:
                current_t = block_timesteps[:, index]
                time_emb = self.t_embedder(current_t, dtype=hidden_states.dtype)
                hidden_states = hidden_states + time_emb.unsqueeze(1)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
            )
            hidden_states = layer_outputs[0]

            layer_time_emb_proj = time_embs[:, index]
            layer_image = self._process_hidden_state_to_image(
                hidden_states, 
                num_tokens, 
                shapes, 
                layer_time_emb_proj,
                input_is_list
            )

            if isinstance(layer_image, list):
                layer_image_cat = torch.cat(layer_image, dim=0) if layer_image[0].dim() == 4 else torch.stack(layer_image, dim=0)
            else:
                layer_image_cat = layer_image

            layer_loss = ((ground_truth_x1 - layer_image_cat) ** 2).mean()

            total_loss = total_loss + layer_weights[index] * layer_loss

        hidden_states = self.llm.norm(hidden_states)

        if input_is_list:
            max_tokens = max(num_tokens)
            image_embedding = hidden_states[:, -max_tokens:]
            final_time_emb = self.t_embedder(timestep, dtype=x[0].dtype)
            final_proj = self.final_layer(image_embedding, final_time_emb)
            
            latents = []
            for i, (nt, shape) in enumerate(zip(num_tokens, shapes)):
                latent = final_proj[i:i+1, :nt]
                latent = self.unpatchify(latent, shape[0], shape[1])
                latents.append(latent)
        else:
            image_embedding = hidden_states[:, -num_tokens:]
            final_time_emb = self.t_embedder(timestep, dtype=x.dtype)
            final_proj = self.final_layer(image_embedding, final_time_emb)
            latents = self.unpatchify(final_proj, shapes[0], shapes[1])

        if input_is_list:
            final_cat = torch.cat(latents, dim=0) if latents[0].dim() == 4 else torch.stack(latents, dim=0)
        else:
            final_cat = latents
        
        final_loss = ((ground_truth_x1 - final_cat) ** 2).mean()
        total_loss = total_loss + layer_weights[-1] * final_loss
        
        return latents, total_loss
        
    @torch.no_grad()
    def generate(self, x: torch.Tensor, input_ids: torch.Tensor, input_img_latents: Optional[torch.Tensor], input_image_sizes: dict, attention_mask: torch.Tensor, position_ids: torch.Tensor, guidance_scale: float = 1.0, generator: Optional[torch.Generator] = None):
        B = x.shape[0]
        device = x.device

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
            offload_model=False,
        )

        intermediate_results = [pred.clone() for pred in intermediate_preds]

        return final_pred, intermediate_results

def noise_training_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None):
    """Loss based on iterative noise levels
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

def isl_training_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None, main_loss_scale=5):
    """x1 prediction Loss for training the score model
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
    t = torch.ones(B, device=device, dtype=model_dtype)

    t_view = t.view(-1, 1, 1, 1)
    xt = t_view * x0 + (1 - t_view) * x1
    xt = xt.to(model_dtype)

    _, hidden_states = model(xt, t, **model_kwargs)
    hidden_states = torch.stack(hidden_states, dim=0)

    terms = {}

    num_layers = hidden_states.size(0)
    batch_size = x1.shape[0]

    layer_weights = torch.ones(num_layers, device=device, dtype=model_dtype)
    layer_weights[-1] = 5

    hidden_states = hidden_states.view(-1, *hidden_states.shape[2:])
    x1_expanded = x1.repeat(num_layers, *([1] * (x1.dim() - 1)))

    squared_diff = (x1_expanded - hidden_states) ** 2
    spatial_dims = tuple(range(1, squared_diff.dim())) # All dims except dim=0
    loss = squared_diff.mean(dim=spatial_dims)

    layer_weights = layer_weights.repeat_interleave(batch_size)
    weighted_loss = loss * layer_weights

    loss = weighted_loss.mean()

    terms["loss"] = loss

    return terms

def isl_training_losses_streaming(model: EffISLOmniGen, x1, model_kwargs=None, main_loss_scale=5):
    """
    Args:
        model: UltimateMemoryEfficientOmniGen wrapped in DeepSpeed
        x1: Ground truth clean images
        model_kwargs: Additional model arguments
        main_loss_scale: Weight for final layer loss
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    if isinstance(x1, list):
        x1 = torch.cat(x1, dim=0) if x1[0].dim() == 4 else torch.stack(x1, dim=0)
    
    device = x1.device
    model_dtype = next(model.parameters()).dtype
    
    B = x1.shape[0]
    x0 = torch.randn_like(x1).to(model_dtype)
    x1 = x1.to(model_dtype)
    
    t = torch.ones(B, device=device, dtype=model_dtype)
    t_view = t.view(-1, 1, 1, 1)
    xt = t_view * x0 + (1 - t_view) * x1

    num_layers = model.module.num_layers if hasattr(model, 'module') else model.num_layers
    layer_weights = torch.ones(num_layers, device=device, dtype=torch.float32)
    layer_weights[-1] = main_loss_scale

    final_output, total_loss = model.forward_with_loss_callback(xt, t, ground_truth_x1=x1, layer_weights=layer_weights, **model_kwargs)

    total_loss = total_loss / layer_weights.sum()

    return {"loss": total_loss}

def isl_training_losses_scheduled(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None, main_loss_scale=5):
    """x1 prediction Loss for training the score model
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
    t = torch.ones(B, device=device, dtype=model_dtype)

    t_view = t.view(-1, 1, 1, 1)
    xt = t_view * x0 + (1 - t_view) * x1
    xt = xt.to(model_dtype)

    _, hidden_states = model.scheduled(xt, **model_kwargs)
    hidden_states = torch.stack(hidden_states, dim=0)

    terms = {}

    num_layers = hidden_states.size(0)
    batch_size = x1.shape[0]

    layer_weights = torch.ones(num_layers, device=device, dtype=model_dtype)
    layer_weights[-1] = 5

    hidden_states = hidden_states.view(-1, *hidden_states.shape[2:])
    x1_expanded = x1.repeat(num_layers, *([1] * (x1.dim() - 1)))

    squared_diff = (x1_expanded - hidden_states) ** 2
    spatial_dims = tuple(range(1, squared_diff.dim())) # All dims except dim=0
    loss = squared_diff.mean(dim=spatial_dims)

    layer_weights = layer_weights.repeat_interleave(batch_size)
    weighted_loss = loss * layer_weights

    loss = weighted_loss.mean()

    terms["loss"] = loss

    return terms

def isl_flow_losses(model, x1, model_kwargs=None, snr_type='uniform', patch_weight=None):
    """Flow Matching loss for training the score model
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

    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
    target_vector_field = x1 - x0

    num_layers = model.module.num_layers # changed for deepspeed
    num_transformer_layers = num_layers - 1 # exclude final layer
    intermediate_layer_indices = list(range(num_transformer_layers))
    per_layer_weights = [1 for _ in intermediate_layer_indices] # tilde w
    model_output, hidden_states = model(xt, t, **model_kwargs)

    if isinstance(model_output, list):
        if model_output[0].dim() == 4:
            model_output = torch.cat(model_output, dim=0)
        else:
            model_output = torch.stack(model_output, dim=0)

    terms = {}
    total_loss = 0.0

    main_loss = torch.stack([((target_vector_field[i] - model_output[i])**2).mean() for i in range(B)], dim=0)

    intermediate_losses = []
    for layer_idx in intermediate_layer_indices:
        hidden_state = hidden_states[layer_idx]

        if isinstance(hidden_state, list):
            if hidden_state[0].dim() == 4:
                hidden_state = torch.cat(hidden_state, dim=0)
            else:
                hidden_state = torch.stack(hidden_state, dim=0)

        layer_loss = torch.stack([per_layer_weights[layer_idx] * ((target_vector_field[i] - hidden_state[i])**2).mean() for i in range(B)], dim=0)
        
        intermediate_losses.append(layer_loss)

    total_loss = main_loss + sum(intermediate_losses)
    terms["loss"] = total_loss.mean()
    terms["main_loss"] = main_loss.mean()
    terms["intermediate_loss"] = sum([loss.mean() for loss in intermediate_losses])
    
    return terms