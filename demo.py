import torch
import torch.nn as nn
from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

# using https://huggingface.co/stabilityai/sdxl-vae
class IntermediateSupervision(nn.Module):
    def __init__(self, text_llama, text_tokenizer):
        super().__init__()
        self.text_llama = text_llama
        self.text_tokenizer = text_tokenizer
        self.autoencoder = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        self.diffusion_batch_mul = 4
        self.tht_token_dim = self.autoencoder.config.latent_channels

    def forward(self, input_ids_q, output_ids):
        bs = input_ids_q.shape[0]
        device = input_ids_q.device

        with torch.no_grad():
            thought_tokens = self.autoencoder._compress(output_ids)

        thought_tokens = thought_tokens.reshape(bs, -1, self.tht_token_dim)
        N = thought_tokens.shape[1]

        gt_thought_tokens = thought_tokens.repeat(self.diffusion_batch_mul, 1, 1)
        noise = torch.randn_like(gt_thought_tokens)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (gt_thought_tokens.shape[0],), device=device)

        noisy_thoughts = self.noise_scheduler.add_noise(gt_thought_tokens, noise, timesteps)

        special_tokens = torch.tensor(
            [self.text_tokenizer.bot_token_id,
            self.text_tokenizer.time_token_id] +
            [self.text_tokenizer.tht_token_id] * N +
            [self.text_tokenizer.eot_token_id],
            device=device
        ).unsqueeze(0).expand(bs * self.diffusion_batch_mul, -1)

        full_input_ids = torch.cat([
            input_ids_q.repeat(self.diffusion_batch_mul, 1),
            special_tokens
        ], dim=1)

        inputs_embeds = self.text_llama.get_input_embeddings()(full_input_ids)

        tht_mask = full_input_ids == self.text_tokenizer.tht_token_id
        inputs_embeds[tht_mask] = noisy_thoughts.reshape(-1, self.tht_token_dim)

        outputs = self.text_llama(
            inputs_embeds=inputs_embeds,
            attention_mask=full_input_ids != self.text_tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict=True
        )

        intermediate_layers = [10, 20, 25] # layers for eval (proof of concept), layer 32 evaled at end
        total_loss = 0

        for i, layer_idx in enumerate(intermediate_layers):
            layer_hidden = outputs.hidden_states[layer_idx]

            layer_pred = layer_hidden[tht_mask].reshape(bs * self.diffusion_batch_mul, N, -1)

            layer_level = 1 - (i+1) / (len(intermediate_layers)+1) # progressive noise levels (proof of concept)
            layer_target = noise * layer_level

            total_loss += nn.functional.mse_loss(layer_pred.float(), layer_target.float())

        last_hidden = outputs.hidden_states[-1]
        model_pred = last_hidden[tht_mask].reshape(bs * self.diffusion_batch_mul, N, -1)

        total_loss += nn.functional.mse_loss(model_pred.float(), gt_thought_tokens.float()) # predict fully unnoised result
        
        return total_loss