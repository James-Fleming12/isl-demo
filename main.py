import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from PIL import Image
from diffusers.models import AutoencoderKL
from datasets import load_dataset

from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator
from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler

from omni_cust import CustomOmniGen
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.utils import vae_encode, vae_encode_list
from OmniGenCode.OmniGen.train_helper import training_losses

from transformers import Phi3Config

def main():
    batch_size = 2
    lr = 1e-4
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = Phi3Config(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
    )
    model = CustomOmniGen(config)
    model.to(device)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model.train()

    processor = OmniGenProcessor.from_pretrained("Shitao/OmniGen-v1")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = load_dataset("yzwang/X2I-text-to-image", split="train", streaming=True)
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):
            output_images = data['output_images']
            with torch.no_grad():
                output_images = vae.encode(output_images).latent_dist.sample()
            model_kwargs = dict(
                input_ids=data['input_ids'],
                input_img_latents=None,
                input_image_sizes=data['input_image_sizes'],
                attention_mask=data['attention_mask'],
                position_ids=data['position_ids'],
                padding_latent=data.get('padding_images', None),
                past_key_values=None,
                return_past_key_values=False
            )

            loss_dict = training_losses(model, output_images, model_kwargs)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__=="__main__":
    main()