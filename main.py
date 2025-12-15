import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import transforms

from PIL import Image
from diffusers.models import AutoencoderKL
from datasets import load_dataset

from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator
from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler

from omni_cust import CustomOmniGen, JsonFolderDataset
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

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = JsonFolderDataset("00000", processor, image_transform)
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch, data in enumerate(dataloader):
            output_images = [img.to(device) for img in data['output_images']]
            with torch.no_grad():
                # output_images = vae.encode(output_images).latent_dist.sample()
                output_images = vae_encode_list(vae, output_images, model.llm.dtype)
            padding_latent = data.get("padding_images", None)
            if padding_latent is not None:
                padding_latent = [p.to(device=output_images[0].device) if p is not None else None for p in padding_latent]
            model_kwargs = dict(
                input_ids=data['input_ids'].to(device),
                input_img_latents=None,
                input_image_sizes=data['input_image_sizes'],
                attention_mask=data['attention_mask'].to(device),
                position_ids=data['position_ids'].to(device),
                padding_latent=padding_latent,
                past_key_values=None,
                return_past_key_values=False
            )

            loss_dict = training_losses(model, output_images, model_kwargs)
            loss = loss_dict["loss"].mean()
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss {total_loss}")

if __name__=="__main__":
    main()