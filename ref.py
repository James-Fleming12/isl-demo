import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision import transforms

from diffusers.models import AutoencoderKL

from OmniGenCode.OmniGen.model import OmniGen
from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator

from omni_cust import JsonFolderDataset
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.utils import vae_encode, vae_encode_list
from OmniGenCode.OmniGen.train_helper import training_losses

from transformers import Phi3Config

def main():
    batch_size = 2
    lr = 1e-4
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_file = os.path.join("logs", f"ref_log.txt")
    
    config = Phi3Config(
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8
    )
    model = OmniGen(config)
    model.to(device)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model.train()

    processor = OmniGenProcessor.from_pretrained("Shitao/OmniGen-v1")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = JsonFolderDataset("00000", processor, image_transform, small_subset=True)
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
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

            loss_dict = training_losses(model, output_images, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            total_loss += loss
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch {batch}: Loss {loss}")

        avg_loss = total_loss / num_batches

        with open(log_file, 'a') as f:
            f.write(f"{epoch} {avg_loss}\n")

        if avg_loss < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                'config': config,
            }, 'omnigen_ref_model.pth')
            best_loss = avg_loss
            print(f"Best Model saved with loss: {avg_loss}")
        else:
            print(f"Epoch {epoch}: Loss {avg_loss}")

if __name__=="__main__":
    main()