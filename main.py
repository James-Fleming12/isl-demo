from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from PIL import Image
from diffusers.models import AutoencoderKL

from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler

from omni_cust import CustomOmniGen
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.utils import vae_encode, vae_encode_list
from OmniGenCode.OmniGen.train_helper import training_losses

from transformers import Phi3Config

class SimpleTextToImageDataset(Dataset):
    def __init__(self):
        self.samples = [
            ("A white cat resting on a picnic table.", "cat.png"),
            ("A person walking on a suspension bridge.", "walking.png"),
        ]

        self.image_dir = Path("/home/james/Research/QLab/isl-demo/OmniGenCode/toy_data/images")
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        caption, image_file = self.samples[idx]
        image = Image.open(self.image_dir / image_file).convert("RGB")
        image = self.transform(image)

        return image, caption

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
    model = CustomOmniGen(config).to(device)
    model.train()

    processor = OmniGenProcessor.from_pretrained("Shitao/OmniGen-v1")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = SimpleTextToImageDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for batch, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            if isinstance(images, list):
                images_latents = vae_encode_list(vae, images, torch.float32).to(device)
            else:
                images_latents = vae_encode(vae, images, torch.float32).to(device)

            processed = processor(instructions=captions)
            input_ids = processed["input_ids"].to(device)
            attention_mask = processed["attention_mask"].to(device)
            position_ids = processed["position_ids"].to(device)
            input_image_sizes = {i: [(0, images_latents.shape[1])] for i in range(batch_size)}
            
            model_kwargs = dict(input_ids=input_ids, input_img_latents=images_latents, input_image_sizes=input_image_sizes, attention_mask=attention_mask, position_ids=position_ids)

            loss_dict = training_losses(model, images_latents, model_kwargs)
            loss = loss_dict["loss"].mean()

            print(loss)

            break
        break


if __name__=="__main__":
    main()