import json
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import argparse

from diffusers.models import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler

from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator

from omni_cust import CustomOmniGen, JsonFolderDataset, isl_training_losses, isl_training_losses_scheduled
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.utils import vae_encode, vae_encode_list
from transformers import Phi3Config

import deepspeed

def get_titles(num_blocks):
    return ['Noisy Input'] + [f'Layer {i+1}' for i in range(num_blocks)] + ['Ground Truth']

def visualize_block_progression(noisy_input, block_outputs, ground_truths=None, titles=None):
    """
    Create a labeled image showing progression through blocks
    noisy_input: Initial noisy image [B, C, H, W] or list
    block_outputs: List of 4 images from each block
    ground_truths: Optional list of ground truth targets for each block
    """
    titles = get_titles(len(block_outputs)) if titles is None else titles
    
    num_images = len(block_outputs) + 2
    cols = min(8, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flat
    
    if isinstance(noisy_input, list):
        img0 = noisy_input[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        img0 = noisy_input[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()

    img0 = (img0 - img0.min()) / (img0.max() - img0.min())
    axes[0].imshow(img0)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    normalized_blocks = []
    for i, block_img in enumerate(block_outputs):
        if isinstance(block_img, list):
            img = block_img[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
        else:
            img = block_img[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()

        img = (img - img.min()) / (img.max() - img.min())
        normalized_blocks.append(img)
        
        axes[i+1].imshow(img)
        axes[i+1].set_title(titles[i+1])
        axes[i+1].axis('off')

    gt_img = None
    if ground_truths is not None and len(ground_truths) > 0:
        if isinstance(ground_truths[0], list):
            gt_img = ground_truths[0][0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
        else:
            gt_img = ground_truths[0][0].detach().squeeze().permute(1, 2, 0).cpu().numpy()

        gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min())
        gt_idx = len(block_outputs) + 1
        axes[gt_idx].imshow(gt_img)
        axes[gt_idx].set_title(titles[gt_idx])
        axes[gt_idx].axis('off')

    plt.tight_layout()
    plt.savefig("inference_check.png")

def inference_check(model: CustomOmniGen, data: DataLoader, vae, device=None):
    """
    vae is the model for decoding latents back to images
    """
    num_layers = model.num_layers
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = next(iter(data))

    output_latent = batch["output_images"][0].to(device)

    if output_latent.dim() == 3: # just in case
        output_latent = output_latent.unsqueeze(0)

    padding_latent = batch.get("padding_images", None)
    if padding_latent is not None:
        padding_latent = [p.to(device=output_latent.device) if p is not None else None for p in padding_latent]

    model_kwargs = dict(
        input_ids=batch['input_ids'][0:1].to(device),
        input_img_latents=None,
        input_image_sizes=batch['input_image_sizes'],
        attention_mask=batch['attention_mask'][0:1].to(device),
        position_ids=batch['position_ids'][0:1].to(device)
    )
    
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        gt_image = vae.decode(output_latent.float() / vae.config.scaling_factor).sample

    model_input = torch.randn_like(output_latent).to(model_dtype)

    with torch.no_grad():
        # generated, intermediate_gen = model.generate(model_input, guidance_scale=1.0, **model_kwargs)
        generated, intermediate_gen = model.scheduled_generate(model_input, guidance_scale=1.0, **model_kwargs)
    intermediate_gen = intermediate_gen[:-1]  # removes output layer

    decoded_blocks = []

    with torch.no_grad():
        for block_latent in intermediate_gen:
            decoded = vae.decode(
                block_latent.float() / vae.config.scaling_factor
            ).sample
            decoded_blocks.append(decoded)

        final_decoded = vae.decode(
            generated.float() / vae.config.scaling_factor
        ).sample

        decoded_noise = vae.decode(
            model_input.float() / vae.config.scaling_factor
        ).sample

    decoded_blocks.append(final_decoded)

    visualize_block_progression(
        noisy_input=decoded_noise,
        block_outputs=decoded_blocks,
        ground_truths=[gt_image],
        titles=None
    )

def main():
    batch_size = 4 # temporary for DeepSpeed memory issues
    lr = 1e-4
    epochs = 1000

    num_gpus = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    # parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    args = parser.parse_args()

    deepspeed.init_distributed()

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = CustomOmniGen.from_pretrained("Shitao/OmniGen-v1")
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    
    processor = OmniGenProcessor.from_pretrained("Shitao/OmniGen-v1")
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    vae = None

    if local_rank == 0:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
        vae.eval()

        dataset = JsonFolderDataset("00000", processor, vae=vae, device=device, image_transform=image_transform, small_subset=True, use_preencoded=True)

    torch.distributed.barrier() # wait for dataset preprocessing

    if local_rank != 0:
        dataset = JsonFolderDataset("00000", processor, vae=None,device=device, image_transform=image_transform, small_subset=True, use_preencoded=True)

    sampler = DistributedSampler(dataset)
    collate_fn = TrainDataCollator(pad_token_id=processor.text_tokenizer.eos_token_id, hidden_size=model.llm.config.hidden_size, keep_raw_resolution=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, shuffle=False,num_workers=0,pin_memory=True)

    if local_rank == 0:
        best_loss = float('inf')
        os.makedirs("logs", exist_ok=True)
        log_file = os.path.join("logs", f"log.txt")

    with open("ds_config.json", 'r') as f:
        deepspeed_config = json.load(f)

    if deepspeed_config.get("train_micro_batch_size_per_gpu") == "auto":
        deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size
    if deepspeed_config.get("train_batch_size") == "auto":
        deepspeed_config["train_batch_size"] = batch_size * num_gpus

    params_to_freeze = [
        'input_x_embedder.proj.weight',
        'input_x_embedder.proj.bias',
    ]

    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in params_to_freeze):
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    model_engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=trainable_params, config=deepspeed_config)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        dataloader.sampler.set_epoch(epoch)

        if local_rank == 0: end_time = time.perf_counter()

        for batch_idx, data in enumerate(dataloader):
            model_dtype = next(model_engine.parameters()).dtype

            output_images = data['output_images']
            if isinstance(output_images, list):
                output_images = torch.cat(output_images, dim=0)
            output_images = output_images.to(device=device, dtype=model_dtype)

            padding_latent = data.get("padding_images", None)
            if padding_latent is not None:
                padding_latent = [p.to(device=device, dtype=model_dtype) if p is not None else None for p in padding_latent]

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

            loss_dict = isl_training_losses(model_engine, output_images, model_kwargs=model_kwargs)
            loss = loss_dict["loss"]

            model_engine.backward(loss)
            model_engine.step()

            # loss_tensor = torch.tensor([loss.item()], device=device)
            # torch.distributed.all_reduce(loss_tensor)
            # avg_loss_across = loss_tensor.item() / torch.distributed.get_world_size()

            # total_loss += avg_loss_across
            num_batches += 1

        avg_loss = total_loss / num_batches

        if local_rank == 0:
            print(f"Epoch {epoch} Loss: {avg_loss}")
            with open(log_file, 'a') as f:
                f.write(f"{epoch} {avg_loss}\n")

    if local_rank == 0:
        os.makedirs("models", exist_ok=True)
        torch.save(model_engine.module.state_dict(), f'models/final_model_epoch_{epoch}.pth')
        print(f"Final model saved with loss: {avg_loss:.6f}")

        if vae is None:
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
            vae.eval()
        
        model_engine.module.eval()
        with torch.no_grad():
            inference_check(model_engine.module, dataloader, vae, device=device)

if __name__=="__main__":
    main()
