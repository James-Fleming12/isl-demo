import json
import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import argparse

from diffusers.models import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler

from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator

from omni_cust import CustomOmniGen, JsonFolderDataset, isl_training_losses
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.utils import vae_encode, vae_encode_list
from transformers import Phi3Config

import deepspeed

def visualize_block_progression(noisy_input, block_outputs, ground_truths=None, titles=None):
    """
    Create a labeled image showing progression through blocks
    noisy_input: Initial noisy image [B, C, H, W] or list
    block_outputs: List of 4 images from each block
    ground_truths: Optional list of ground truth targets for each block
    """
    if titles is None:
        titles = ['Noisy Input', 'Block 1', 'Block 2', 'Block 3', 'Block 4', 'Ground Truth']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flat
    
    if isinstance(noisy_input, list):
        img0 = noisy_input[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        img0 = noisy_input[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()

    img0 = (img0 - img0.min()) / (img0.max() - img0.min())
    axes[0].imshow(img0)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    for i, block_img in enumerate(block_outputs):
        if isinstance(block_img, list):
            img = block_img[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
        else:
            img = block_img[0].detach().squeeze().permute(1, 2, 0).cpu().numpy()
        
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
        axes[5].imshow(gt_img)
        axes[5].set_title(titles[5])
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig("inference_check.png")

def inference_check(model: CustomOmniGen, data: DataLoader, device = None):
    num_layers = model.num_layers
    intermediate_layer_indices = [num_layers//4, num_layers//2, num_layers*3//4]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = next(iter(data))
    output_image = [img.to(device) for img in batch['output_images']]
    output_image = output_image[0]

    padding_latent = batch.get("padding_images", None)
    if padding_latent is not None:
        padding_latent = [p.to(device=output_image.device) if p is not None else None for p in padding_latent]

    model_kwargs = dict(
        input_ids=batch['input_ids'][0:1].to(device),
        input_img_latents=None,
        input_image_sizes=batch['input_image_sizes'],
        attention_mask=batch['attention_mask'][0:1].to(device),
        position_ids=batch['position_ids'][0:1].to(device),
        padding_latent=padding_latent,
        past_key_values=None,
        return_past_key_values=False
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval()
    
    model_dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        gt_latent = vae.encode(output_image).latent_dist.sample()
        gt_latent_scaled = gt_latent * vae.config.scaling_factor

    model_input = torch.randn_like(gt_latent_scaled).to(model_dtype)

    model_output, hidden_states = model.inference(model_input, torch.ones(1, device=device, dtype=model_dtype), **model_kwargs)

    decoded_blocks = []

    with torch.no_grad():
        for idx in intermediate_layer_indices:
            decoded = vae.decode(
                hidden_states[idx].float() / vae.config.scaling_factor
            ).sample
            decoded_blocks.append(decoded)

        final_decoded = vae.decode(
            model_output.float() / vae.config.scaling_factor
        ).sample
        decoded_blocks.append(final_decoded)

        decoded_noise = vae.decode(
            model_input.float() / vae.config.scaling_factor
        ).sample

    visualize_block_progression(
        noisy_input=decoded_noise,
        block_outputs=decoded_blocks,
        ground_truths=[output_image],
        titles=[
            f'Noisy Input (t=1)',
            'Block 1 Output',
            'Block 2 Output', 
            'Block 3 Output',
            'Block 4 Output',
            'Ground Truth'
        ]
    )

def main():
    batch_size = 1
    lr = 1e-4
    epochs = 300
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    args = parser.parse_args()

    deepspeed.init_distributed()

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # model = CustomOmniGen.from_pretrained_other("Shitao/OmniGen-v1")
    config = Phi3Config(
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8
    )
    model = CustomOmniGen(config)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    
    processor = OmniGenProcessor.from_pretrained("Shitao/OmniGen-v1")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.eval()
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset = JsonFolderDataset("00000", processor, image_transform, small_subset=True)
    sampler = DistributedSampler(dataset)
    collate_fn = TrainDataCollator(
        pad_token_id=processor.text_tokenizer.eos_token_id,
        hidden_size=model.llm.config.hidden_size,
        keep_raw_resolution=True
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    if local_rank == 0:
        best_loss = float('inf')
        log_file = os.path.join("logs", f"log.txt")

    with open(args.deepspeed_config, 'r') as f:
        deepspeed_config = json.load(f)

    deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size

    params_to_freeze = [ # freeizing params for deepspeed error
        'input_x_embedder.proj.weight',
        'input_x_embedder.proj.bias',
    ]

    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in params_to_freeze):
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_params,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(dataloader):
            output_images = [img.to(device) for img in data['output_images']]
            
            with torch.no_grad():
                output_images = vae_encode_list(vae, output_images, model.llm.dtype)

            model_dtype = next(model_engine.parameters()).dtype
            output_images = [img.to(model_dtype) for img in output_images]

            padding_latent = data.get("padding_images", None)
            if padding_latent is not None:
                padding_latent = [p.to(device=output_images[0].device) if p is not None else None for p in padding_latent]

            model_kwargs = dict(
                input_ids=data['input_ids'].to(device),
                block_inputs=None,
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

            loss_tensor = torch.tensor([loss.item()], device=device)
            torch.distributed.all_reduce(loss_tensor)
            avg_loss_across = loss_tensor.item() / torch.distributed.get_world_size()

            total_loss += avg_loss_across
            num_batches += 1

        avg_loss = total_loss / num_batches

        if local_rank == 0:
            print(f"Epoch {epoch} Loss: {avg_loss}")
            with open(log_file, 'a') as f:
                f.write(f"{epoch} {avg_loss}\n")

    if local_rank == 0:
        torch.save(model_engine.module.state_dict(), f'models/final_model_epoch_{epoch}.pth')
        print(f"Final model saved with loss: {avg_loss:.6f}")
        model_engine.module.eval()
        with torch.no_grad():
            inference_check(model_engine.module, dataloader, device=device)

if __name__=="__main__":
    main()