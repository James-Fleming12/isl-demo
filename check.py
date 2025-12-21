import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers.models import AutoencoderKL

from omni_cust import CustomOmniGen, JsonFolderDataset
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator

from main import visualize_block_progression, inference_check

def convert_checkpoint_to_pretrained_format(checkpoint_path, output_dir):
    """
    Convert a training checkpoint to the format expected by from_pretrained
    """
    print(f"Converting checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_state_dict = checkpoint['model_state_dict']

    torch.save(model_state_dict, os.path.join(output_dir, 'model.pt'))

def main():
    checkpoint_path = "models/best_model_epoch_299.pth"
    converted_model_dir = "models"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(os.path.join(converted_model_dir, 'model.pt')):
        convert_checkpoint_to_pretrained_format(checkpoint_path, converted_model_dir)

    model = CustomOmniGen.from_pretrained(converted_model_dir)
    
    params_to_freeze = [
        'input_x_embedder.proj.weight',
        'input_x_embedder.proj.bias',
    ]
    
    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in params_to_freeze):
            param.requires_grad = False
    
    model.to(device)
    model.eval()

    processor = OmniGenProcessor.from_pretrained("Shitao/OmniGen-v1")
    
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
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    
    with torch.no_grad():
        inference_check(model, dataloader, device=device)

if __name__ == "__main__":
    main()