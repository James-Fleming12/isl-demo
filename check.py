import torch
from OmniGenCode.OmniGen.processor import OmniGenProcessor
from OmniGenCode.OmniGen.train_helper.data import TrainDataCollator
from omni_cust import CustomOmniGen, JsonFolderDataset
from main import inference_check, visualize_block_progression

from torchvision import transforms

from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "models/final_model_epoch_299.pth"
    model = CustomOmniGen.from_pretrained()

    params_to_freeze = [
        'input_x_embedder.proj.weight',
        'input_x_embedder.proj.bias',
    ]
    
    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in params_to_freeze):
            print(f'Freezing: {name}')
            param.requires_grad = False
    
    model.to(device)
    model.eval()

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

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

if __name__=="__main__":
    main()