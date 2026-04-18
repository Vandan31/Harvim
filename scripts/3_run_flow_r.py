import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from harvim.realnvp_2 import create_harvim_realnvp
from harvim.attacks import FlowR

IMAGE_SIZE = 64

def run_flow_r_attack(watermarked_path, mask_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(watermarked_path) or not os.path.exists(mask_path):
        print("Missing required images. Run scripts/1_run_harvim.py first.")
        return

    # Load and preprocess watermarked image
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    y = transform(Image.open(watermarked_path).convert("RGB")).unsqueeze(0).to(device)
    
    # Load mask - represents where watermark is present (1 = watermark region, 0 = original image)
    mask_img = Image.open(mask_path).convert("L")  
    print(f"Mask image loaded with size: {mask_img.size}")
    mask_resized = mask_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    A_m = transform(mask_resized).unsqueeze(0).to(device)
    
    # Threshold mask to binary (0 or 1)
    A_m = (A_m > 0.5).float()
    
    # 1. Initialize and load trained RealNVP prior
    print("Loading trained Real-NVP prior...")
    prior = create_harvim_realnvp(image_size=IMAGE_SIZE).to(device)
    
    checkpoint_path = "checkpoints/real_ckp/realnvp_epoch_4.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        prior.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded RealNVP checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
    
    prior.eval()
    
    # 2. Run Flow-R attack
    print("Running Flow-R reconstruction attack...")
    attacker = FlowR(generative_prior=prior, sigma_sq=0.01)
    
    # Reconstruct image with gradient optimization
    # Higher lam emphasizes prior, lower lam emphasizes data fidelity
    reconstructed_x = attacker.remove_watermark(y, A_m, lam=10.0, steps=1000, lr=0.05)
    
    # 3. Save output
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    
    # Ensure values are in valid range
    reconstructed_x = torch.clamp(reconstructed_x, 0, 1)
    
    out_path = os.path.join(output_dir, "flow_r_reconstructed.png")
    to_pil(reconstructed_x.squeeze(0).cpu()).save(out_path)
    print(f"Flow-R reconstruction saved to: {out_path}")

if __name__ == "__main__":
    # Ensure this runs on outputs generated from step 1
    run_flow_r_attack(
        watermarked_path="data/watermarked_output.png", 
        mask_path="data/raw_watermark.png", 
        output_dir="data"
    )
