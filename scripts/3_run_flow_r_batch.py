import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from harvim.realnvp_2 import RealNVP, create_harvim_realnvp
from harvim.attacks import FlowR
from harvim.utils import create_differentiable_mask

def process_flow_r_directory(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    watermarked_images = glob.glob(os.path.join(output_dir, "*_watermarked.png"))
    if not watermarked_images:
        print(f"No watermarked images found in {output_dir}")
        return
        
    print(f"Found {len(watermarked_images)} watermarked images. Initializing Flow-R prior...")
    prior = create_harvim_realnvp(image_size=IMAGE_SIZE if "IMAGE_SIZE" in globals() else 64).to(device)
    attacker = FlowR(generative_prior=prior, sigma_sq=0.01)
    
    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    for watermarked_path in tqdm(watermarked_images, desc="Running Flow-R Attack"):
        basename = watermarked_path.replace("_watermarked.png", "")
        mask_path = basename + "_mask.png"
        
        if not os.path.exists(mask_path):
            continue
            
        y = transform(Image.open(watermarked_path).convert("RGB")).unsqueeze(0).to(device)
        mask_img = Image.open(mask_path).convert("L")  
        A_m_inverted = transform(mask_img).unsqueeze(0).to(device)
        
        m = A_m_inverted
        A_m = create_differentiable_mask(m, alpha=0.15, beta=0.01)
        
        # Optimize to remove watermark
        # Lowered steps compared to single image for slightly faster batch processing
        reconstructed_x = attacker.remove_watermark(y, A_m, lam=1.0, steps=200, lr=0.05)
        
        out_path = basename + "_flow_r.png"
        to_pil(reconstructed_x.squeeze(0).cpu()).save(out_path)

if __name__ == "__main__":
    process_flow_r_directory("outputs")