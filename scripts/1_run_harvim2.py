import os
import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from harvim.core import HARVIM
from harvim.watermark_generator import WatermarkCVAE, LearneableWatermark
from harvim.realnvp import RealNVP
from harvim.utils import create_differentiable_mask

# GLOBAL IMAGE_SIZE CONFIGURATION
IMAGE_SIZE = 64

def process_images_in_directory(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Models
    condition_dim = 12
    latent_dim = 16
    cvae = WatermarkCVAE(condition_dim=condition_dim, latent_dim=latent_dim, image_size=(IMAGE_SIZE, IMAGE_SIZE)).to(device)
    
    if os.path.exists("checkpoints/watermark_cvae.pth"):
        cvae.load_state_dict(torch.load("checkpoints/watermark_cvae.pth", map_location=device))
    else:
        print("CVAE weights not found. Using untrained CVAE.")
        
    print("Initializing Real-NVP prior...")
    prior = RealNVP(num_channels=3, num_layers=4).to(device)
    
    # 2. Gather images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found. Please create it and add images.")
        return

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Processing HARVIM iteratively...")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()
    
    for img_path in tqdm(image_paths, desc="Running HARVIM"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")
        x_T = transform(img).unsqueeze(0).to(device)
        
        # Reset Learnable parameter states correctly for each independent image
        class_idx = 7
        class_one_hot = F.one_hot(torch.tensor([class_idx]), num_classes=10).float().to(device)
        initial_pad = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)
        initial_c = torch.cat([class_one_hot, initial_pad], dim=-1)
        
        learnable_watermark = LearneableWatermark(cvae, initial_c, class_dim=10).to(device)
        
        # Initialize HARVIM config (Table 2 config)
        harvim_pipeline = HARVIM(
            generative_prior=prior,
            watermark_generator=learnable_watermark,
            sigma_sq=0.01,
            alpha=0.15,
            beta=0.01,
            reg_coeff=0.001
        )
        
        # Run Optimization
        optimal_watermark = harvim_pipeline.run(
            x_T=x_T,
            target_lambda=1.0,
            T_steps=50, 
            K_unroll=1, 
            lr=0.05
        )
        
        # 6. Apply watermark and save
        A_m = create_differentiable_mask(optimal_watermark, alpha=0.15, beta=0.01)
        
        if optimal_watermark.shape[1] == 1 and x_T.shape[1] == 3:
            A_m = A_m.repeat(1, 3, 1, 1)

        watermarked_image = (1 - A_m) * x_T + A_m * 1.0 # White watermark overlay
        watermarked_image = torch.clamp(watermarked_image, 0, 1)
        
        # Save output structured neatly by basename
        to_pil(watermarked_image.squeeze(0).cpu()).save(os.path.join(output_dir, f"{basename}_watermarked.png"))
        to_pil(optimal_watermark.squeeze(0).cpu()).save(os.path.join(output_dir, f"{basename}_mask.png"))
        to_pil(x_T.squeeze(0).cpu()).save(os.path.join(output_dir, f"{basename}_original.png"))

if __name__ == "__main__":
    input_directory = "data/imagenet-sample-images-master"
    output_directory = "outputs"
    process_images_in_directory(input_directory, output_directory)