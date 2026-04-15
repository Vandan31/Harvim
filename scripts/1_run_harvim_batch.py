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
from harvim.realnvp_2 import RealNVP, create_harvim_realnvp
from harvim.utils import create_differentiable_mask

# GLOBAL IMAGE_SIZE CONFIGURATION
IMAGE_SIZE = 64

def process_images_in_directory(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Models
    condition_dim = 12
    latent_dim = 16
    
    # We must instantiate CVAE at the original resolution it was trained on (64)
    # Otherwise its FC layer weights won't match the checkpoint.
    cvae = WatermarkCVAE(condition_dim=condition_dim, latent_dim=latent_dim, image_size=(64, 64)).to(device)
    
    if os.path.exists("checkpoints/watermark_cvae.pth"):
        cvae.load_state_dict(torch.load("checkpoints/watermark_cvae.pth", map_location=device))
    else:
        print("CVAE weights not found. Using untrained CVAE.")
        
    print("Initializing Real-NVP prior...")
    prior = create_harvim_realnvp(image_size=IMAGE_SIZE if "IMAGE_SIZE" in globals() else 64).to(device)
    
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
        
        # We need the generated watermark to match the target IMAGE_SIZE (128x128)
        # So we wrap the learned watermark generation in a resizer if it differs from 64
        class ResizedLearneableWatermark(nn.Module):
            def __init__(self, base_wm, target_size):
                super().__init__()
                self.base_wm = base_wm
                self.target_size = target_size
                
            def forward(self):
                # The base generator makes a 64x64 mask, we interpolate it up to target_size
                m = self.base_wm()
                return F.interpolate(m, size=self.target_size, mode='bilinear', align_corners=False)
                
        # Inject the resize wrapper if necessary
        if IMAGE_SIZE != 64:
            watermark_model = ResizedLearneableWatermark(learnable_watermark, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            watermark_model = learnable_watermark
        
        # Initialize HARVIM config (Table 2 config)
        harvim_pipeline = HARVIM(
            generative_prior=prior,
            watermark_generator=watermark_model,
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

        # Apply a semi-transparent blending factor (opacity) to improve readability
        # Lower opacity (e.g., 0.5 - 0.7) prevents the watermark from looking purely saturated white
        opacity = 1.0 
        watermark_color = 1.0 # 1.0 for White, 0.0 for Black
        
        watermarked_image = (1 - opacity * A_m) * x_T + (opacity * A_m * watermark_color)
        watermarked_image = torch.clamp(watermarked_image, 0, 1)
        
        # Save output structured neatly by basename
        to_pil(watermarked_image.squeeze(0).cpu()).save(os.path.join(output_dir, f"{basename}_watermarked.png"))
        to_pil(optimal_watermark.squeeze(0).cpu()).save(os.path.join(output_dir, f"{basename}_mask.png"))
        to_pil(x_T.squeeze(0).cpu()).save(os.path.join(output_dir, f"{basename}_original.png"))

if __name__ == "__main__":
    input_directory = "data/imagenet_test"
    output_directory = "outputs"
    process_images_in_directory(input_directory, output_directory)