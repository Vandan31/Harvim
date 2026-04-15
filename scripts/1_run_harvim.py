import os
import torch
import torch.nn as nn
from harvim.core import HARVIM
from harvim.watermark_generator import WatermarkCVAE, LearneableWatermark
from harvim.realnvp import RealNVP
from harvim.prior import MLEObjective
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from harvim.realnvp_2 import  create_harvim_realnvp
IMAGE_SIZE=128
def run_harvim_on_image(image_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load an image (x_T)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    # If using dummy image because path doesn't exist
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found. Using a random dummy image.")
        x_T = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    else:
        img = Image.open(image_path).convert("RGB")
        x_T = transform(img).unsqueeze(0).to(device)

    # 2. Setup Watermark Generator
    condition_dim = 12
    latent_dim = 16
    cvae = WatermarkCVAE(condition_dim=condition_dim, latent_dim=latent_dim, image_size=(IMAGE_SIZE, IMAGE_SIZE)).to(device)
    
    if os.path.exists(f"checkpoints/watermark_cvae_{IMAGE_SIZE}.pth"):
        cvae.load_state_dict(torch.load(f"checkpoints/watermark_cvae_{IMAGE_SIZE}.pth", map_location=device))
    else:
        print("CVAE weights not found. Using untrained CVAE.")
    
    # We want digit '7' for example, with initial padding mostly centered (0.5, 0.5)
    class_idx = 7
    class_one_hot = F.one_hot(torch.tensor([class_idx]), num_classes=10).float().to(device)
    initial_pad = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)
    initial_c = torch.cat([class_one_hot, initial_pad], dim=-1)
    
    # The learnable wrapper enables tuning the latent code and the location
    learnable_watermark = LearneableWatermark(cvae, initial_c, class_dim=10).to(device)
    
    # 3. Setup Prior
    print("Initializing Real-NVP prior...")
    # prior = RealNVP().to(device)
    prior = create_harvim_realnvp(image_size=IMAGE_SIZE if "IMAGE_SIZE" in globals() else 64).to(device)
    
    # 4. Initialize HARVIM (Table 2 config)
    harvim_pipeline = HARVIM(
        generative_prior=prior,
        watermark_generator=learnable_watermark,
        sigma_sq=0.01,
        alpha=0.15,
        beta=0.01,
        reg_coeff=0.001
    )
    
    # 5. Run Optimization
    print("Running HARVIM Optimization...")
    # T_steps corresponds to the number of metadata/lambda updates
    optimal_watermark = harvim_pipeline.run(
        x_T=x_T,
        target_lambda=0.5,
        T_steps=100, # Reduced from 100 for faster demonstration
        K_unroll=1, # Meta step K=1 as per Table 2
        lr=0.05
    )
    
    print("Optimization finished.")
    # 6. Apply watermark and save
    from harvim.utils import create_differentiable_mask
    A_m = create_differentiable_mask(optimal_watermark, alpha=0.15, beta=0.01)
    
    # Ensure watermark is scaled to target image channels if it's 1 channel
    if optimal_watermark.shape[1] == 1 and x_T.shape[1] == 3:
        A_m = A_m.repeat(1, 3, 1, 1)
    print(A_m.min(), A_m.max())
    watermarked_image = (A_m) * x_T + optimal_watermark# White watermark for simplicity
    watermarked_image = torch.clamp(watermarked_image, 0, 1)
    
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    to_pil(watermarked_image.squeeze(0).cpu()).save(os.path.join(output_dir, "watermarked_output.png"))
    to_pil(optimal_watermark.squeeze(0).cpu()).save(os.path.join(output_dir, "raw_watermark.png"))
    to_pil(x_T.squeeze(0).cpu()).save(os.path.join(output_dir, "original_image.png"))
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    run_harvim_on_image("./data/harvim_dog.png", "./data")