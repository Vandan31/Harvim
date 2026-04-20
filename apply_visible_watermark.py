#!/usr/bin/env python
"""
Wrapper script for HARVIM + StegaStamp watermarking with command-line arguments.
Applies both visible (HARVIM) and invisible (StegaStamp) watermarks to an image.

Usage:
    python apply_visible_watermark.py --image <image_path> --output <output_dir> --secret <message> [--save-original]
    
Example:
    python apply_visible_watermark.py --image photo.jpg --output ./watermarked_output --secret "hello123"
    python apply_visible_watermark.py --image photo.jpg --output ./watermarked_output --secret "hello" --save-original
"""

import os
import sys
import argparse
import subprocess
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harvim.core import HARVIM
from harvim.watermark_generator import WatermarkCVAE, LearneableWatermark
from harvim.realnvp_2 import create_harvim_realnvp
from harvim.utils import create_differentiable_mask

IMAGE_SIZE = 400
torch.manual_seed(42)


def apply_visible_watermark(image_path, output_dir, secret=None, save_original=False):
    """
    Apply visible watermark using HARVIM, then invisible watermark using StegaStamp.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save watermarked output
        secret: Secret message for invisible watermark (optional)
        save_original: Whether to also save the original image
    
    Returns:
        dict with paths to generated images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate input image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading image: {image_path}")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    transform_f = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    x_T = transform(img).unsqueeze(0).to(device)
    
    # Setup Watermark Generator (CVAE)
    print("Setting up Watermark Generator (CVAE)...")
    condition_dim = 12
    latent_dim = 16
    cvae = WatermarkCVAE(condition_dim=condition_dim, latent_dim=latent_dim, image_size=(64, 64)).to(device)
    
    # Load CVAE weights
    cvae_path = "checkpoints/watermark_cvae_64.pth"
    if os.path.exists(cvae_path):
        cvae.load_state_dict(torch.load(cvae_path, map_location=device))
        print(f"Loaded CVAE from {cvae_path}")
    else:
        print(f"Warning: CVAE weights not found at {cvae_path}. Using untrained CVAE.")
    
    # Setup learnable watermark
    class_idx = 0
    class_one_hot = F.one_hot(torch.tensor([class_idx]), num_classes=10).float().to(device)
    initial_pad = torch.tensor([[0, 1]], dtype=torch.float32).to(device)
    initial_c = torch.cat([class_one_hot, initial_pad], dim=-1)
    
    learnable_watermark = LearneableWatermark(cvae, initial_c, class_dim=10).to(device)
    
    # Setup Prior (Real-NVP)
    print("Initializing Real-NVP prior...")
    prior = create_harvim_realnvp(image_size=64).to(device)
    
    # Load prior weights
    prior_path = "checkpoints/real_ckp/realnvp_epoch_28.pt"
    if os.path.exists(prior_path):
        checkpoint = torch.load(prior_path, map_location=device)
        prior.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded prior from {prior_path}")
    else:
        raise FileNotFoundError(f"Prior model not found at {prior_path}")
    
    # Initialize HARVIM
    print("Initializing HARVIM pipeline...")
    harvim_pipeline = HARVIM(
        generative_prior=prior,
        watermark_generator=learnable_watermark,
        sigma_sq=0.01,
        alpha=0.15,
        beta=0.01,
        reg_coeff=0.001
    )
    
    # Run optimization
    print("Running HARVIM optimization (this may take a minute)...")
    optimal_watermark = harvim_pipeline.run(
        x_T=x_T,
        target_lambda=1,
        T_steps=50,
        K_unroll=1,
        lr=0.05
    )
    
    print("Optimization finished.")
    
    # Apply watermark
    print("Generating watermarked output...")
    optimal_watermark = F.interpolate(
        optimal_watermark,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode='bilinear',
        align_corners=False
    )
    A_m = create_differentiable_mask(optimal_watermark, alpha=0.15, beta=0.01)
    
    # Ensure proper channel dimensions
    if optimal_watermark.shape[1] == 1 and x_T.shape[1] == 3:
        A_m = A_m.repeat(1, 3, 1, 1)
    
    x_f = transform_f(img).unsqueeze(0).to(device)
    watermarked_image = A_m * x_f + (1 - A_m) * x_f * 0.45 + optimal_watermark
    watermarked_image = torch.clamp(watermarked_image, 0, 1)
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    
    # Save visible watermarked image
    watermarked_path = os.path.join(output_dir, "visible_watermarked.png")
    to_pil(watermarked_image.squeeze(0).cpu()).save(watermarked_path)
    print(f"✓ Visible watermarked image saved: {watermarked_path}")
    
    # Save raw watermark
    raw_watermark_path = os.path.join(output_dir, "raw_watermark.png")
    to_pil(optimal_watermark.squeeze(0).cpu()).save(raw_watermark_path)
    print(f"✓ Raw watermark saved: {raw_watermark_path}")
    
    # Save original if requested
    original_path = None
    if save_original:
        original_path = os.path.join(output_dir, "original_image.png")
        to_pil(x_T.squeeze(0).cpu()).save(original_path)
        print(f"✓ Original image saved: {original_path}")
    
    final_output_path = watermarked_path
    
    # Apply invisible watermark if secret is provided
    if secret:
        print(f"\nApplying invisible watermark with secret: '{secret}'")
        
        # Run StegaStamp encoder
        cmd = [
            'python', '-m', 'StegaStamp-pytorch.stegastamp.encode_image',
            '--model', './StegaStamp-pytorch/asset/best.pth',
            '--image', watermarked_path,
            '--save_dir', output_dir,
            '--secret', secret,
            '--height', '400',
            '--width', '400',
            '--secret_size', '100'
        ]
        
        print(f"Running StegaStamp encoder...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            print(f"⚠ Warning: Invisible watermarking failed: {result.stderr}")
            print(f"Continuing with visible watermark only...")
        else:
            # Find the encoded output
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if 'encoded' in file.lower() and file.endswith('.png'):
                        output_files.append(os.path.join(root, file))
            
            if output_files:
                # Rename to final output
                final_output_path = os.path.join(output_dir, "watermarked_output.png")
                import shutil
                shutil.move(output_files[0], final_output_path)
                print(f"✓ Invisible watermarked image saved: {final_output_path}")
            else:
                print(f"⚠ Warning: No encoded output found, using visible watermark only")
    else:
        # Just rename the visible watermark to standard output name
        final_output_path = os.path.join(output_dir, "watermarked_output.png")
        import shutil
        shutil.move(watermarked_path, final_output_path)
    
    results = {
        'watermarked': final_output_path,
        'visible_watermarked': watermarked_path if secret else None,
        'raw_watermark': raw_watermark_path,
        'original': original_path
    }
    
    print(f"\n✓ All results saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Apply visible (HARVIM) and invisible (StegaStamp) watermarks to image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python apply_visible_watermark.py --image photo.jpg --output ./watermarked --secret "hello123"
  python apply_visible_watermark.py --image photo.jpg --output ./watermarked --secret "test" --save-original
  python apply_visible_watermark.py --image photo.jpg --output ./watermarked  # Visible only
        """
    )
    
    parser.add_argument(
        '--image',
        required=True,
        help='Path to input image (PNG, JPG, etc.)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for watermarked image'
    )
    parser.add_argument(
        '--secret',
        default=None,
        help='Secret message for invisible watermark (optional)'
    )
    parser.add_argument(
        '--save-original',
        action='store_true',
        help='Also save the original image'
    )
    
    args = parser.parse_args()
    
    try:
        results = apply_visible_watermark(
            image_path=args.image,
            output_dir=args.output,
            secret=args.secret,
            save_original=args.save_original
        )
        print("\n✓ Success!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
