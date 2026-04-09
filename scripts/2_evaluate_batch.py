import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
import skimage.metrics as metrics
import numpy as np
from harvim.utils import compute_psnr

def evaluate_directory(output_dir, target_suffix="_watermarked.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    transform = transforms.ToTensor()
    
    orig_images = glob.glob(os.path.join(output_dir, "*_original.png"))
    
    if not orig_images:
        print(f"No original images found in {output_dir}")
        return
        
    psnr_list, ssim_list, lpips_list = [], [], []
    
    print(f"Evaluating {len(orig_images)} images against suffix '{target_suffix}'...")
    for orig_path in orig_images:
        basename = orig_path.replace("_original.png", "")
        target_path = basename + target_suffix
        
        if not os.path.exists(target_path):
            continue
            
        img_orig = Image.open(orig_path).convert("RGB")
        img_recon = Image.open(target_path).convert("RGB")
        
        t_orig = transform(img_orig).unsqueeze(0).to(device)
        t_recon = transform(img_recon).unsqueeze(0).to(device)
        
        # PSNR
        psnr_val = compute_psnr(t_recon, t_orig).item()
        psnr_list.append(psnr_val)
        
        # SSIM
        np_orig = t_orig.squeeze().permute(1, 2, 0).cpu().numpy()
        np_recon = t_recon.squeeze().permute(1, 2, 0).cpu().numpy()
        ssim_val = metrics.structural_similarity(np_orig, np_recon, channel_axis=2, data_range=1.0)
        ssim_list.append(ssim_val)
        
        # LPIPS
        t_orig_lpips = t_orig * 2.0 - 1.0
        t_recon_lpips = t_recon * 2.0 - 1.0
        lpips_val = loss_fn_vgg(t_orig_lpips, t_recon_lpips).item()
        lpips_list.append(lpips_val)

    if not psnr_list:
        print("No valid pairs found to evaluate.")
        return
        
    print("-" * 40)
    print("Average Evaluation Results")
    print(f"Evaluated {len(psnr_list)} image pairs.")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {np.mean(psnr_list):.4f} dB")
    print(f"SSIM (Structural Similarity):     {np.mean(ssim_list):.4f}")
    print(f"LPIPS (Perceptual Similarity):    {np.mean(lpips_list):.4f}")
    print("-" * 40)

if __name__ == "__main__":
    # Change target_suffix to "_flow_r.png" to evaluate attack reconstructions instead
    evaluate_directory("outputs", target_suffix="_watermarked.png")