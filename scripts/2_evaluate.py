import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
import skimage.metrics as metrics
from harvim.utils import compute_psnr

def evaluate_metrics(orig_path, recon_path):
    """
    Evaluates PSNR, SSIM, and LPIPS between original and reconstructed image.
    Higher PSNR/SSIM and lower LPIPS means better reconstruction (weaker protection).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(orig_path) or not os.path.exists(recon_path):
        print("Original or reconstructed image missing. Run HARVIM and an inpainting method first.")
        return
        
    img_orig = Image.open(orig_path).convert("RGB")
    img_recon = Image.open(recon_path).convert("RGB")
    
    transform = transforms.ToTensor()
    t_orig = transform(img_orig).unsqueeze(0).to(device)
    t_recon = transform(img_recon).unsqueeze(0).to(device)
    
    # 1. PSNR
    psnr_val = compute_psnr(t_recon, t_orig).item()
    
    # 2. SSIM
    # Converting to numpy (H, W, C) for scikit-image
    np_orig = t_orig.squeeze().permute(1, 2, 0).cpu().numpy()
    np_recon = t_recon.squeeze().permute(1, 2, 0).cpu().numpy()
    
    ssim_val = metrics.structural_similarity(
        np_orig, np_recon, channel_axis=2, data_range=1.0
    )
    
    # 3. LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # LPIPS expects input in [-1, 1]
    t_orig_lpips = t_orig * 2.0 - 1.0
    t_recon_lpips = t_recon * 2.0 - 1.0
    lpips_val = loss_fn_vgg(t_orig_lpips, t_recon_lpips).item()
    
    print("-" * 40)
    print("Evaluation Results")
    print("PSNR (Peak Signal-to-Noise Ratio): {:.4f} dB".format(psnr_val))
    print("SSIM (Structural Similarity):     {:.4f}".format(ssim_val))
    print("LPIPS (Perceptual Similarity):    {:.4f}".format(lpips_val))
    print("-" * 40)
    print("Note: In HARVIM, we want *lower* PSNR/SSIM and *higher* LPIPS to indicate successful protection (poor reconstruction by attackers).")

if __name__ == "__main__":
    # For demonstration, compare the watermarked image to original, 
    # to simulate a failed "blind" removal.
    evaluate_metrics("outputs/original_image.png", "outputs/watermarked_output.png")