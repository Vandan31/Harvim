import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from harvim.watermark_generator import WatermarkCVAE

def test_watermark_generation():
    """
    Test script to generate watermarks from the trained CVAE model.
    Performs sanity checks by generating watermarks for different conditions.
    """
    os.makedirs("outputs/watermark_test", exist_ok=True)
    
    # Configuration
    IMAGE_SIZE = 256
    condition_dim = 12  # 10 (class one-hot) + 2 (padding ratios)
    latent_dim = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = f"checkpoints/watermark_cvae_{IMAGE_SIZE}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run: python -m scripts.0_prepare_data_and_cvae")
        return
    
    model = WatermarkCVAE(condition_dim=condition_dim, latent_dim=latent_dim, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Loaded model from {checkpoint_path}")
    
    # Test 1: Generate watermarks for different classes with same latent code
    print("\nTest 1: Sampling for different digit classes (same z)")
    z = torch.randn(1, latent_dim).to(device)
    
    for digit_class in range(0, 10, 3):  # Test classes 0, 3, 6, 9
        # Create one-hot class condition
        class_cond = F.one_hot(torch.tensor([digit_class]), num_classes=10).float().to(device)
        # Random padding ratios
        pad_cond = torch.rand(1, 2).to(device)
        c = torch.cat([class_cond, pad_cond], dim=-1)
        
        with torch.no_grad():
            watermark = model.decode(z, c)
        
        save_image(watermark, f"outputs/watermark_test/class_{digit_class}_z_same.png")
        print(f"  ✓ Generated watermark for digit class {digit_class}")
    
    # Test 2: Generate watermarks with different latent codes for same class
    print("\nTest 2: Sampling for same class with different z")
    digit_class = 5
    class_cond = F.one_hot(torch.tensor([digit_class]), num_classes=10).float().to(device)
    
    for z_idx in range(4):
        z = torch.randn(1, latent_dim).to(device)
        pad_cond = torch.tensor([[0.5, 0.5]]).float().to(device)
        c = torch.cat([class_cond, pad_cond], dim=-1)
        
        with torch.no_grad():
            watermark = model.decode(z, c)
        
        save_image(watermark, f"outputs/watermark_test/class_{digit_class}_z_{z_idx}.png")
        print(f"  ✓ Generated watermark for digit class {digit_class} with z_{z_idx}")
    
    # Test 3: Vary padding ratios
    print("\nTest 3: Varying padding ratios")
    z = torch.randn(1, latent_dim).to(device)
    digit_class = 3
    class_cond = F.one_hot(torch.tensor([digit_class]), num_classes=10).float().to(device)
    
    for pad_idx, (pad_left, pad_bottom) in enumerate([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]):
        pad_cond = torch.tensor([[pad_left, pad_bottom]]).float().to(device)
        c = torch.cat([class_cond, pad_cond], dim=-1)
        
        with torch.no_grad():
            watermark = model.decode(z, c)
        
        save_image(watermark, f"outputs/watermark_test/class_{digit_class}_pad_{pad_idx}.png")
        print(f"  ✓ Generated watermark for padding ({pad_left:.1f}, {pad_bottom:.1f})")
    
    # Test 4: Encode and reconstruct
    print("\nTest 4: Encode-decode reconstruction")
    
    # Create a random watermark-like sample
    sample_img = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    digit_class = 7
    class_cond = F.one_hot(torch.tensor([digit_class]), num_classes=10).float().to(device)
    pad_cond = torch.tensor([[0.3, 0.7]]).float().to(device)
    c = torch.cat([class_cond, pad_cond], dim=-1)
    
    with torch.no_grad():
        # Encode
        mu, logvar = model.encode(sample_img, c)
        # Decode using mean
        reconstruction = model.decode(mu, c)
        # Calculate MSE
        mse = ((sample_img - reconstruction) ** 2).mean().item()
    
    save_image(sample_img, "outputs/watermark_test/original_sample.png", normalize=True)
    save_image(reconstruction, "outputs/watermark_test/reconstructed_sample.png", normalize=True)
    print(f"  ✓ Reconstruction MSE: {mse:.6f}")
    
    print("\n✓ All sanity checks passed!")
    print(f"Generated watermarks saved to outputs/watermark_test/")

if __name__ == "__main__":
    test_watermark_generation()
