import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import random

from harvim.watermark_generator import WatermarkCVAE

class PaddedMNIST(Dataset):
    def __init__(self, mnist_dataset, image_size=(64, 64)):
        self.mnist = mnist_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # MNIST is 28x28, we need to pad it to image_size (e.g., 64x64)
        # Randomly choose padding left and bottom
        max_pad_h = self.image_size[0] - img.shape[1]
        max_pad_w = self.image_size[1] - img.shape[2]
        
        pad_left = random.randint(0, max_pad_w)
        pad_bottom = random.randint(0, max_pad_h)
        
        pad_right = max_pad_w - pad_left
        pad_top = max_pad_h - pad_bottom
        
        # pad is (padding_left, padding_right, padding_top, padding_bottom)
        padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
        
        # Calculate padding ratios in [0, 1]
        pad_left_ratio = pad_left / max_pad_w if max_pad_w > 0 else 0.0
        pad_bottom_ratio = pad_bottom / max_pad_h if max_pad_h > 0 else 0.0
        
        # Create conditional vector (10 classes for digits + 2 padding ratios)
        loc_cond = torch.tensor([pad_left_ratio, pad_bottom_ratio], dtype=torch.float32)
        
        return padded_img, label, loc_cond

def train_cvae():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Prepare Data
    print("Downloading and preparing MNIST for CVAE training...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    padded_dataset = PaddedMNIST(mnist_train, image_size=(64, 64))
    dataloader = DataLoader(padded_dataset, batch_size=128, shuffle=True)

    # 2. Initialize Model
    # Condition dim = 10 (one-hot digit) + 2 (padding ratios)
    condition_dim = 12
    latent_dim = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = WatermarkCVAE(condition_dim=condition_dim, latent_dim=latent_dim, image_size=(64, 64)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x.view(-1, 64*64), x.view(-1, 64*64), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # 3. Train
    print("Training CVAE...")
    epochs = 10
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels, loc_cond) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            # Create one-hot class condition
            class_cond = F.one_hot(labels, num_classes=10).float().to(device)
            loc_cond = loc_cond.to(device)
            
            c = torch.cat([class_cond, loc_cond], dim=-1)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, c)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch {epoch+1} loss: {train_loss / len(dataloader.dataset):.4f}")

    torch.save(model.state_dict(), "checkpoints/watermark_cvae.pth")
    print("CVAE training complete and saved to checkpoints/watermark_cvae.pth")

if __name__ == "__main__":
    train_cvae()