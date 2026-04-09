import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkCVAE(nn.Module):
    """
    Conditional VAE for generating watermarks (MNIST digits or letters).
    Parameterized by a three-hidden-layer MLP as per Section A.1.
    """
    def __init__(self, condition_dim: int, latent_dim: int, image_size: tuple = (64, 64)):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        flat_dim = image_size[0] * image_size[1]
        
        # Encoder: takes flattened image + conditions (e.g., class, padding_x, padding_y)
        self.enc_fc1 = nn.Linear(flat_dim + condition_dim, 512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_fc3 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder: takes latent code + conditions
        self.dec_fc1 = nn.Linear(latent_dim + condition_dim, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_fc3 = nn.Linear(256, 512)
        self.fc_out = nn.Linear(512, flat_dim)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        x_flat = x.view(x.size(0), -1)
        inputs = torch.cat([x_flat, c], dim=-1)
        h = F.relu(self.enc_fc1(inputs))
        h = F.relu(self.enc_fc2(h))
        h = F.relu(self.enc_fc3(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        inputs = torch.cat([z, c], dim=-1)
        h = F.relu(self.dec_fc1(inputs))
        h = F.relu(self.dec_fc2(h))
        h = F.relu(self.dec_fc3(h))
        out_flat = torch.sigmoid(self.fc_out(h))
        return out_flat.view(out_flat.size(0), 1, self.image_size[0], self.image_size[1])

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

class LearneableWatermark(nn.Module):
    """
    Wraps the trained CVAE so HARVIM can optimize the latent code and the padding ratios.
    """
    def __init__(self, cvae: WatermarkCVAE, initial_c: torch.Tensor, class_dim: int):
        super().__init__()
        self.cvae = cvae
        
        for param in self.cvae.parameters():
            param.requires_grad = False
            
        # Initialize latent code z arbitrarily
        self.z = nn.Parameter(torch.randn(1, cvae.latent_dim))
        
        # The condition vector consists of [class_one_hot, padding_left, padding_bottom]
        # We only want to optimize the padding ratios
        self.class_cond = initial_c[:, :class_dim].clone().detach() # Fixed
        self.padding_cond = nn.Parameter(initial_c[:, class_dim:].clone().detach()) # Learnable

    def forward(self):
        # Constrain padding inside [0, 1] gracefully using sigmoid or clamping
        pad_clamped = torch.clamp(self.padding_cond, 0.0, 1.0)
        c = torch.cat([self.class_cond, pad_clamped], dim=-1)
        m = self.cvae.decode(self.z, c)
        return m
