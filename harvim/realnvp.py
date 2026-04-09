import torch
import torch.nn as nn
import numpy as np

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class AffineCouplingLayer(nn.Module):
    def __init__(self, num_channels, mask_type='channel', mask_even=True):
        super().__init__()
        self.mask_type = mask_type
        self.mask_even = mask_even
        
        # A simple ResNet to compute scale and translation
        if mask_type == 'channel':
            if mask_even:
                in_channels = num_channels // 2
                out_channels = (num_channels - num_channels // 2) * 2
            else:
                in_channels = num_channels - num_channels // 2
                out_channels = (num_channels // 2) * 2
        else:
            in_channels = num_channels
            out_channels = num_channels * 2
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResNetBlock(64),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def get_mask(self, x):
        b, c, h, w = x.shape
        mask = torch.zeros((1, c, h, w), device=x.device)
        
        if self.mask_type == 'channel':
            if self.mask_even:
                mask[:, :c//2, :, :] = 1
            else:
                mask[:, c//2:, :, :] = 1
        elif self.mask_type == 'checkerboard':
            # Create checkerboard
            coords = torch.arange(h, device=x.device).unsqueeze(-1) + torch.arange(w, device=x.device).unsqueeze(0)
            if self.mask_even:
                idx = (coords % 2 == 0).float()
            else:
                idx = (coords % 2 == 1).float()
            mask[:, :, :, :] = idx.unsqueeze(0).unsqueeze(0)
            
        return mask

    def forward(self, x):
        mask = self.get_mask(x)
        x_masked = x * mask
        
        if self.mask_type == 'channel':
            x_in = x[:, :x.shape[1]//2] if self.mask_even else x[:, x.shape[1]//2:]
        else:
            x_in = x_masked
            
        out = self.net(x_in)
        
        if self.mask_type == 'channel':
            s, t = out.chunk(2, dim=1)
            # Expand to full shape
            s_full = torch.zeros_like(x)
            t_full = torch.zeros_like(x)
            if self.mask_even:
                s_full[:, x.shape[1]//2:] = s
                t_full[:, x.shape[1]//2:] = t
            else:
                s_full[:, :x.shape[1]//2] = s
                t_full[:, :x.shape[1]//2] = t
            s, t = s_full, t_full
        else:
            s, t = out.chunk(2, dim=1)
            
        s = self.scale * torch.tanh(s)
        s = s * (1 - mask)
        t = t * (1 - mask)
        
        z = x_masked + (x * torch.exp(s) + t) * (1 - mask)
        log_det_jacobian = s.view(s.shape[0], -1).sum(dim=-1)
        
        return z, log_det_jacobian

class RealNVP(nn.Module):
    def __init__(self, num_channels=3, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # Interleave masking strategies
        for i in range(num_layers):
            mask_type = 'checkerboard' if i < 2 else 'channel'
            mask_even = (i % 2 == 0)
            self.layers.append(AffineCouplingLayer(num_channels, mask_type=mask_type, mask_even=mask_even))
            
        # Base distribution (Standard Normal)
        self.register_buffer('base_mean', torch.zeros(1))
        self.register_buffer('base_var', torch.ones(1))

    def prior_log_prob(self, z):
        # log p(z) for standard normal
        log_prob = -0.5 * (z ** 2 + np.log(2 * np.pi))
        return log_prob.view(z.shape[0], -1).sum(dim=-1)

    def forward(self, x):
        """
        Computes log p_G(x)
        """
        z = x
        total_log_det = 0
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det += log_det
            
        log_p_z = self.prior_log_prob(z)
        log_p_x = log_p_z + total_log_det
        return log_p_x
