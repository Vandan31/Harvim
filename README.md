# HARVIM: Hard-to-Remove Visible Watermarks

A PyTorch implementation of the HARVIM algorithm, which generates visible watermarks that are robust against inpainting-based watermark removal methods.

## Usage

```python
import torch
import torch.nn as nn
from harvim import HARVIM

# Define your prior and watermark generator models
class DummyPrior(nn.Module):
    def forward(self, x):
        return -torch.sum(x**2) # Dummy log p(x)

class DummyWatermarkGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.m = nn.Parameter(torch.rand(size))
    def forward(self):
        return self.m

# Load your image
x_T = torch.rand(1, 3, 64, 64) # Dummy image

prior = DummyPrior()
wm_gen = DummyWatermarkGenerator(x_T.shape)

# Initialize HARVIM
harvim_pipeline = HARVIM(
    generative_prior=prior,
    watermark_generator=wm_gen,
    sigma_sq=0.01,
    alpha=0.15,
    beta=0.01,
    reg_coeff=0.001
)

# Run HARVIM to generate the optimal hard-to-remove watermark
learned_watermark = harvim_pipeline.run(
    x_T=x_T,
    target_lambda=1.0,
    T_steps=100,
    K_unroll=1,
    lr=0.05
)

print(learned_watermark.shape)
```
