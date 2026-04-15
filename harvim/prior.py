import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class MLEObjective(nn.Module):
    def __init__(self, prior_model: nn.Module, sigma_sq: float):
        """
        Args:
            prior_model (nn.Module): The pre-trained Deep Generative Model (e.g., Real-NVP).
                                     Must return log p_G(x).
            sigma_sq (float): known variance of the isotropic Gaussian noise.
        """
        super().__init__()
        self.prior = prior_model
        self.sigma_sq = sigma_sq
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, A_m: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Computes the log-posterior: log p(x | y; \lambda) 
        = log p_e(y - A_m x) + \lambda log p_G(x)
        """
        # Data term: log-likelihood of Gaussian noise e = y - A_m * x
        residual = y - A_m * x
        # log p_e = - ||residual||^2 / (2 * sigma^2) + const
        log_pe = -torch.sum(residual ** 2) / (2 * self.sigma_sq)
        img = (A_m*x).squeeze(0).cpu().detach().numpy().transpose(1,2,0)
        img = img - img.min()
        img = img / img.max()
        plt.imsave("watermarked_image.png", img, cmap='gray')
        # Prior term: \lambda * log p_G(x)
        log_pG = self.prior(x)[0]
        
        return log_pe + lam * log_pG