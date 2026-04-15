import torch
import torch.nn.functional as F

def create_differentiable_mask(m: torch.Tensor, alpha: float = 0.15, beta: float = 0.01) -> torch.Tensor:
    """
    Creates the differentiable diagonal mask matrix A_m from Eq (4).
    A_m = diag(sig((m - alpha) / beta))
    
    Args:
        m (torch.Tensor): Continuous-valued watermark tensor in [0, 1].
        alpha (float): Smoothing factor, default 0.15.
        beta (float): Smoothing factor, default 0.01.
        
    Returns:
        torch.Tensor: The differentiable mask tensor with values squashed between 0 and 1.
    """
    return 1-torch.sigmoid((m - alpha) / beta)

def watermark_regularizer(m: torch.Tensor, coeff: float = 0.001) -> torch.Tensor:
    """
    Computes the regularization R(m) = ||m||_1 to penalize excessive watermark size.
    
    Args:
        m (torch.Tensor): Watermark tensor.
        coeff (float): Regularization strength, default 0.001.
        
    Returns:
        torch.Tensor: L1 regularization loss.
    """
    return coeff * torch.norm(m, p=1)

def compute_psnr(x_recon: torch.Tensor, x_target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR). HARVIM seeks to minimize this.
    """
    mse = F.mse_loss(x_recon, x_target)
    if mse == 0:
        return torch.tensor(float('inf')).to(x_recon.device)
    return 20 * torch.log10(max_val / torch.sqrt(mse))