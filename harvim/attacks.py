import torch
from torch.optim import Adam
from harvim.prior import MLEObjective

class FlowR:
    """
    Implements Flow-R watermark removal attack (worst-case class).
    It has access to the ground truth watermark mask (A_m) and uses the same 
    flow-based generative model G to solve the inpainting inverse problem.
    """
    def __init__(self, generative_prior: torch.nn.Module, sigma_sq: float = 0.01):
        self.objective_fn = MLEObjective(generative_prior, sigma_sq)

    def remove_watermark(self, 
                         y: torch.Tensor, 
                         A_m: torch.Tensor, 
                         lam: float = 1.0, 
                         steps: int = 1000, 
                         lr: float = 0.01) -> torch.Tensor:
        """
        Removes the watermark by optimizing a randomly initialized image x 
        to maximize the posterior probability log p(x | y; \lambda).
        
        Args:
            y: The watermarked observation (x_T * A_m + noise)
            A_m: The ground truth location of watermarks (binary or near-binary mask)
            lam: Hyperparameter controlling the weight of the prior G
            steps: Number of gradient descent steps
            lr: Learning rate
            
        Returns:
            The reconstructed image without the watermark.
        """
        # "Flow-R uses the same flow-based model G to solve the inpainting task 
        # with random initialized x [48]"
        x_opt = torch.randn_like(y, requires_grad=True)
        optimizer = Adam([x_opt], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            
            # log_p computes log p_e(y - A_m * x) + \lambda log p_G(x)
            log_p = self.objective_fn(x_opt, y, A_m, lam)
            
            # We want to MAXIMIZE the log posterior, so we minimize its negative
            loss = -log_p
            loss.backward()
            optimizer.step()
            
            # Optional: clamp x to valid image range during optimization
            with torch.no_grad():
                x_opt.clamp_(0, 1)

        return x_opt.detach()
