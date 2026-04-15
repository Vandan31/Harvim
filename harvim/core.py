import torch
from torch.optim import AdamW
import torch.autograd as autograd
from tqdm import tqdm
from .utils import create_differentiable_mask, watermark_regularizer, compute_psnr
from .prior import MLEObjective

class HARVIM:
    def __init__(self, 
                 generative_prior: torch.nn.Module, 
                 watermark_generator: torch.nn.Module,
                 sigma_sq: float = 0.01,
                 alpha: float = 0.15,
                 beta: float = 0.01,
                 reg_coeff: float = 0.001):
        """
        Initializes the HARVIM pipeline.
        """
        self.objective_fn = MLEObjective(generative_prior, sigma_sq)
        self.watermark_generator = watermark_generator
        self.sigma_sq = sigma_sq
        self.alpha = alpha
        self.beta = beta
        self.reg_coeff = reg_coeff

    def run(self, 
            x_T: torch.Tensor, 
            target_lambda: float = 1.0, 
            T_steps: int = 100, 
            K_unroll: int = 1, 
            lr: float = 0.05) -> torch.Tensor:
        """
        Algorithm 1: HARVIM algorithm
        
        Args:
            x_T: Copyrighted image x_T
            target_lambda: Final prior weight \lambda > 0
            T_steps: Number of lambda update steps
            K_unroll: Meta-learning unrolled gradient descent steps
            lr: Learning rate for watermark parameters
            
        Returns:
            m_t: The learned optimal watermark
        """
        device = x_T.device
        
        # Parameterize m using the CVAE (watermark generator framework)
        # Assuming the generator takes latent codes & location params
        m_params = [p for p in self.watermark_generator.parameters() if p.requires_grad]
        optimizer = AdamW(m_params, lr=lr)
        
        # Initialize: lambda_0 = 0
        lam_t = 0.0
        lam_step = target_lambda / T_steps
        
        # Generate initial m_0
        m_t = self.watermark_generator()
        
        # Find MLE solution x_0 for y_0 with lambda=0 (independent of m0)
        x_tilde = x_T.clone().detach().requires_grad_(True)
        # simplified assumed mle burn-in: 
        # x_tilde could be solved by standard MAP optimization prior to loop
        pbar = tqdm(range(1, T_steps + 1), desc="Running HARVIM Optimization")
        for t in pbar:
            optimizer.zero_grad()
            
            # Step 5: Treat y_{t-1}(m_{t-1}) = A_m * x_T + e as function of m_{t-1}
            m_t = self.watermark_generator()
            A_m = create_differentiable_mask(m_t, self.alpha, self.beta)
            e = torch.randn_like(x_T) * torch.sqrt(torch.tensor(self.sigma_sq))
            y_t_minus_1 = A_m * x_T + e
            
            # Step 6: Update lambda
            lam_t += lam_step
            
            # Unrolled inner optimization for \tilde{x}_t
            # We must maintain the computation graph connecting x_tilde to m_t
            x_tilde_k = x_tilde.clone().detach().requires_grad_(True)
            
            # Step 8-10: K-step gradient descent to approximate \tilde{x}_t(m_{t-1}, \lambda_t)
            for k in range(K_unroll):
                log_p = self.objective_fn(x_tilde_k, y_t_minus_1, A_m, lam_t)
                
                # Compute gradient wrt x: \nabla_x log p(x | y; \lambda)
                grad_x = autograd.grad(log_p, x_tilde_k, create_graph=True)[0]
                
                # Gradient ASCENT step because we are doing MLE/MAP (maximizing log p)
                # Using a fixed step size for inner loop (could be parameterized)
                inner_lr = 0.01 
                x_tilde_k = x_tilde_k + inner_lr * grad_x
            
            # Step 11: Current solution \tilde{x}_t
            x_recon = x_tilde_k
            
            # Step 12: Upper-level update
            # s(\tilde{x}_t(m_{t-1}; \lambda_t), x_T) + R(m_{t-1})
            # We use PSNR as similarity but we want to minimize it (destroy reconstruction)
            loss_similarity = compute_psnr(x_recon, x_T)
            loss_reg = watermark_regularizer(m_t, self.reg_coeff)
            
            total_loss = loss_similarity + loss_reg
            pbar.set_postfix({
                "lambda": lam_t,
                "PSNR": loss_similarity.item(),
                "Reg": loss_reg.item()
            })
            # Backpropagate through the unrolled graph to update watermark generator
            total_loss.backward()
            optimizer.step()
            
            # Set \tilde{x}_t for next iteration (detached to truncate graph between outer iterations)
            x_tilde = x_recon.detach()

        return self.watermark_generator().detach()