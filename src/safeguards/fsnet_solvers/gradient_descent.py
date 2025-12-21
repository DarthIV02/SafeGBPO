import torch
from dataclasses import dataclass
from typing import Callable

@dataclass
class GradientDescentSolverConfig:
    """Configuration for FSNet Gradient Descent Solver."""
    def __init__(self,
        k: int = 10,
        kp: int = 5,
        eta: float = 0.1,
        val_tol: float = 1e-4,
        **kwargs
    ):
        self.k = k
        self.kp = kp
        self.eta = eta
        self.val_tol = val_tol



@torch.enable_grad()
def gradient_descent_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: GradientDescentSolverConfig = None,
    **kwargs
) -> torch.Tensor:
    """
    Feasibility Seeking Step using Gradient Descent on Violation Function.
    
    Args:
        a0: Initial action (batch_size, action_dim)
        phi_fn: Function that takes action and returns violation scalar (or batch)
        config: Solver configuration
    """
    if config is None:
        config = GradientDescentSolverConfig(**kwargs)

    is_training = y_init.requires_grad
    K = config.k
    Kp = config.kp
    eta = config.eta

   
    if is_training:
        a = y_init 
    else:
        a = y_init.detach().requires_grad_(True)

    
    for _ in range(min(K, Kp)):
        phi = _create_objective_function(x, data, 1.0)(a).mean()
        grad_a, = torch.autograd.grad(phi, a, create_graph=is_training) 
        a = a - eta * grad_a

    a_diff = a 
    a_nd = a.detach()

    
    for _ in range(Kp, K):
        a_nd.requires_grad_(True)
        
        phi_batch = _create_objective_function(None, data, 1.0)(a_nd)
        
        if phi_batch.max() < config.val_tol:
            break
            
        phi = phi_batch.mean()
        
       
        grad_a, = torch.autograd.grad(phi, a_nd, create_graph=False)
        
        with torch.no_grad():
            a_nd = a_nd - eta * grad_a

    a_hat = a_diff + (a_nd - a_diff).detach()
    
    return a_hat

def _create_objective_function(x: torch.Tensor, data, scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create objective function closure."""
    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq_residual = (data.eq_resid(x, y) ** 2).sum(dim=1).mean(0)
        ineq_residual = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
        return scale * (eq_residual + ineq_residual)
    return _obj