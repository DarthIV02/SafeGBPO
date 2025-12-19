from torch.optim import LBFGS
from dataclasses import dataclass
from typing import Callable
import torch

class LBFGSSolverConfig:
    """Configuration for FSNet L-BFGS Solver."""
    def __init__(self,
        memory: int = 10,
        lr: float = 1.0,
        max_norm: float = 2.0,
        max_iter: int = 10,
        max_diff_iter: int = 5,
        **kwargs
    ):
        self.memory = memory
        self.lr = lr
        self.max_norm = max_norm
        self.max_iter = max_iter
        self.max_diff_iter = max_diff_iter

def _create_objective_function(x: torch.Tensor, data, scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create objective function closure."""
    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq_residual = (data.eq_resid(x, y) ** 2).sum(dim=1).mean(0)
        ineq_residual = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
        return scale * (eq_residual + ineq_residual)
    return _obj

def _lbfgs_step(
    y_init: torch.Tensor,
    obj_fn: Callable,
    max_norm: float,
    max_iter: int,
    lr: float,
    memory: int,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Single LBFGS optimization step.
    
    Args:
        y_init: Initial point
        obj_fn: Objective function
        max_norm: Gradient clipping norm
        max_iter: Max iterations
        lr: Learning rate
        memory: LBFGS history size
        create_graph: Whether to create backward graph
    """
    a = y_init.clone().detach().requires_grad_(True)
    
    optimizer = LBFGS([a], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        phi = obj_fn(a)
        
        # Check for non-finite values in loss
        if not torch.isfinite(phi).all():
            return torch.tensor(1e10, device=phi.device, dtype=phi.dtype)
        
        # Use autograd.grad instead of backward to avoid memory leak
        if create_graph:
            grad = torch.autograd.grad(phi.mean(), a, create_graph=True)[0]
        else:
            grad = torch.autograd.grad(phi.mean(), a, create_graph=False)[0]
        
        # Check for non-finite gradients and sanitize
        if not torch.isfinite(grad).all():
            grad = grad.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip gradient and assign to a.grad
        grad_norm = grad.norm()
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        
        a.grad = grad
        
        return phi
    
    optimizer.step(closure)
    return a

def lbfgs_torch_solve(    
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: LBFGSSolverConfig = None,
    **kwargs
) -> torch.Tensor:
    """
    Hybrid Feasibility Seeking Step using L-BFGS on Violation Function.
    
    Runs differentiable L-BFGS for max_diff_iter iterations, then
    non-differentiable L-BFGS for remaining iterations. This saves
    memory while maintaining gradient flow through early iterations.
    
    Args:
        x: Input tensor (batch_size, input_dim)
        y_init: Initial action (batch_size, action_dim)
        data: Data object with eq_resid and ineq_resid methods
        config: Solver configuration
    """
    if config is None:
        config = LBFGSSolverConfig(**kwargs)

    obj_fn = _create_objective_function(x, data, 1.0)
    
    # Differentiable phase
    a_diff = _lbfgs_step(
        y_init,
        obj_fn,
        config.max_norm,
        config.max_diff_iter,
        config.lr,
        config.memory,
        create_graph=True
    )
    
    # Non-differentiable phase (if more iterations needed)
    if config.max_iter > config.max_diff_iter:
        remaining_iter = config.max_iter - config.max_diff_iter
        a_nondiff = _lbfgs_step(
            a_diff.detach(),
            obj_fn,
            config.max_norm,
            remaining_iter,
            config.lr,
            config.memory,
            create_graph=False
        )
        # Connect gradients only through differentiable phase
        return a_diff + (a_nondiff - a_diff).detach()
    else:
        return a_diff   

def nondiff_lbfgs_torch_solve(    
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: LBFGSSolverConfig = None,
    **kwargs
) -> torch.Tensor:
    """
    Non-differentiable version (no backward graph).
    
    Args:
        x: Input tensor (batch_size, input_dim)
        y_init: Initial action (batch_size, action_dim)
        data: Data object with eq_resid and ineq_resid methods
        config: Solver configuration
    """
    if config is None:
        config = LBFGSSolverConfig(**kwargs)

    obj_fn = _create_objective_function(x, data, 1.0)
    
    a = _lbfgs_step(
        y_init,
        obj_fn,
        config.max_norm,
        config.max_iter,
        config.lr,
        config.memory,
        create_graph=False
    )
    
    return a.detach()