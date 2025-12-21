from torch.optim import LBFGS
from dataclasses import dataclass
from typing import Callable
import torch

from safeguards.fsnet_solvers.lbfgs import lbfgs_solve

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
    """
    Create objective function closure. Importet from FSNet codebase. 
    Even though x is not used in the function, it is kept 
    for compatibility with FSNet structure.
    
    Args:
        x:      Input tensor (batch_size, input_dim)
        y:      Action tensor (batch_size, action_dim)
        data:   Data object with eq_resid and ineq_resid methods
        scale:  Scaling factor for the objective

    Returns:
        Objective function that takes y and returns the scaled residual loss
        """

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
    memory: int
    ) -> torch.Tensor:
    """
    LBFGS step that preserves gradient connection to y_init.
    
    Key idea: Optimize a delta (correction) tensor, then add it to y_init.
    This keeps y_init in the computation graph!
    
    Args:
        y_init: Initial point (can be non-leaf, gradient connection preserved!)
        obj_fn: Objective function
        max_norm: Gradient clipping norm
        max_iter: Max iterations
        lr: Learning rate
        memory: LBFGS history size
        create_graph: Whether to create backward graph
    """
    # Optimizing delta for y_init + delta
    # y_init stays in the graph, delta is the leaf tensor for optimizer
    delta = torch.zeros_like(y_init, requires_grad=True)
    
    optimizer = LBFGS([delta], lr=lr, max_iter=max_iter, 
                      history_size=memory, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        y_current = y_init + delta
        phi = obj_fn(y_current)
        
        # Check for non-finite values in loss
        if not torch.isfinite(phi).all():
            return torch.tensor(1e10, device=phi.device, dtype=phi.dtype)
        
        # Compute gradient w.r.t. delta
        grad = torch.autograd.grad(phi.mean(), delta, create_graph=False)[0]
        
        # Check for non-finite gradients and sanitize
        if not torch.isfinite(grad).all():
            grad = grad.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip gradient and assign to delta.grad
        grad_norm = grad.norm()
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        
        delta.grad = grad
        
        return phi
    
    optimizer.step(closure)
    
    # Return y_init + delta as the optimized point
    return y_init + delta

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
    memory while maintaining gradient flow through y_init.
    
    Args:
        x: Input tensor (batch_size, input_dim)
        y_init: Initial action (batch_size, action_dim)
        data: Data object with eq_resid and ineq_resid methods
        config: Solver configuration
    """
    if config is None:
        config = LBFGSSolverConfig(**kwargs)

    obj_fn = _create_objective_function(x, data, 1.0)
    
    # use torch implementation of LBFGS if no differentiable steps needed
    if config.max_diff_iter == 0:
        
        a_diff = _lbfgs_step(
            y_init,
            obj_fn,
            config.max_norm,
            config.max_iter,
            config.lr,
            config.memory
        )
        return a_diff
    
    # Differentiable phase via LBFGS Solver from FSNet codebases 
    # there is no way to get the differentiable steps only from the torch LBFGS implementation
    # so we use the original FSNet lbfgs_solve for the differentiable part
    lbfgs_kwargs = kwargs.copy()
    lbfgs_kwargs['max_iter'] = config.max_diff_iter
    a_diff = lbfgs_solve(
            x,
            y_init,
            data,
            **lbfgs_kwargs
        )
    
    # Non-differentiable phase (if more iterations needed)
    if config.max_iter > config.max_diff_iter:
        a_nondiff = _lbfgs_step(
            a_diff.detach(),
            obj_fn,
            config.max_norm,
            config.max_iter - config.max_diff_iter,
            config.lr,
            config.memory
        )
        # Connect gradients through differentiable phase
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
        config.memory
    )
    
    return a.detach()