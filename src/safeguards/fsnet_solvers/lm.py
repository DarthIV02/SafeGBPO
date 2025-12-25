import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List


class LMSolverConfig:
    """Configuration for Levenberg-Marquardt Solver."""
    def __init__(
        self,
        max_iter: int = 5,
        damping_init: float = 1e-3,
        damping_up: float = 10.0,
        damping_down: float = 0.1,
        val_tol: float = 1e-6,
        grad_tol: float = 1e-6,
        verbose: bool = False
    ):
        self.max_iter = max_iter
        self.damping_init = damping_init
        self.damping_up = damping_up
        self.damping_down = damping_down
        self.val_tol = val_tol
        self.grad_tol = grad_tol
        self.verbose = verbose


def compute_batched_jacobian(residual_fn, y, x):
    """
    Fast batched Jacobian computation using torch.func.jacrev with proper batching.
    
    For residual function r: (B, n) -> (B, m), computes J where J[b, i, j] = dr_i/dy_j for batch b.
    """
    from torch.func import vmap, jacrev
    
    # Define per-sample Jacobian
    def single_jacobian(y_single, x_single):
        y_single = y_single.unsqueeze(0).requires_grad_(True)
        x_single = x_single.unsqueeze(0) if x_single is not None else None
        r = residual_fn(x_single, y_single).squeeze(0)
        return r
    
    # Use jacrev which is efficient for m > n (typical case)
    jac_fn = jacrev(single_jacobian)
    
    # Apply to batch
    if x is None:
        x_dummy = torch.zeros(y.shape[0], 1, device=y.device, dtype=y.dtype)
        J = vmap(jac_fn, in_dims=(0, None))(y, x_dummy)
    else:
        J = vmap(jac_fn, in_dims=(0, 0))(y, x)
    
    return J


def batch_lm_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LMSolverConfig] = None,
    debug_trajectory: bool = False,  # Added argument
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:  # Updated return type
    """
    GPU-optimized Levenberg-Marquardt solver - uses direct linear algebra.
    """
    # Clean kwargs to avoid passing debug_trajectory to Config
    if "debug_trajectory" in kwargs:
        kwargs.pop("debug_trajectory")

    if config is None:
        config = LMSolverConfig(**kwargs)

    trajectory = []

    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    y = y_init.clone()
    
    # Capture initial state
    if debug_trajectory:
        trajectory.append(y.detach().cpu().clone())
    
    # Pre-allocate identity matrix once (GPU optimization)
    eye_n = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)  # (1, n, n)
    
    # Damping parameters per batch sample - shape (B, 1, 1) for broadcasting
    lambdas = torch.full((B, 1, 1), config.damping_init, device=device, dtype=dtype)
    
    # Get constraint matrices
    if hasattr(data, 'C') and data.C is not None:
        C = data.C  # (B, m_eq, n) - equality constraint matrix
        d = data.d  # (B, m_eq, 1) - equality constraint RHS
    else:
        C = None
        d = None
    
    A = data.A  # (B, m_ineq, n) - inequality constraint matrix
    b = data.b  # (B, m_ineq, 1) - inequality constraint RHS
    
    for k in range(config.max_iter):
        # Compute residuals (vectorized)
        y_expanded = y.unsqueeze(2)  # (B, n, 1) - expand once
        
        # Equality constraints: C @ y - d = 0
        if C is not None:
            eq_r = (C @ y_expanded - d).squeeze(2)  # (B, m_eq)
        else:
            eq_r = torch.zeros((B, 0), device=device, dtype=dtype)  # (B, 0) - no eq constraints
        
        # Inequality constraints: A @ y - b <= 0
        ineq_raw = (A @ y_expanded - b).squeeze(2)  # (B, m_ineq)
        
        # Active set mask for inequality constraints
        active_mask = (ineq_raw > 0).unsqueeze(2)  # (B, m_ineq, 1)
        
        # Use fused ReLU (GPU-optimized kernel)
        ineq_r = F.relu(ineq_raw)  # (B, m_ineq)
        
        # Stack residuals
        r = torch.cat([eq_r, ineq_r], dim=1)  # (B, m)
        current_loss = 0.5 * (r ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        
        # Build Jacobian analytically
        A_masked = A * active_mask  # (B, m_ineq, n) - broadcast multiply
        if C is not None:
            J = torch.cat([C, A_masked], dim=1)  # (B, m, n)
        else:
            J = A_masked  # (B, m_ineq, n)
        
        # Compute gradient and Gauss-Newton Hessian
        JT = J.transpose(1, 2)  # (B, n, m)
        JTr = torch.bmm(JT, r.unsqueeze(2)).squeeze(2)  # (B, n)
        JTJ = torch.bmm(JT, J)  # (B, n, n)
        
        # Add damping (lambdas already shaped for broadcasting)
        H_damped = JTJ + lambdas * eye_n
        
        # Add numerical stability to ensure positive definiteness
        H_damped = H_damped + 1e-8 * eye_n
        
        # Solve for update (batched Cholesky is GPU-optimized)
        try:
            L = torch.linalg.cholesky(H_damped)
            delta = torch.cholesky_solve(-JTr.unsqueeze(2), L).squeeze(2)
        except RuntimeError:
            # Fallback: add more regularization
            H_damped = H_damped + 1e-4 * eye_n
            delta = torch.linalg.solve(H_damped, -JTr.unsqueeze(2)).squeeze(2)
        
        # Evaluate new point
        y_new = y + delta
        y_new_expanded = y_new.unsqueeze(2)
        
        # Compute new residuals
        if C is not None:
            eq_r_new = (C @ y_new_expanded - d).squeeze(2)
        else:
            eq_r_new = torch.zeros((B, 0), device=device, dtype=dtype)
        
        ineq_r_new = F.relu((A @ y_new_expanded - b).squeeze(2))
        r_new = torch.cat([eq_r_new, ineq_r_new], dim=1)
        new_loss = 0.5 * (r_new ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        
        # Accept/reject (fully parallel per batch)
        improved = (new_loss < current_loss).squeeze(1)  # (B,)
        y = torch.where(improved.unsqueeze(1), y_new, y)
        
        # Capture updated state
        if debug_trajectory:
            trajectory.append(y.detach().cpu().clone())

        # Update damping (vectorized, no CPU sync)
        lambdas = torch.where(
            improved.view(B, 1, 1),
            lambdas * config.damping_down,
            lambdas * config.damping_up
        )
        lambdas = torch.clamp(lambdas, min=1e-8, max=1e5)
        
        if config.verbose and k % 2 == 0:
            print(f"Iter {k:3d}: loss = {current_loss.mean().item():.3e}, "
                  f"grad_norm = {JTr.abs().max().item():.3e}")
    
    # Optional: check convergence at the end (single sync)
    if config.verbose:
        final_loss = current_loss.mean().item()
        final_grad = JTr.abs().max().item()
        if final_loss < config.val_tol or final_grad < config.grad_tol:
            print(f"Converged: loss={final_loss:.3e}, grad={final_grad:.3e}")
    
    if debug_trajectory:
        return y, trajectory
    return y


def nondiff_lm_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LMSolverConfig] = None,
    debug_trajectory: bool = False, # Added argument
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Non-differentiable LM solver."""
    if "debug_trajectory" in kwargs:
        kwargs.pop("debug_trajectory")

    if config is None:
        config = LMSolverConfig(**kwargs)
    
    with torch.no_grad():
        return batch_lm_solve(x, y_init, data, config, debug_trajectory=debug_trajectory)


def hybrid_lm_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_diff_iter: int = 5,
    config: Optional[LMSolverConfig] = None,
    debug_trajectory: bool = False, # Added argument
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Hybrid solver: differentiable LM warm-start + non-differentiable refinement.
    """
    if "debug_trajectory" in kwargs:
        kwargs.pop("debug_trajectory")

    if config is None:
        config = LMSolverConfig(**kwargs)
    
    trajectory = []

    # Warm start with few differentiable LM iterations
    diff_config = LMSolverConfig(
        max_iter=max_diff_iter,
        damping_init=config.damping_init,
        damping_up=config.damping_up,
        damping_down=config.damping_down,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        verbose=False
    )
    
    # Run differentiable LM (keeps gradient graph)
    # We capture trajectory here too if debugging
    if debug_trajectory:
        y_warm, traj_warm = batch_lm_solve(x, y_init, data, diff_config, debug_trajectory=True)
        trajectory.extend(traj_warm)
    else:
        y_warm = batch_lm_solve(x, y_init, data, diff_config)
    
    # Refine with non-differentiable solver for remaining iterations
    remaining_config = LMSolverConfig(
        max_iter=config.max_iter - max_diff_iter,
        damping_init=config.damping_init,
        damping_up=config.damping_up,
        damping_down=config.damping_down,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        verbose=config.verbose
    )
    
    if debug_trajectory:
        y_refined, traj_refined = nondiff_lm_solve(x, y_warm.detach(), data, remaining_config, debug_trajectory=True)
        # Avoid duplicating the connection point if it exists
        if len(trajectory) > 0 and len(traj_refined) > 0:
             trajectory.extend(traj_refined[1:])
        else:
             trajectory.extend(traj_refined)
        
        final_action = y_warm + (y_refined - y_warm).detach()
        return final_action, trajectory
    else:
        y_refined = nondiff_lm_solve(x, y_warm.detach(), data, remaining_config)
        return y_warm + (y_refined - y_warm).detach()


def torch_opt_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_iter: int = 10,
    **kwargs
) -> torch.Tensor:
    """Drop-in replacement."""
    config = LMSolverConfig(max_iter=max_iter, **kwargs)
    return batch_lm_solve(x, y_init, data, config)