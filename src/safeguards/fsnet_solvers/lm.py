import torch
import torch.nn.functional as F
from typing import Optional


class LMSolverConfig:
    """Configuration for Levenberg-Marquardt Solver."""
    def __init__(
        self,
        max_iter: int = 5,
        damping_init: float = 1e-3,
        damping_up: float = 20.0,  # Faster rejection of bad steps
        damping_down: float = 0.1,
        val_tol: float = 1e-6,
        grad_tol: float = 1e-6,
        verbose: bool = False,
        **kwargs
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
    **kwargs
) -> torch.Tensor:
    """
    GPU-optimized Levenberg-Marquardt solver - uses direct linear algebra.
    
    Key optimizations:
    - Pre-allocated identity matrix (no recreation per iteration)
    - Proper lambda broadcasting shape (B, 1, 1)
    - Fused ReLU instead of torch.where
    - Deferred convergence check to avoid CPU-GPU sync
    - Numerical stability improvements
    """
    if config is None:
        config = LMSolverConfig(**kwargs)

    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    y = y_init.clone()
    
    # Pre-allocate identity matrix once (GPU optimization)
    eye_n = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)  # (1, n, n)
    
    # Damping parameters per batch sample - shape (B, 1, 1) for broadcasting
    lambdas = torch.full((B, 1, 1), config.damping_init, device=device, dtype=dtype)
    
    # Get constraint matrices
    # For ZonotopeData: eq_resid = C @ y - d, ineq_resid = relu(A @ y - b)
    # For PolytopeData: no eq constraints, ineq_resid = relu(A @ y - b)
    # Jacobian of eq_resid is just C
    # Jacobian of ineq_resid is A where A @ y > b
    
    if hasattr(data, 'C') and data.C is not None:
        C = data.C  # (B, m_eq, n) - equality constraint matrix
        d = data.d  # (B, m_eq, 1) - equality constraint RHS
    else:
        C = None
        d = None
    
    A = data.A  # (B, m_ineq, n) - inequality constraint matrix
    b = data.b  # (B, m_ineq, 1) - inequality constraint RHS
    
    # Get dimensions
    m_ineq = A.shape[1]
    m_eq = C.shape[1] if C is not None else 0
    
    for k in range(config.max_iter):
        # Compute residuals (vectorized)
        y_expanded = y.unsqueeze(2)  # (B, n, 1)
        
        # Equality constraints: C @ y - d = 0
        if C is not None:
            eq_r = (C @ y_expanded - d).squeeze(2)  # (B, m_eq)
        else:
            eq_r = torch.empty((B, 0), device=device, dtype=dtype)  # (B, 0)
        
        # Inequality constraints: A @ y - b <= 0
        ineq_raw = (A @ y_expanded - b).squeeze(2)  # (B, m_ineq)
        
        # Active set mask for inequality constraints
        active_mask = (ineq_raw > 0)  # (B, m_ineq) - boolean mask
        
        # Use fused ReLU (GPU-optimized kernel)
        ineq_r = F.relu(ineq_raw)  # (B, m_ineq)
        
        # Stack residuals
        r = torch.cat([eq_r, ineq_r], dim=1)  # (B, m)
        current_loss = 0.5 * (r ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        
        # Build Jacobian analytically
        # J_eq = C for equality constraints
        # J_ineq = A where active, 0 elsewhere (efficient masking)
        A_masked = A * active_mask.unsqueeze(2)  # (B, m_ineq, n)
        if C is not None:
            J = torch.cat([C, A_masked], dim=1)  # (B, m, n)
        else:
            J = A_masked  # (B, m_ineq, n)
        
        # Compute gradient and Gauss-Newton Hessian (fused operations)
        JT = J.transpose(1, 2)  # (B, n, m)
        JTr = torch.bmm(JT, r.unsqueeze(2)).squeeze(2)  # (B, n)
        
        # Early termination check (avoid expensive solve if converged)
        grad_norm_sq = (JTr ** 2).sum(dim=1, keepdim=True)
        if (grad_norm_sq < config.grad_tol ** 2).all():
            break
        
        JTJ = torch.bmm(JT, J)  # (B, n, n)
        
        # Add damping + stability (proper broadcasting)
        H_damped = JTJ + lambdas * eye_n + 1e-8 * eye_n
        
        # Use LU decomposition (faster than Cholesky for ill-conditioned systems)
        try:
            delta = torch.linalg.solve(H_damped, -JTr.unsqueeze(2)).squeeze(2)
        except RuntimeError:
            # Fallback: use gradient descent step with diagonal approximation
            diag = torch.diagonal(JTJ, dim1=1, dim2=2) + lambdas.squeeze() + 1e-4
            delta = -JTr / diag
        
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
        
        # Update damping (vectorized, no CPU sync)
        lambdas = torch.where(
            improved.view(B, 1, 1),
            lambdas * config.damping_down,
            lambdas * config.damping_up
        )
        lambdas = torch.clamp(lambdas, min=1e-8, max=1e5)
        
        # DEFERRED: convergence check removed to avoid .all() CPU-GPU sync
        # Check only happens at the end or when verbose printing
        
        if config.verbose and k % 2 == 0:
            # Only sync for printing (when needed)
            print(f"Iter {k:3d}: loss = {current_loss.mean().item():.3e}, "
                  f"grad_norm = {JTr.abs().max().item():.3e}")
    
    # Optional: check convergence at the end (single sync)
    if config.verbose:
        final_loss = current_loss.mean().item()
        final_grad = JTr.abs().max().item()
        if final_loss < config.val_tol or final_grad < config.grad_tol:
            print(f"Converged: loss={final_loss:.3e}, grad={final_grad:.3e}")
    
    return y


def nondiff_lm_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LMSolverConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Non-differentiable LM solver with aggressive in-place optimizations.
    
    Much faster than batch_lm_solve due to:
    - Pre-allocated buffers reused across iterations
    - In-place operations (no autograd overhead)
    - Reduced memory allocations
    """
    if config is None:
        config = LMSolverConfig(**kwargs)
    
    with torch.no_grad():
        B, n = y_init.shape
        device, dtype = y_init.device, y_init.dtype
        y = y_init.clone()
        
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
        
        # Pre-allocate buffers for reuse (avoid repeated allocations)
        m_ineq = A.shape[1]
        m_eq = C.shape[1] if C is not None else 0
        m = m_eq + m_ineq
        
        # Pre-allocate working tensors
        y_expanded = torch.empty((B, n, 1), device=device, dtype=dtype)
        ineq_raw = torch.empty((B, m_ineq), device=device, dtype=dtype)
        r = torch.empty((B, m), device=device, dtype=dtype)
        J = torch.empty((B, m, n), device=device, dtype=dtype)
        
        # Pre-compute equality part of Jacobian (constant)
        if C is not None:
            J[:, :m_eq, :] = C
        
        for k in range(config.max_iter):
            # Reuse y_expanded buffer (in-place update)
            y_expanded.copy_(y.unsqueeze(2))
            
            # Equality constraints: C @ y - d = 0 (in-place)
            if C is not None:
                torch.bmm(C, y_expanded, out=r[:, :m_eq].unsqueeze(2)).squeeze_(2).sub_(d.squeeze(2))
            
            # Inequality constraints: A @ y - b <= 0 (in-place)
            torch.bmm(A, y_expanded, out=ineq_raw.unsqueeze(2)).squeeze_(2).sub_(b.squeeze(2))
            
            # Active set mask and ReLU (combined)
            active_mask = (ineq_raw > 0)  # (B, m_ineq) - boolean mask
            
            # Use ReLU and copy to residual buffer
            r[:, m_eq:] = F.relu(ineq_raw)
            
            # Current loss
            current_loss = 0.5 * (r ** 2).sum(dim=1, keepdim=True)  # (B, 1)
            
            # Build Jacobian inequality part (in-place masking)
            # Only update inequality part (equality part is pre-computed)
            J[:, m_eq:, :] = A
            J[:, m_eq:, :].mul_(active_mask.unsqueeze(2))  # Zero out inactive constraints
            
            # Compute gradient and Gauss-Newton Hessian (fused operations)
            JT = J.transpose(1, 2)  # (B, n, m)
            JTr = torch.bmm(JT, r.unsqueeze(2)).squeeze(2)  # (B, n)
            
            # Early termination check
            grad_norm_sq = (JTr ** 2).sum(dim=1, keepdim=True)
            if (grad_norm_sq < config.grad_tol ** 2).all():
                break
            
            JTJ = torch.bmm(JT, J)  # (B, n, n)
            
            # Add damping + stability (proper broadcasting)
            H_damped = JTJ + lambdas * eye_n + 1e-8 * eye_n
            
            # Use LU decomposition (faster than Cholesky for ill-conditioned systems)
            try:
                delta = torch.linalg.solve(H_damped, -JTr.unsqueeze(2)).squeeze(2)
            except RuntimeError:
                # Fallback: use gradient descent step with diagonal approximation
                diag = torch.diagonal(JTJ, dim1=1, dim2=2) + lambdas.squeeze() + 1e-4
                delta = -JTr / diag
            
            # Evaluate new point
            y_new = y + delta
            y_new.unsqueeze_(2)  # In-place: (B, n, 1)
            
            # Compute new residuals (reuse buffers where possible)
            r_new = torch.empty_like(r)
            if C is not None:
                torch.bmm(C, y_new, out=r_new[:, :m_eq].unsqueeze(2)).squeeze_(2).sub_(d.squeeze(2))
            
            torch.bmm(A, y_new, out=r_new[:, m_eq:].unsqueeze(2)).squeeze_(2).sub_(b.squeeze(2))
            r_new[:, m_eq:] = F.relu(r_new[:, m_eq:])
            
            y_new.squeeze_(2)  # Back to (B, n)
            new_loss = 0.5 * (r_new ** 2).sum(dim=1, keepdim=True)  # (B, 1)
            
            # Accept/reject (in-place update, fully parallel)
            improved = (new_loss < current_loss).squeeze(1)  # (B,)
            improved_mask = improved.unsqueeze(1)  # (B, 1)
            
            # In-place update for y
            y[improved_mask.expand_as(y)] = y_new[improved_mask.expand_as(y)]
            
            # Update damping (in-place, vectorized)
            lambdas.mul_(torch.where(
                improved.view(B, 1, 1),
                torch.tensor(config.damping_down, device=device, dtype=dtype),
                torch.tensor(config.damping_up, device=device, dtype=dtype)
            ))
            lambdas.clamp_(min=1e-8, max=1e5)
            
            if config.verbose and k % 2 == 0:
                print(f"Iter {k:3d}: loss = {current_loss.mean().item():.3e}, "
                      f"grad_norm = {JTr.abs().max().item():.3e}")
        
        # Optional: check convergence at the end (single sync)
        if config.verbose:
            final_loss = current_loss.mean().item()
            final_grad = JTr.abs().max().item()
            if final_loss < config.val_tol or final_grad < config.grad_tol:
                print(f"Converged: loss={final_loss:.3e}, grad={final_grad:.3e}")
        
        return y


def hybrid_lm_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_diff_iter: int = 5,
    config: Optional[LMSolverConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Hybrid solver: differentiable LM warm-start + non-differentiable refinement.
    
    Uses actual Levenberg-Marquardt in differentiable phase for faster convergence.
    """
    if config is None:
        config = LMSolverConfig(**kwargs)
    
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
    
    y_refined = nondiff_lm_solve(x, y_warm.detach(), data, remaining_config)
    
    # Connect gradient only through differentiable warm-start
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