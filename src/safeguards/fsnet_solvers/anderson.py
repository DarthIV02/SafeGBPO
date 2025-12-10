import torch
from typing import Optional, Callable
from safeguards.fsnet_solvers.lbfgs import LBFGSConfig

# Reuse the same objective closure pattern as in lbfgs.py
def _create_objective_function(x: torch.Tensor, data, scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create objective function closure (same as lbfgs.py)."""
    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq_residual = (data.eq_resid(x, y) ** 2).sum(dim=1).mean(0)
        ineq_residual = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
        return scale * (eq_residual + ineq_residual)
    return _obj


def _check_convergence(f_val: torch.Tensor, g: torch.Tensor, config) -> torch.Tensor:
    """Same convergence predicate as in your LBFGS code."""
    # support either LBFGSConfig or AndersonConfig with same fields
    val_converged = f_val / getattr(config, "scale", 1.0) < getattr(config, "val_tol", 1e-6)
    grad_converged = g.norm(dim=1) < getattr(config, "grad_tol", 1e-6)
    return val_converged | grad_converged


class AndersonConfig:
    """Configuration for Anderson solver. Field names parallel LBFGSConfig where sensible."""
    def __init__(
        self,
        max_iter: int = 50,
        m: int = 5,                    # history depth
        val_tol: float = 1e-6,
        grad_tol: float = 1e-6,
        scale: float = 1.0,
        alpha: float = 1.0,            # fixed-point step size (y <- y - alpha * grad)
        reg: float = 1e-8,             # regularization for least-squares
        verbose: bool = False
    ):
        self.max_iter = max_iter
        self.m = m
        self.val_tol = val_tol
        self.grad_tol = grad_tol
        self.scale = scale
        self.alpha = alpha
        self.reg = reg
        self.verbose = verbose


def anderson_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[AndersonConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Differentiable Anderson Acceleration solver (vectorized, GPU-friendly).

    Matches the public signature of lbfgs_solve: (x, y_init, data, config=...).

    This function is differentiable (creates backward graph), i.e. uses create_graph=True
    when computing gradients so you can unroll it inside training.

    Algorithm notes:
    - Fixed-point operator: G(y) = y - alpha * grad f(y), where f is your squared-residual objective.
    - We store recent y_k and f_k = G(y_k) - y_k, then apply Anderson mixing using differences:
         dY = Y[t+1] - Y[t],  dF = F[t+1] - F[t]
      solve normal equations (dF^T dF + reg I) w = dF^T f_k  (batched)
      y_{k+1} = G(y_k) - dY @ w
    - If history is too short or (m < 2), we fall back to plain fixed-point iteration.
    """
    if config is None:
        # allow passing LBFGSConfig or kwargs for compatibility
        config = LBFGSConfig(**kwargs)

    # Convert LBFGSConfig to AndersonConfig if needed
    if not isinstance(config, AndersonConfig):
        cfg = AndersonConfig(
            max_iter=getattr(config, "max_iter", 50),
            m=getattr(config, "memory", 5),
            val_tol=getattr(config, "val_tol", 1e-6),
            grad_tol=getattr(config, "grad_tol", 1e-6),
            scale=getattr(config, "scale", 1.0),
            alpha=getattr(config, "scale", 1.0),  # default alpha fallback
            reg=1e-8,
            verbose=getattr(config, "verbose", False)
        )
        config = cfg

    # Initialize
    y = y_init.clone()
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype

    obj_func = _create_objective_function(x, data, config.scale)

    # initial evaluation: scalar objective and its gradient (B,n)
    f_val = obj_func(y)
    g = torch.autograd.grad(f_val, y, create_graph=True)[0]  # shape (B,n)

    # Anderson storage: store last m y and f = G(y) - y
    Y_hist = []  # list of tensors (B,n)
    F_hist = []  # list of tensors (B,n)

    m = config.m
    alpha = config.alpha
    reg = config.reg

    for k in range(config.max_iter):
        # convergence check
        if _check_convergence(f_val, g, config).all():
            if config.verbose:
                print(f"Anderson converged at iter {k}")
            break

        # compute fixed-point operator G(y) = y - alpha * g
        G = y - alpha * g       # (B,n)
        f_k = (G - y)           # residual for AA, shape (B,n)

        # Compute Anderson update if we have enough history (m >= 2)
        if len(F_hist) >= 1:
            # append current to history (we keep up to m entries)
            Y_hist.append(y)
            F_hist.append(f_k)

            if len(F_hist) > m:
                Y_hist.pop(0)
                F_hist.pop(0)

            cur_len = len(F_hist)  # between 1 and m

            if cur_len >= 2:
                # Build difference tensors: dY (cur_len-1, B, n), dF same
                # Shape them to (B, n, p) where p = cur_len-1 for batched linear solves
                dY = torch.stack([Y_hist[i+1] - Y_hist[i] for i in range(cur_len-1)], dim=0)  # (p,B,n)
                dF = torch.stack([F_hist[i+1] - F_hist[i] for i in range(cur_len-1)], dim=0)  # (p,B,n)

                # permute to (B, n, p)
                dF_b = dF.permute(1, 2, 0)  # (B, n, p)
                dY_b = dY.permute(1, 2, 0)  # (B, n, p)

                # current residual f_k: (B, n) -> (B, n, 1)
                f_k_b = f_k.unsqueeze(-1)

                # Solve least squares for each batch:
                # For each batch b: A = dF_b[b] (n x p), solve (A^T A + reg I) w = A^T f_k
                # Compute ATA: (B, p, p) = dF_b.transpose(-1,-2) @ dF_b
                ATA = torch.matmul(dF_b.transpose(-1, -2), dF_b)  # (B, p, p)
                # Regularize diagonal
                diag_idx = torch.arange(ATA.shape[-1], device=device)
                ATA[:, diag_idx, diag_idx] += reg

                # Compute ATf: (B, p, 1)
                ATf = torch.matmul(dF_b.transpose(-1, -2), f_k_b)  # (B, p, 1)

                # Solve for w (B, p, 1)
                # torch.linalg.solve expects (..., p, p) and (..., p, k)
                try:
                    w = torch.linalg.solve(ATA, ATf)  # (B, p, 1)
                except RuntimeError:
                    # fallback to pinv multiplication in rare singular cases
                    # w = pinv(ATA) @ ATf
                    ATA_flat = ATA.reshape(-1, ATA.shape[-2], ATA.shape[-1])
                    ATf_flat = ATf.reshape(-1, ATf.shape[-2], ATf.shape[-1])
                    w_list = []
                    for i in range(ATA_flat.shape[0]):
                        w_i = torch.linalg.lstsq(ATA_flat[i], ATf_flat[i]).solution
                        w_list.append(w_i)
                    w = torch.stack(w_list, dim=0).reshape(ATA.shape[0], ATA.shape[1], 1)

                # Compute correction: dY_b @ w -> (B, n, 1) -> squeeze -> (B, n)
                corr = torch.matmul(dY_b, w).squeeze(-1)

                # new iterate:
                y_next = G - corr
            else:
                # not enough differences yet -> plain fixed-point
                y_next = G
        else:
            # first iteration: no history yet
            Y_hist.append(y)
            F_hist.append(f_k)
            y_next = G

        # Prepare for next iter: evaluate objective and gradient (keep graph for differentiable)
        f_next = obj_func(y_next)
        g_next = torch.autograd.grad(f_next, y_next, create_graph=True)[0]

        # Logging
        if config.verbose and (k % 5 == 0):
            print(f"Anderson iter {k:3d}: f = {f_next.item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}")

        # update state
        y = y_next
        f_val = f_next.clone()
        g = g_next.clone()

    return y


def nondiff_anderson_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[AndersonConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Non-differentiable Anderson solver (no backward graph) matching nondiff_lbfgs_solve behavior.

    Uses create_graph=False for gradient computations and detaches history entries to avoid storing graph.
    """
    if config is None:
        
        config = LBFGSConfig(**kwargs)

    if not isinstance(config, AndersonConfig):
        cfg = AndersonConfig(
            max_iter=getattr(config, "max_iter", 50),
            m=getattr(config, "memory", 5),
            val_tol=getattr(config, "val_tol", 1e-6),
            grad_tol=getattr(config, "grad_tol", 1e-6),
            scale=getattr(config, "scale", 1.0),
            alpha=1.0,
            reg=1e-8,
            verbose=getattr(config, "verbose", False)
        )
        config = cfg

    # initialize
    y = y_init.detach().clone().requires_grad_(True)
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype

    obj_func = _create_objective_function(x, data, config.scale)

    f_val = obj_func(y)
    g = torch.autograd.grad(f_val, y, create_graph=False)[0]

    Y_hist = []
    F_hist = []
    m = config.m
    alpha = config.alpha
    reg = config.reg

    for k in range(config.max_iter):
        # disable grad for iteration computations (we only want gradient for next evaluation)
        y.requires_grad_(False)
        g = g.detach()

        if _check_convergence(f_val, g, config).all():
            if config.verbose:
                print(f"Non-diff Anderson converged at iter {k}")
            break

        G = y - alpha * g
        f_k = (G - y).detach()  # keep residual detached for history ops

        # Manage history
        Y_hist.append(y.detach())
        F_hist.append(f_k)
        if len(Y_hist) > m:
            Y_hist.pop(0)
            F_hist.pop(0)

        cur_len = len(F_hist)
        if cur_len >= 2:
            dY = torch.stack([Y_hist[i+1] - Y_hist[i] for i in range(cur_len-1)], dim=0)  # (p,B,n)
            dF = torch.stack([F_hist[i+1] - F_hist[i] for i in range(cur_len-1)], dim=0)  # (p,B,n)

            dF_b = dF.permute(1, 2, 0)  # (B, n, p)
            dY_b = dY.permute(1, 2, 0)  # (B, n, p)
            f_k_b = f_k.unsqueeze(-1)

            ATA = torch.matmul(dF_b.transpose(-1, -2), dF_b)  # (B, p, p)
            diag_idx = torch.arange(ATA.shape[-1], device=device)
            ATA[:, diag_idx, diag_idx] += reg

            ATf = torch.matmul(dF_b.transpose(-1, -2), f_k_b)  # (B, p, 1)
            # Solve
            try:
                w = torch.linalg.solve(ATA, ATf)  # (B, p, 1)
            except RuntimeError:
                # fallback to least-squares per-batch
                ATA_flat = ATA.reshape(-1, ATA.shape[-2], ATA.shape[-1])
                ATf_flat = ATf.reshape(-1, ATf.shape[-2], ATf.shape[-1])
                w_list = []
                for i in range(ATA_flat.shape[0]):
                    w_i = torch.linalg.lstsq(ATA_flat[i], ATf_flat[i]).solution
                    w_list.append(w_i)
                w = torch.stack(w_list, dim=0).reshape(ATA.shape[0], ATA.shape[1], 1)

            corr = torch.matmul(dY_b, w).squeeze(-1)
            y_next = (G - corr).detach()  # detach to avoid building graph
        else:
            y_next = G.detach()

        # now prepare next iter with gradient eval
        y_next = y_next.clone().requires_grad_(True)
        f_next = obj_func(y_next)
        g_next, = torch.autograd.grad(f_next, y_next, create_graph=False)

        if config.verbose and (k % 5 == 0):
            print(f"Non-diff Anderson iter {k:3d}: f = {f_next.item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}")

        y = y_next.detach()
        y = y.clone().requires_grad_(True)
        f_val = f_next.clone()
        g = g_next.clone()

    return y.detach()


# Optionally, a hybrid variant similar to your hybrid_lbfgs_solve can be written
# to do a few differentiable Anderson iterations and then switch to the non-diff version.
def hybrid_anderson_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_diff_iter: int = 10,
    config: Optional[AndersonConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Run a short differentiable Anderson phase, then continue non-differentiable.
    Returns y connected only to the differentiable part (same pattern as hybrid_lbfgs_solve).
    """
    if config is None:
        cfg = LBFGSConfig(**kwargs)
        config = AndersonConfig(
            max_iter=getattr(cfg, "max_iter", 50),
            m=getattr(cfg, "memory", 5),
            val_tol=getattr(cfg, "val_tol", 1e-6),
            grad_tol=getattr(cfg, "grad_tol", 1e-6),
            scale=getattr(cfg, "scale", 1.0),
            alpha=1.0,
            reg=1e-8,
            verbose=getattr(cfg, "verbose", False)
        )

    # differentiable phase
    diff_cfg = AndersonConfig(
        max_iter=max_diff_iter,
        m=config.m,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        alpha=config.alpha,
        reg=config.reg,
        verbose=config.verbose
    )

    y_diff = anderson_solve(x, y_init, data, config=diff_cfg)
    # Continue with nondiff using stored state is complex; instead call nondiff starting from y_diff
    remaining_config = AndersonConfig(
        max_iter=max(0, config.max_iter - max_diff_iter),
        m=config.m,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        alpha=config.alpha,
        reg=config.reg,
        verbose=config.verbose
    )
    y_nondiff = nondiff_anderson_solve(x, y_diff.detach(), data, config=remaining_config)

    # keep gradient only through the differentiable phase
    return y_diff + (y_nondiff - y_diff).detach()
