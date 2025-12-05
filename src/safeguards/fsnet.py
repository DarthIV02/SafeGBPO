import cvxpy as cp
from torch import Tensor
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, jaxtyped
from typing import Dict, Tuple, Callable, Optional
import torch

from safeguards.interfaces.safeguard import Safeguard, SafeEnv


class FSNetSafeguard(Safeguard):
    """
    Projecting an unsafe action to the closest safe action.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, 
                 env: SafeEnv, 
                 regularisation_coefficient: float,
                 eq_pen_coefficient: float,
                 ineq_pen_coefficient: float,
                 val_tol: float,
                 memory_size: int,
                 max_iter: int,
                 max_diff_iter:int,
                 scale : float,
                 **kwargs):
        Safeguard.__init__(self, env)

        self.boundary_layer = None
        self.regularisation_coefficient = regularisation_coefficient
        self.eq_pen_coefficient = eq_pen_coefficient
        self.ineq_pen_coefficient = ineq_pen_coefficient 

        self.config_method = {
            'val_tol': val_tol,
            'memory_size': memory_size,
            'max_iter': max_iter,
            'max_diff_iter': max_diff_iter,
            'scale': scale,
        }
        if self.env.polytope:
            self.data = PolytopeData(env)
        else:
            self.data = ZonotopeData(env)

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Safeguard the action to ensure safety.

        Args:
            action: The action to safeguard.

        Returns:
            The safeguarded action.
        """
        self.data.setup_resid(action)
        processed_action = self.data.pre_process_action(action)
        self.pre_eq_violation = self.data.eq_resid(None, processed_action).square().sum(dim=1)
        self.pre_ineq_violation = self.data.ineq_resid(None, processed_action).square().sum(dim=1)
        
        # Use non-differentiable solver during evaluation (no grad mode),
        # hybrid solver during training (with grad)
        if torch.is_grad_enabled():
            safe_action = hybrid_lbfgs_solve(
                None,
                processed_action,
                self.data,
                val_tol=self.config_method['val_tol'],
                memory=self.config_method['memory_size'],
                max_iter=self.config_method['max_iter'],
                max_diff_iter=self.config_method['max_diff_iter'],
                scale=self.config_method['scale'],
            )
        else:
            with torch.enable_grad():
                safe_action = nondiff_lbfgs_solve(
                    None,
                    processed_action,
                    self.data,
                    val_tol=self.config_method['val_tol'],
                    memory=self.config_method['memory_size'],
                    max_iter=self.config_method['max_iter'],
                    scale=self.config_method['scale'],
                )
    

        self.post_eq_violation = self.data.eq_resid(None, safe_action).square().sum(dim=1)
        self.post_ineq_violation = self.data.ineq_resid(None, safe_action).square().sum(dim=1)

        safe_action = self.data.post_process_action(safe_action)
        if torch.isnan(safe_action).any():
            print("FSNetSafeguard: safe_action has NaN values", safe_action)
        return safe_action

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:

        if self.pre_eq_violation.mean() >= 1e3 or self.pre_ineq_violation.mean() >= 1e3:
            loss = self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action) +\
                   self.eq_pen_coefficient * self.pre_eq_violation + \
                   self.ineq_pen_coefficient * self.pre_ineq_violation
        else:
            loss = self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action) 
        return loss
    
    def safeguard_metrics(self):
        return  {
            "pre_eq_violation": self.pre_eq_violation.mean().item(),
            "pre_ineq_violation": self.pre_ineq_violation.mean().item(),
            "post_eq_violation": self.post_eq_violation.mean().item(),
            "post_ineq_violation": self.post_ineq_violation.mean().item(),
        }
    
class DataInterface:

    def __init__(self,env):
        self.env = env
    
    def setup_resid(self, action):
        pass

    def pre_process_action(self, action):
        return action
    
    def post_process_action(self, action):
        return action
    
    def ineq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(Y)

    def eq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(Y)

class PolytopeData(DataInterface):

    def setup_resid(self, action):
        self.A, self.b = self.env.compute_A_b()
        self.A = self.A.to(dtype=action.dtype, device=action.device).detach()
        self.b = self.b.to(dtype=action.dtype, device=action.device).detach().unsqueeze(2)
        
    def ineq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.A @ Y.unsqueeze(2) - self.b)

    
class BoxData(DataInterface):

    def setup_resid(self, action):
        box = self.env.safe_action_set().box()
        center = box.center
        gen = box.generator

        if gen.dim() == 3:
            half_extents = gen.abs().sum(dim=2)  # (batch_box, dim)
        else:
            half_extents = gen.abs()
        
        self.box_min = (center - half_extents).to(dtype=action.dtype, device=action.device).detach()
        self.box_max = (center + half_extents).to(dtype=action.dtype, device=action.device).detach()
        
    def ineq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.relu(self.box_min - Y), torch.relu(Y - self.box_max)], dim=1)

class ZonotopeData(DataInterface):

    def setup_resid(self, action):
        zonotope = self.env.safe_action_set()
        self.center = zonotope.center
        self.gen = zonotope.generator
        if torch.isnan(self.center).any():
            print("ZonotopeData: center has NaN values", self.center)
        if torch.isnan(self.gen).any():
            print("ZonotopeData: generator has NaN values", self.gen)
        batch_dim, dim, num_generators = self.gen.shape

        # for eq constraints Ax = b
        # A with shape (batch_dim, dim, dim + num_generators)
        # b with shape (batch_dim, dim, 1)
        
        self.A = torch.cat([
                    torch.eye(dim, dtype=action.dtype, device=action.device).expand(batch_dim, dim, dim), 
                    -self.gen
                ], dim=2).detach() 
        self.b = self.center.unsqueeze(2).detach() 
        
        # for ineq constraints Kx <= h
        # K with shape (batch_dim, 2 * num_generators, dim + num_generators)
        # h with shape (batch_dim, 2 * num_generators, 1)

        K_half = torch.cat([
                    torch.zeros((batch_dim, num_generators, dim), dtype=action.dtype, device=action.device),
                    torch.eye(num_generators, dtype=action.dtype, device=action.device).expand(batch_dim, num_generators, num_generators)
                ], dim=2)
        
        self.K = torch.cat([K_half, -K_half], dim=1).detach()  
        self.h = torch.ones((batch_dim, 2 * num_generators, 1), dtype=action.dtype, device=action.device).detach()  

    def pre_process_action(self, action):
        if torch.isnan(action).any():
            print("ZonotopeData: action has NaN values before pre_process_action", action)
        # z with shape (batch_dim, dim + num_generators)
        batch_dim, _, num_generators = self.gen.shape
        gamma = torch.randn((batch_dim, num_generators), dtype=action.dtype, device=action.device).detach()
        z = torch.cat([action, gamma], dim=1)
        if torch.isnan(z).any():
            print("ZonotopeData: z has NaN values after pre_process_action", z)
        return z
    
    def post_process_action(self, action):
        return action[:, :self.env.action_dim]

    def eq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        resid = self.A @ Y.unsqueeze(2) - self.b
        if torch.isnan(resid).any():
            print("ZonotopeData: eq_resid has NaN values", resid)
        return resid

    def ineq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        resid = torch.relu(self.K @ Y.unsqueeze(2) - self.h)
        if torch.isnan(resid).any():
            print("ZonotopeData: ineq_resid has NaN values", resid)
        return resid

# ----------------------------------------------------------------------
#  Code from FSNet/utils/lbfgs.py
# ----------------------------------------------------------------------

# Differentiable and nondifferentiable L-BFGS solver

@torch.jit.script
def _search_direction(
    g: torch.Tensor,               # (B, n)
    S: torch.Tensor,               # (m, B, n) stacked s‑vectors
    Y: torch.Tensor,               # (m, B, n) stacked y‑vectors
    gamma: torch.Tensor            # (B, 1) or scalar
) -> torch.Tensor:                 # returns d (B, n)
    """
    Compute d = −H_k^{-1} g_k for L‑BFGS in batch mode using two-loop recursion.

    Parameters
    ----------
    g : torch.Tensor
        Current gradient, shape (B, n)
    S : torch.Tensor
        History of s_i vectors, shape (m, B, n)
    Y : torch.Tensor
        History of y_i vectors, shape (m, B, n)
    gamma : torch.Tensor
        Scalar or (B,1) scaling for the initial Hessian approximation

    Returns
    -------
    torch.Tensor
        Search direction, shape (B, n)
    """
    m = S.shape[0]  # history length
    eps = 1e-10
    rho = 1.0 / ((S * Y).sum(dim=2, keepdim=True) + eps)  # (m,B,1)

    # First loop (reverse order)
    q = g.clone()
    alphas = []
    for i in range(m - 1, -1, -1):
        alpha_i = rho[i] * (S[i] * q).sum(dim=1, keepdim=True)  # (B,1)
        alphas.append(alpha_i)
        q = q - alpha_i * Y[i]

    # Apply initial Hessian approximation: gamma * I
    r = gamma * q

    # Second loop (forward order)
    alphas = alphas[::-1]
    for i in range(m):
        beta = rho[i] * (Y[i] * r).sum(dim=1, keepdim=True)
        r = r + S[i] * (alphas[i] - beta)

    return -r


@torch.jit.script
def compute_gamma(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the initial Hessian scaling factor γ = s^T y / y^T y.
    
    Parameters
    ----------
    S : torch.Tensor
        History of s vectors, shape (m, B, n)
    Y : torch.Tensor
        History of y vectors, shape (m, B, n)
        
    Returns
    -------
    torch.Tensor
        Scaling factor, shape (B, 1)
    """
    eps = 1e-10
    s_dot_y = (S[-1] * Y[-1]).sum(dim=1, keepdim=True)
    y_dot_y = (Y[-1] * Y[-1]).sum(dim=1, keepdim=True) + eps
    return s_dot_y / y_dot_y


class LBFGSConfig:
    """Configuration class for L-BFGS parameters."""
    def __init__(
        self,
        max_iter: int = 20,
        memory: int = 20,
        val_tol: float = 1e-6,
        grad_tol: float = 1e-6,
        scale: float = 1.0,
        c: float = 1e-4,
        rho_ls: float = 0.5,
        max_ls_iter: int = 10,
        verbose: bool = False
    ):
        self.max_iter = max_iter
        self.memory = memory
        self.val_tol = val_tol
        self.grad_tol = grad_tol
        self.scale = scale
        self.c = c
        self.rho_ls = rho_ls
        self.max_ls_iter = max_ls_iter
        self.verbose = verbose


def _create_objective_function(x: torch.Tensor, data, scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create objective function closure."""
    def _obj(y: torch.Tensor) -> torch.Tensor:
        eq_residual = (data.eq_resid(x, y) ** 2).sum(dim=1).mean(0)
        ineq_residual = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
        return scale * (eq_residual + ineq_residual)
    return _obj


def _check_convergence(f_val: torch.Tensor, g: torch.Tensor, config: LBFGSConfig) -> torch.Tensor:
    """Check convergence criteria."""
    val_converged = f_val / config.scale < config.val_tol
    grad_converged = g.norm(dim=1) < config.grad_tol
    return val_converged | grad_converged


def _backtracking_line_search(
    y: torch.Tensor,
    d: torch.Tensor,
    g: torch.Tensor,
    f_val: torch.Tensor,
    obj_func: Callable,
    config: LBFGSConfig
) -> float:
    """Perform backtracking line search."""
    step = 1.0
    dir_deriv = (g * d).sum()
    
    with torch.no_grad():
        for _ in range(config.max_ls_iter):
            y_trial = y + step * d
            f_trial = obj_func(y_trial)
            if (f_trial <= f_val + config.c * step * dir_deriv).all():
                break
            step *= config.rho_ls
    
    return step


def lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LBFGSConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Differentiable L‑BFGS solver with vectorized two‑loop recursion.
    
    Parameters
    ----------
    y_init : torch.Tensor
        Initial guess, shape (B, n)
    x : torch.Tensor
        Input data
    data : object
        Data object with eq_resid and ineq_resid methods
    config : LBFGSConfig, optional
        Configuration object. If None, uses default parameters from kwargs.
    **kwargs
        Additional parameters if config is not provided
        
    Returns
    -------
    torch.Tensor
        Solution, shape (B, n)
    """
    if config is None:
        config = LBFGSConfig(**kwargs)
    
    # Initialize
    y = y_init.clone()
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    # History buffers
    S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    # Create objective function
    obj_func = _create_objective_function(x, data, config.scale)
    
    # Initial evaluation
    f_val = obj_func(y)
    g = torch.autograd.grad(f_val, y, create_graph=True)[0]
    
    for k in range(config.max_iter):
        # Check convergence
        if _check_convergence(f_val, g, config).all():
            if config.verbose:
                print(f"Converged at iteration {k}")
            break
        
        # Compute search direction
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g  # Steepest descent for first iteration
        
        # Line search
        step = _backtracking_line_search(y, d, g, f_val, obj_func, config)
        
        # Update solution
        y_next = y + step * d
        f_next = obj_func(y_next)
        g_next = torch.autograd.grad(f_next, y_next, create_graph=True)[0]
        
        # Update history
        S_hist[hist_ptr] = y_next - y
        Y_hist[hist_ptr] = g_next - g
        hist_ptr = (hist_ptr + 1) % config.memory
        hist_len = min(hist_len + 1, config.memory)
        
        # Prepare for next iteration
        y = y_next
        f_val = f_next.clone()
        g = g_next.clone()
        
        if config.verbose and k % 5 == 0:
            print(f"Iter {k:3d}: f = {f_next.item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step = {step:.3e}")
    
    return y


def nondiff_lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config: Optional[LBFGSConfig] = None,
    S_hist: Optional[torch.Tensor] = None,
    Y_hist: Optional[torch.Tensor] = None,
    hist_len: int = 0,
    hist_ptr: int = 0,
    **kwargs
) -> torch.Tensor:
    """
    Non-differentiable L‑BFGS solver that doesn't build backward graph.
    
    Parameters
    ----------
    y_init : torch.Tensor
        Initial guess, shape (B, n)
    x : torch.Tensor
        Input data
    data : object
        Data object with eq_resid and ineq_resid methods
    config : LBFGSConfig, optional
        Configuration object
    S_hist, Y_hist : torch.Tensor, optional
        Pre-existing history buffers
    hist_len, hist_ptr : int
        History tracking variables
    **kwargs
        Additional parameters if config is not provided
        
    Returns
    -------
    torch.Tensor
        Solution, shape (B, n)
    """
    if config is None:
        config = LBFGSConfig(**kwargs)
    
    # Initialize without gradient tracking
    y = y_init.detach().clone().requires_grad_(True)
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    # Initialize history buffers if not provided
    if S_hist is None:
        S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype)
        Y_hist = torch.zeros_like(S_hist)
        hist_len = 0
        hist_ptr = 0
    
    obj_func = _create_objective_function(x, data, config.scale)
    
    f_val = obj_func(y)
    g = torch.autograd.grad(f_val, y, create_graph=False)[0]
    
    for k in range(config.max_iter):
        y.requires_grad_(False)
        g = g.detach()
        
        # Check convergence
        if _check_convergence(f_val, g, config).all():
            if config.verbose:
                print(f"Converged at iteration {k}")
            break
        
        # Compute search direction
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g
        
        # Line search
        step = _backtracking_line_search(y, d, g, f_val, obj_func, config)
        
        y_next = y + step * d
        
        # Update history with detached tensors
        y_next.requires_grad_(True)
        f_next = obj_func(y_next)
        g_next, = torch.autograd.grad(f_next, y_next, create_graph=False)
        
        S_hist[hist_ptr] = (y_next - y).detach()
        Y_hist[hist_ptr] = (g_next - g).detach()
        hist_ptr = (hist_ptr + 1) % config.memory
        hist_len = min(hist_len + 1, config.memory)
        
        y = y_next.detach()
        f_val = f_next.clone()
        g = g_next.clone()
        
        if config.verbose and k % 5 == 0:
            print(f"Iter {k:3d}: f = {f_next.item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step = {step:.3e}")
    
    return y


def hybrid_lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    max_diff_iter: int = 20,
    config: Optional[LBFGSConfig] = None,
    **kwargs
) -> torch.Tensor:
    """
    Hybrid L‑BFGS solver with truncated backpropagation.
    
    Starts with differentiable L‑BFGS and switches to non-differentiable
    after max_diff_iter iterations for memory efficiency.
    
    Parameters
    ----------
    y_init : torch.Tensor
        Initial guess, shape (B, n)
    x : torch.Tensor
        Input data
    data : object
        Data object with eq_resid and ineq_resid methods
    max_diff_iter : int
        Number of differentiable iterations before switching
    config : LBFGSConfig, optional
        Configuration object
    **kwargs
        Additional parameters if config is not provided
        
    Returns
    -------
    torch.Tensor
        Solution with gradient connection to first max_diff_iter steps
    """
    if config is None:
        config = LBFGSConfig(**kwargs)
    
    # Create a config for the differentiable phase
    diff_config = LBFGSConfig(
        max_iter=max_diff_iter,
        memory=config.memory,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        c=config.c,
        rho_ls=config.rho_ls,
        max_ls_iter=config.max_ls_iter,
        verbose=config.verbose
    )
    
    # Run differentiable phase (shortened version of lbfgs_solve)
    y = y_init.clone()
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    obj_func = _create_objective_function(x, data, config.scale)
    f_val = obj_func(y)
    g = torch.autograd.grad(f_val, y, create_graph=True)[0]
    
    for k in range(max_diff_iter):
        if _check_convergence(f_val, g, diff_config).all():
            if config.verbose:
                print(f"Converged in differentiable phase at iteration {k}")
            return y
        
        # Search direction
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]
            Y = Y_hist[idx]
            gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else:
            d = -0.1 * g
        
        # Line search
        step = _backtracking_line_search(y, d, g, f_val, obj_func, diff_config)
        
        # Update
        y_next = y + step * d
        f_next = obj_func(y_next)
        g_next = torch.autograd.grad(f_next, y_next, create_graph=True)[0]
        
        # Update history
        S_hist[hist_ptr] = y_next - y
        Y_hist[hist_ptr] = g_next - g
        hist_ptr = (hist_ptr + 1) % config.memory
        hist_len = min(hist_len + 1, config.memory)
        
        y = y_next
        f_val = f_next.clone()
        g = g_next.clone()
        
        if config.verbose and k % 5 == 0:
            print(f"Diff iter {k:3d}: f = {f_next.item()/config.scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step = {step:.3e}")
    
    # Switch to non-differentiable phase
    remaining_config = LBFGSConfig(
        max_iter=config.max_iter - max_diff_iter,
        memory=config.memory,
        val_tol=config.val_tol,
        grad_tol=config.grad_tol,
        scale=config.scale,
        c=config.c,
        rho_ls=config.rho_ls,
        max_ls_iter=config.max_ls_iter,
        verbose=config.verbose
    )
    
    y_nondiff = nondiff_lbfgs_solve(
        x, y, data, remaining_config,
        S_hist=S_hist,
        Y_hist=Y_hist,
        hist_len=hist_len,
        hist_ptr=hist_ptr
    )
    
    # Return with gradient connection only to differentiable phase
    return y + (y_nondiff - y).detach()
