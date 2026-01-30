import torch
import sys
import os
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Tuple, Optional, Callable, Union, List, Any

# =============================================================================
# Path & Imports
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Base Implementation
try:
    from safeguards.fsnet import FSNetSafeguard
    from safeguards.interfaces.safeguard import SafeEnv
    from src.sets.zonotope import Zonotope
    
    # Handle HPolytope/Polytope naming
    try:
        from src.sets.polytope import HPolytope
        PolytopeClass = HPolytope
    except ImportError:
        from src.sets.polytope import Polytope
        PolytopeClass = Polytope

except ImportError as e:
    # Allow partial imports for standalone inspection
    print(f"[WARNING] Could not import project modules: {e}")
    # Mock classes for linting
    class Safeguard:
        def __init__(self, env): pass
    class SafeEnv: pass
    class PolytopeClass: pass
    class Zonotope: pass
    class FSNetSafeguard(Safeguard):
        def __init__(self, env, **kwargs): pass

# =============================================================================
# L-BFGS Solver Implementation (Differentiable & Trajectory Tracking)
# =============================================================================

@torch.jit.script
def _search_direction(g: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    m = S.shape[0]; eps = 1e-10
    rho = 1.0 / ((S * Y).sum(dim=2, keepdim=True) + eps)
    q = g.clone(); alphas = []
    for i in range(m - 1, -1, -1):
        alpha_i = rho[i] * (S[i] * q).sum(dim=1, keepdim=True)
        alphas.append(alpha_i)
        q = q - alpha_i * Y[i]
    r = gamma * q
    alphas = alphas[::-1]
    for i in range(m):
        beta = rho[i] * (Y[i] * r).sum(dim=1, keepdim=True)
        r = r + S[i] * (alphas[i] - beta)
    return -r

@torch.jit.script
def compute_gamma(S: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    s_dot_y = (S[-1] * Y[-1]).sum(dim=1, keepdim=True)
    y_dot_y = (Y[-1] * Y[-1]).sum(dim=1, keepdim=True) + eps
    return s_dot_y / y_dot_y

class LBFGSConfig:
    def __init__(self, max_iter: int=20, memory: int=20, val_tol: float=1e-6, grad_tol: float=1e-6, 
                 scale: float=1.0, c: float=1e-4, rho_ls: float=0.5, max_ls_iter: int=10, verbose: bool=False, **kwargs):
        self.max_iter = max_iter; self.memory = memory; self.val_tol = val_tol; self.grad_tol = grad_tol
        self.scale = scale; self.c = c; self.rho_ls = rho_ls; self.max_ls_iter = max_ls_iter; self.verbose = verbose

def _create_objective_function(x: torch.Tensor, data, scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    # Dynamic Check for API compatibility
    has_resid = hasattr(data, "eq_resid")
    if has_resid:
        def _obj(y: torch.Tensor) -> torch.Tensor:
            eq = (data.eq_resid(x, y) ** 2).sum(dim=1).mean(0)
            ineq = (data.ineq_resid(x, y) ** 2).sum(dim=1).mean(0)
            return scale * (eq + ineq)
    else:
        def _obj(y: torch.Tensor) -> torch.Tensor:
            eq = (data.equality_constraint_violation(x, y) ** 2).sum(dim=1).mean(0)
            ineq = (data.inequality_constraint_violation(x, y) ** 2).sum(dim=1).mean(0)
            return scale * (eq + ineq)
    return _obj

def _check_convergence(f_val: torch.Tensor, g: torch.Tensor, config: LBFGSConfig) -> torch.Tensor:
    val_converged = f_val / config.scale < config.val_tol
    grad_converged = g.norm(dim=1) < config.grad_tol
    return val_converged | grad_converged

def _backtracking_line_search(y: torch.Tensor, d: torch.Tensor, g: torch.Tensor, f_val: torch.Tensor, 
                              obj_func: Callable, config: LBFGSConfig) -> float:
    step = 1.0; dir_deriv = (g * d).sum()
    with torch.no_grad():
        for _ in range(config.max_ls_iter):
            y_trial = y + step * d
            f_trial = obj_func(y_trial)
            if (f_trial <= f_val + config.c * step * dir_deriv).all(): break
            step *= config.rho_ls
    return step

def nondiff_lbfgs_solve_viz(x: torch.Tensor, y_init: torch.Tensor, data, config: Optional[LBFGSConfig]=None,
                        S_hist: Optional[torch.Tensor]=None, Y_hist: Optional[torch.Tensor]=None,
                        hist_len: int=0, hist_ptr: int=0, debug_trajectory: bool=False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    if "debug_trajectory" in kwargs: kwargs.pop("debug_trajectory")
    if config is None: config = LBFGSConfig(**kwargs)
    trajectory = []
    y = y_init.detach().clone().requires_grad_(True)
    if debug_trajectory: trajectory.append(y.detach().cpu().clone())
    B, n = y_init.shape; device, dtype = y_init.device, y_init.dtype
    if S_hist is None:
        S_hist = torch.zeros(config.memory, B, n, device=device, dtype=dtype); Y_hist = torch.zeros_like(S_hist)
        hist_len = 0; hist_ptr = 0
    obj_func = _create_objective_function(x, data, config.scale)
    f_val = obj_func(y); g = torch.autograd.grad(f_val, y, create_graph=False)[0]
    
    for k in range(config.max_iter):
        y.requires_grad_(False); g = g.detach()
        if _check_convergence(f_val, g, config).all(): break
        if hist_len > 0:
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.memory
            S = S_hist[idx]; Y = Y_hist[idx]; gamma = compute_gamma(S, Y)
            d = _search_direction(g, S, Y, gamma)
        else: d = -0.1 * g
        step = _backtracking_line_search(y, d, g, f_val, obj_func, config)
        y_next = y + step * d
        if debug_trajectory: trajectory.append(y_next.detach().cpu().clone())
        y_next.requires_grad_(True); f_next = obj_func(y_next); g_next, = torch.autograd.grad(f_next, y_next, create_graph=False)
        S_hist[hist_ptr] = (y_next - y).detach(); Y_hist[hist_ptr] = (g_next - g).detach()
        hist_ptr = (hist_ptr + 1) % config.memory; hist_len = min(hist_len + 1, config.memory)
        y = y_next.detach(); f_val = f_next.clone(); g = g_next.clone()
    if debug_trajectory: return y, trajectory
    return y

# =============================================================================
# 3. VizFSNetSafeguard (Wrapper)
# =============================================================================

class VizFSNetSafeguard(FSNetSafeguard):
    """
    A thin wrapper around the actual FSNetSafeguard to enable trajectory visualization.
    Inherits from FSNetSafeguard to use its __init__ logic for arguments,
    but overrides the solver setup.
    """

    def __init__(self, env: SafeEnv, **kwargs):
        # 1. Force trajectory storage
        kwargs['store_trajectory'] = True
        
        # 2. Store config locally to ensure it exists before any delegation
        self.config_method = kwargs
        
        # 3. Provide default attributes if missing (Safety fallback)
        self.regularisation_coefficient = kwargs.get('regularisation_coefficient', 0.1)
        self.boundary_layer = None
        
        # 4. Initialize Base Class
        # Note: We catch TypeError in case base class signature differs slightly
        try:
            super().__init__(env, **kwargs)
        except TypeError:
            # Fallback: Try calling without kwargs if base is strict
            super().__init__(env)

        # 5. HOT-SWAP SOLVER for Visualization
        self.nondiff_solver = nondiff_lbfgs_solve_viz
        
        # 6. Initialize Viz Buffers
        self.debug_mode = True
        self.last_trajectory = None
        self.last_unsafe_action = None
        self.last_safe_set_info = {}

    def __getattr__(self, name):
        # If config_method is missing in self but requested, return local cache
        if name == 'config_method' and 'config_method' in self.__dict__:
             return self.__dict__['config_method']
        # Delegate to environment
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        
        self.data = self.safe_action_set()
        
        # Robust Type Check (Handle Reload Mismatches)
        if not isinstance(self.data, (PolytopeClass, Zonotope)):
             if not (type(self.data).__name__ in ["HPolytope", "Polytope", "Zonotope"]):
                raise NotImplementedError(f"VizFSNet only supports Polytope/Zonotope. Got: {type(self.data)}")

        # Handle API mismatch (setup_resid vs setup_constraints)
        if hasattr(self.data, "setup_resid"):
            self.data.setup_resid()
        elif hasattr(self.data, "setup_constraints"):
            self.data.setup_constraints()
        
        processed_action = self.data.pre_process_action(action)
        
        # Calculate residuals (Support both APIs)
        if hasattr(self.data, "eq_resid"):
            self.pre_eq_violation = self.data.eq_resid(None, processed_action).square().sum(dim=1)
            self.pre_ineq_violation = self.data.ineq_resid(None, processed_action).square().sum(dim=1)
        else:
            self.pre_eq_violation = self.data.equality_constraint_violation(None, processed_action).square().sum(dim=1)
            self.pre_ineq_violation = self.data.inequality_constraint_violation(None, processed_action).square().sum(dim=1)
            
        tol = 1e-6
        is_safe_mask = (self.pre_eq_violation <= tol) & (self.pre_ineq_violation <= tol)

        # --- VISUALIZATION SOLVE ---
        with torch.enable_grad():
            result = self.nondiff_solver(
                None,
                processed_action,
                self.data,
                debug_trajectory=True, 
                **self.config_method
            )
        
        if isinstance(result, tuple):
            safe_action, trajectory = result
            self.last_trajectory = [self.data.post_process_action(t) for t in trajectory]
            self.last_unsafe_action = action.detach().cpu()
        else:
            safe_action = result
            self.last_trajectory = [self.data.post_process_action(safe_action).detach().cpu()]

        # Capture Safe Set Geometry
        if hasattr(self.data, "A"):
            self.last_safe_set_info = {
                "safe_set_A": self.data.A.detach().cpu(),
                "safe_set_b": self.data.b.detach().cpu()
            }
        elif hasattr(self.data, "generator"):
            self.last_safe_set_info = {
                "safe_set_center": self.data.center.detach().cpu(),
                "safe_set_generators": self.data.generator.detach().cpu()
            }

        # Post logs
        if hasattr(self.data, "eq_resid"):
            self.post_eq_violation = self.data.eq_resid(None, safe_action).square().sum(dim=1)
            self.post_ineq_violation = self.data.ineq_resid(None, safe_action).square().sum(dim=1)
        else:
            self.post_eq_violation = self.data.equality_constraint_violation(None, safe_action).square().sum(dim=1)
            self.post_ineq_violation = self.data.inequality_constraint_violation(None, safe_action).square().sum(dim=1)

        safe_action = self.data.post_process_action(safe_action)

        if is_safe_mask.ndim == 1:
            mask = is_safe_mask.unsqueeze(1).expand_as(action)
        else:
            mask = is_safe_mask.expand_as(action)
            
        safe_action = torch.where(mask, action, safe_action)

        return safe_action

    def get_visualization_data(self):
        if self.last_trajectory is None: return None
        if isinstance(self.last_trajectory, list):
            traj_stack = torch.stack(self.last_trajectory).detach().cpu()
        else:
            traj_stack = self.last_trajectory.detach().cpu().unsqueeze(0)
        return {
            "trajectory": traj_stack,
            "unsafe_action": self.last_unsafe_action.detach().cpu(),
            **self.last_safe_set_info
        }