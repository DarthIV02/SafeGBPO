from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Any, Dict, Tuple, Callable, Optional
from types import SimpleNamespace
import torch

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

from src.sets.polytope import Polytope
from src.sets.zonotope import Zonotope

 
class FSNetSafeguard(Safeguard):
    """
    Projecting an unsafe action to the closest safe action.
    Safeguard implementation based on FSNet

    min y∈Rn f(y; x) s.t. g(y; x) ≤ 0, h(y; x) = 0

    Algorithm of FSNet Training Algorithm
        1: init. NN weights θ, learning rate
        2: repeat
        3:      sample x ∼ D                -> state from the rl environment (not used in the safeguard directly)
        4:      predict yθ(x) via NN        -> action of the policy
        5:      compute yˆθ(x) = FS(yθ(x); x) 
                        via minimzing ∥h(yθ(x); x)∥^2_2 + ∥g(yθ(x); x)∥^2_2 
                        with gradient descent and  yθ(x) as the initial value 
                                            -> LBFGS to get the safe action
        6:      update θ with ∇θF(yθ(x), yˆθ(x)) 
                        via the loss f(ˆyθ(x); x) + ρ/2∥yθ(x) − yˆθ(x)∥^2_2 
                        (+ρ/2 * residual penalties for practical efficiency) 
                                            -> add the safeguard loss to the policy loss
        7: until convergence

    Reference:

    @article{nguyen2025fsnet,
        title={FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees}, 
        author={Hoang T. Nguyen and Priya L. Donti},
        year={2025},
        journal={arXiv preprint arXiv:2506.00362},
    }
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, 
                 env: SafeEnv, 
                 regularisation_coefficient: float,
                 penalty_addition_threshold: float,
                 residual_coefficient: float,
                 **kwargs):
        Safeguard.__init__(self, env, regularisation_coefficient)

        # assume the remaining kwargs are solver config parameters
        self.method_config = kwargs
        self.penalty_addition_threshold = penalty_addition_threshold
        self.residual_coefficient = residual_coefficient

        self.solver =    lbfgs_torch_solve
        self.nondiff_solver =   nondiff_lbfgs_torch_solve

        if not self.env.safe_action_polytope:
            raise Exception("Polytope attribute has to be True")
        if not hasattr(self.env, "safe_action_set"):
            raise Exception("Environment must implement safe_action_set method")

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

        # for FSNet create a data object that contains the constraint representation and residual functions
        # for this implementation we implement it from the safe action set in the env, using convex set constraints
        self.data = self.safe_action_set()

        self.data.setup_constraints()

        # prepocess the action to accommate the different safe set representations
        processed_action = self.data.pre_process_action(action)

        # compute pre safeguard violations for logging and loss
        self.pre_eq_violation = self.data.equality_constraint_violation(None, processed_action).square().sum(dim=1)
        self.pre_ineq_violation = self.data.inequality_constraint_violation(None, processed_action).square().sum(dim=1)
        
        # Use non-differentiable solver during evaluation (no grad mode),
        # hybrid solver during training (with grad) for better backpropagation

        if torch.is_grad_enabled(): # training mode
            safe_action = self.solver(
                None,
                processed_action,
                self.data,
                **self.method_config
            )
        else: # evaluation mode
            with torch.enable_grad(): # ensure grad is enabled for solvers just like in FSNet codebase
                safe_action = self.nondiff_solver(
                    None,
                    processed_action,
                    self.data,
                    **self.method_config
                )
    
            # compute post safeguard violations for logging
            self.post_eq_violation = self.data.equality_constraint_violation(None, safe_action).square().sum(dim=1)
            self.post_ineq_violation = self.data.inequality_constraint_violation(None, safe_action).square().sum(dim=1)

        # return the safe action to original space
        safe_action = self.data.post_process_action(safe_action)
        return safe_action

    def regularisation(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safeguard_metrics: Optional[Dict[str, Any]] = None
                        ) -> Tensor:
        """
        Compute the safeguard regularisation loss for FSNet.
        Args:
            action: The original action before safeguarding.
            safe_action: The safeguarded action.
            safeguard_metrics: Optional dictionary containing safeguard metrics such as pre-violation values.
        Returns:
            The safeguard regularisation loss consisting loss f(ˆyθ(x); x) + ρ/2∥yθ(x) − yˆθ(x)∥^2_2 (+ρ/2 * residual penalties for practical efficiency)
        """

        loss = self.regularisation_coefficient/2 *  (torch.norm(action - safe_action, dim=-1, keepdim=True) ** 2)

        if safeguard_metrics:
            pre_eq_violation = safeguard_metrics["pre_eq_violation"].tensor
            pre_ineq_violation = safeguard_metrics["pre_ineq_violation"].tensor

            loss +=  self.residual_coefficient * torch.where(pre_eq_violation > self.penalty_addition_threshold,
                                                                pre_eq_violation, 
                                                                torch.zeros_like(pre_eq_violation))
            loss +=  self.residual_coefficient * torch.where(pre_ineq_violation > self.penalty_addition_threshold,
                                                                pre_ineq_violation,
                                                                torch.zeros_like(pre_ineq_violation))
        return loss.mean()
    
    def safeguard_metrics(self, **kwargs) -> dict[str, Any]:
        """
            Metrics to monitor the safeguard performance residual violations 

            Args:
                keyword arguments for compatibility
            Returns:
                dict[str, Any]: dictionary of safeguard metrics
        """

        safe_guard_metrics_training = {
        "pre_eq_violation":     self.pre_eq_violation,
        "pre_ineq_violation":   self.pre_ineq_violation}
        
        safe_guard_metrics_evaluation = safe_guard_metrics_training.copy() 
        
        if not torch.is_grad_enabled(): # evaluation mode
            safe_guard_metrics_evaluation |= {
                "post_eq_violation":    self.post_eq_violation,
                "post_ineq_violation":  self.post_ineq_violation}
        return super().safeguard_metrics(safeguard_metrics_dict=safe_guard_metrics_evaluation,
                                        safeguard_metrics_dict_training_only=safe_guard_metrics_training)
    
#################################################
# LBFGS solver functions 
#################################################


def lbfgs_torch_solve(    
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config = None,
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
        data: Data object with equality_constraint_violation and inequality_constraint_violation methods
        config: Solver configuration
    """
    if config is None:
        config = SimpleNamespace(**kwargs)

    obj_fn = _create_objective_function(x, data, 1.0)
    
    # use torch implementation of LBFGS if no differentiable steps needed
    if config.max_diff_iter == 0:
        
        a_diff = _lbfgs_step(
            y_init,
            obj_fn,
            config.gradient_clipping_max_norm,
            config.max_iter,
            config.lbfgs_torch_learning_rate,
            config.lbfgs_history_size
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
            config.gradient_clipping_max_norm,
            config.max_iter - config.max_diff_iter,
            config.lbfgs_torch_learning_rate,
            config.lbfgs_history_size
        )
        # Connect gradients through differentiable phase
        return a_diff + (a_nondiff - a_diff).detach()
    else:
        return a_diff


def nondiff_lbfgs_torch_solve(    
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config = None,
    **kwargs
) -> torch.Tensor:
    """
    Non-differentiable version (no backward graph).
    
    Args:
        x: Input tensor (batch_size, input_dim)
        y_init: Initial action (batch_size, action_dim)
        data: Data object with equality_constraint_violation and inequality_constraint_violation methods
        config: Solver configuration
    """
    if config is None:
        config = SimpleNamespace(**kwargs)

    obj_fn = _create_objective_function(x, data, config.objective_scale)
    
    a = _lbfgs_step(
        y_init,
        obj_fn,
        config.gradient_clipping_max_norm,
        config.max_iter,
        config.lbfgs_torch_learning_rate,
        config.lbfgs_history_size
    )
    
    return a.detach()

def _lbfgs_step(
    y_init: torch.Tensor,
    obj_fn: Callable,
    gradient_clipping_max_norm: float,
    max_iter: int,
    lbfgs_torch_learning_rate: float,
    lbfgs_history_size: int
    ) -> torch.Tensor:
    """
    LBFGS step that preserves gradient connection to y_init.
    
    It optimizes a delta (correction) tensor, then adds it to y_init.
    This keeps y_init in the computation graph while allowing LBFGS to optimize.
    
    Args:
        y_init: Initial point (can be non-leaf, gradient connection preserved!)
        obj_fn: Objective function
        gradient_clipping_max_norm: Gradient clipping norm
        max_iter: Max iterations
        lbfgs_torch_learning_rate: Learning rate
        lbfgs_history_size: LBFGS history size
        create_graph: Whether to create backward graph
    """
    # Optimizing delta for y_init + delta
    # y_init stays in the graph, delta is the leaf tensor for optimizer
    delta = torch.zeros_like(y_init, requires_grad=True)
    
    optimizer = torch.optim.LBFGS([delta], lr=lbfgs_torch_learning_rate, max_iter=max_iter, 
                      history_size=lbfgs_history_size, line_search_fn='strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        y_current = y_init + delta
        phi = obj_fn(y_current)
        
        # Compute gradient w.r.t. delta
        grad = torch.autograd.grad(phi.mean(), delta, create_graph=False)[0]
        
        # Clip gradient and assign to delta.grad
        grad_norm = grad.norm()
        if grad_norm > gradient_clipping_max_norm:
            grad = grad * (gradient_clipping_max_norm / grad_norm)
        
        delta.grad = grad
        
        return phi
    
    optimizer.step(closure)
    
    # Return y_init + delta as the optimized point
    return y_init + delta

#################################################
# functions and classes from original LBFGS solver
#################################################

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




def _create_objective_function(x: torch.Tensor, data, objective_scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create objective function closure."""
    def _obj(y: torch.Tensor) -> torch.Tensor:
        equality_constraint_violationual = (data.equality_constraint_violation(x, y) ** 2).sum(dim=1).mean(0)
        inequality_constraint_violationual = (data.inequality_constraint_violation(x, y) ** 2).sum(dim=1).mean(0)
        return objective_scale * (equality_constraint_violationual + inequality_constraint_violationual)
    return _obj


def _check_convergence(f_val: torch.Tensor, g: torch.Tensor, config) -> torch.Tensor:
    """Check convergence criteria."""
    val_converged = f_val / config.objective_scale < config.convergence_value_tolerance
    grad_converged = g.norm(dim=1) < config.convergence_gradient_tolerance
    return val_converged | grad_converged


def _backtracking_line_search(
    y: torch.Tensor,
    d: torch.Tensor,
    g: torch.Tensor,
    f_val: torch.Tensor,
    obj_func: Callable,
    config,
) -> float:
    """Perform backtracking line search."""
    step = 1.0
    dir_deriv = (g * d).sum()
    
    with torch.no_grad():
        for _ in range(config.line_search_max_iter):
            y_trial = y + step * d
            f_trial = obj_func(y_trial)
            if (f_trial <= f_val + config.line_search_armijo_c * step * dir_deriv).all():
                break
            step *= config.line_search_rho
    
    return step


def lbfgs_solve(
    x: torch.Tensor,
    y_init: torch.Tensor,
    data,
    config = None,
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
        Data object with equality_constraint_violation and inequality_constraint_violation methods
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
        config = SimpleNamespace(**kwargs)
    
    # Initialize
    y = y_init.clone()
    B, n = y_init.shape
    device, dtype = y_init.device, y_init.dtype
    
    # History buffers
    S_hist = torch.zeros(config.lbfgs_history_size, B, n, device=device, dtype=dtype)
    Y_hist = torch.zeros_like(S_hist)
    hist_len = 0
    hist_ptr = 0
    
    # Create objective function
    obj_func = _create_objective_function(x, data, config.objective_scale)
    
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
            idx = (hist_ptr - hist_len + torch.arange(hist_len, device=device)) % config.lbfgs_history_size
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
        hist_ptr = (hist_ptr + 1) % config.lbfgs_history_size
        hist_len = min(hist_len + 1, config.lbfgs_history_size)
        
        # Prepare for next iteration
        y = y_next
        f_val = f_next.clone()
        g = g_next.clone()
        
        if config.verbose and k % 5 == 0:
            print(f"Iter {k:3d}: f = {f_next.item()/config.objective_scale:.3e}, "
                  f"|g| = {g_next.norm():.3e}, step = {step:.3e}")
    
    return y


 
