from time import time
from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Dict, Tuple, Callable, Optional
import torch

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

from sets.polytope import HPolytope
from sets.zonotope import Zonotope

# original file from FSNet codebase with kwargs extension
from safeguards.fsnet_solvers.lbfgs import hybrid_lbfgs_solve, nondiff_lbfgs_solve 
# PyTorch implementation of LBFGS solver for FSNet (Yasin's addition)
from safeguards.fsnet_solvers.lbfgs_torch_opt import lbfgs_torch_solve, nondiff_lbfgs_torch_solve 
from safeguards.fsnet_solvers.gradient_descent import gradient_descent_solve 

class FSNetSafeguard(Safeguard):
    """
    Projecting an unsafe action to the closest safe action.
    Safeguard implementation based on FSNet
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, 
                 env: SafeEnv, 
                 regularisation_coefficient: float,
                 **kwargs):
        Safeguard.__init__(self, env)

        self.boundary_layer = None
        self.regularisation_coefficient = regularisation_coefficient

        # assume the remaining kwargs are solver config parameters
        self.config_method = kwargs

        # torch opt LBFGS solver should be more efficient on GPU 
        if torch.get_default_device() == 'cpu': # an option to switch solvers based on device
            self.solver = hybrid_lbfgs_solve
            self.nondiff_solver = nondiff_lbfgs_solve
        else:
            self.solver = lbfgs_torch_solve  
            self.nondiff_solver = nondiff_lbfgs_torch_solve

        # Visualization Flags
        self.debug_mode = False
        self.last_trajectory = None
        self.last_unsafe_action = None
        self.last_safe_set_info = {}

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:

        self.data = self.safe_action_set()
        if not isinstance(self.data, (HPolytope, Zonotope)):
            raise NotImplementedError("FSNet only supports Polytope and Zonotope safe action sets.")

        # first setup the residuals by computing A and b matrices (Ay ≤ b) and equality matrices (Cy = d)
        self.data.setup_resid()

        # prepocess the action to accommate the different safe set representations
        processed_action = self.data.pre_process_action(action)

        # compute pre safeguard violations for logging
        self.pre_eq_violation = self.data.eq_resid(None, processed_action).square().sum(dim=1)
        self.pre_ineq_violation = self.data.ineq_resid(None, processed_action).square().sum(dim=1)
        
        # --- Optimization: Check if actions are already safe ---
        tol = 1e-6
        is_safe_mask = (self.pre_eq_violation <= tol) & (self.pre_ineq_violation <= tol)
        
        # If all samples in the batch are safe, return immediately (unless debugging)
        if is_safe_mask.all() and not self.debug_mode:
             self.post_eq_violation = self.pre_eq_violation
             self.post_ineq_violation = self.pre_ineq_violation
             return action

        # Use non-differentiable solver during evaluation (no grad mode),
        # hybrid solver during training (with grad) for better backpropagation

        # --- Visualization Logic Start ---
        if self.debug_mode:
            # Use the local LBFGS solver for visualization
            from safeguards.fsnet_solvers.lbfgs import nondiff_lbfgs_solve as debug_solver
            
            with torch.enable_grad():
                result = debug_solver(
                    None,
                    processed_action,
                    self.data,
                    debug_trajectory=True, 
                    **self.config_method
                )
            
            if isinstance(result, tuple):
                safe_action, trajectory = result
                # Convert trajectory back to original space
                self.last_trajectory = [self.data.post_process_action(t) for t in trajectory]
                self.last_unsafe_action = action.detach().cpu()
            else:
                safe_action = result
                self.last_trajectory = [self.data.post_process_action(safe_action).detach().cpu()]

            # Save Safe Set Info
            if isinstance(self.data, Zonotope):
                self.last_safe_set_info = {
                    "safe_set_center": self.data.center.detach().cpu(),
                    "safe_set_generators": self.data.generator.detach().cpu()
                }
            elif hasattr(self.data, "A"): # For HPolytope
                self.last_safe_set_info = {
                    "safe_set_A": self.data.A.detach().cpu(),
                    "safe_set_b": self.data.b.detach().cpu()
                }
        # --- Visualization Logic End ---

        elif torch.is_grad_enabled():
            # Training phase: usually no debug needed, keep it fast
            safe_action = self.solver(
                None,
                processed_action,
                self.data,
                **self.config_method
            )
        else:
            with torch.enable_grad(): # ensure grad is enabled for solvers just like in FSNet codebase
                safe_action = self.nondiff_solver(
                    None,
                    processed_action,
                    self.data,
                    **self.config_method
                )
    
        # compute post safeguard violations for logging
        self.post_eq_violation = self.data.eq_resid(None, safe_action).square().sum(dim=1)
        self.post_ineq_violation = self.data.ineq_resid(None, safe_action).square().sum(dim=1)

        # return the safe action to original space
        safe_action = self.data.post_process_action(safe_action)

        # --- Optimization: Restore originally safe actions ---
        if is_safe_mask.ndim == 1:
            mask = is_safe_mask.unsqueeze(1).expand_as(action)
        else:
            mask = is_safe_mask.expand_as(action)
            
        safe_action = torch.where(mask, action, safe_action)

        return safe_action

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:
        """
        Compute the safeguard loss for FSNet.
        """
        # compute the safeguard loss
        loss = self.regularisation_coefficient/2 * torch.nn.functional.mse_loss(safe_action, action) 
        
        # add penalty for residual violations as defined in FSNet paper practical implementation
        if self.pre_eq_violation.mean() > 1e-3:
            loss = loss + self.regularisation_coefficient * self.pre_eq_violation.mean()
        if self.pre_ineq_violation.mean() > 1e-3:
            loss = loss + self.regularisation_coefficient * self.pre_ineq_violation.mean()
        return loss
    
    def safeguard_metrics(self):
        return  super().safeguard_metrics() | {
            "pre_eq_violation":     self.pre_eq_violation.mean().item() if type(self.pre_eq_violation) == torch.Tensor else self.pre_eq_violation,
            "pre_ineq_violation":   self.pre_ineq_violation.mean().item() if type(self.pre_ineq_violation) == torch.Tensor else self.pre_ineq_violation,
            "post_eq_violation":    self.post_eq_violation.mean().item() if type(self.post_eq_violation) == torch.Tensor else self.post_eq_violation,
            "post_ineq_violation":  self.post_ineq_violation.mean().item() if type(self.post_ineq_violation) == torch.Tensor else self.post_ineq_violation,
        }

    def get_visualization_data(self):
        if self.last_trajectory is None:
            return None
        
        traj_stack = torch.stack(self.last_trajectory).detach().cpu()
        unsafe_cpu = self.last_unsafe_action.detach().cpu()

        return {
            "trajectory": traj_stack,
            "unsafe_action": unsafe_cpu,
            **self.last_safe_set_info
        }