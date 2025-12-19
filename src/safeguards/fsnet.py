import cvxpy as cp
from torch import Tensor
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, jaxtyped
from typing import Dict, Tuple, Callable, Optional
import torch
from time import time

from safeguards.interfaces.safeguard import Safeguard, SafeEnv
from sets.polytope import HPolytope
from sets.zonotope import Zonotope

from safeguards.fsnet_solvers.lbfgs import hybrid_lbfgs_solve, nondiff_lbfgs_solve
from safeguards.fsnet_solvers.lbfgs_torch_opt import lbfgs_torch_solve, nondiff_lbfgs_torch_solve

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
                 **kwargs):
        Safeguard.__init__(self, env)

        self.boundary_layer = None
        self.regularisation_coefficient = regularisation_coefficient
        self.eq_pen_coefficient = eq_pen_coefficient
        self.ineq_pen_coefficient = ineq_pen_coefficient 

        self.config_method = kwargs

        if torch.get_default_device() == 'cpu':
            self.solver = hybrid_lbfgs_solve
            self.nondiff_solver = nondiff_lbfgs_solve
        else:
            self.solver =     lbfgs_torch_solve  
            self.nondiff_solver =   nondiff_lbfgs_torch_solve

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
        self.data = self.safe_action_set()
        if not isinstance(self.data, (HPolytope, Zonotope)):
            raise NotImplementedError("FSNet only supports Polytope and Zonotope safe action sets.")


        self.data.setup_resid()
        processed_action = self.data.pre_process_action(action)
        self.pre_eq_violation = self.data.eq_resid(None, processed_action).square().sum(dim=1)
        self.pre_ineq_violation = self.data.ineq_resid(None, processed_action).square().sum(dim=1)
        
        # Use non-differentiable solver during evaluation (no grad mode),
        # hybrid solver during training (with grad)

        if torch.is_grad_enabled():
            safe_action = self.solver(
                None,
                processed_action,
                self.data,
                **self.config_method
            )
        else:
            with torch.enable_grad():
                safe_action = self.nondiff_solver(
                    None,
                    processed_action,
                    self.data,
                    **self.config_method
                )
    
        self.post_eq_violation = self.data.eq_resid(None, safe_action).square().sum(dim=1)
        self.post_ineq_violation = self.data.ineq_resid(None, safe_action).square().sum(dim=1)

        safe_action = self.data.post_process_action(safe_action)
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
        return  super().safeguard_metrics() | {
            "pre_eq_violation": self.pre_eq_violation.mean().item(),
            "pre_ineq_violation": self.pre_ineq_violation.mean().item(),
            "post_eq_violation": self.post_eq_violation.mean().item(),
            "post_ineq_violation": self.post_ineq_violation.mean().item(),
        }
    

 