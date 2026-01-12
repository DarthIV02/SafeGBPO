from torch import Tensor
from beartype import beartype
from jaxtyping import Float, jaxtyped
from typing import Dict, Tuple, Callable, Optional
import torch

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

from src.sets.polytope import Polytope
from src.sets.zonotope import Zonotope

from safeguards.fsnet_solvers.lbfgs import hybrid_lbfgs_solve, nondiff_lbfgs_solve #original file from FSNet codebase with kwargs extension
from safeguards.fsnet_solvers.lbfgs_torch_opt import lbfgs_torch_solve, nondiff_lbfgs_torch_solve # PyTorch implementation of LBFGS solver for FSNet
from safeguards.fsnet_solvers.gradient_descent import gradient_descent_solve # simple gradient descent solver

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
        if not isinstance(self.data, (Polytope, Zonotope)):
            raise NotImplementedError("FSNet only supports Polytope and Zonotope safe action sets.")

        # first setup the residuals by computing A and b matrices (Ay ≤ b) and equality matrices (Cy = d)
        self.data.setup_resid()

        # prepocess the action to accommate the different safe set representations
        processed_action = self.data.pre_process_action(action)

        # compute pre safeguard violations for logging
        self.pre_eq_violation = self.data.eq_resid(None, processed_action).square().sum(dim=1)
        self.pre_ineq_violation = self.data.ineq_resid(None, processed_action).square().sum(dim=1)
        
        # Use non-differentiable solver during evaluation (no grad mode),
        # hybrid solver during training (with grad) for better backpropagation

        if torch.is_grad_enabled():
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
        return safe_action

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:
        """
        Compute the safeguard loss for FSNet.
        Args:
            action: The original action before safeguarding.
            safe_action: The safeguarded action.
        Returns:
            The safeguard loss consisting loss f(ˆyθ(x); x) + ρ/2∥yθ(x) − yˆθ(x)∥^2_2 (+ρ/2 * residual penalties for practical efficiency)
        """

        # compute the safeguard loss
        loss = self.regularisation_coefficient/2 * torch.nn.functional.mse_loss(safe_action, action) 
        
        # add penalty for residual violations as defined in FSNet paper practical implementation
        # for good backpropagation, only add penalty if the mean violation is significant
        if self.pre_eq_violation.mean() > 1e-3:
            loss = loss + self.regularisation_coefficient * self.pre_eq_violation.mean()
        if self.pre_ineq_violation.mean() > 1e-3:
            loss = loss + self.regularisation_coefficient * self.pre_ineq_violation.mean()
        return loss
    
    def safeguard_metrics(self):
        """
            Metrics to monitor the safeguard performance residual violations 
        """

        return  super().safeguard_metrics() | {
            "pre_eq_violation":     self.pre_eq_violation.mean().item() if type(self.pre_eq_violation) == torch.Tensor else self.pre_eq_violation,
            "pre_ineq_violation":   self.pre_ineq_violation.mean().item() if type(self.pre_ineq_violation) == torch.Tensor else self.pre_ineq_violation,
            "post_eq_violation":    self.post_eq_violation.mean().item() if type(self.post_eq_violation) == torch.Tensor else self.post_eq_violation,
            "post_ineq_violation":  self.post_ineq_violation.mean().item() if type(self.post_ineq_violation) == torch.Tensor else self.post_ineq_violation,
        }
    

 