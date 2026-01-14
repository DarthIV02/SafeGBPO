from typing import Self
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog

from safeguards.interfaces.safeguard import Safeguard, SafeEnv
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Union
import time
import os
from torch.func import jacrev, vmap
import torch.nn.functional as F

###########################################
# Constraint primitives
###########################################

@dataclass
class HyperplaneConstraint:
    """
    Equality constraint of the form A x = b.

    Used for projecting lifted variables onto the ADMM equality subspace.
    """
    A: Tensor  # (B, m, n)
    b: Tensor  # (B, m, 1)
    
    def __init__(self, A, b):
        with torch.no_grad():
            Apinv = torch.linalg.pinv(A)
            B, m, n = A.shape
            # ensure identity is on the same device and dtype as A to avoid device mismatches
            self.P = torch.eye(n, device=A.device, dtype=A.dtype).expand(B, n, n) - torch.bmm(Apinv, A)
            self.c = torch.bmm(Apinv, b)
        self.P = self.P.detach()    
        self.c = self.c.detach()

    def project(self, y: Tensor) -> Tensor:
        """
        Project onto the affine subspace {x | A x = b}.

        Args:
            y: Input tensor.

        Returns:
            Projected tensor satisfying A x = b.
        """
        return torch.bmm(self.P, y) + self.c

@dataclass
class BoxConstraint:
    """
    Axis-aligned box constraint lb <= x <= ub.
    """
    lb: Tensor  
    ub: Tensor  

    def __init__(self, lb, ub):
        self.lb = lb.detach()
        self.ub = ub.detach()

    def project(self, y: Tensor) -> Tensor:
        """
        Clamp input to the box bounds.

        Args:
            y: Input tensor.

        Returns:
            Box-projected tensor.
        """
        return torch.clamp(y, self.lb, self.ub)

# ---------------------------------------------------------------------------
#                            PiNet Safeguard
# ---------------------------------------------------------------------------
class PinetSafeguard(Safeguard):

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        env: SafeEnv,
        regularisation_coefficient: float,
        n_iter_admm: int,
        n_iter_bwd: int,
        sigma: float = 1.0,
        omega: float = 1.7,
        bwd_method: str = "implicit",
        fpi: bool = False,
        **kwargs
    ):
        super().__init__(env)

        self.regularisation_coefficient = regularisation_coefficient
        self.n_iter_admm = n_iter_admm
        self.n_iter_bwd = n_iter_bwd

        self.sigma = sigma
        self.omega = omega
        self.bwd_method = bwd_method
        self.fpi = fpi        
        self.save_dim = False

        if not self.env.polytope:
            raise Exception("Polytope attribute has to be True")
        
        # --- Visualization Flags ---
        self.debug_mode = False
        self.last_trajectory = None
        self.last_unsafe_action = None
        self.last_safe_set_info = {} 
    
    @jaxtyped(typechecker=beartype)
    def safeguard(
        self,
        action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]
    ) -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        
        """
        Project actions onto the safe set using ADMM.

        Args:
            action: Raw action tensor.

        Returns:
            Safeguarded action.
        """

        action_lifted = action.unsqueeze(2)

        # Linear constraints A x <= b
        A, b = self.env.compute_A_b()

        if not self.save_dim:
            # Save frequently used variables
            Bbatch, m, D = A.shape
            total_dim = D + m
            
            # create tensors on the same device and dtype as A
            E = torch.zeros(Bbatch, D, D, device=A.device, dtype=A.dtype)
            Z = torch.zeros(Bbatch, D, m, device=A.device, dtype=A.dtype)
            self.negI = -torch.eye(m, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(Bbatch, 1, 1)

            self.top = torch.cat([E, Z], dim=2)

            self.beq = torch.zeros(Bbatch, total_dim, 1, device=A.device, dtype=A.dtype)
            self.lb = torch.full((Bbatch, total_dim, 1), -torch.inf, device=A.device, dtype=A.dtype)
            
            self.save_dim = True

        # --- Visualization Logic ---
        if self.debug_mode:
            y_safe, trajectory_lifted = self._run_projection(
                action=action_lifted,
                A=A, 
                b=b,
                debug=True
            )
            
            D = self.action_dim
            self.last_trajectory = [t[:, :D, :].squeeze(2).detach().cpu() for t in trajectory_lifted]
            self.last_unsafe_action = action.detach().cpu()

            self.last_safe_set_info = {
                "safe_set_A": A.detach().cpu(),
                "safe_set_b": b.detach().cpu()
            }

        else:
            y_safe = self._run_projection(
                action=action_lifted,
                A=A, 
                b=b,
                debug=False 
            )

        return y_safe

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:

        return self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action)
    
    def _run_admm(self, sk, y_raw, steps, debug=False):
        D = self.action_dim
        sk_iter = sk.clone()
        trajectory = [] # For visualization

        y_raw_D = y_raw[:, :D, :]
        denom = 1 / (1 + 2 * self.sigma)
        addition = 2 * self.sigma * y_raw_D
        
        # --- Visualization Fix: Save Init point ---
        if debug:
            zk_init = self.eq.project(sk_iter)
            trajectory.append(zk_init.detach().clone())

        for _ in range(steps):
            zk = self.eq.project(sk_iter)
            reflect = 2 * zk - sk_iter
            numerator = addition + reflect[:, :D, :]
            reflect[:, :D, :] = numerator * denom
            tk = self.box.project(reflect)
            sk_iter = sk_iter + self.omega * (tk - zk)

            if debug:
                zk_next = self.eq.project(sk_iter)
                trajectory.append(zk_next.detach().clone())

        if debug:
            return sk_iter, trajectory
        return sk_iter

    def _elevate(self, x: Tensor, A: Tensor) -> Tensor:
        """
        Lift primal variable into augmented ADMM space.
        """
        Ax = torch.bmm(A, x)
        return torch.cat([x, Ax], dim=1)

    # -------------------------------------------------------------------------------
    # ADMM projection wrapped in a torch.autograd.Function with implicit backward
    # -------------------------------------------------------------------------------
    def _run_projection(
        self,
        action: Tensor,         # (B, D, 1)
        A: Tensor,              # (B, m, D)
        b: Tensor,              # (B, m)
        debug: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Run ADMM projection with chosen backward method.

        Returns:
            Safe action tensor.
        """
        Bbatch, m, D = A.shape
        total_dim = D + m

        bottom = torch.cat([A, self.negI], dim=2)
        Aeq = torch.cat([self.top, bottom], dim=1)

        ub = torch.cat([
            torch.full((Bbatch, D, 1),  torch.inf, device=A.device, dtype=A.dtype),
            b.unsqueeze(2)
        ], dim=1)

        self.eq = HyperplaneConstraint(Aeq, self.beq)
        self.box = BoxConstraint(self.lb, ub)

        # --- Visualization Logic (Force Unroll) ---
        if debug:
            y_safe, trajectory = self.project_unroll(
                yraw = self._elevate(action, A),
                step_iteration = self._run_admm,
                step_final = self.eq.project,
                og_dim = int(D),                 
                n_iter = self.n_iter_admm,       
                debug = True
            )
            return y_safe, trajectory

        # --- Normal Logic ---
        if self.bwd_method == "implicit":
            y_safe = _ProjectImplicitFn.apply(
                self._elevate(action, A),       # yraw
                self._run_admm,                 # step_iteration
                self.eq.project,                # step_final
                int(D),                         # og_dim
                total_dim,                      # dim_lifted
                self.n_iter_admm,               # n_iter
                self.n_iter_bwd,                # n_iter_bwd
                self.fpi,                       # fpi
            )
        
        elif self.bwd_method == "unroll":
            y_safe = self.project_unroll(
                yraw = self._elevate(action, A), 
                step_iteration = self._run_admm, 
                step_final = self.eq.project,    
                og_dim = int(D),                 
                n_iter = self.n_iter_admm,       
                debug = False
            )
        
        else:
            raise ValueError(f"Unknown bwd_method: {self.bwd_method}")

        return y_safe

    def project_unroll(self, yraw,
                step_iteration, step_final, og_dim, n_iter, debug=False, **kwargs):

        """
        Fully unrolled ADMM projection (autograd-tracked).
        """
        with torch.enable_grad():
            # --- Algorithm Change: Warm Start (Critical for Visualization) ---
            # Initialize with yraw (unsafe action) instead of zeros.
            # This makes the trajectory start FROM the unsafe action.
<<<<<<< HEAD
            sK = torch.zeros_like(yraw, requires_grad=True) 
=======
            sK = torch.zeros_like(yraw, requires_grad=True)
>>>>>>> 6aee374f1053edc2c315664ba493be5420e3b7e0
            
            if debug:
                sK, trajectory = step_iteration(sK, yraw, n_iter, debug=True)
            else:
                sK = step_iteration(sK, yraw, n_iter, debug=False)
            
            y = step_final(sK)
            result = y[:, :og_dim].squeeze(2)

        if debug:
            return result, trajectory
        return result
    
    def get_visualization_data(self):
        """Helper to extract data for plotting later."""
        if self.last_trajectory is None:
            return None
        
        traj_stack = torch.stack(self.last_trajectory)
        unsafe_cpu = self.last_unsafe_action.detach().cpu()

        return {
            "trajectory": traj_stack,
            "unsafe_action": unsafe_cpu,
            **self.last_safe_set_info 
        }

class _ProjectImplicitFn(torch.autograd.Function):

    """
    Custom autograd Function implementing implicit differentiation
    through the ADMM fixed point.
    """

    @staticmethod
    def forward(ctx, yraw,
                step_iteration, step_final, og_dim, dim_lifted, n_iter, n_iter_bwd, fpi):
        
        # --- Fix applied here as well for consistency during training ---
        # Initialize with yraw (Warm Start) to match project_unroll behavior
        sK = yraw.clone() 
        
        with torch.no_grad():
            sK = step_iteration(sK, yraw, n_iter) # Default debug=False

        y = step_final(sK).detach()

        ctx.save_for_backward(sK.clone().detach(), yraw.clone().detach())
        ctx.step_iteration = step_iteration
        ctx.step_final = step_final
        ctx.n_iter_bwd = n_iter_bwd
        ctx.dim_lifted = dim_lifted
        ctx.fpi = fpi

        return y[:, :og_dim].squeeze(2)

    @staticmethod
    def backward(ctx, grad_y):
        """
        Implicit backward solve for ADMM with implicit differentiation.
        """
        sK, yraw = ctx.saved_tensors
        step_iteration = ctx.step_iteration
        step_final = ctx.step_final
        n_iter_bwd = ctx.n_iter_bwd
        dim_lifted = ctx.dim_lifted
        fpi = ctx.fpi

        batch, out_dim = grad_y.shape
        grad_y = grad_y.unsqueeze(2).clone().detach()

        y_for_vjp = torch.cat([
            grad_y,
            torch.zeros(batch, dim_lifted - out_dim, 1, device=grad_y.device)
        ], dim=1)

        sK_bwd = sK.requires_grad_(True)
        yraw_bwd = yraw.requires_grad_(True)

        with torch.enable_grad():
            y_final = step_final(sK_bwd)

        vjp = torch.autograd.grad(
            outputs=y_final,
            inputs=sK_bwd,
            grad_outputs=y_for_vjp,
            retain_graph=False,
            allow_unused=False
        )[0]
      
        def iteration_vjp(v):
            with torch.enable_grad():
                # debug=False implicitly
                admm_plus = step_iteration(sK_bwd, yraw_bwd, 1)
            return torch.autograd.grad(
                outputs=admm_plus,
                inputs=sK_bwd,
                grad_outputs=v,
                retain_graph=False,
                allow_unused=False
            )[0]

        if fpi:
            g = torch.zeros_like(vjp)
            for _ in range(n_iter_bwd):
                g = iteration_vjp(g) + vjp
        else:
            g = vjp.clone()
            for _ in range(n_iter_bwd):
                g = g + 0.2 * (vjp - (g - iteration_vjp(g)))

        def final_fn(y_in):
            with torch.enable_grad():
                # debug=False implicitly
                s = step_iteration(sK_bwd.detach(), y_in, 1)
            return step_final(s)

        _, grad_yraw = torch.autograd.functional.jvp(final_fn, yraw_bwd, g)

        return grad_yraw, None, None, None, None, None, None, None, None, None, None
