import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import time
import os

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

@dataclass
class HyperplaneConstraint:
    A: Tensor  # (B, m, n)
    b: Tensor  # (B, m, 1)
    
    def __init__(self, A, b):
        # store constants without gradients
        self.A = A.detach()
        self.b = b.detach()

        # pseudo-inverse as constant
        with torch.no_grad():
            self.Apinv = torch.linalg.pinv(self.A).detach()

    def project(self, y: Tensor) -> Tensor:
        """
        Projection onto { x | A x = b }:
            x_proj = x − (A^+)(Ax − b)
        """

        # grad flows ONLY through y
        correction = torch.bmm(self.A, y) - self.b
        correction_proj = torch.bmm(self.Apinv, correction)

        # this has requires_grad == True because y does
        projected_x = y - correction_proj

        return projected_x

@dataclass
class BoxConstraint:
    lb: Tensor  # (B, n, 1)
    ub: Tensor  # (B, n, 1)

    def __init__(self, lb, ub):
        self.lb = lb.detach()
        self.ub = ub.detach()

    def project(self, y: Tensor) -> Tensor:
        return torch.clamp(y, self.lb, self.ub)

def _save_log(stage: str, data: dict, logfile_base: str):
    """
    Save dictionary of tensors to a .pt file.
    Stage is either 'forward' or 'backward'.
    Creates two separate files:
        <logfile_base>_forward.pt
        <logfile_base>_backward.pt
    """
    base = logfile_base
    if base is None:
        return
    
    # remove .pt if user put it in the base name
    if base.endswith(".pt"):
        base = base[:-3]

    # choose file based on stage
    outfile = f"{base}_{stage}.pt"
    torch.save(data, outfile)

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
        debug: bool = False,
        **kwargs
    ):
        super().__init__(env)

        self.regularisation_coefficient = regularisation_coefficient
        self.debug = debug
        self.n_iter_admm = n_iter_admm
        self.n_iter_bwd = n_iter_bwd

        self.sigma = sigma
        self.omega = omega
        self.bwd_method = bwd_method
        self.log_file = f"logs/{self.bwd_method}_admm_iter_{self.n_iter_admm}_bwd_{self.n_iter_bwd}"

    @jaxtyped(typechecker=beartype)
    def safeguard(
        self,
        action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]
    ) -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:

        action = action.unsqueeze(2)  # match (B, D, 1)

        # ----- Build Ax ≤ b -----
        A, b = self.env.compute_A_b()

        # ADMM projection function with implicit backward that returns the final safe action.
        y_safe = self._project_with_implicit(
            action=action,  # (B, D, 1)
            A=A,            # (B, m, D)
            b=b            # (B, m)
        )  # returns (B, D)

        return y_safe  # already squeezed to (B, D)

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:

        return self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action)

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:
        return self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action)
    
    # ----------------------------------------------------------------------
    # Ruiz scaling
    # ----------------------------------------------------------------------
    def _ruiz(self, A, max_iter=10, eps=1e-9):
        B, m, n = A.shape

        d_r = torch.ones(B, m, 1, device=A.device, dtype=A.dtype)
        d_c = torch.ones(B, 1, n, device=A.device, dtype=A.dtype)

        M = A.clone()

        for _ in range(max_iter):
            row_norm = torch.norm(M, p=1, dim=2, keepdim=True).clamp_min(eps)
            M = M / row_norm
            d_r = d_r / row_norm

            col_norm = torch.norm(M, p=1, dim=1, keepdim=True).clamp_min(eps)
            M = M / col_norm
            d_c = d_c / col_norm
        return M.detach(), d_r.detach(), d_c.detach()

    def _run_admm(self, sk, y_raw, scale):
        sk_iter = sk.clone()
        sigma = torch.tensor(self.sigma, device=y_raw.device, dtype=y_raw.dtype)
        omega = torch.tensor(self.omega, device=y_raw.device, dtype=y_raw.dtype)
        D = self.action_dim
        
        # Clone scale slice to ensure it's a distinct tensor
        scale_sub = scale[:, :D, :].clone() 

        for _ in range(self.n_iter_admm):
            zk = self.eq.project(sk_iter)
            reflect = 2*zk - sk_iter

            # 1. Force clones on slices to break "view" dependencies in the graph
            # 2. Pre-calculate denominator
            
            y_raw_D = y_raw[:, :D, :].clone()
            reflect_D = reflect[:, :D, :].clone()
            reflect_rest = reflect[:, D:, :].clone()
            
            scale_sq = scale_sub ** 2
            denom = 1 + 2 * sigma * scale_sq
            
            # Use multiplication by reciprocal instead of division to avoid DivBackward issues
            inv_denom = torch.reciprocal(denom) 
            
            numerator = (2 * sigma * scale_sub * y_raw_D + reflect_D)
            first = numerator * inv_denom

            to_box = torch.cat([first, reflect_rest], dim=1)
            tk = self.box.project(to_box)

            sk_iter = sk_iter.clone() + omega*(tk - zk)  # avoid in-place updates

        return sk_iter

    def _elevate(self, x: Tensor, A: Tensor) -> Tensor:
        Ax = torch.bmm(A, x)
        elevated = torch.cat([x, Ax], dim=1)
        return elevated

    # ----------------------------------------------------------------------
    # ADMM projection wrapped in a torch.autograd.Function with implicit backward
    # ----------------------------------------------------------------------
    def _project_with_implicit(
        self,
        action: Tensor,         # (B, D, 1)
        A: Tensor,              # (B, m, D)
        b: Tensor,              # (B, m)
    ) -> Tensor:
        """
        Runs the ADMM projection and returns the safe action (B, D) with implicit backward.
        """
        # Build constraints
        Bbatch, m, D = A.shape
        total_dim = D + m

        E = torch.zeros(Bbatch, D, D, device=A.device, dtype=A.dtype)
        Z = torch.zeros(Bbatch, D, m, device=A.device, dtype=A.dtype)
        negI = -torch.eye(m, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(Bbatch, 1, 1)

        top = torch.cat([E, Z], dim=2)
        bottom = torch.cat([A, negI], dim=2)
        Aeq = torch.cat([top, bottom], dim=1)  # (B, total_dim, total_dim?)

        beq = torch.zeros(Bbatch, total_dim, 1, device=A.device, dtype=A.dtype)

        # Ruiz scaling
        _, _, d_c = self._ruiz(Aeq)
        scale = d_c.transpose(1, 2)

        # Box constraint
        lb = torch.full((Bbatch, total_dim, 1), -torch.inf, device=A.device, dtype=A.dtype)
        ub = torch.cat([
            torch.full((Bbatch, D, 1),  torch.inf, device=A.device, dtype=A.dtype),
            b.unsqueeze(2)
        ], dim=1)

        self.eq = HyperplaneConstraint(Aeq, beq)
        self.box = BoxConstraint(lb, ub)

        # Call custom autograd Function
        if self.bwd_method == "implicit": 

            _ProjectImplicitFn.debug = self.debug
            _ProjectImplicitFn.logfile = self.log_file

            y_safe = _ProjectImplicitFn.apply(
                self._elevate(action, A.detach()),               # yraw
                scale,                # d_c
                self._run_admm,       # step_iteration
                self.eq.project,      # step_final
                int(D),               # og_dim
                int(total_dim),       # dim_lifted
                self.n_iter_admm,     # n_iter
                self.n_iter_bwd,      # n_iter_bwd
                False,                # fpi
            )
        
        elif self.bwd_method == "unroll":
            y_safe = self.project_complete(
                self._elevate(action, A.detach()),               # yraw
                scale,                # d_c
                self._run_admm,       # step_iteration
                self.eq.project,      # step_final
                int(D),               # og_dim
                self.n_iter_admm,     # n_iter
            )
        
        else:
            raise ValueError(f"Unknown bwd_method: {self.bwd_method}")

        return y_safe

    def project_complete(self, yraw, d_c,
                step_iteration, step_final, og_dim, n_iter):

        # Forward: run ADMM iterations tracking all the gradients
        with torch.enable_grad():
            sK = torch.zeros_like(yraw, requires_grad=True)

            for _ in range(n_iter):
                sK = step_iteration(sK, yraw, d_c)

            y = step_final(sK)

            y_scaled = y * d_c

            result = y_scaled[:, :og_dim].squeeze(2)

        if self.debug:
            # storage dict for backward gradients
            self._unroll_backward_grads = {}

            def save_grad(name):
                return lambda grad: self._unroll_backward_grads.__setitem__(name, grad.detach().cpu())

            # register hook on y_scaled BEFORE returning
            if not y_scaled.requires_grad:
                raise RuntimeError(f"y_scaled does not require grad: {y_scaled!r}")
            y_scaled.retain_grad()
            y_scaled.register_hook(save_grad("grad_y_scaled"))

            # also store forward values
            _save_log("forward", {
                "yraw": yraw.detach().cpu(),
                "d_c": d_c.detach().cpu(),
                "sK_final": sK.detach().cpu(),
                "y_final": y.detach().cpu(),
                "y_scaled": y_scaled.detach().cpu(),
            }, self.log_file)

            def save_final_grad(grad):
                _save_log("backward", {
                    "grad_y_scaled": self._unroll_backward_grads.get("grad_y_scaled", None),
                    "grad_output": grad.detach().cpu(),
                }, self.log_file)
            
            result.register_hook(save_final_grad)

        return result

# --------------------------
# Autograd Function
# --------------------------
class _ProjectImplicitFn(torch.autograd.Function):
    debug = False
    logfile = None

    @staticmethod
    def forward(ctx, yraw, d_c,
                step_iteration, step_final, og_dim, dim_lifted, n_iter, n_iter_bwd, fpi):

        # Forward: run ADMM iterations without tracking gradients
        sK = torch.zeros_like(yraw)
        with torch.no_grad():
            for _ in range(n_iter):
                sK = step_iteration(sK, yraw, d_c)

        y = step_final(sK).detach()
        y_scaled = y * d_c

        # Save for backward

        ctx.save_for_backward(sK.clone().detach(), yraw.clone().detach(), d_c.clone().detach())
        ctx.step_iteration = step_iteration
        ctx.step_final = step_final
        ctx.n_iter_bwd = n_iter_bwd
        ctx.dim_lifted = dim_lifted
        ctx.fpi = fpi

        if _ProjectImplicitFn.debug:
            _save_log("forward", {
                "yraw": yraw.detach().cpu(),
                "d_c": d_c.detach().cpu(),
                "sK_final": sK.detach().cpu(),
                "y_final": y.detach().cpu(),
                "y_scaled": y_scaled.detach().cpu(),
            }, _ProjectImplicitFn.logfile)

        return y_scaled[:, :og_dim].squeeze(2)

    @staticmethod
    def backward(ctx, grad_y):
        sK, yraw, d_c = ctx.saved_tensors
        step_iteration = ctx.step_iteration
        step_final = ctx.step_final
        n_iter_bwd = ctx.n_iter_bwd
        dim_lifted = ctx.dim_lifted
        fpi = ctx.fpi

        batch, out_dim = grad_y.shape
        grad_y = grad_y.unsqueeze(2).clone().detach()  # (B, D, 1)

        # Scale gradient back
        grad_z = grad_y * d_c[:, :out_dim, :]
        y_for_vjp = torch.cat([
            grad_z,
            torch.zeros(batch, dim_lifted - out_dim, 1, device=grad_y.device)
        ], dim=1)

        # --- Recompute sK and yraw for backward graph ---
        sK_bwd = sK.clone().detach().requires_grad_(True)
        yraw_bwd = yraw.clone().detach().requires_grad_(True)

        # Gradient through final projection
        with torch.enable_grad():
            y_final = step_final(sK_bwd)

        vjp = torch.autograd.grad(
            outputs=y_final,
            inputs=sK_bwd,
            grad_outputs=y_for_vjp,
            retain_graph=False,
            allow_unused=False
        )[0]

        # Implicit backward solve: (I - J_iteration)^T g = vjp       
        def iteration_vjp(v):
            with torch.enable_grad():
                admm_plus = step_iteration(sK_bwd, yraw_bwd, d_c.clone())  # recompute fresh
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
                g = iteration_vjp(g).clone() + vjp
        else:
            g = vjp.clone()
            for _ in range(n_iter_bwd):
                g = g + 0.1 * (vjp - (g - iteration_vjp(g).clone()))

        # Gradient w.r.t input yraw
        with torch.enable_grad():
            admm_plus_last = step_iteration(sK_bwd, yraw_bwd, d_c.clone())
        
        grad_yraw = torch.autograd.grad(
            outputs=admm_plus_last,
            inputs=yraw_bwd,
            grad_outputs=g,
            retain_graph=False,
            allow_unused=False
        )[0]

        if _ProjectImplicitFn.debug:
            _save_log("backward", {
                "grad_y_in": grad_y.detach().cpu(),
                "grad_z": grad_z.detach().cpu(),
                "vjp_projection": vjp.detach().cpu(),
                "implicit_solution_g": g.detach().cpu(),
                "grad_yraw": None if grad_yraw is None else grad_yraw.detach().cpu(),
            }, _ProjectImplicitFn.logfile)

        return grad_yraw, None, None, None, None, None, None, None, None, None, None