from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from jaxtyping import Float, jaxtyped
from beartype import beartype

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

# ============================================================
# JAX / Torch interoperability utilities
# ============================================================

def torch_to_jax(x: Tensor) -> jnp.ndarray:
    """
    Convert a PyTorch tensor to a JAX array via DLPack.

    Args:
        x: Input PyTorch tensor.

    Returns:
        A JAX array sharing memory with the input tensor.
    """
    return jax.dlpack.from_dlpack(x.detach())


def jax_to_torch(x: jnp.ndarray, device: torch.device | None) -> Tensor:
    """
    Convert a JAX array to a PyTorch tensor via DLPack.

    Args:
        x: Input JAX array.
        device: Target PyTorch device.

    Returns:
        A PyTorch tensor.
    """
    t = torch.utils.dlpack.from_dlpack(x.__dlpack__())
    return t.to(device) if device is not None else t


# ============================================================
# ADMM solver (fully JAX-based)
# ============================================================

@partial(jax.jit, static_argnames=("steps", "D", "omega", "sigma"))
def admm_run(
    sk: jnp.ndarray,
    yraw: jnp.ndarray,
    Aeq: jnp.ndarray,
    Apinv: jnp.ndarray,
    beq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    *,
    sigma: float,
    omega: float,
    D: int,
    steps: int,
) -> jnp.ndarray:
    """
    Run a fixed number of ADMM iterations for box + equality constraints.

    Args:
        sk: Initial iterate.
        yraw: Raw input (lifted variables).
        Aeq: Equality constraint matrix.
        Apinv: Pseudoinverse of Aeq.
        beq: Equality constraint vector.
        lb: Lower bounds.
        ub: Upper bounds.
        sigma: Quadratic regularization weight.
        omega: Relaxation parameter.
        D: Action dimension.
        steps: Number of ADMM iterations.

    Returns:
        Final ADMM iterate.
    """
    denom = 1.0 / (1.0 + 2.0 * sigma)
    addition = 2.0 * sigma * yraw[:, :D, :]

    def body(sk, _):
        correction = Aeq @ sk - beq
        zk = sk - Apinv @ correction
        reflect = 2.0 * zk - sk
        reflect_D = (reflect[:, :D, :] + addition) * denom
        reflect = reflect.at[:, :D, :].set(reflect_D)
        tk = jnp.clip(reflect, lb, ub)
        sk = sk + omega * (tk - zk)
        return sk, None

    sk, _ = jax.lax.scan(body, sk, None, length=steps)
    return sk

# ============================================================
# Custom JAX projection with implicit differentiation
# ============================================================

@jax.custom_vjp
def project_jax(
    yraw: jnp.ndarray,
    Aeq: jnp.ndarray,
    Apinv: jnp.ndarray,
    beq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    sigma: float,
    omega: float,
    D: int,
    steps: int,
    fpi: bool,
    n_iter_bwd: int,
    damping: float,
) -> jnp.ndarray:
    """
    Project lifted variables onto the feasible set using ADMM.

    Returns:
        Projected action (only the first D dimensions).
    """
    sk0 = jnp.zeros_like(yraw)
    sk = admm_run(
        sk0, yraw, Aeq, Apinv, beq, lb, ub,
        sigma=sigma, omega=omega, D=D, steps=steps
    )
    sk = sk - Apinv @ (Aeq @ sk - beq)
    return sk[:, :D, 0]


def project_fwd(*args):
    out = project_jax(*args)
    return out, args

def project_bwd(res, g):
    (
        yraw, Aeq, Apinv, beq, lb, ub,
        sigma, omega, D, steps, fpi, n_iter_bwd, damping
    ) = res

    def one_step(x):
        return admm_run(
            x, yraw, Aeq, Apinv, beq, lb, ub,
            sigma=sigma, omega=omega, D=D, steps=1
        )

    sk = jnp.zeros_like(yraw)
    _, vjp_fn = jax.vjp(one_step, sk)
    vjp = vjp_fn(g)[0]

    def iteration(v):
        _, fn = jax.vjp(one_step, sk)
        return fn(v)[0]

    if fpi:
        gsol = jnp.zeros_like(vjp)
        for _ in range(n_iter_bwd):
            gsol = iteration(gsol) + vjp
    else:
        gsol = vjp
        for _ in range(n_iter_bwd):
            gsol = gsol + damping * (vjp - (gsol - iteration(gsol)))

    _, vjp_y = jax.vjp(lambda y: one_step(sk), yraw)
    grad_yraw = vjp_y(gsol)[0]

    return grad_yraw, *([None] * (len(res) - 1))


project_jax.defvjp(project_fwd, project_bwd)

# ============================================================
# Torch-facing safeguard
# ============================================================

@jaxtyped(typechecker=beartype)
class PinetJAXSafeguard(Safeguard):
    """
    PiNet safeguard implemented via a fully JAX-based ADMM projection,
    wrapped for PyTorch autograd compatibility.
    """

    def __init__(
        self,
        env: SafeEnv,
        regularisation_coefficient: float,
        n_iter_admm: int,
        n_iter_bwd: int,
        sigma: float = 1.0,
        omega: float = 1.7,
        fpi: bool = True,
        **kwargs,
    ):
        """
        Args:
            env: Safe environment providing linear constraints.
            regularisation_coefficient: Weight of safeguard loss.
            n_iter_admm: Number of ADMM iterations.
            n_iter_bwd: Number of backward implicit iterations.
            sigma: ADMM penalty parameter.
            omega: ADMM relaxation parameter.
            fpi: Whether to use fixed-point iteration in backward pass.
        """
        super().__init__(env)
        self.regularisation_coefficient = regularisation_coefficient
        self.n_iter_admm = n_iter_admm
        self.n_iter_bwd = n_iter_bwd
        self.sigma = sigma
        self.omega = omega
        self.fpi = fpi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jax.config.update("jax_enable_x64", True)
        self.save_dim = False

        if not self.env.safe_action_polytope:
            raise Exception("Polytope attribute has to be True")


    def safeguard(self, action: Tensor) -> Tensor:
        """
        Project an action onto the safe set.

        Args:
            action: Raw action tensor.

        Returns:
            Safeguarded action.
        """
        action = action.unsqueeze(2)
        A, b = self.env.compute_A_b()

        self.pre_constraint_violation = torch.clamp(
            torch.bmm(A, action) - b.unsqueeze(2),
            min=0.0
        ).square().sum(dim=1).squeeze(1)

        if not self.save_dim:
            self.B, self.D, _ = action.shape
            self.m = A.shape[1]
            self.beq = jnp.zeros((self.B, self.D + self.m, 1))
            self.lb = jnp.full((self.B, self.D + self.m, 1), -jnp.inf)
            self.eye = -jnp.eye(self.m)
            self.save_dim = True

        yraw = torch_to_jax(action)
        A_j = torch_to_jax(A)
        b_j = torch_to_jax(b).reshape(self.B, self.m, 1)

        Aeq = jnp.zeros((self.B, self.D + self.m, self.D + self.m))
        Aeq = Aeq.at[:, self.D:, :self.D].set(A_j)
        Aeq = Aeq.at[:, self.D:, self.D:].set(self.eye)

        Apinv = jnp.linalg.pinv(Aeq)

        ub = jnp.concatenate(
            [jnp.full((self.B, self.D, 1), jnp.inf), b_j], axis=1
        )

        lifted = jnp.concatenate([yraw, A_j @ yraw], axis=1)

        out = ProjectJAXFunction.apply(
            lifted,
            self.device,
            Aeq, Apinv, self.beq, self.lb, ub,
            self.sigma, self.omega, self.D,
            self.n_iter_admm, self.fpi, self.n_iter_bwd, 0.2,
        )

        safe_action = out[:, :self.D]
        safe_action = jax_to_torch(safe_action, None).requires_grad_(True)

        self.post_constraint_violation = torch.clamp(
            torch.bmm(A, safe_action.unsqueeze(2)) - b.unsqueeze(2),
            min=0.0
        ).square().sum(dim=1).squeeze(1)

        return safe_action

    def safe_guard_loss(self, action: Tensor, safe_action: Tensor) -> Tensor:
        """
        MSE-based safeguard regularization loss.
        """
        return self.regularisation_coefficient * F.mse_loss(safe_action, action)
    
    def safeguard_metrics(self):
        return super().safeguard_metrics() | {
            "pre_ineq_violation": self.pre_constraint_violation.mean().item(),
            "post_ineq_violation": self.post_constraint_violation.mean().item(),
        }


class ProjectJAXFunction(torch.autograd.Function):
    """
    Torch autograd wrapper around the JAX projection.
    """

    @staticmethod
    def forward(ctx: Any, yraw: Tensor, device: torch.device, *params: Any) -> Tensor:
        out = project_jax(yraw, *params)
        ctx.save_for_backward(yraw)
        ctx.params = params
        return jax_to_torch(out, device)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor):
        (yraw,) = ctx.saved_tensors
        params = ctx.params
        g_jax = torch_to_jax(grad_out)

        _, vjp_fn = jax.vjp(lambda y: project_jax(y, *params), yraw)
        grad_y = vjp_fn(g_jax)[0]

        return jax_to_torch(grad_y, None), *([None] * len(params))