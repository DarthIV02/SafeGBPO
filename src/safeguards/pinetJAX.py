# ============================================================
# Fully JAX-based PiNet ADMM Safeguard (Torch Wrapper)
# ============================================================

from safeguards.interfaces.safeguard import Safeguard, SafeEnv
import torch
from torch import Tensor
import torch.utils.dlpack
import jax
import jax.numpy as jnp
from jaxtyping import Float, jaxtyped
from beartype import beartype
from functools import partial
from dataclasses import dataclass
from typing import Tuple
import torch.nn.functional as F
import numpy as np

# ============================================================
# JAX utilities
# ============================================================

def torch_to_jax(x: torch.Tensor) -> jnp.ndarray:
    return jax.dlpack.from_dlpack(x.detach())

def jax_to_torch(x: jnp.ndarray) -> torch.Tensor:
   return torch.tensor(np.array(x), device="cpu", dtype=torch.float64)

# ============================================================
# JAX constraints
# ============================================================

@dataclass
class JAXHyperplane:
    A: jnp.ndarray      # (B, m, n)
    Apinv: jnp.ndarray  # (B, n, m)
    b: jnp.ndarray      # (B, m, 1)

    def project(self, x):
        return x - self.Apinv @ (self.A @ x - self.b)

@dataclass
class JAXBox:
    lb: jnp.ndarray
    ub: jnp.ndarray

    def project(self, x):
        return jnp.clip(x, self.lb, self.ub)

# ============================================================
# JAX ADMM iteration
# ============================================================

@partial(jax.jit, static_argnames=("steps", "D"))
def admm_run(
    sk,
    yraw,
    scale,
    Aeq,
    Apinv,
    beq,
    lb,
    ub,
    *,
    sigma: float,
    omega: float,
    D: int,
    steps: int
):
    scale_sub = scale[:, :D, :]
    denom = 1.0 / (1.0 + 2.0 * sigma * scale_sub**2)
    addition = 2.0 * sigma * scale_sub * yraw[:, :D, :]

    def body(sk, _):
        # Hyperplane projection
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

def ruiz_equilibration_jax(A: jnp.ndarray, max_iter: int = 10, eps: float = 1e-9):
    """
    JAX version of Ruiz equilibration (row and column scaling)
    A: (B, n, n)
    Returns: scaled matrix M, row scales d_r, column scales d_c
    """
    B, n, _ = A.shape
    d_r = jnp.ones((B, n, 1))
    d_c = jnp.ones((B, 1, n))
    M = A.copy()

    def body(_, state):
        M, d_r, d_c = state
        # row scaling
        row_norm = jnp.linalg.norm(M, ord=1, axis=2, keepdims=True).clip(min=eps)
        M = M / row_norm
        d_r = d_r / row_norm
        # column scaling
        col_norm = jnp.linalg.norm(M, ord=1, axis=1, keepdims=True).clip(min=eps)
        M = M / col_norm
        d_c = d_c / col_norm
        return M, d_r, d_c

    M, d_r, d_c = jax.lax.fori_loop(0, max_iter, body, (M, d_r, d_c))
    return M, d_r, d_c

# Example: JIT and partial to fix max_iter
ruiz_jit = jax.jit(partial(ruiz_equilibration_jax, max_iter=10))

# ============================================================
# JAX projection with custom_vjp (arrays only)
# ============================================================

@jax.custom_vjp
def project_jax(yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, omega, D, steps, fpi, n_iter_bwd, damping):
    """
    Fully JAX-compatible ADMM projection.
    yraw: (B, D, 1)
    scale: (B, D + m, 1)
    Aeq: (B, D + m, D + m)
    Apinv: (B, D + m, D + m)
    beq: (B, D + m, 1)
    lb, ub: (B, D + m, 1)
    """
    sk0 = jnp.zeros_like(yraw)
    sk = admm_run(
        sk0, yraw, scale,
        Aeq, Apinv, beq, lb, ub,
        sigma=sigma,
        omega=omega,
        D=D,
        steps=steps
    )
    correction = Aeq @ sk - beq
    sk = sk - Apinv @ correction
    sk_scaled = sk * scale_norm
    return sk_scaled[:, :D].squeeze(2)

def project_fwd(yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, omega, D, steps,
                fpi=True, n_iter_bwd=25, damping=0.2):
    # Forward pass: compute sk
    sk = project_jax(yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, omega, D, steps, fpi, n_iter_bwd, damping)

    # Save everything needed for backward
    return sk, (sk, yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, omega, D, steps,
                fpi, n_iter_bwd, damping)

def project_bwd(res, g):
    sk, yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, omega, D, steps, fpi, n_iter_bwd, damping = res    
    # Single ADMM step function
    def iter_fn(x):
        return admm_run(
            x, yraw, scale,
            Aeq, Apinv, beq, lb, ub,
            sigma=sigma,
            omega=omega,
            D=D,
            steps=1
        )

    sk = sk.unsqueeze(2) * scale_norm[:, :D, :]
    _, vjp_fn = jax.vjp(iter_fn, sk)
    vjp = vjp_fn(g)[0]

    # Iteration operator for implicit solve
    def iteration_vjp(v):
        _, fn = jax.vjp(iter_fn, sk)
        return fn(v)[0]

    # Fixed Point Iteration
    if fpi:
        gsol = jnp.zeros_like(vjp)
        for _ in range(n_iter_bwd):
            gsol = iteration_vjp(gsol) + vjp
    else:
        gsol = vjp
        for _ in range(n_iter_bwd):
            gsol = gsol + damping * (vjp - (gsol - iteration_vjp(gsol)))

    # Gradient w.r.t yraw using a single ADMM step
    def final_fn(y_in):
        return admm_run(
            sk, y_in, scale,
            Aeq, Apinv, beq, lb, ub,
            sigma=sigma,
            omega=omega,
            D=D,
            steps=1
        )

    _, vjp_yraw = jax.vjp(final_fn, yraw)
    grad_yraw = vjp_yraw(gsol)[0]

    # Only yraw has gradient
    return grad_yraw, None, None, None, None, None, None, None, None, None, None, None, None

# Link the forward and backward functions
project_jax.defvjp(project_fwd, project_bwd)

# ============================================================
# Torch-facing Safeguard
# ============================================================

class PinetJAXSafeguard(Safeguard):

    @jaxtyped(typechecker=beartype)
    def __init__(self,
        env: SafeEnv,
        regularisation_coefficient: float,
        n_iter_admm: int,
        n_iter_bwd: int,
        sigma: float = 1.0,
        omega: float = 1.7,
        bwd_method: str = "implicit",
        debug: bool = False,
        fpi: bool = False):

        super().__init__(env)

        self.regularisation_coefficient = regularisation_coefficient
        self.debug = debug
        self.n_iter_admm = n_iter_admm
        self.n_iter_bwd = n_iter_bwd

        self.sigma = sigma
        self.omega = omega
        self.bwd_method = bwd_method
        self.fpi = fpi
        self.log_file = f"logs/{self.bwd_method}_admm_iter_{self.n_iter_admm}_bwd_{self.n_iter_bwd}"
        jax.config.update("jax_enable_x64", True)
    
    @jaxtyped(typechecker=beartype)
    def safeguard(
        self,
        action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]
    ) -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        action: (B, D)
        A:      (B, m, D)
        b:      (B, m)
        """
        B, D = action.shape

        # ----- Build Ax ≤ b -----
        A, b = self.env.compute_A_b()
        m = A.shape[1]

        action = action.unsqueeze(2)

        self.pre_constraint_violation = torch.clamp(torch.bmm(A, action) - b.unsqueeze(2), min=0.0).squeeze(2)

        # Torch → JAX
        yraw = torch_to_jax(action)
        A_j = torch_to_jax(A)
        b_j = torch_to_jax(b).reshape(B, m, 1)

        # Build lifted system
        zero_DD = jnp.zeros((B, D, D))
        zero_Dm = jnp.zeros((B, D, m))
        negI = -jnp.eye(m)[None, :, :].repeat(B, axis=0)

        Aeq = jnp.concatenate(
            [jnp.concatenate([zero_DD, zero_Dm], axis=2),
             jnp.concatenate([A_j, negI], axis=2)],
            axis=1,
            dtype=jnp.float64
        )

        beq = jnp.zeros((B, D + m, 1), dtype=jnp.float64)

        Apinv = jnp.linalg.pinv(Aeq)

        #eq = JAXHyperplane(Aeq, Apinv, beq)

        lb = jnp.concatenate(
            [jnp.full((B, D, 1), -jnp.inf),
             jnp.full((B, m, 1), -jnp.inf)],
            axis=1,
            dtype=jnp.float64
        )
        ub = jnp.concatenate(
            [jnp.full((B, D, 1), jnp.inf),
             b_j],
            axis=1,
            dtype=jnp.float64
        )

        #box = JAXBox(lb, ub)

        # Ruiz equilibration (like PyTorch)
        _, _, d_c = ruiz_jit(Aeq)  # d_c: (B, 1, D+m) (24, 1, 11)
        scale = jnp.transpose(d_c, (0, 2, 1))  # shape: (B, D+m, 1)

        # Optionally normalize per batch to avoid very large/small numbers
        scale_max = scale.max(axis=1, keepdims=True)  # shape: (B, 1, 1)
        scale_norm = scale / scale_max                # shape: (24, 11, 1)

        sk = project_jax(
            jnp.concatenate([yraw, A_j @ yraw], axis=1),  # yraw lifted
            scale,
            scale_norm,                                    # scaling array
            Aeq=Aeq,                                      # lifted system matrix
            Apinv=Apinv,                                  # pseudo-inverse
            beq=beq,                                      # RHS
            lb=lb,                                        # lower bound
            ub=ub,                                        # upper bound
            sigma=self.sigma,
            omega=self.omega,
            D=D,
            steps=self.n_iter_admm,
            fpi=self.fpi,
            n_iter_bwd=self.n_iter_bwd,
            damping=0.2,  # or another parameter 
        )

        safe_action = sk[:, :D, 0]
        safe_action = jax_to_torch(safe_action).to(action.device)

        self.post_constraint_violation = torch.clamp(torch.bmm(A, safe_action.unsqueeze(2)) - b.unsqueeze(2), min=0.0).squeeze(2)

        return safe_action
    
    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:

        return self.regularisation_coefficient * F.mse_loss(safe_action, action)
    
    def safeguard_metrics(self):
        return  {
            "pre_ineq_violation": self.pre_constraint_violation.mean().item(),
            "post_ineq_violation": self.post_constraint_violation.mean().item(),
        }
