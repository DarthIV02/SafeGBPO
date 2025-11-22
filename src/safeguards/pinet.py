import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype
from dataclasses import dataclass

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

@dataclass
class HyperplaneConstraint:
    A: Tensor  # (B, m, n)
    b: Tensor  # (B, m, 1)

    def project(self, y: Tensor) -> Tensor:
        """
        Projection onto { x | A x = b }.
        Uses least-squares formula:
            x_proj = x - A^T (A A^T)^{-1} (A x - b)
        """
        A, b = self.A, self.b
        Ax = A @ y
        res = Ax - b
        # Solve (A A^T)^{-1} A res
        lhs = A @ A.transpose(1, 2)  # (B, m, m)
        solve = torch.linalg.solve(lhs, res)  # (B, m, 1)
        correction = A.transpose(1, 2) @ solve
        return y - correction


@dataclass
class BoxConstraint:
    lb: Tensor  # (B, n, 1)
    ub: Tensor  # (B, n, 1)

    def project(self, y: Tensor) -> Tensor:
        return torch.max(torch.min(y, self.ub), self.lb)


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
        sigma: float = 1.0,
        omega: float = 1.7,
        **kwargs
    ):
        super().__init__(env)

        self.regularisation_coefficient = regularisation_coefficient
        self.n_iter_admm = n_iter_admm

        self.sigma = sigma
        self.omega = omega

    @jaxtyped(typechecker=beartype)
    def safeguard(
        self,
        action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]
    ) -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:

        B = self.batch_dim
        D = self.action_dim
        action = action.unsqueeze(2)  # match (B, D, 1)

        # ----- Build Ax ≤ b -----
        A, b = self.env.compute_A_b()
        m = A.shape[1]

        # ----- Build equality + box constraints in lifted dim -----
        hyperplane, box, scale = self._build_constraints(A, b)

        # ----- Elevate action into y = [x; Ax] -----
        y_raw = self._elevate(action, A)

        # ----- Run ADMM -----
        y = self._run_admm(y_raw, hyperplane, box)

        # ----- Extract first D coords then unscale -----
        y_proj = hyperplane.project(y)[:, :D, :]
        y_proj = y_proj / scale[:, :D, :]  # undo Ruiz scaling

        return y_proj.squeeze(2)

    # ----------------------------------------------------------------------
    # Build equality + box constraint
    # ----------------------------------------------------------------------
    def _build_constraints(self, A: Tensor, b: Tensor):
        B, m, D = A.shape
        total_dim = D + m

        # Equality constraint: [0 0; A -I] y = 0
        E = torch.zeros(B, D, D, device=A.device, dtype=A.dtype)
        Z = torch.zeros(B, D, m, device=A.device, dtype=A.dtype)
        negI = -torch.eye(m, device=A.device, dtype=A.dtype).expand(B, m, m)

        top = torch.cat([E, Z], dim=2)
        bottom = torch.cat([A, negI], dim=2)
        Aeq = torch.cat([top, bottom], dim=1)

        beq = torch.zeros(B, total_dim, 1, device=A.device, dtype=A.dtype)

        # Ruiz scaling
        _, _, d_c = self._ruiz(Aeq)
        scale = d_c.transpose(1, 2)  # (B, total_dim, 1)

        # Box constraint: x free, Ax ≤ b
        lb = torch.full((B, total_dim, 1), -torch.inf, device=A.device, dtype=A.dtype)
        ub = torch.cat([
            torch.full((B, D, 1),  torch.inf, device=A.device, dtype=A.dtype),
            b.unsqueeze(2)
        ], dim=1)

        return (
            HyperplaneConstraint(Aeq, beq),
            BoxConstraint(lb, ub),
            scale
        )

    # ----------------------------------------------------------------------
    # Elevation
    # ----------------------------------------------------------------------
    def _elevate(self, x: Tensor, A: Tensor) -> Tensor:
        Ax = A @ x
        return torch.cat([x, Ax], dim=1)

    # ----------------------------------------------------------------------
    # ADMM loop
    # ----------------------------------------------------------------------
    def _run_admm(self, y_raw, eq, box):
        B, N, _ = y_raw.shape
        sk = torch.zeros_like(y_raw)

        sigma = torch.tensor(self.sigma, device=y_raw.device, dtype=y_raw.dtype)
        omega = torch.tensor(self.omega, device=y_raw.device, dtype=y_raw.dtype)

        D = self.action_dim

        for _ in range(self.n_iter_admm):
            zk = eq.project(sk)
            reflect = 2 * zk - sk

            scale = self.scale_t[:, :D, :]

            first = (
                2 * sigma * scale * y_raw[:, :D, :] + reflect[:, :D, :]
            ) / (1 + 2 * sigma * scale**2)

            second = reflect[:, D:, :]
            to_box = torch.cat([first, second], dim=1)

            tk = box.project(to_box)
            sk = sk + omega * (tk - zk)

        return sk

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
            d_r /= row_norm
            M /= row_norm

            col_norm = torch.norm(M, p=1, dim=1, keepdim=True).clamp_min(eps)
            d_c /= col_norm
            M /= col_norm

        return M, d_r, d_c