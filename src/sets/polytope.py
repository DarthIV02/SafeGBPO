from typing import Self
import cvxpy as cp
import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from sets.interface.convex_set import ConvexSet


class HPolytope(ConvexSet):
    """
    Batched H-Polytope representation defined as:
        P_i = { x | A_i x <= b_i } for i in [0, batch_dim)

    Attributes:
        A: Tensor of shape [batch_dim, num_constraints, dim]
        b: Tensor of shape [batch_dim, num_constraints]
    """
    A: Tensor
    b: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        A: Float[Tensor, "batch_dim num_constraints dim"],
        b: Float[Tensor, "batch_dim num_constraints"],
    ) -> None:
        super().__init__(batch_dim=A.shape[0], dim=A.shape[2])
        if A.ndim != 3 or b.ndim != 2:
            raise ValueError("A must be [B, C, D] and b must be [B, C].")
        if A.shape[1] != b.shape[1]:
            raise ValueError("A and b must have same num_constraints.")
        self.A = A
        self.b = b
        self.batch_dim = A.shape[0]
        self.num_constraints = A.shape[1]
        self.dim = A.shape[2]
        self._centers: list[Tensor | None] = [None] * self.batch_dim

    def __iter__(self):
        return iter((self.A, self.b))

    def __getitem__(self, idx) -> Self:
        """Return a single HPolytope instance or a batched subset."""
        if isinstance(idx, int):
            return HPolytope(self.A[idx].unsqueeze(0), self.b[idx].unsqueeze(0))
        elif isinstance(idx, Tensor) or isinstance(idx, slice):
            return HPolytope(self.A[idx], self.b[idx])
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

    @staticmethod
    def from_unit_box(batch_dim: int, dim: int) -> Self:
        """Create batched unit boxes [-1, 1]^dim."""
        eye = torch.eye(dim)
        A_single = torch.cat([eye, -eye], dim=0)               # [2*dim, dim]
        b_single = torch.ones(2 * dim)                         # [2*dim]
        A = A_single.unsqueeze(0).repeat(batch_dim, 1, 1)
        b = b_single.unsqueeze(0).repeat(batch_dim, 1)
        return HPolytope(A, b)

    @jaxtyped(typechecker=beartype)
    def contains_point(
        self, points: Float[Tensor, "batch_dim dim"]
    ) -> Bool[Tensor, "batch_dim"]:
        """Check if each batched point lies inside its corresponding polytope."""
        if points.shape != (self.batch_dim, self.dim):
            raise ValueError(f"points must be [{self.batch_dim}, {self.dim}]")

        # Compute A_i @ x_i <= b_i for each batch element
        Ax = torch.einsum("bij,bj->bi", self.A, points)
        return (Ax <= self.b + 1e-8).all(dim=1)

    def contains(self, points: Float[Tensor, "batch_dim dim"]) -> Bool[Tensor, "batch_dim"]:
        """Alias for contains_point()."""
        return self.contains_point(points)

    def vertices(self) -> list[Float[Tensor, "num_vert dim"]]:
        """Compute vertices for each polytope using pypoman."""
        import pypoman
        verts_all = []
        for i in range(self.batch_dim):
            A_np = self.A[i].detach().cpu().numpy()
            b_np = self.b[i].detach().cpu().numpy()
            verts_np = np.array(pypoman.compute_polytope_vertices(A_np, b_np))
            verts_all.append(torch.tensor(verts_np, dtype=self.A.dtype, device=self.A.device))
        return verts_all

    def center(self, idx: int | None = None) -> Float[Tensor, "dim"]:
        """Compute or return cached approximate center of a given batch polytope."""
        if idx is not None:
            if self._centers[idx] is None:
                verts = self.vertices()[idx]
                self._centers[idx] = verts.mean(dim=0)
            return self._centers[idx]
        else:
            # Compute all centers
            return torch.stack([self.center(i) for i in range(self.batch_dim)], dim=0)

    def draw(self, ax=None, batch_idx: int = 0, color="blue", **kwargs):
        """Draw a specific 2D polytope in the batch."""
        if self.dim != 2:
            raise NotImplementedError("Can only draw 2D polytopes.")
        if ax is None:
            ax = plt.gca()
        verts = self.vertices()[batch_idx].cpu().numpy()
        polygon = Polygon(verts, closed=True, fill=False, edgecolor=color, **kwargs)
        ax.add_patch(polygon)
        ax.autoscale()
        return ax

    @jaxtyped(typechecker=beartype)
    def boundary_point(
        self, directions: Float[Tensor, "batch_dim dim"]
    ) -> Float[Tensor, "batch_dim dim"]:
        """
        Compute a boundary point along given directions for each batch.
        Solves: max l s.t. A (center + l * dir) <= b
        """
        results = []
        for i in range(self.batch_dim):
            A_np = self.A[i].cpu().numpy()
            b_np = self.b[i].cpu().numpy()
            dir_np = directions[i].cpu().numpy()
            center_np = self.center(i).cpu().numpy()

            x = cp.Variable(self.dim)
            l = cp.Variable()
            constraints = [A_np @ x <= b_np, x == center_np + l * dir_np]
            prob = cp.Problem(cp.Maximize(l), constraints)
            prob.solve(solver=cp.ECOS, verbose=False)

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise RuntimeError(f"LP failed for batch {i}: {prob.status}")

            boundary_np = center_np + l.value * dir_np
            results.append(torch.tensor(boundary_np, dtype=self.A.dtype, device=self.A.device))

        return torch.stack(results, dim=0)

    @jaxtyped(typechecker=beartype)
    def sample(self) -> Float[torch.Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample a point approximately uniformly from the polytope Ax <= b.
        """
        batch, num_constraints, dim = self.A.shape

        # Start at center
        x = self.center()

        # Random direction per batch
        dir_vec = torch.randn(batch, dim, device=device)
        dir_vec = dir_vec / (torch.norm(dir_vec, dim=1, keepdim=True) + 1e-8)

        # Compute intersection bounds t_lower <= t <= t_upper
        Ad = torch.einsum("bij, bj -> bi", self.A, dir_vec)          # (batch, num_constraints)
        Ax = torch.einsum("bij, bj -> bi", self.A, x)                # (batch, num_constraints)
        c = self.b - Ax

        t_lower = torch.full((batch,), -float("inf"), device=device)
        t_upper = torch.full((batch,), float("inf"), device=device)

        # For positive directions: A_i * dir > 0 → t ≤ (b_i - A_i x) / (A_i dir)
        pos_mask = Ad > 1e-8
        neg_mask = Ad < -1e-8

        t_upper[pos_mask.any(dim=1)] = torch.min(
            torch.where(pos_mask, c / (Ad + 1e-8), torch.full_like(c, float("inf"))),
            dim=1
        ).values

        # For negative directions: A_i * dir < 0 → t ≥ (b_i - A_i x) / (A_i dir)
        t_lower[neg_mask.any(dim=1)] = torch.max(
            torch.where(neg_mask, c / (Ad + 1e-8), torch.full_like(c, -float("inf"))),
            dim=1
        ).values

        # Sample t uniformly within [t_lower, t_upper]
        t = torch.rand(batch, device=device) * (t_upper - t_lower) + t_lower

        # New sampled point
        sample = x + t.unsqueeze(1) * dir_vec
        return sample

    # WARNING: Should be double checked
    @jaxtyped(typechecker=beartype)
    def intersects(self, other: ConvexSet) -> Bool[Tensor, "{self.batch_dim}"]:
        """
        Check if another convex set intersects with the zonotope.

        Args:
            other: The convex set to check for intersection.

        Returns:
            True if other intersects with the zonotope, False otherwise.
        """
        import sets as sets

        if isinstance(other, sets.Ball):
            return other.intersects(self)
        elif isinstance(other, sets.Box):
            return other.intersects(self)
        elif isinstance(other, sets.Capsule):
            return other.intersects(self)
        elif isinstance(other, sets.Zonotope):
            # Overapproximation 
            return other.box().intersects(self)
        elif isinstance(other, sets.Polytope):
            batch, dim = self.center.shape # p1
            epsilon = 1e-8

            # 1. Ray directions: from P1 center to P2 center
            ray_dir = other.center - self.center               # (batch, dim)
            ray_norm = torch.norm(ray_dir, dim=1, keepdim=True)
            ray_unit = ray_dir / (ray_norm + 1e-8)       # normalized direction

            # 2. Compute intersection bounds for each polytope along the ray
            t1_lower, t1_upper = ray_hyperplane_intersections_parallel(
                c=self.center,
                d=ray_unit,
                A=self.A,
                b=self.b,
                epsilon=epsilon
            )

            t2_lower, t2_upper = ray_hyperplane_intersections_parallel(
                c=other.center,
                d=-ray_unit,    # opposite direction to compute interval from P2 center
                A=other.A,
                b=other.b,
                epsilon=epsilon
            )

            # Shift P2 intervals to the ray originating at P1 center
            t2_lower_shifted = -t2_upper
            t2_upper_shifted = -t2_lower

            # 3. Check interval overlap
            intersects = (t1_upper >= t2_lower_shifted) & (t2_upper_shifted >= t1_lower)
            
            return intersects
        
        else:
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")

    def ray_hyperplane_intersections_parallel(
        c: torch.Tensor,  # (batch, dim)
        d: torch.Tensor,  # (batch, dim)
        A: torch.Tensor,  # (num_hyperplanes, dim)
        b: torch.Tensor,  # (num_hyperplanes,)
        epsilon: float = 1e-8
    ):
        """
        Compute lower and upper t values along ray directions for multiple hyperplanes in parallel.
        """
        batch = c.shape[0]
        num_planes = A.shape[0]

        # Compute dot products: (batch, num_planes)
        denom = torch.matmul(d, A.T)  # d_i · a_j
        numer = b.unsqueeze(0) - torch.matmul(c, A.T)  # b_j - a_j · c_i

        # Avoid division by zero
        parallel_mask = torch.abs(denom) < epsilon
        denom_safe = denom.clone()
        denom_safe[parallel_mask] = 1.0  # temporarily avoid div by zero
        t = numer / denom_safe

        # Handle rays parallel to hyperplanes
        t[parallel_mask & (numer < 0)] = -float("inf")  # ray outside → no intersection
        t[parallel_mask & (numer >= 0)] = float("inf")  # ray inside → no constraint

        # Compute lower and upper bounds per batch
        t_lower = torch.max(torch.where(denom < -epsilon, t, torch.full_like(t, -float("inf"))), dim=1).values
        t_upper = torch.min(torch.where(denom > epsilon, t, torch.full_like(t, float("inf"))), dim=1).values

        return t_lower, t_upper