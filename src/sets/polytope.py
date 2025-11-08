from typing import Self
import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
from sets.interface.convex_set import ConvexSet


class HPolytope(ConvexSet):
    """
    H-Polytope representation of a convex set defined by linear inequalities.

    Defined as:
        P = { x | A x <= b }

    Attributes:
        A: Matrix defining the halfspace constraints.
        b: Vector defining the bounds of the halfspaces.
    """

    A: Tensor
    b: Tensor

    @jaxtyped(typechecker=beartype)
    def __init__(self, 
                 A: Float[Tensor, "num_constraints dim"], 
                 b: Float[Tensor, "num_constraints"]) -> None:
        """
        Initialize an H-polytope with constraint matrix A and vector b.
        """
        super().__init__(dim=A.shape[1])
        if A.ndim != 2 or b.ndim != 1:
            raise ValueError("A must be 2D and b must be 1D tensors.")
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b must have the same number of rows.")
        self.A = A
        self.b = b
        self._center: Tensor | None = None

    def __iter__(self):
        return iter((self.A, self.b))

    def __getitem__(self, item) -> Self:
        return HPolytope(self.A[item], self.b[item])

    @staticmethod
    def from_unit_box(dim: int) -> Self:
        """Create a unit box polytope: [-1, 1]^dim."""
        eye = torch.eye(dim)
        A = torch.cat([eye, -eye], dim=0)
        b = torch.ones(2 * dim)
        return HPolytope(A, b)

    @jaxtyped(typechecker=beartype)
    def contains_point(self, point: Float[Tensor, "{self.dim}"]) -> Bool[Tensor, ""]:
        """
        Check if a single point is inside the polytope.
        
        Equivalent to the procedural:
            polytope_contains(polytope, point)
        
        Args:
            point: A 1D tensor of shape (dim,).
        
        Returns:
            True if A x <= b, False otherwise.
        """
        if point.ndim != 1 or point.shape[0] != self.dim:
            raise ValueError(f"Expected point of shape ({self.dim},), got {tuple(point.shape)}.")
        Ax = self.A @ point
        return torch.all(Ax <= self.b)

    @jaxtyped(typechecker=beartype)
    def contains(self, x: Float[Tensor, "{self.dim}"]) -> Bool[Tensor, ""]:
        """Alias for contains_point(), for compatibility."""
        return self.contains_point(x)

    @jaxtyped(typechecker=beartype)
    def vertices(self) -> Float[Tensor, "num_vert dim"]:
        """Compute the vertices of the H-polytope using pypoman."""
        import pypoman
        A_np = self.A.detach().cpu().numpy()
        b_np = self.b.detach().cpu().numpy()
        verts = np.array(pypoman.compute_polytope_vertices(A_np, b_np))
        return torch.tensor(verts, dtype=torch.float32)

    @jaxtyped(typechecker=beartype)
    def center(self) -> Float[Tensor, "{self.dim}"]:
        """Compute or return cached center (approximate if necessary)."""
        if self._center is not None:
            return self._center
        center = torch.mean(self.vertices(), dim=0)
        self._center = center
        return center

    @jaxtyped(typechecker=beartype)
    def draw(self, ax=None, color="blue", **kwargs):
        """Draw the polytope in 2D (only works for dim=2)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        if self.dim != 2:
            raise NotImplementedError("Can only draw 2D polytopes.")

        if ax is None:
            ax = plt.gca()

        verts = self.vertices().numpy()
        polygon = Polygon(verts, closed=True, fill=False, edgecolor=color, **kwargs)
        ax.add_patch(polygon)
        ax.autoscale()
        return ax

    @jaxtyped(typechecker=beartype)
    def boundary_point(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Compute a boundary point along the given direction using CVXPY.
        """
        direction_np = direction.detach().cpu().numpy()
        center_np = self.center().detach().cpu().numpy()

        A_np = self.A.detach().cpu().numpy()
        b_np = (self.b - A_np @ center_np).detach().cpu().numpy()  # zero-centered

        n = self.dim

        x = cp.Variable(n)
        l = cp.Variable()

        objective = cp.Maximize(l)  # maximize l is equivalent to minimize -l

        constraints = [
            A_np @ x <= b_np,
            -x + l * direction_np == 0
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"CVXPY failed to solve LP: {problem.status}")

        boundary_np = center_np + l.value * direction_np
        boundary_tensor = torch.tensor(boundary_np, dtype=self.A.dtype, device=self.A.device)
        return boundary_tensor