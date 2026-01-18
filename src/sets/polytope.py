from typing import Self
import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog

from sets.interface.convex_set import ConvexSet
from scipy.spatial import HalfspaceIntersection

class Polytope(ConvexSet):
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

        self.C, self.d = None, None

        self.batch_dim = A.shape[0]
        self.num_constraints = A.shape[1]
        self.dim = A.shape[2]
        self.device = A.device
        self._centers: list[Tensor | None] = [None] * self.batch_dim

    def __iter__(self):
        return iter((self.A, self.b))

    def __getitem__(self, idx) -> Self:
        """Return a single HPolytope instance or a batched subset."""
        if isinstance(idx, int):
            return Polytope(self.A[idx].unsqueeze(0), self.b[idx].unsqueeze(0))
        elif isinstance(idx, Tensor) or isinstance(idx, slice):
            return Polytope(self.A[idx], self.b[idx])
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

    def vertices(self) -> np.array:
        A = np.asarray(self.A[0].cpu())
        b = np.asarray(self.b[0].cpu())
        n = self.dim

        # Variables: x (n dims) and t (slack)
        c = np.zeros(n + 1)
        c[-1] = -1  # maximize t -> minimize -t

        b_ub = b[np.isfinite(b)]
        unpadded_shape = b_ub.shape[0]

        A_ub = np.hstack([A[:unpadded_shape], np.ones((unpadded_shape, 1))])     

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * n + [(0, None)])
        if not res.success or res.x[-1] <= 1e-9:
            return np.empty((0, n))  # or return boundary-only

        interior_point = res.x[:-1]
        halfspaces = np.hstack([A, -b.reshape(-1, 1)])

        try:
            hs = HalfspaceIntersection(halfspaces, interior_point)
            V = hs.intersections
            V = np.reshape(V, (len(V), n))
        except Exception:
            # In rare cases QHull fails for degenerate polytope
            V = np.empty((0, n))

        return self.order_vertices_ccw(V).T
    
    def order_vertices_ccw(self, points):
        """Order them clockwise -> prevent bowtie shape"""
        transposed = False
        if points.shape[0] == 2:
            points = points.T
            transposed = True

        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1],
                            points[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = points[order]

        return ordered.T if transposed else ordered

    def center(self, idx: int | None = None) -> Float[Tensor, "batch_dim dim"]:
        """Computation of the Chebyshev center of an HPolyhedron HP via linear programming.
        Defined as the center with the ball of largest radius contained in HP.

        Raises:
            EmptySetError: Set is empty.
            UnboundedSetError: Set is unbounded. LP could not converge.

        Returns:
            np.ndarray: Chebyshev center.
        """

        B, num_constraints, dim = self.A.shape
        
        if idx is not None:
            if self._centers[idx] is not None:
                return self._centers[idx]

            # objective function
            c = np.hstack((-1, np.zeros(dim)))

            mask = torch.isfinite(self.b[idx])
            A_i = self.A[idx, mask]
            b_i = self.b[idx, mask] 

            if mask.sum() == 0:
                raise ValueError("No finite constraints — Chebyshev center undefined")
            b_i = torch.clamp(b_i, min=-1e8)

            b_ub = np.hstack((0, b_i.cpu().detach()))

            norms = torch.linalg.norm(A_i, dim=1)
            norms_np = norms.detach().cpu().numpy()
            if torch.any((norms == 0) & (b_i < 0)):
                raise ValueError("Zero row in A with negative b")

            A_ub = np.vstack(
                (
                    np.hstack((-1, np.zeros(dim))),
                    np.hstack(
                        (
                            np.reshape(
                                norms_np, (sum(mask), 1)
                            ),
                            A_i.cpu().detach(),
                        )
                    ),
                )
            )
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] + [(None, None)] * dim)

            if res.status == 2:  # infeasible -> empty
                self._centers[idx] = torch.zeros(dim, device=self.device)
            elif res.status == 3:  # unbounded
                self._centers[idx] = torch.zeros(dim, device=self.device)
            else:
                self._centers[idx] = torch.Tensor(res.x[1:]).to(self.device)
            
            return self._centers[idx]
        
        else:
            update = []
            for i in range(self.batch_dim):
                try:
                    update.append(self.center(i))
                except Exception:
                    update.append(torch.zeros(dim, device=self.device))  # How to resolve this issue?
            return torch.stack(update, dim=0)

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
    def sample(self) -> Float[torch.Tensor, "{self.batch_dim} {self.dim}"]:
        """
        Sample a point approximately uniformly from the polytope Ax <= b.
        """
        raise NotImplementedError(
                f"Sample method not implemented for Polytope")

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
             raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")
        else:
            raise NotImplementedError(
                f"Intersection check not implemented for {type(other)}")
    
    def setup_constraints(self):
        """
        Setup the equality and inequality residual functions for FSNet.
        The equality constraints are Cy = d which are trivially satisfied for H-Polytopes. C=d= None
        The inequality constraints are Ay <= b.
        Setup detaches A and b from the computation graph for faster computation during FSNet safeguarding
        """
        self.A = self.A.detach()
        self.b = self.b.detach()
        self.C = torch.zeros((self.batch_dim, 1, self.dim), device=self.A.device) 
        self.d = torch.zeros((self.batch_dim, 1), device=self.A.device)