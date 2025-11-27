from dataclasses import dataclass


@dataclass
class SafeguardConfig:
    regularisation_coefficient: float

    @property
    def name(self) -> str:
        return self.__class__.__name__[:-6]


@dataclass
class BoundaryProjectionConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1


@dataclass
class RayMaskConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    linear_projection: bool = True
    zonotopic_approximation: bool = False
    polytopic_approximation: bool = True
    passthrough: bool = False

@dataclass
class FSNetConfig(SafeguardConfig): #TODO: fill default valueS
    regularisation_coefficient: float = 0.0
    eq_pen_coefficient: float = 1.0
    ineq_pen_coefficient: float = 1.0
    # val_tol: float = 1e-6,
    # memory_size: int = 10,
    # maxmax_iter_iter: int = 50,
    # max_diff_iter = 10,
    # scale : float = 1.0,

@dataclass
class PinetConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    n_iter_admm: int = 10
    n_iter_bwd: int = 10