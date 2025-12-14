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
    regularisation_coefficient: float = 0.1
    eq_pen_coefficient: float = 0.025
    ineq_pen_coefficient: float = 0.025

    # generic solver
    val_tol: float = 1e-4
    grad_tol: float = 1e-4
    max_iter: int = 3
    max_diff_iter : int = 2
    # lbfgs specific
    # memory: int = 10
    # max_ls_iter: int = 5
     # scale : float = 1.0
    
    # lm specific
    damping_init: float = 1e-3
    damping_up: float = 10.0
    damping_down: float = 0.1

@dataclass
class PinetConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    n_iter_admm: int = 10
    n_iter_bwd: int = 10
    bwd_method: str = "implicit"  # "implicit" or "unroll"
    fpi: bool = False