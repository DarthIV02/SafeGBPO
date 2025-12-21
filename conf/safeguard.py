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

    # FSNet genera solver config parameters
    memory: int = 10
    max_iter: int = 10
    max_diff_iter: int = 5

    # fsnet lbfgs torch opt solver config parameters
    lr: float = 1.0
    max_norm: float = 2.0

    # fsnet lbfgs original solver config parameters
    val_tol: float = 1e-6
    grad_tol: float = 1e-6
    scale: float = 1.0
    c: float = 1e-4
    rho_ls: float = 0.5
    max_ls_iter: int = 10
    verbose: bool = False


@dataclass
class PinetConfig(SafeguardConfig):
    regularisation_coefficient: float = 0.1
    n_iter_admm: int = 10
    n_iter_bwd: int = 10
    bwd_method: str = "implicit"  # "implicit" or "unroll"
    fpi: bool = False