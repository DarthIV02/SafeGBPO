import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

# --- Feasibility Seeking Logic ---
@torch.enable_grad()
def fs_unroll(a0, phi_fn, K, Kp, eta):
    """
    Feasibility Seeking Step using Gradient Descent on Violation Function.
    """
    is_training = a0.requires_grad

    if is_training:
        a = a0 
    else:
        a = a0.detach().requires_grad_(True)

    for _ in range(min(K, Kp)):
        phi = phi_fn(a).mean()
        
        grad_a, = torch.autograd.grad(phi, a, create_graph=is_training) 
        
        a = a - eta * grad_a

    
    a_diff = a 
    a_nd = a.detach()

    for _ in range(Kp, K):
        a_nd.requires_grad_(True)
        
        phi_batch = phi_fn(a_nd)
        
        if phi_batch.max() < 1e-4:
            break
            
        phi = phi_batch.mean()
        
        grad_a, = torch.autograd.grad(phi, a_nd, create_graph=False)
        
        with torch.no_grad():
            a_nd = a_nd - eta * grad_a

    a_hat = a_diff + (a_nd - a_diff).detach()
    
    return a_hat

    
class FSNetTestSafeguard(Safeguard):
    """
    Safeguard that uses Feasibility-Seeking Neural Network (FSNet) logic.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: SafeEnv,
                 regularisation_coefficient: float,
                 fs_k: int = 10,       
                 fs_kp: int = 5,       
                 fs_eta: float = 0.1,  
                 # Dummy args for compatibility
                 eq_pen_coefficient: float = 1.0,
                 ineq_pen_coefficient: float = 1.0,
                 val_tol: float = 1e-4,
                 memory_size: int = 10,
                 maxmax_iter_iter: int = 50,
                 max_diff_iter: int = 10,
                 scale: float = 1.0,
                 **kwargs):
        super().__init__(env)
        self.regularisation_coefficient = regularisation_coefficient
        self.fs_k = fs_k
        self.fs_kp = fs_kp
        self.fs_eta = fs_eta

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        
        safe_set_zonotope = self.env.safe_action_set()

        def phi_fn(a: Tensor) -> Tensor:
            return safe_set_zonotope.validation(a, p=2)

        safe_action = fs_unroll(
            a0=action,
            phi_fn=phi_fn,
            K=self.fs_k,
            Kp=self.fs_kp,
            eta=self.fs_eta
        )

        return safe_action

    @jaxtyped(typechecker=beartype)
    def safe_guard_loss(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"],
                        safe_action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) -> Tensor:
        """
        FSNet regularization term: rho/2 * ||y - y_hat||^2
        """
        return self.regularisation_coefficient * 0.5 * torch.nn.functional.mse_loss(safe_action, action)