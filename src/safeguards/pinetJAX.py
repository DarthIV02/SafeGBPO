# ============================================================ 
# # Fully JAX-based PiNet ADMM Safeguard # 
# ============================================================ 
from safeguards.interfaces.safeguard import Safeguard, SafeEnv 
import torch 
from torch import Tensor 
import torch.nn.functional as F 
import torch.utils.dlpack 
import numpy as np 
import jax 
import jax.numpy as jnp 
from jaxtyping import Float, jaxtyped 
from beartype import beartype 
from functools import partial 
from dataclasses import dataclass 

# ============================================================ 
#  JAX utilities # 
# ============================================================ 
def torch_to_jax(x: torch.Tensor) -> jnp.ndarray: 
    return jax.dlpack.from_dlpack(x.detach()) 
    
def jax_to_torch(x: jnp.ndarray, device: torch.device) -> torch.Tensor: 
    return torch.utils.dlpack.from_dlpack(x.__dlpack__()).to(device) 
    
# ============================================================ 
# # JAX constraints # 
# ============================================================ 

@dataclass 
class JAXHyperplane: 
    A: jnp.ndarray # (B, m, n) 
    Apinv: jnp.ndarray # (B, n, m) 
    b: jnp.ndarray # (B, m, 1) 
    
    def project(self, x): 
        return x - self.Apinv @ (self.A @ x - self.b) 
        
@dataclass 
class JAXBox: 
    lb: jnp.ndarray 
    ub: jnp.ndarray 
    
    def project(self, x): 
        return jnp.clip(x, self.lb, self.ub) 
        
# ============================================================ 
# # JAX ADMM iteration 
# # ============================================================ 
@partial(jax.jit, static_argnames=("steps", "D")) 
def admm_run( sk, yraw, scale, Aeq, Apinv, beq, lb, ub, *, 
             sigma: float, omega: float, D: int, steps: int, ): 
    scale_sub = scale[:, :D, :] 
    denom = 1.0 / (1.0 + 2.0 * sigma * scale_sub**2) 
    addition = 2.0 * sigma * scale_sub * yraw[:, :D, :] 
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

def ruiz_equilibration_jax(A, max_iter=10, eps=1e-9): 
    B, n, _ = A.shape 
    d_r = jnp.ones((B, n, 1)) 
    d_c = jnp.ones((B, 1, n)) 
    M = A 
    
    def body(_, state): 
        M, d_r, d_c = state 
        row_norm = jnp.linalg.norm(M, ord=1, axis=2, keepdims=True).clip(eps) 
        M = M / row_norm 
        d_r = d_r / row_norm 
        col_norm = jnp.linalg.norm(M, ord=1, axis=1, keepdims=True).clip(eps) 
        M = M / col_norm 
        d_c = d_c / col_norm 
        return M, d_r, d_c 
    
    return jax.lax.fori_loop(0, max_iter, body, (M, d_r, d_c)) 

ruiz_jit = jax.jit(partial(ruiz_equilibration_jax, max_iter=10))

 # ============================================================ 
 # # JAX projection with custom_vjp (arrays only) 
 # ============================================================ 
 
@jax.custom_vjp 
def project_jax( yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, 
                omega, D, steps, fpi, n_iter_bwd, damping ): 
    sk0 = jnp.zeros_like(yraw) 
    sk = admm_run( sk0, yraw, scale, Aeq, Apinv, beq, lb, ub, sigma=sigma, omega=omega, D=D, steps=steps ) 
    sk = sk - Apinv @ (Aeq @ sk - beq) 
    sk = sk * scale_norm 
    return sk[:, :D, 0] 

def project_fwd(*args): 
    out = project_jax(*args) 
    return out, args 
    
def project_bwd(res, g): 
    sk, yraw, scale, scale_norm, Aeq, Apinv, beq, lb, ub, sigma, omega, D, steps, fpi, n_iter_bwd, damping = res 
    # Single ADMM step function 
    def iter_fn(x): 
        return admm_run( x, yraw, scale, Aeq, Apinv, beq, lb, ub, sigma=sigma, omega=omega, D=D, steps=1 ) 
    
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
        return admm_run( sk, y_in, scale, Aeq, Apinv, beq, lb, ub, sigma=sigma, omega=omega, D=D, steps=1 )
    
    _, vjp_yraw = jax.vjp(final_fn, yraw) 
    
    grad_yraw = vjp_yraw(gsol)[0]
    # Only yraw has gradient 
    return grad_yraw, None, None, None, None, None, None, None, None, None, None, None, None 

# Link the forward and backward functions 
project_jax.defvjp(project_fwd, project_bwd) 

# ============================================================ 
# # Torch-facing Safeguard 
# # ============================================================ 

class PinetJAXSafeguard(Safeguard): 
    @jaxtyped(typechecker=beartype) 
    def __init__( self, env: SafeEnv, 
                 regularisation_coefficient: float, 
                 n_iter_admm: int, n_iter_bwd: int, 
                 sigma: float = 1.0, omega: float = 1.7, 
                 fpi: bool = True, **kwargs ): 
                 
        super().__init__(env) 
        self.regularisation_coefficient = regularisation_coefficient 
        self.n_iter_admm = n_iter_admm 
        self.n_iter_bwd = n_iter_bwd 
        self.sigma = sigma 
        self.omega = omega 
        self.fpi = fpi 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        jax.config.update("jax_enable_x64", False) # Check how much it differs... 
    
    def safeguard(self, action: Tensor) -> Tensor: 
        B, D = action.shape 
        A, b = self.env.compute_A_b() 
        m = A.shape[1] 
        action = action.unsqueeze(2) 
        self.pre_constraint_violation = torch.clamp(torch.bmm(A, action) - b.unsqueeze(2), min=0.0).squeeze(2) 
        
        yraw = torch_to_jax(action) 
        A_j = torch_to_jax(A) 
        b_j = torch_to_jax(b).reshape(B, m, 1) 
        
        # Build once and cache 
        Aeq = jnp.zeros((B, D+m, D+m)) 
        Aeq = Aeq.at[:, D:, :D].set(A_j) 
        Aeq = Aeq.at[:, D:, D:].set(-jnp.eye(m)) 
        beq = jnp.zeros((B, D + m, 1)) 
        Apinv = jnp.linalg.pinv(Aeq) 
        lb = jnp.full((B, D + m, 1), -jnp.inf) 
        ub = jnp.concatenate([jnp.full((B, D, 1), jnp.inf), b_j], axis=1) 
        _, _, d_c = ruiz_jit(Aeq) 
        scale = d_c.transpose(0, 2, 1) 
        scale_norm = scale / scale.max(axis=1, keepdims=True) 
        
        out = ProjectJAXFunction.apply( 
            jnp.concatenate([yraw, A_j @ yraw], axis=1), # yraw lifted, 
            self.device, scale, scale_norm, Aeq, Apinv, 
            beq, lb, ub, self.sigma, self.omega, D,
            self.n_iter_admm, self.fpi, self.n_iter_bwd, 0.2 ) 
        
        safe_action = out[:, :D] 
        safe_action = jax_to_torch(safe_action, action.device).requires_grad_(True) 
        self.post_constraint_violation = torch.clamp(torch.bmm(A, safe_action.unsqueeze(2).to(torch.double)) - b.unsqueeze(2), min=0.0).squeeze(2) 
        return safe_action 
    
    def safe_guard_loss(self, action, safe_action): 
        return self.regularisation_coefficient * F.mse_loss(safe_action, action) 
    
    def safeguard_metrics(self):
         return { "pre_ineq_violation": self.pre_constraint_violation.mean().item(), 
                 "post_ineq_violation": self.post_constraint_violation.mean().item(), } 
         
class ProjectJAXFunction(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx, yraw, device, *params): 
        out_jax = project_jax(yraw, *params) 
        # forward uses custom VJP 
        D = out_jax.shape[1] 
        ctx.save_for_backward(yraw[:, :D]) 
        ctx.device = device 
        ctx.params = params 
        return jax_to_torch(out_jax, device) 
    
    @staticmethod 
    def backward(ctx, grad_out): 
        (y_jax, ) = ctx.saved_tensors 
        device = ctx.device 
        params = ctx.params 
        g_jax = torch_to_jax(grad_out) 
        
        # Use the custom VJP you defined with 
        project_jax.defvjp(project_fwd, project_bwd) 
        _, vjp_fn = jax.vjp(lambda y: project_jax(y, *params), y_jax) 
        grad_y_jax = vjp_fn(g_jax)[0] 
        return jax_to_torch(grad_y_jax, device), *([None] * len(params))