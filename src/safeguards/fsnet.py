import cvxpy as cp
from torch import Tensor
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, jaxtyped
from typing import Dict, Tuple, Callable, Optional
import torch
from time import time

from safeguards.interfaces.safeguard import Safeguard, SafeEnv

from safeguards.fsnet_solvers.lbfgs import hybrid_lbfgs_solve, nondiff_lbfgs_solve
from safeguards.fsnet_solvers.anderson import hybrid_anderson_solve, nondiff_anderson_solve
from safeguards.fsnet_solvers.lm import  hybrid_lm_solve, nondiff_lm_solve, LMSolverConfig, batch_lm_solve

class FSNetSafeguard(Safeguard):
    """
    Projecting an unsafe action to the closest safe action.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, 
                 env: SafeEnv, 
                 regularisation_coefficient: float,
                 eq_pen_coefficient: float,
                 ineq_pen_coefficient: float,
                 **kwargs):
        Safeguard.__init__(self, env)

        self.boundary_layer = None
        self.regularisation_coefficient = regularisation_coefficient
        self.eq_pen_coefficient = eq_pen_coefficient
        self.ineq_pen_coefficient = ineq_pen_coefficient 

        self.regularisation_coefficient = regularisation_coefficient
        self.eq_pen_coefficient = eq_pen_coefficient
        self.ineq_pen_coefficient = ineq_pen_coefficient 
        self.config_method = kwargs

        if self.env.polytope:
            self.data = PolytopeData(env)
        else:
            self.data = ZonotopeData(env)

        if torch.get_default_device() == 'cpu':
            self.solver = hybrid_lbfgs_solve
            self.nondiff_solver = nondiff_lbfgs_solve
        else:
            self.solver =     hybrid_lm_solve
            self.nondiff_solver =   nondiff_lm_solve

        self.debug_mode = False
        self.last_trajectory = None
        self.last_unsafe_action = None

    @jaxtyped(typechecker=beartype)
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Safeguard the action to ensure safety.

        Args:
            action: The action to safeguard.

        Returns:
            The safeguarded action.
        """
        self.data.setup_resid()
        processed_action = self.data.pre_process_action(action)
        self.pre_eq_violation = self.data.eq_resid(None, processed_action).square().sum(dim=1)
        self.pre_ineq_violation = self.data.ineq_resid(None, processed_action).square().sum(dim=1)
        
        # Use non-differentiable solver during evaluation (no grad mode),
        # hybrid solver during training (with grad)
        start_time = time()

        if torch.is_grad_enabled():
            # Training phase: usually no debug needed, keep it fast
            safe_action = self.solver(
                None,
                processed_action,
                self.data,
                **self.config_method
            )
        else:
            # Eval phase: enable debug if requested
            with torch.enable_grad():
                result = self.nondiff_solver(
                    None,
                    processed_action,
                    self.data,
                    debug_trajectory=self.debug_mode, # Pass flag
                    **self.config_method
                )
            
            if self.debug_mode and isinstance(result, tuple):
                safe_action, trajectory = result
                self.last_trajectory = trajectory # Store trajectory list
                self.last_unsafe_action = action.detach().cpu()
            else:
                safe_action = result
        #print(f"FSNet safeguard time: {time() - start_time:.4f} seconds")
    

        self.post_eq_violation = self.data.eq_resid(None, safe_action).square().sum(dim=1)
        self.post_ineq_violation = self.data.ineq_resid(None, safe_action).square().sum(dim=1)

        safe_action = self.data.post_process_action(safe_action)
        return safe_action

    def safe_guard_loss(self, action: Float[Tensor, "{batch_dim} {action_dim}"],
                        safe_action: Float[Tensor, "{batch_dim} {action_dim}"]) -> Tensor:

        if self.pre_eq_violation.mean() >= 1e3 or self.pre_ineq_violation.mean() >= 1e3:
            loss = self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action) +\
                   self.eq_pen_coefficient * self.pre_eq_violation + \
                   self.ineq_pen_coefficient * self.pre_ineq_violation
        else:
            loss = self.regularisation_coefficient * torch.nn.functional.mse_loss(safe_action, action) 
        return loss
    
    def safeguard_metrics(self):
        return  {
            "pre_eq_violation": self.pre_eq_violation.mean().item(),
            "pre_ineq_violation": self.pre_ineq_violation.mean().item(),
            "post_eq_violation": self.post_eq_violation.mean().item(),
            "post_ineq_violation": self.post_ineq_violation.mean().item(),
        }

    def get_visualization_data(self):
        """Helper to extract data for plotting later."""
        if self.last_trajectory is None:
            return None
        
        # リスト内のTensorを結合してCPUに移す (List[Tensor] -> Tensor)
        # shape: (Steps, Batch, Dim)
        traj_stack = torch.stack(self.last_trajectory).detach().cpu()
        
        # Unsafe Action も CPUへ
        unsafe_cpu = self.last_unsafe_action.detach().cpu()

        # 安全領域のデータ (数値のみ抽出)
        # ※ self.data オブジェクトそのものは保存しない！
        safe_set_info = {}
        
        # Zonotopeの場合 (center, generators)
        # 環境から直接値を取り出して Tensor として保存する
        try:
            # 環境のメソッドを呼び出して値だけコピーする
            zonotope = self.env.safe_action_set()
            safe_set_info["safe_set_center"] = zonotope.center.detach().cpu()
            safe_set_info["safe_set_generators"] = zonotope.generator.detach().cpu()
        except Exception:
            # Zonotopeでない場合、あるいは取得失敗時のフォールバック
            pass

        # 辞書を作成 (ここにはクラスインスタンスを含めない)
        return {
            "trajectory": traj_stack,
            "unsafe_action": unsafe_cpu,
            **safe_set_info  # 辞書を結合
        }
    
class DataInterface:

    def __init__(self,env):
        self.env = env
    
    def setup_resid(self):
        pass

    def pre_process_action(self, action):
        return action
    
    def post_process_action(self, action):
        return action
    
    def eq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.C is None or self.d is None:
            return torch.zeros_like(Y)
        return self.C @ Y.unsqueeze(2) - self.d

    def ineq_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.A is None or self.b is None:
            return torch.zeros_like(Y)
        return torch.relu(self.A @ Y.unsqueeze(2) - self.b)

class PolytopeData(DataInterface):

    def setup_resid(self):
        polytope = self.env.safe_action_set()
        self.A = polytope.A.detach()
        self.b = polytope.b.detach().unsqueeze(2)
        

class BoxData(DataInterface):

    def setup_resid(self):
        box = self.env.safe_action_set().box()
        center = box.center
        gen = box.generator

        if gen.dim() == 3:
            half_extents = gen.abs().sum(dim=2)  # (batch_box, dim)
        else:
            half_extents = gen.abs()
        
        box_min = (center - half_extents)
        box_max = (center + half_extents)

        A_half = torch.eye(self.env.action_dim).expand(self.env.batch_size, self.env.action_dim, self.env.action_dim)
        
        self.A = torch.cat([A_half, -A_half], dim=1).detach()
        self.b = torch.cat([box_max, -box_min], dim=1).unsqueeze(2).detach()
        
 

class ZonotopeData(DataInterface):

    def setup_resid(self):
        zonotope = self.env.safe_action_set()
        center = zonotope.center
        self.gen = zonotope.generator

        batch_dim, dim, num_generators = self.gen.shape

        # for eq constraints Cx = d
        # C with shape (batch_dim, dim, dim + num_generators)
        # d with shape (batch_dim, dim, 1)
        
        self.C = torch.cat([
                torch.eye(dim).expand(batch_dim, dim, dim), 
                -self.gen
            ], dim=2).detach() 
        self.d = center.unsqueeze(2).detach() 
        
        # for ineq constraints  Ax <= b
        # A with shape (batch_dim, 2 * num_generators, dim + num_generators)
        # b with shape (batch_dim, 2 * num_generators, 1)

        A_half = torch.cat([
                torch.zeros((batch_dim, num_generators, dim)),
                torch.eye(num_generators).expand(batch_dim, num_generators, num_generators)
            ], dim=2)
        
        self.A = torch.cat([A_half, -A_half], dim=1).detach()  
        self.b = torch.ones((batch_dim, 2 * num_generators, 1)).detach()  

    def pre_process_action(self, action):
        # z with shape (batch_dim, dim + num_generators)
        batch_dim, _, num_generators = self.gen.shape
        gamma = torch.randn((batch_dim, num_generators), dtype=action.dtype, device=action.device).detach()
        z = torch.cat([action, gamma], dim=1)
        return z
    
    def post_process_action(self, action):
        return action[:, :self.env.action_dim]


 