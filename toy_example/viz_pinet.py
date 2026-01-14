import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# ==========================================
# 1. Path Setup & Imports
# ==========================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

try:
    from src.safeguards.pinet import PinetSafeguard
    import sets as sets
except ImportError as e:
    print(f"[Error] Failed to import PinetSafeguard: {e}")
    sys.exit(1)

from envs.interfaces.safe_action_env import SafeActionEnv
from envs.simulators.interfaces.simulator import Simulator

# ==========================================
# 2. Mock Environment with Complex Polytope
# ==========================================
class PolytopeSafeEnv(Simulator, SafeActionEnv):
    """
    複雑な形状(Polytope)のSafe Setを持つダミー環境
    """
    def __init__(self):
        SafeActionEnv.__init__(self, num_action_gens=2)
        num_envs = 1
        action_dim = 2
        state_dim = 2
        self.polytope = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Safe Set Definition: Ax <= b ---
        # 5つの不等式で囲まれた領域を作成
        A = [
            [1.0, 1.0],   # x + y <= 0.2  (右上カット)
            [-1.0, 0.0],  # -x <= 0.7     (左端 x >= -0.7)
            [0.0, -1.0],  # -y <= 0.7     (下端 y >= -0.7)
            [1.0, -1.0],  # x - y <= 0.6  (右下カット)
            [-1.0, 1.0]   # -x + y <= 0.6 (左上カット)
        ]
        b = [0.2, 0.7, 0.7, 0.6, 0.6]

        self.A = torch.tensor(A, dtype=torch.float64, device=self.device)
        self.b = torch.tensor(b, dtype=torch.float64, device=self.device)

        # PiNet初期化に必要なダミー設定
        center = torch.zeros((num_envs, state_dim), dtype=torch.float64, device=self.device)
        generator = torch.diag_embed(torch.ones((num_envs, state_dim), dtype=torch.float64, device=self.device))
        box_set = sets.AxisAlignedBox(center, generator)

        Simulator.__init__(self, action_dim=action_dim, state_set=box_set, noise_set=box_set,
                           observation_set=box_set, num_envs=num_envs)

    def compute_A_b(self):
        return self.A.unsqueeze(0), self.b.unsqueeze(0)

    def safe_action_set(self):
        return sets.HPolytope(self.A.unsqueeze(0), self.b.unsqueeze(0))

    # --- Minimal Implementations for Abstract Methods ---
    def reward(self, action): 
        return torch.zeros(self.num_envs, device=self.device)
    
    def episode_ending(self): 
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def unbatched_dynamics(self, s, a, n): return s
    
    def linear_dynamics(self):
        # Dummy linear dynamics
        batch = self.num_envs
        z = torch.zeros
        return z((batch, 2), device=self.device), z((batch, 2, 2), device=self.device), \
               z((batch, 2, 2), device=self.device), z((batch, 2, 2), device=self.device)

    # ★ これがないとエラーになります
    def render(self) -> list[torch.Tensor]:
        return []

# ==========================================
# 3. Optimization Setup
# ==========================================
def run_pinet_optimization():
    env = PolytopeSafeEnv()
    device = env.device
    print(f"Using device: {device}")

    pinet = PinetSafeguard(
        env=env,
        regularisation_coefficient=10.0,
        n_iter_admm=20,     # 高速化のため減らす
        n_iter_bwd=10,
        sigma=1.0,
        omega=1.7,
        bwd_method="unroll",
        fpi=False
    )
    # 念のため
    pinet.action_dim = 2
    pinet.batch_dim = 1

    # 初期値 (右上) -> Target (左下)
    u_param = torch.tensor([[0.8, 0.8]], requires_grad=True, dtype=torch.float64, device=device)
    target = torch.tensor([[-0.4, -0.4]], dtype=torch.float64, device=device)
    
    # Adamの方が収束が早い傾向があります
    optimizer = torch.optim.Adam([u_param], lr=0.05)
    
    u_hist, s_hist = [], []
    
    print("Starting Optimization...")
    steps = 50
    for i in range(steps):
        u_hist.append(u_param.detach().cpu().numpy()[0])
        
        # Forward
        u_safe = pinet.safeguard(u_param)
        s_hist.append(u_safe.detach().cpu().numpy()[0])
        
        # Loss: Safe距離 + 補助項
        loss = torch.sum((u_safe - target)**2) + 0.05 * torch.sum((u_param - target)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iter {i}: Loss={loss.item():.4f}")

    return np.array(u_hist), np.array(s_hist), target.cpu().numpy()[0]

# ==========================================
# 4. Visualization
# ==========================================
def plot_results(unsafe_traj, safe_traj, target):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Background
    x_range = np.linspace(-1.2, 1.2, 100)
    X, Y = np.meshgrid(x_range, x_range)
    Z = np.sqrt((X - target[0])**2 + (Y - target[1])**2)
    ax.contourf(X, Y, -Z, levels=50, cmap='GnBu', alpha=0.5)
    
    # Safe Set Polygon (頂点を定義して描画)
    # A, b に基づく頂点座標
    verts = [(-0.7, -0.7), (-0.1, -0.7), (0.4, -0.2), (-0.2, 0.4), (-0.7, -0.1)]
    poly = Polygon(verts, closed=True, facecolor='none', edgecolor='black', 
                   linewidth=2, linestyle='--', label='Safe Set')
    ax.add_patch(poly)
    
    # Trajectory Line
    ax.plot(unsafe_traj[:, 0], unsafe_traj[:, 1], 'k:', linewidth=1, alpha=0.5)

    # Arrows (間引いて描画)
    indices = np.linspace(0, len(unsafe_traj)-1, 6, dtype=int)
    for i in indices:
        u_pt = unsafe_traj[i]
        s_pt = safe_traj[i]
        
        ax.plot(u_pt[0], u_pt[1], 'kp', markersize=10, markeredgecolor='white', zorder=5) # Unsafe
        ax.plot(s_pt[0], s_pt[1], 'o', color='royalblue', markersize=6, zorder=4) # Safe
        ax.annotate("", xy=s_pt, xytext=u_pt,
                    arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.5, alpha=0.7))

    ax.scatter(target[0], target[1], marker='x', s=200, color='red', lw=3, label='Target', zorder=10)
    
    ax.legend(loc='upper right')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.set_title("PiNet Optimization (Complex Polytope)")
    
    plt.tight_layout()
    plt.savefig("pinet_polytope_opt.png", dpi=300)
    print("Saved: pinet_polytope_opt.png")

if __name__ == "__main__":
    u_h, s_h, t = run_pinet_optimization()
    plot_results(u_h, s_h, t)