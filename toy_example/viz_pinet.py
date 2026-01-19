import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

# ==========================================
# 1. Path Setup & Imports
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

try:
    from safeguards.pinet import PinetSafeguard
    import sets as sets
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

from envs.interfaces.safe_action_env import SafeActionEnv
from envs.simulators.interfaces.simulator import Simulator

# ==========================================
# 2. Helper: Automatic Polytope Visualization
# ==========================================
def get_polytope_patch(A, b, color):
    if isinstance(A, torch.Tensor): A = A.detach().cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.detach().cpu().numpy()
    
    norm_A = np.linalg.norm(A, axis=1)
    c_obj = np.zeros(A.shape[1] + 1)
    c_obj[-1] = -1 
    A_ub = np.hstack([A, norm_A[:, np.newaxis]])
    b_ub = b
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
    
    interior_point = np.zeros(A.shape[1])
    if res.success: interior_point = res.x[:-1]
    
    halfspaces = np.hstack((A, -b[:, np.newaxis]))
    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        hull = ConvexHull(hs.intersections)
        ordered_vertices = hs.intersections[hull.vertices]
        return Polygon(ordered_vertices, closed=True, fc=color, alpha=0.1, ec=color, lw=2.5, ls='--', label='Safe Set')
    except Exception:
        return None

# ==========================================
# 3. Environment with Complex Constraints
# ==========================================
class ComplexPolytopeEnv(Simulator, SafeActionEnv):
    def __init__(self):
        SafeActionEnv.__init__(self, num_action_gens=2)
        num_envs = 1
        action_dim = 2
        self.polytope = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        A = [
            [ 0.8,  1.0], [-1.0,  0.5], [-1.0, -0.2],
            [-0.2, -1.0], [ 1.0, -0.8], [ 1.0,  0.0]
        ]
        b = [0.5, 0.6, 0.7, 0.6, 0.5, 0.7]

        self.A = torch.tensor(A, dtype=torch.float64, device=self.device)
        self.b = torch.tensor(b, dtype=torch.float64, device=self.device)

        box = sets.AxisAlignedBox(torch.zeros((num_envs, 2), device=self.device), torch.eye(2, device=self.device).unsqueeze(0))
        Simulator.__init__(self, action_dim=action_dim, state_set=box, noise_set=box, observation_set=box, num_envs=num_envs)

    def compute_A_b(self): return self.A.unsqueeze(0), self.b.unsqueeze(0)
    def safe_action_set(self): return sets.HPolytope(self.A.unsqueeze(0), self.b.unsqueeze(0))
    def reward(self, action): return torch.zeros(1, device=self.device)
    def episode_ending(self): return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    def unbatched_dynamics(self, s, a, n): return s
    def render(self): return []

# ==========================================
# 4. Optimization Setup
# ==========================================
def run_pinet_optimization():
    env = ComplexPolytopeEnv()
    device = env.device

    pinet = PinetSafeguard(
        env=env,
        regularisation_coefficient=10.0,
        n_iter_admm=20, 
        n_iter_bwd=10,
        bwd_method="unroll"
    )

    u_param = torch.tensor([[0.9, 0.9]], requires_grad=True, dtype=torch.float64, device=device)
    target = torch.tensor([[-0.15, -0.15]], dtype=torch.float64, device=device)
    
    optimizer = torch.optim.Adam([u_param], lr=0.01, betas=(0.9, 0.999))
    
    u_hist, s_hist = [], []
    steps = 500
    
    print(f"Starting Optimization ({steps} steps)...")
    
    for i in range(steps):
        u_hist.append(u_param.detach().cpu().numpy().copy()[0])
        u_safe = pinet.safeguard(u_param)
        s_hist.append(u_safe.detach().cpu().numpy().copy()[0])
        
        diff = u_safe - target
        loss_attr = torch.sum(diff**2)
        curl_loss = 0.1 * torch.abs(diff[:, 0] * 0.6 + diff[:, 1] * (-0.4))
        loss_raw = 0.005 * torch.sum((u_param - target)**2)
        
        loss = loss_attr + curl_loss + loss_raw
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.array(u_hist), np.array(s_hist), target.cpu().numpy()[0], env

# ==========================================
# 5. Visualization (Poster Style)
# ==========================================
def plot_results(u_traj, s_traj, target, env):
    # --- Color Palette (Magma / Flare Style) ---
    cmap = plt.get_cmap('magma')
    c_path_dark = cmap(0.15)  # (Unsafe Trajectory)
    c_proj_link = cmap(0.55)  # (Projection Link)
    c_safe_set  = cmap(0.75)  # (Safe Set)
    c_target    = cmap(0.65)  # (Target)
    c_safe_pt   = cmap(0.85)  # (Projected Point)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Background
    ax.set_facecolor('white')

    # 2. Safe Set
    poly_patch = get_polytope_patch(env.A, env.b, color=c_safe_set)
    if poly_patch:
        ax.add_patch(poly_patch)

    # 3. Sampling Strategy (Manual Adjustment for better spacing)
    indices = [0, 15, 50, 100, 180, 280, 400, 499]
    print(f"Plotting steps: {indices}")

    # 4. Plotting
    for k, i in enumerate(indices):
        if i >= len(u_traj): break
        u, s = u_traj[i], s_traj[i]
        
        # --- A. Unsafe Parameter Evolution  ---
        if k < len(indices) - 1:
            next_i = indices[k+1]
            u_next = u_traj[next_i]
            arrow = FancyArrowPatch(
                posA=u, posB=u_next,
                arrowstyle='-|>,head_length=6,head_width=3',
                connectionstyle="arc3,rad=0", 
                color=c_path_dark, lw=2, alpha=0.8, zorder=3
            )
            ax.add_patch(arrow)

        # --- B. Projection Link ---
        ax.plot([u[0], s[0]], [u[1], s[1]], linestyle=':', color=c_proj_link, lw=1.5, alpha=0.7, zorder=2)

        # --- C. Markers ---
        # Unsafe (X marker)
        ax.plot(u[0], u[1], marker='x', ms=8, color=c_path_dark, mew=2.5, zorder=5)
        # Safe (Circle marker)
        ax.plot(s[0], s[1], marker='o', ms=6, color=c_safe_pt, mec=c_path_dark, mew=1, zorder=4)

    # 5. Target (Safe Set)
    ax.scatter(target[0], target[1], marker='*', s=300, color=c_target, edgecolors='black', lw=1.5, label='Target', zorder=10)

    # 6. Formatting
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    
    ax.set_title("Optimization Trajectory", fontsize=16, fontweight='bold', color='#333333')
    
    ax.grid(True, linestyle='-', alpha=0.15, color='gray')

    legend_elements = [
        Line2D([0], [0], color='white', marker='s', markerfacecolor=c_safe_set, alpha=0.3, markersize=10, markeredgecolor=c_safe_set, linestyle='--', lw=2, label='Safe Set'),
        Line2D([0], [0], color=c_path_dark, lw=2, marker='x', markersize=8, label='Unsafe Param Evolution'),
        Line2D([0], [0], color=c_proj_link, lw=1.5, linestyle=':', marker='o', markerfacecolor=c_safe_pt, markersize=6, label='Projection Mapping'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=c_target, markeredgecolor='black', markersize=12, label='Target')
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    
    # SVG保存
    plt.savefig("pinet_optimization_magma.svg", format='svg', bbox_inches='tight')
    plt.savefig("pinet_optimization_magma.png", dpi=300, bbox_inches='tight') # 確認用PNG
    print("Saved: pinet_optimization_magma.svg & .png")

if __name__ == "__main__":
    u_h, s_h, t, env_obj = run_pinet_optimization()
    plot_results(u_h, s_h, t, env_obj)