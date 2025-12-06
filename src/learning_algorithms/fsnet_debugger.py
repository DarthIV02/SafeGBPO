import torch
from beartype import beartype
from jaxtyping import jaxtyped
from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from envs.simulators.interfaces.simulator import Simulator

class FSNetDebugger(LearningAlgorithm):
    """
    FSNetの動作確認用エージェント。
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 env: Simulator,
                 policy_kwargs: dict,
                 policy_optim_kwargs: dict,
                 **kwargs): 
        
        super().__init__(
            env=env, 
            policy_kwargs=policy_kwargs, 
            policy_optim_kwargs=policy_optim_kwargs, 
            vf_kwargs={},
            vf_optim_kwargs={}, 
            regularisation_coefficient=0.0, 
            q_function=False 
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 【修正】 Tanhのリミッターを解除する ===
        # 既存の self.policy (Tanh付き) を上書きして、素のLinear層にする。
        # これで出力は 5.0 でも 100.0 でも出せるようになり、確実に制約違反できる。
        import torch.nn as nn
        self.policy = nn.Sequential(
            nn.Linear(self.env.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.env.action_dim)
            # 最後に Tanh を入れない！
        )
        self.policy.to(self.device)
        
        # Optimizerも新しいPolicyのパラメータで作り直す
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), **policy_optim_kwargs)

        
    @jaxtyped(typechecker=beartype)
    def _learn_episode(self, eps: int) -> tuple[float, float, float]:
        obs, info = self.env.reset()
        total_distance_loss = 0.0
        
        steps = self.env.num_steps if hasattr(self.env, "num_steps") else 200

        for t in range(steps):
            self.policy_optim.zero_grad()
            
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=self.device, dtype=torch.float64)
            else:
                obs = obs.to(self.device)
            
            # 1. Policy出力
            a_raw = self.policy(obs)
            
            # === 【追加】 ダミーの目標（危険な行動）を作成 ===
            # 例えば、常に [5.0, 5.0, ...] のような大きな値を目指させる。
            # これにより、Policyは制約の外に出ようとする。
            fake_target = torch.ones_like(a_raw) * 5.0 
            
            # 2. FSNet Safeguard適用
            a_safe = self.env.safeguard(a_raw)
            
            # 3. Loss計算
            # Task Loss: 危険な目標に近づきたい
            task_loss = torch.nn.functional.mse_loss(a_raw, fake_target)
            
            # Safety Loss: FSNetによる修正を最小化したい (これが本題)
            # 係数(10.0など)を掛けて、Safetyを優先させる
            safety_loss = 10.0 * torch.nn.functional.mse_loss(a_raw, a_safe)
            
            # Total Loss
            loss = task_loss + safety_loss
            
            # 4. 更新
            loss.backward()
            
            # --- 【デバッグ用】 勾配チェック ---
            if t == 0 and eps % 10 == 0:
                # Policyの重みに勾配が入っているか確認
                grad_norm = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                print(f"  [Grad Check] Policy Grad Norm: {grad_norm:.6f}")
                print(f"  [Val Check] Raw: {a_raw[0].detach().cpu().numpy().round(2)} -> Safe: {a_safe[0].detach().cpu().numpy().round(2)}")
            # ----------------------------------

            self.policy_optim.step()
            
            next_obs, reward, terminated, truncated, info = self.env.step(a_safe)
            
            if isinstance(next_obs, torch.Tensor):
                obs = next_obs.detach()
            else:
                obs = next_obs
            
            # ログには Safety Loss (修正量) だけ記録して減少を見る
            total_distance_loss += safety_loss.item()
            
            if isinstance(terminated, torch.Tensor):
                done = terminated.any() or truncated.any()
            else:
                done = terminated or truncated

            if done:
                obs, _ = self.env.reset()

        avg_dist_loss = total_distance_loss / steps
        
        return -avg_dist_loss, avg_dist_loss, 0.0