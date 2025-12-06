import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) # /workspaces/SafeGBPO/src
project_root = os.path.dirname(current_dir)              # /workspaces/SafeGBPO
sys.path.append(project_root)

import torch
from dataclasses import asdict

from conf.envs import NavigateSeekerConfig
from conf.safeguard import FSNetConfig
from conf.learning_algorithms import FSNetDebuggerConfig
from envs.navigate_seeker import NavigateSeekerEnv
from safeguards.fsnet_test import FSNetSafeguard 
from learning_algorithms.fsnet_debugger import FSNetDebugger

class ConsoleLogger:
    def log(self, metrics: dict):
        pass

def run_debug():
    print("=== Setting up Configuration ===")
    env_cfg = NavigateSeekerConfig()
    safeguard_cfg = FSNetConfig(
        regularisation_coefficient=0.5,
        fs_k=10,
        fs_kp=5,
        fs_eta=0.1
    )
    algo_cfg = FSNetDebuggerConfig(
        policy_kwargs={"net_arch": [64, 64]}, 
        policy_optim_kwargs={"lr": 0.001}
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    print("=== Building Environment ===")
    env = NavigateSeekerEnv(**asdict(env_cfg))
    
    print("=== Wrapping with FSNet Safeguard ===")
    env = FSNetSafeguard(env, **asdict(safeguard_cfg))

    print("=== Initializing FSNet Debugger Agent ===")
    agent = FSNetDebugger(
        env=env, 
        policy_kwargs=algo_cfg.policy_kwargs, 
        policy_optim_kwargs=algo_cfg.policy_optim_kwargs
    )

    print("=== Starting Debug Loop ===")
    
    num_episodes = 50
    for eps in range(num_episodes):
        reward, loss, _ = agent._learn_episode(eps)
        print(f"Episode {eps+1}/{num_episodes} | Loss: {loss:.6f} (Should decrease)")

    print("=== Debug Finished ===")

if __name__ == "__main__":
    run_debug()