import torch
import wandb
import torchvision.utils
from beartype import beartype
from jaxtyping import jaxtyped
from optuna import Trial, TrialPruned
from wandb.sdk.wandb_run import Run

from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from envs.simulators.interfaces.simulator import Simulator

import time
import psutil
import csv
import os
import sys

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

class Logger:
    """
    Logs evaluation and training data to Weights and Biases, CSV, and Terminal.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 agent: LearningAlgorithm,
                 env: Simulator,
                 eval_env: Simulator,
                 wandb_run: Run,
                 optuna_trial: Trial | None,
                 eval_freq: int,
                 fast_eval: bool):
        """
        Initializes the Logger.
        """
        self.model = agent
        self.env = env
        self.eval_env = eval_env
        self.wandb_run = wandb_run
        self.optuna_trial = optuna_trial
        self.eval_freq = eval_freq
        self.fast_eval = fast_eval

        self.best_reward = -torch.inf
        self.log_data = {}
        self.last_eval = 0

        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                nvmlInit()
                self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass
        # ==================================
            
        self.intermediate_time = time.time()
        self.process = psutil.Process()

        # === CSV Logging Setup ===
        self.csv_path = "training_log.csv"
        self.csv_file = open(self.csv_path, mode='w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        
        self.header = ["Episode", "Step", "AvgReward", "PolicyLoss", "ValueLoss", "Interventions", "Time(s)"]
        self.writer.writerow(self.header)
        self.csv_file.flush()
        
        print(f"[Logger] Local logging started: {os.path.abspath(self.csv_path)}")


    @jaxtyped(typechecker=beartype)
    def on_learning_episode(self,
                            eps: int,
                            average_reward: float,
                            policy_loss: float,
                            value_loss: float,
                            num_learn_episodes: int):
        """
        Callback call to log and evaluate.
        """
        self.log_performance()
        
        interventions = 0
        if hasattr(self.env, "interventions"):
            interventions = self.env.interventions
            self.log_data["train/Interventions"] = interventions
            
        self.log_data["train/Average Reward"] = average_reward
        
        if hasattr(self.model.policy, "log_std"):
            for i, val in enumerate(self.model.policy.log_std.detach().cpu().numpy()):
                self.log_data[f"train/log(std_{i})"] = val
            
        self.log_data["train/Policy Loss"] = policy_loss
        self.log_data["train/Value Loss"] = value_loss

        samples = eps * self.model.interactions_per_episode

        # --- 1. Terminal Output ---
        print(f"[Ep {eps}/{num_learn_episodes}] Step: {samples} | Reward: {average_reward:.4f} | P_Loss: {policy_loss:.6f} | V_Loss: {value_loss:.6f} | Intv: {interventions}")
        sys.stdout.flush()

        # --- 2. CSV Output ---
        current_time = time.time()
        row = [eps, samples, average_reward, policy_loss, value_loss, interventions, f"{current_time - self.intermediate_time:.2f}"]
        self.writer.writerow(row)
        self.csv_file.flush()

        # --- 3. Evaluation & W&B Logging ---
        if samples - self.last_eval >= self.eval_freq or eps == num_learn_episodes - 1:
            self.last_eval = samples
            print(f"--- Evaluating at episode {eps} ---")
            eval_reward = self.evaluate_policy(eps, num_learn_episodes)

            if self.optuna_trial is not None:
                self.optuna_trial.report(eval_reward, eps)

        self.wandb_run.log(data=self.log_data, step=samples, commit=True)
        self.log_data = {}

        if self.optuna_trial is not None and self.optuna_trial.should_prune():
            self.wandb_run.finish()
            self.csv_file.close()
            raise TrialPruned()
            
        self.intermediate_time = time.time()

    @jaxtyped(typechecker=beartype)
    def evaluate_policy(self, eps: int, num_learn_episodes: int) -> float:
        eval_reward = 0
        record = not self.fast_eval or eps == num_learn_episodes - 1
        frames = []

        observation, info = self.eval_env.eval_reset()
        terminal = False
        steps = 0
        while not terminal:
            if hasattr(self.model.policy, "predict"):
                action = self.model.policy.predict(observation, deterministic=True)
            else:
                action = self.model.policy(observation)

            observation, reward, terminated, truncated, info = self.eval_env.step(action)
            terminal = (terminated | truncated)[0].item()
            if record:
                frame = torchvision.utils.make_grid(torch.stack(self.eval_env.render()),
                                                    nrow=int(torch.sqrt(torch.tensor([self.eval_env.num_envs]))))
                frames += [frame]

            eval_reward += reward.sum().item()
            steps += 1

        avg_eval_reward = eval_reward  / self.eval_env.num_envs / steps

        self.log_data["eval/Average Reward"] = avg_eval_reward
        print(f"    >>> Eval Reward: {avg_eval_reward:.4f}")

        if record and frames[0].numel() != 0:
            frames = torch.stack(frames).numpy()
            self.log_data["eval/Video"] = wandb.Video(frames, fps=60, format="mp4")

        if avg_eval_reward > self.best_reward:
            self.best_reward = avg_eval_reward

        return avg_eval_reward


    def log_performance(self):
        # === 【修正】 GPU情報の取得 ===
        if self.gpu_handle and PYNVML_AVAILABLE:
            try:
                gpu_util = nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                self.log_data["performance/gpu_utilization_mean"] = gpu_util
            except Exception:
                pass
        # ============================
        
        cpu_util = psutil.cpu_percent(interval=None)
        elapsed = time.time() - self.intermediate_time
        if elapsed > 0:
            steps_per_sec =  self.model.interactions_per_episode / elapsed
            episodes_per_sec = 1 / elapsed
        else:
            steps_per_sec = 0
            episodes_per_sec = 0

        self.log_data["performance/episodes_per_second"] = episodes_per_sec
        self.log_data["performance/steps_per_second"] = steps_per_sec
        self.log_data["performance/cpu_utilization_mean"] = cpu_util
    
    def __del__(self):
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()