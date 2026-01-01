import torch
import wandb
import torchvision.utils
from beartype import beartype
from jaxtyping import jaxtyped
from optuna import Trial, TrialPruned
from wandb.sdk.wandb_run import Run

from learning_algorithms.interfaces.learning_algorithm import LearningAlgorithm
from envs.simulators.interfaces.simulator import Simulator
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
import time
import psutil
import sys

class Logger:
    """
    Logs evaluation and training data to Weights and Biases.
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
        Args:
            agent: The model to train.
            env: The training environment.
            eval_env: The environment to evaluate the policy on.
            wandb_run: The Weights and Biases run object.
            optuna_trial: The Optuna trial object.
            eval_freq: The frequency at which to evaluate the policy.
            fast_eval (bool): Whether to produce video only on the final episode
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

        nvmlInit()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
        self.intermediate_time_train = time.time()
        self.process = psutil.Process()  



    @jaxtyped(typechecker=beartype)
    def on_learning_episode(self,
                            eps: int,
                            average_reward: float,
                            policy_loss: float,
                            value_loss: float,
                            num_learn_episodes: int,
                            additional_metrics: dict[str, float] = {}):
        """
        Callback call to log and evaluate.

        Args:
            eps: Number of the current learning episode
            average_reward: The average reward of the current episode
            policy_loss: The policy loss
            value_loss: The value loss
            num_learn_episodes: The total number of learning episodes
        """
        self.log_performance()
        self.log_data["train/Average Reward"] = average_reward
        if hasattr(self.env, "interventions"):
            self.log_data["train/Interventions"] = self.env.interventions
        for i, val in enumerate(self.model.policy.log_std.detach().cpu().numpy()):
            self.log_data[f"train/log(std_{i})"] = val
        self.log_data["train/Policy Loss"] = policy_loss
        self.log_data["train/Value Loss"] = value_loss

        for key, value in additional_metrics.items():
            self.log_data[f"train/{key}"] = value

        samples = eps * self.model.interactions_per_episode
        if samples - self.last_eval >= self.eval_freq or eps == num_learn_episodes - 1:
            self.last_eval = samples
            eval_reward = self.evaluate_policy(eps, num_learn_episodes)

            if self.optuna_trial is not None:
                self.optuna_trial.report(eval_reward, eps)

        self.wandb_run.log(data=self.log_data, step=samples, commit=True)
        self.log_data = {}

        if self.optuna_trial is not None and self.optuna_trial.should_prune():
            self.wandb_run.finish()
            raise TrialPruned()
        self.intermediate_time_train = time.time()

    @jaxtyped(typechecker=beartype)
    def evaluate_policy(self, eps: int, num_learn_episodes: int) -> float:
        """
        Evaluates the current policy on the evaluation environment.

        Args:
            eps: The current episode number.
            num_learn_episodes: The total number of learning episodes.
        Returns:
            float: The average reward obtained during the evaluation.
        """
        eval_reward = 0
        record = not self.fast_eval or eps == num_learn_episodes - 1
        frames = []

        observation, info = self.eval_env.eval_reset()
        terminal = False
        steps = 0
        self.intermediate_time_eval = time.time()

        # If there is no safeguard -> at least check the pre and post violation
        if not hasattr(self.eval_env, "safeguard_metrics"):
            def safeguard_metrics():
                return {
                    "pre_eq_violation":     self.eval_env.pre_eq_violation,
                    "pre_ineq_violation":   self.eval_env.pre_ineq_violation,
                    "pre_contraint_violation": self.eval_env.pre_constraint_violation
                }
            self.eval_env.safeguard_metrics = safeguard_metrics

        store_violation = {"pre_eq": 0, "pre_ineq": 0, "post_eq": 0, "post_ineq": 0, "dif": 0, "pre_cv": 0, "post_cv": 0}

        while not terminal:
            action = self.model.policy.predict(observation, deterministic=True) # Unsafe action
            
            observation, reward, terminated, truncated, info = self.eval_env.step(action) # Action becomes safe due to Gymnasium actions
            
            terminal = (terminated | truncated)[0].item()
            if record:
                frame = torchvision.utils.make_grid(torch.stack(self.eval_env.render()),
                                                    nrow=int(torch.sqrt(torch.tensor([self.eval_env.num_envs]))))
                frames += [frame]

            eval_reward += reward.sum().item()

            safe_action_set = self.eval_env.safe_action_set()
            safe_action_set.setup_resid()
            processed_action = safe_action_set.pre_process_action(action)
            store_violation["pre_eq"] += safe_action_set.eq_resid(None, processed_action).square().mean().item()
            store_violation["pre_ineq"] += safe_action_set.ineq_resid(None, processed_action).square().mean().item()
            store_violation["pre_cv"] += safe_action_set.constraint_violation(None, processed_action).square().mean().item()
            
            if hasattr(self.eval_env, "safe_action"):
                safe_action = self.eval_env.safe_action
                processed_safe_action = safe_action_set.pre_process_action(safe_action)
                store_violation["post_eq"] += safe_action_set.eq_resid(None, processed_safe_action).square().mean().item()
                store_violation["post_ineq"] += safe_action_set.ineq_resid(None, processed_safe_action).square().mean().item()
                store_violation["dif"] += torch.norm(safe_action - action, dim=1).mean().item()
                store_violation["post_cv"] += safe_action_set.constraint_violation(None, processed_action).square().mean().item()

            steps += 1

        self.eval_env.pre_eq_violation = store_violation["pre_eq"] / steps
        self.eval_env.pre_ineq_violation = store_violation["pre_ineq"] / steps
        self.eval_env.pre_constraint_violation = store_violation["pre_cv"] / steps
        
        if hasattr(self.eval_env, "safe_action"):
            self.eval_env.post_eq_violation = store_violation["post_eq"] / steps
            self.eval_env.post_ineq_violation = store_violation["post_ineq"] / steps
            self.eval_env.dist_safe_action = store_violation["dif"] / steps
            self.eval_env.post_constraint_violation = store_violation["post_cv"] / steps

        avg_eval_reward = eval_reward  / self.eval_env.num_envs / steps
        
        self.log_performance_eval()
        if hasattr(self.eval_env, "safeguard_metrics"):
            for key, value in self.eval_env.safeguard_metrics().items():
                self.log_data[f"eval/{key}"] = value

        self.log_data["eval/Average Reward"] = avg_eval_reward

        if record and frames[0].numel() != 0:
            frames = torch.stack(frames).numpy()
            self.log_data["eval/Video"] = wandb.Video(frames, fps=60, format="mp4")

        if avg_eval_reward > self.best_reward:
            self.best_reward = avg_eval_reward

        return avg_eval_reward


    def log_performance(self):
        gpu_util = nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
        cpu_util = psutil.cpu_percent(interval=None)
        steps_per_sec =  self.model.interactions_per_episode / (time.time() - self.intermediate_time_train)
        sec_per_episode = (time.time() - self.intermediate_time_train)
        self.log_data["performance_train/seconds_per_episode"] = sec_per_episode
        self.log_data["performance_train/steps_per_second"] = steps_per_sec
        self.log_data["performance_train/gpu_utilization_mean"] = gpu_util
        self.log_data["performance_train/cpu_utilization_mean"] = cpu_util

    def log_performance_eval(self):
        gpu_util = nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
        cpu_util = psutil.cpu_percent(interval=None)
        sec_per_evaluation = (time.time() - self.intermediate_time_eval)
        self.log_data["performance_eval/seconds_per_evaluation"] = sec_per_evaluation
        self.log_data["performance_eval/gpu_utilization_mean"] = gpu_util
        self.log_data["performance_eval/cpu_utilization_mean"] = cpu_util
 