import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pathlib import Path
from dataclasses import asdict
from typing import Optional

import torch
import wandb
import optuna # used to train the hyperparameters

from logger import Logger
from utils import categorise_run, import_module, gather_custom_modules
from conf.experiment import Experiment

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def run_experiment(cfg: Experiment, trial: Optional[optuna.Trial] = None) -> float:
    if trial is not None:
        cfg.learning_algorithm.vary(trial, cfg)

    group, tags = categorise_run(cfg)

    run = wandb.init(project="Leveraging Analytical Gradients in Provably Safe Reinforcement Learning",
                     config=asdict(cfg),
                     monitor_gym=True,
                     group=group,
                     tags=tags)

    if trial is not None:
        run.name = f"trial/{trial.number}/{run.name}"
        run.config.update(trial.params)

    run.config["config"] = asdict(cfg)

    modules = gather_custom_modules(Path(__file__).parent / "envs", "Env")
    modules |= gather_custom_modules(Path(__file__).parent / "safeguards", "Safeguard")
    modules |= gather_custom_modules(Path(__file__).parent / "learning_algorithms", "LearningAlgorithm")

    ## Yasin note: 
    ##  the real example enviroments consist of the 2 important interfaces, 
    ##  the first is the simulator where one is asked to define the reset, observation, reward, dynamics of the system, episode_ending (if the episode ended ) and how the simulation is rendered
    ##  an implemented simulator defines first the feasible observation, noise and  state as axis aligned boxes.  
    ##  the second is defining what kind of Safety we enforce. Either Safe Action set, safe states or both via RCI (Robust Control Invariance) defined on a Zonotope 


    env_class = import_module(modules, cfg.env.name + "Env")
    env = env_class(**asdict(cfg.env))
    cfg.env.num_envs = env.EVAL_ENVS
    eval_env = env_class(**asdict(cfg.env))

    if cfg.safeguard:
        ## Yasin note: here the enviroment is packaged into the Safeguard such that it is encapsulated and has the same properties as env.
        safeguard_class = import_module(modules, cfg.safeguard.name + "Safeguard")
        env = safeguard_class(env, **asdict(cfg.safeguard))
        eval_env = safeguard_class(eval_env, **asdict(cfg.safeguard))

    agent = import_module(modules, cfg.learning_algorithm.name)(**vars(cfg.learning_algorithm), env=env)
    logger = Logger(agent, env, eval_env, run, trial, cfg.eval_freq, cfg.fast_eval)

    ## Yasin note: 
    ## all the things before just loaded the config information given with cfg with is the Experiment. Only this actually trains this stuff now
    ## The agent is the main program where the env, saveguard and can be found in learning_algorithms/interfaces/learning_algorithms.

    agent.learn(interactions=cfg.interactions, logger=logger)

    run.finish()

    return logger.best_reward


if __name__ == "__main__":
    from conf.envs import *
    from conf.safeguard import *
    from conf.learning_algorithms import *

    wandb.login()

    experiment_queue = []

    
    # experiment_queue.extend([
    #     Experiment(num_runs=1, learNavigateSeekning_algorithm=SHACConfig(), env=erConfig(), safeguard=None, interactions=15_000, eval_freq=5_000, fast_eval=False),
    #     Experiment(num_runs=1, learning_algorithm=SHACConfig(), env=NavigateSeekerConfig(), safeguard=BoundaryProjectionConfig(), interactions=15_000, eval_freq=5_000, fast_eval=False),
    #     Experiment(num_runs=1, learning_algorithm=SHACConfig(), env=NavigateSeekerConfig(), safeguard=RayMaskConfig(zonotopic_approximation=True), interactions=15_000, eval_freq=5_000, fast_eval=False),
    # ])

    ## decide randomly
    experiment_queue.append(
        Experiment(
            num_runs=1,
            
            learning_algorithm=SHACConfig(
                len_trajectories=64, 
            ),
            
            env=NavigateSeekerConfig(
                num_envs=64,
                num_steps=400,
                draw_safe_action_set=False 
            ),
            
            safeguard=FSNetTestConfig(
                regularisation_coefficient=0.5,
                fs_k=10,
                fs_kp=5,
                fs_eta=0.1,
                eq_pen_coefficient=1.0,
                ineq_pen_coefficient=1.0
            ),
            
            interactions=2_000_000,
            eval_freq=200_000,
            fast_eval=False
        )
    )

    for i, experiment in enumerate(experiment_queue):
        if experiment.num_runs == 0:
            print("[STATUS] Running hyperparameter search")
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        storage=f"sqlite:///hyperparameters/{experiment.env.name}/study.sqlite3",
                                        study_name=experiment.learning_algorithm.name)

            pre_valued_objective = lambda trial: run_experiment(experiment, trial)
            study.optimize(pre_valued_objective, n_trials=100, n_jobs=1)
            print(f"Best value: {study.best_value} (params: {study.best_params})")

        else:
            print(f"[STATUS] Running experiment [{i + 1}/{len(experiment_queue)}]")
            for j in range(experiment.num_runs):
                run_experiment(experiment)