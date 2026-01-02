from pathlib import Path
from dataclasses import asdict
from typing import Optional

import torch
import wandb
import optuna # used to train the hyperparameters

from logger import Logger
from utils import categorise_run, import_module, gather_custom_modules
from conf.experiment import Experiment

torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)

def run_experiment(cfg: Experiment, trial: Optional[optuna.Trial] = None) -> float:
    if trial is not None:
        cfg.learning_algorithm.vary(trial, cfg)

    group, tags = categorise_run(cfg)

    # ----------------- LOGGER -------------------------

    run = wandb.init(project="Leveraging Analytical Gradients in Provably Safe Reinforcement Learning",
                     config=asdict(cfg),
                     monitor_gym=True,
                     group=group,
                     tags=tags)

    if trial is not None:
        run.name = f"trial/{trial.number}/{run.name}"
        run.config.update(trial.params)

    run.config["config"] = asdict(cfg)

    # ---------------------------------------------------------


    modules = gather_custom_modules(Path(__file__).parent / "envs", "Env")
    modules |= gather_custom_modules(Path(__file__).parent / "safeguards", "Safeguard")
    modules |= gather_custom_modules(Path(__file__).parent / "learning_algorithms", "LearningAlgorithm")

    ## Yasin note: 
    ##  the real example enviroments consist of the 2 important interfaces, 
    ##  the first is the simulator where one is asked to define the reset, observation, reward, 
    ##  dynamics of the system, episode_ending (if the episode ended ) and how the simulation is rendered
    ##  an implemented simulator defines first the feasible observation, noise and state as axis aligned boxes.  
    ##  the second is defining what kind of Safety we enforce. Either Safe Action set, safe states or both via 
    ##  RCI (Robust Control Invariance) defined on a Zonotope 


    env_class = import_module(modules, cfg.env.name + "Env")
    env = env_class(**asdict(cfg.env))
    cfg.env.num_envs = env.EVAL_ENVS
    eval_env = env_class(**asdict(cfg.env))

    if cfg.safeguard:
        ## Yasin note: here the enviroment is packaged into the Safeguard such that it is encapsulated and has the same properties as env.
        print("Main: ", cfg.safeguard.name)
        safeguard_class = import_module(modules, cfg.safeguard.name + "Safeguard")
        env = safeguard_class(env, **asdict(cfg.safeguard))
        eval_env = safeguard_class(eval_env, **asdict(cfg.safeguard))

    agent = import_module(modules, cfg.learning_algorithm.name)(**vars(cfg.learning_algorithm), env=env)
    # ----------------- LOGGER -------------------------
    logger = Logger(agent, env, eval_env, run, trial, cfg.eval_freq, cfg.fast_eval)
    # ---------------------------------------------------------

    ## Yasin note: 
    ## all the things before just loaded the config information given with cfg with is the Experiment. Only this actually trains this stuff now
    ## The agent is the main program where the env, saveguard and can be found in learning_algorithms/interfaces/learning_algorithms.

    agent.learn(interactions=cfg.interactions, logger = logger)

    # ----------------- LOGGER -------------------------
    run.finish()
    # ---------------------------------------------------------

    print("Capturing visualization data...")


    # for caputre several action chunks
    multi_step_data = []
    
    obs, _ = env.reset()

    # unsafe_action_single = torch.tensor([2.0, 2.0], device=obs.device)
    # target_action = unsafe_action_single.unsqueeze(0).expand(env.num_envs, -1)

    env.debug_mode = True 

    with torch.no_grad():
        for t in range(5):
            action = agent.policy(obs)
            
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if hasattr(env, "get_visualization_data"):
                step_data = env.get_visualization_data()
                if step_data is not None:
                    step_data["step_idx"] = t 
                    multi_step_data.append(step_data)

    env.debug_mode = False

    if multi_step_data:
        torch.save(multi_step_data, "5_steps_viz.pt")
        print(f"Saved 5_steps_viz.pt (Total steps: {len(multi_step_data)})")
    else:
        print("[ERROR] No data captured.")

    


    return logger.best_reward


if __name__ == "__main__":
    from conf.envs import *
    from conf.safeguard import *
    from conf.learning_algorithms import *

    wandb.login(key="9487c04b8eff0c16cac4e785f2b57c3a475767d3")

    ## Yasin note: 
    ## we can define multiple Experiment runs in the queue. the Experiment is basically just all the configs that are then loaded in run_experiment()
    ## an Experiment defines mainly the learning algorithm, safeguard and important for us the enviroment  
 
     # this is the experiment for the benchmarking in the paper
    experiment_queue = [
        # Experiment(num_runs=1,
        #            learning_algorithm=SHACConfig(),
        #            env=NavigateSeekerConfig(),
        #            safeguard=None,
        #            interactions=60_000,
        #            eval_freq=5_000,
        #            fast_eval=False),

        # Experiment(num_runs=1,
        #            learning_algorithm=SHACConfig(),
        #            env=NavigateSeekerConfig(),
        #            safeguard=BoundaryProjectionConfig(),
        #            interactions=60_000,
        #            eval_freq=5_000,
        #            fast_eval=False),

        # Experiment(num_runs=1,
        #            learning_algorithm=SHACConfig(),
        #            env=NavigateSeekerConfig(),
        #            safeguard=RayMaskConfig(zonotopic_approximation=False, polytopic_approximation=True),
        #            interactions=60_000,
        #            eval_freq=5_000,
        #            fast_eval=False),
        
        # Experiment(num_runs=1,
        #            learning_algorithm=SHACConfig(),
        #            env=NavigateSeekerConfig(),
        #            safeguard=FSNetConfig(),
        #            interactions=60_000,
        #            eval_freq=5_000,
        #            fast_eval=False),

        # Experiment(num_runs=1,
        #            learning_algorithm=SHACConfig(),
        #            env=NavigateSeekerConfig(),
        #            safeguard=PinetConfig(),
        #            interactions=60_000,
        #            eval_freq=5_000,
        #            fast_eval=False),
        Experiment(
        num_runs=1,
        learning_algorithm=SHACConfig(),
        env=NavigateSeekerConfig(),
        safeguard=FSNetConfig(),
        interactions=10000,
        eval_freq=1000,
        fast_eval=True
    )
    ]

    # this is for testing purposes only
    # experiment_queue = [
    #     Experiment(num_runs=1,
    #                learning_algorithm=SHACConfig(),
    #                env=NavigateSeekerConfig(),
    #                safeguard=  FSNetConfig(),
    #                interactions=60_000,
    #                eval_freq=10_000,
    #                fast_eval=False),
    # ]

    for i, experiment in enumerate(experiment_queue):
        if experiment.num_runs == 0:
            print("[STATUS] Running hyperparameter search")
            ## Yasin note: 
            ##  optuna is used to train the hyperparameters defined in the learning algorithm config
            ## this is not that important for us since we have the hyperparameters already but study.optimize() basically just does the run_experiment()
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
