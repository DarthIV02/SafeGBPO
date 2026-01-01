from abc import ABC, abstractmethod
from typing import Any

import torch
import cvxpy as cp
import numpy as np
from torch import Tensor
from beartype import beartype
from jaxtyping import  Float, jaxtyped
from gymnasium.vector import VectorActionWrapper  # used to parallize the enviroments 

import sets as sets
from envs.simulators.interfaces.simulator import Simulator
from envs.interfaces.safe_state_env import SafeStateEnv
from envs.interfaces.safe_action_env import SafeActionEnv


# Actually its Simulator & (SafeStateEnv | SafeActionEnv) but this is not supported
SafeEnv = Simulator | SafeStateEnv | SafeActionEnv


class Safeguard(VectorActionWrapper, ABC):
    """
    Ensuring safety of actions according to some safe set.
    """
    env: SafeEnv
    solver_args = {"solve_method": "Clarabel"}

    @jaxtyped(typechecker=beartype)
    def __init__(self, env: SafeEnv, **kwargs):
        """
        Args:
            env: A custom secured, pytorch-based environment.
        """
        ## Yasin note: here the wrapper encapsulation happens such that it has the same  variables and parallized functions 
        super().__init__(env)
        self.batch_dim = self.env.num_envs
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.action_constrained = False
        self.state_constrained = False
        if isinstance(env, SafeActionEnv):
            self.action_constrained = True
        if isinstance(env, SafeStateEnv):
            self.state_constrained = True
            self.safe_state_gens = env.num_state_gens

        if self.action_constrained:
            self.safe_action_gens = self.env.num_action_gens
        else:
            self.safe_action_gens = 2 * self.action_dim


        self.constant_mat, self.state_mat, self.action_mat, self.noise_mat = self.env.linear_dynamics()

        self.safe_action = None
        self.interventions = 0

        self.pre_eq_violation = torch.Tensor([0])
        self.pre_ineq_violation = torch.Tensor([0])
        self.post_eq_violation = torch.Tensor([0])
        self.post_ineq_violation = torch.Tensor([0])
        self.dist_safe_action = torch.Tensor([0]) #torch.norm(self.safe_action - action, dim=1).mean().item()

    @jaxtyped(typechecker=beartype)
    def actions(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:

        reachable_set = self.env.reachable_set()

        ### Yasin Tag: 

        ## Yasin note: this function come frome the VectorActionWrapper abstract class and is used to transform the policy before inputing it into the env
        # the simulator env is defined as an Vector env where you have to connect it with execute_action which is the connection here i think
        # the execute_action is then used in the simulator step to then output the reward,
        # Batch of (observations, rewards, terminations, truncations, infos)
        
        # This is an overapproximation so it may not intersect

        ## Yasin note: reachable set is the set of states when doing an action from a safe state which of course can be unsafe
        ## the state_set is the safe set defined by the enviroment. the projectable should then just be aboolean vector i guess that looks if we have to do the safeguard function
        
        ## Paper note: his makes it possible to replace any constraint on a safe action as,i ∈ As by the state constraint Si+1(as,i, si) ⊆ Ss.
        projectable = self.env.state_set.intersects(reachable_set)

        ## Yasin note: this is where the model uses its safeguard optimisation like BP or Rays for actions outside the safe set
        safe_action = torch.where(projectable.unsqueeze(1), action, self.safeguard(action))


        if safe_action.isnan().any() or safe_action.isinf().any():
            raise ValueError(f"""
            Safe action are NaN. 
            {self.env.state[projectable][safe_action.isnan().any(dim=1)]}
            {action[projectable][safe_action.isnan().any(dim=1)]}
            {self.env.state[projectable][safe_action.isinf().any(dim=1)]}
            {action[projectable][safe_action.isinf().any(dim=1)]}
            """)

        self.safe_action = safe_action
        self.interventions += ((~torch.isclose(safe_action, action)).count_nonzero(dim=1) == self.action_dim).sum().item()
        return safe_action

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def safeguard(self, action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]) \
            -> Float[Tensor, "{self.batch_dim} {self.action_dim}"]:
        """
        Safeguard the action to ensure safety.

        Args:
            action: The action to safeguard.

        Returns:
            The safeguarded action.
        """
        pass

    @jaxtyped(typechecker=beartype)
    @abstractmethod
    def safe_guard_loss(self,
                        action: Float[Tensor, "{self.batch_dim} {self.action_dim}"],
                        safe_action: Float[Tensor, "{self.batch_dim} {self.action_dim}"]
                        ) -> Float[Tensor, "{self.batch_dim}"]:
        """
        Compute the safeguard loss for the given action.

        Args:
            action: The action to compute the loss for.
        Returns:
            The safeguard loss.
        """
        pass

    @jaxtyped(typechecker=beartype)
    def safeguard_metrics(self) -> dict[str, Any]:
        """
        Get metrics related to the safeguard.

        Returns:
            A dictionary of metrics.
        """
        return {
            "dist_safe_action":     self.dist_safe_action,
            "pre_eq_violation":     self.pre_eq_violation,
            "pre_ineq_violation":   self.pre_ineq_violation,
            "post_eq_violation":    self.post_eq_violation,
            "post_ineq_violation":  self.post_ineq_violation,
        }

    @jaxtyped(typechecker=beartype)
    def linear_step(self,action: cp.Expression | np.ndarray) \
            -> tuple[cp.Expression | np.ndarray, np.ndarray, list[cp.Parameter]]:
        """
        Propagate the system through its linearised dynamics. It is newly linearised at the current state,
        the noise matrix is assumed to be constant, and the linearisation point of the noise matrix is assumed
        to be the noise centre.

        Args:
            action: The action to take.

        Returns:
            The next state center, generator and the parameters.
        """

        constant_mat = cp.Parameter(self.state_dim)
        action_mat = cp.Parameter((self.state_dim, self.action_dim))
        noise_mat = self.noise_mat[0].cpu().numpy()

        lin_action = self.env.action_set.center[0].cpu().numpy()

        noise_generator = self.env.noise_set.generator[0].cpu().numpy()

        next_state_center = constant_mat + action_mat @ (action - lin_action)
        next_state_generator = noise_mat @ noise_generator

        return next_state_center, next_state_generator, [constant_mat, action_mat]

    @jaxtyped(typechecker=beartype)
    def feasibility_constraints(self, action: cp.Expression | np.ndarray) \
            -> list[bool | cp.Constraint]:
        """
        Construct feasibility constraints by ensuring containment in the feasible action set.

        Args:
            action: The action to constrain.

        Returns:
            list: The constraints.
        """
        ## Yasin note: just looks if the action is inside the in the defined minimum and maximum possivle values
        
        return [
            self.env.action_set.min[0, :].cpu().numpy() <= action,
            self.env.action_set.max[0, :].cpu().numpy() >= action,
        ]

    @jaxtyped(typechecker=beartype)
    def action_safety_constraints(self,
                                  center: cp.Expression | np.ndarray,
                                  generator: cp.Expression | np.ndarray = None) \
            -> tuple[list[cp.Constraint], list[cp.Parameter]]:
        """
        Construct safety constraints by ensuring containment in the safe action set.

        Args:
            center: Center of the safe action.
            generator: Generator of the safe action.

        Returns:
            The constraints and parameters.
        """
        safe_action_center = cp.Parameter(self.action_dim)
        safe_action_generator = cp.Parameter((self.action_dim, self.safe_action_gens))

        if generator is None: ## Yasin note: BP
            constraints = sets.Zonotope.point_containment_constraints(
                center,
                safe_action_center,
                safe_action_generator
            )
        else: ## Yasin note: raymasks
            constraints = sets.Zonotope.zonotope_containment_constraints(
                center,
                generator,
                safe_action_center,
                safe_action_generator
            )
        return constraints, [safe_action_center, safe_action_generator]

    @jaxtyped(typechecker=beartype)
    def state_safety_constraints(self, action: cp.Expression | np.ndarray) \
            -> tuple[list[cp.Constraint], list[cp.Parameter]]:
        """
        Construct safety constraints by ensuring containment in the safe state set.

        Args:
            action: The action to take.

        Returns:
            The constraints and parameters.
        """

        ## Paper note: 
        ## One method we discuss in particular is obtaining safe state sets via robust control invariant sets [46], which
        ## guarantee the existence of an invariance-enforcing controller that can keep all future states within the safe set.
        ## This is achieved by enclosing the dynamics at the current state by a linear transition function with a noise zonotope* W = ⟨cW, GW⟩ ⊂ R* dS , 
        ## such that:*Si+1(ai , si) = M ai ⊕ ⟨c + cW, GW⟩ , (10) where c is the offset and M the Jacobian of the linearisation. 

        safe_state_center = cp.Parameter(self.state_dim)
        safe_state_generator = cp.Parameter((self.state_dim, self.safe_state_gens))

        next_state_center, next_state_generator, parameters = self.linear_step(action)
        constraints = sets.Zonotope.zonotope_containment_constraints(
            next_state_center,
            next_state_generator,
            safe_state_center,
            safe_state_generator
        )

        return constraints, [safe_state_center, safe_state_generator, *parameters]

    @jaxtyped(typechecker=beartype)
    def constraint_parameters(self) -> list[Tensor]:
        parameters = []
        if self.action_constrained:
            parameters += [*self.env.safe_action_set()]
        if self.state_constrained:
            parameters += [*self.env.safe_state_set()]
            constant_mat, _, action_mat, _ = self.env.linear_dynamics()
            parameters += [constant_mat, action_mat]
        return parameters

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def __dir__(self) -> list[str]:
        base = super().__dir__()  # Iterable[str]
        env_attrs = dir(self.env)  # list[str]
        attrs: set[str] = set()
        attrs.update(base)
        attrs.update(env_attrs)
        return sorted(attrs)

    @jaxtyped(typechecker=beartype)
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Tensor, dict[str, Any]]:
        """
        Compatibility shim for Gymnasium's VectorEnv.reset(seed=..., options=...).
        """
        return self.env.reset(seed=seed)


Simulator.register(Safeguard)
