import cvxpy as cp
from torch import Tensor
from beartype import beartype
from cvxpylayers.torch import CvxpyLayer
from jaxtyping import Float, jaxtyped

from safeguards.interfaces.safeguard import Safeguard, SafeEnv


class BoundaryProjectionSafeguard(Safeguard):
    """
    Projecting an unsafe action to the closest safe action.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, env: SafeEnv, **kwargs):
        Safeguard.__init__(self, env)

        self.boundary_layer = None

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

        ### Yasin Tag: 

        ## Yasin note: example of the BP CVXPY optimisation to get the nearest action of the safe set 
        ## min || a_s -a || is computed as a convex optimitation step to compute the nearest possible action
     
        if self.boundary_layer is None:
            cp_action = cp.Parameter(self.action_dim)
            parameters = [cp_action]

            cp_safe_action = cp.Variable(self.action_dim)

            objective = cp.Minimize(cp.sum_squares(cp_action - cp_safe_action))

            ## Yasin note:  
            ## with the feasibility_constraints the model basically constructs the constraint from the zonotope in the env to be used cvxpy
            ## the feasibility_constraints just looks if the action is inside the in the defined minimum and maximum possivle values 
            ## action_safety_constraints: if the point is in the zonotope Z = {c + Gβ | ∥β∥∞ ≤ 1} = ⟨c, G⟩
            ## state_safety_constraints_ or wioth the generator if the generated zonotope of the possible actions is inside the safety zonotope

            ## Paper note: Determining the containment of a zonotope in another
            ## zonotope is co-NP complete [63], but a sufficient condition for Z1 ⊆ Z2 is [64, Eq. 15]
            ## 1 ≥ min γ∈Rn2 ,Γ∈Rn2×n1  ||Γ γ||∞ (8a) 
            ## subject to G1 = G2Γ (8b) 
            ## c2 − c1 = G2γ . (8c)
            ## Both containment problems are linear.


            constraints = self.feasibility_constraints(cp_safe_action)
            if self.action_constrained:
                constraint, params = self.action_safety_constraints(cp_safe_action)
                constraints += constraint
                parameters += params
            if self.state_constrained:
                constraint, params = self.state_safety_constraints(cp_safe_action)
                constraints += constraint
                parameters += params

            problem = cp.Problem(objective, constraints)
            self.boundary_layer = CvxpyLayer(problem, parameters=parameters, variables=[cp_safe_action])

        parameters = [action] + self.constraint_parameters()
        safe_action = self.boundary_layer(*parameters, solver_args=self.solver_args)[0]

        return safe_action


