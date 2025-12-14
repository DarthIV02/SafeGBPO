import numpy as np
import torch
from cvxpygen import cpg
import cvxpy as cp
from cvxpygen_code_unconstrained.cpg_solver import cpg_solve

def cvxpy_solve(X, action, data, scale=1.0, **kwargs):
    action_np = action.detach().cpu().numpy()
    # NOTE: cvxpygen does not support setting an initial guess for the optimization variable y.
    # The variable y is always initialized by the solver itself. action_np is only used for shape and batching.
    # If you need warm-starting, you must use a different solver (e.g., OSQP via cvxpy) or modify the generated code.
    batch_size, n = action_np.shape

    # Get shapes
    C = data.C.cpu().numpy() if data.C is not None else None
    d = data.d.cpu().numpy() if data.d is not None else None
    A = data.A.cpu().numpy() if data.A is not None else None
    b = data.b.cpu().numpy() if data.b is not None else None

    eq_dim   = C.shape[1] if C is not None else 0
    ineq_dim = A.shape[1] if A is not None else 0

    y = cp.Variable(n)
    
    C_param = cp.Parameter((eq_dim, n))
    d_param = cp.Parameter((eq_dim, 1))
    A_param = cp.Parameter((ineq_dim, n))
    b_param = cp.Parameter((ineq_dim, 1))

    eq_resid = C_param @ y - d_param if eq_dim > 0 else 0
    ineq_resid = cp.pos(A_param @ y - b_param) if ineq_dim > 0 else 0

    objective = cp.Minimize(scale * (cp.sum_squares(eq_resid) + cp.sum_squares(ineq_resid)))
    prob = cp.Problem(objective)
    cpg.generate_code(prob, code_dir="cvxpygen_code_unconstrained")
    

    safe_actions = []
    for i in range(batch_size):
        params = {}
        if eq_dim > 0:
            params['C_param'] = C[i]
            params['d_param'] = d[i]
        if ineq_dim > 0:
            params['A_param'] = A[i]
            params['b_param'] = b[i]
        y_sol, _ = cpg_solve(params)
        safe_actions.append(y_sol)
    safe_actions = np.stack(safe_actions, axis=0)
    return torch.tensor(safe_actions, dtype=action.dtype, device=action.device)