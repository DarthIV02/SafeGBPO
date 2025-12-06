import torch
import torch.nn as nn

class FSNetDistance(nn.Module):
    """
    use FSNet instead of CVXPY layer to solve constrained optimization

    max d
    sub.to center + d * direction \in safe action set

    how to train
    1. use dataset collected with CVXPY layer to train FSNet as function approximator
    2. tuning FSNet with, validation(zonotope.validation) and FSNet will learn 
        to predict d with constrains.
    """
    ## just a simple MLP
    def __init__(self, in_dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim

        layers.append(nn.Linear(d, 1))
        layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
