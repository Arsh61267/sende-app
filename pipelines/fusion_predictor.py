import torch
import torch.nn as nn


class FusionMLP(nn.Module):

    def __init__(self, input_dim=2048, hidden_dims=[1024, 256], out_dim=10):
        super().__init__()

        layers = []
        d = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h

        layers.append(nn.Linear(d, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
