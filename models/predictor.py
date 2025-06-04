import torch
import torch.nn as nn

class MLPRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class ConcatMLP(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=128, task='classification'):
        super().__init__()
        input_dim = dim1 + dim2
        if task == 'classification':
            self.model = MLPClassifier(input_dim, hidden_dim)
        else:
            self.model = MLPRegression(input_dim, hidden_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.model(x)
