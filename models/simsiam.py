import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimSiam(nn.Module):
    def __init__(self, encoder, projection_dim=256):
        super().__init__()
        self.encoder = encoder
        self.projector = MLPHead(in_dim=encoder.output_dim, out_dim=projection_dim)
        self.predictor = MLPHead(in_dim=projection_dim, out_dim=projection_dim)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

def D(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def simsiam_loss(p1, z2, p2, z1):
    return 0.5 * (D(p1, z2) + D(p2, z1))
