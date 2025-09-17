import torch
import torch.nn as nn
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.log_std = nn.Linear(hidden, latent_dim)

    def forward(self, s):
        h = self.net(s)
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        log_prob = dist.log_prob(z).sum(-1, keepdim=True)
        return z, log_prob, mu, std

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, latent_dim)
        self.log_std = nn.Linear(hidden, latent_dim)

    def forward(self, z, a):
        h = self.net(torch.cat([z, a], dim=-1))
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        return Normal(mu, std)

class Policy(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, z):
        h = self.net(z)
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        dist = Normal(mu, std)
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(-1, keepdim=True)
        return torch.tanh(a), log_prob

class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))
