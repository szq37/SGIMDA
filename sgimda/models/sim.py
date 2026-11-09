import torch
from torch import nn

class ConcatenationMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, mirna, disease):
        return self.layers(torch.cat((mirna, disease), dim=1)).squeeze(-1)

class StructuredInteractionModule(nn.Module):

    def __init__(self, input_dim: int = 96, hidden_dim: int = 128,
                 interaction_dim: int = 16):
        super().__init__()
        if input_dim <= 0 or input_dim % 3:
            raise ValueError("input_dim must be positive and divisible by three")
        self.input_dim = int(input_dim)
        self.stage_dim = self.input_dim // 3
        self.interaction_dim = int(interaction_dim)
        self.implicit = ConcatenationMLP(self.input_dim, hidden_dim)
        self.m_projection = nn.Linear(self.stage_dim, self.interaction_dim, bias=False)
        self.d_projection = nn.Linear(self.stage_dim, self.interaction_dim, bias=False)
        self.m_norm = nn.LayerNorm(self.interaction_dim)
        self.d_norm = nn.LayerNorm(self.interaction_dim)
        self.explicit_score = nn.Linear(3 * self.interaction_dim, 1)
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _mean(items):
        return torch.stack(items, dim=1).mean(dim=1)

    def _project(self, value, projection, normalization):
        stages = value.reshape(value.shape[0], 3, self.stage_dim)
        return normalization(projection(stages))

    def explicit_features(self, mirna, disease):
        if mirna.ndim != 2 or mirna.shape != disease.shape or mirna.shape[1] != self.input_dim:
            raise ValueError("SIM inputs must have matching [batch, input_dim] shapes")
        m = self._project(mirna, self.m_projection, self.m_norm)
        d = self._project(disease, self.d_projection, self.d_norm)
        same_product = self._mean([m[:, i] * d[:, i] for i in range(3)])
        same_difference = self._mean([torch.abs(m[:, i] - d[:, i]) for i in range(3)])
        cross_product = self._mean(
            [m[:, i] * d[:, j] for i in range(3) for j in range(3) if i != j]
        )
        return torch.cat((same_product, same_difference, cross_product), dim=1)

    def forward(self, mirna, disease):
        implicit = self.implicit(mirna, disease)
        explicit = self.explicit_score(self.explicit_features(mirna, disease)).squeeze(-1)
        return implicit + torch.sigmoid(self.residual_logit) * explicit

    def stabilize_residual_start(self):
        nn.init.zeros_(self.explicit_score.weight)
        nn.init.zeros_(self.explicit_score.bias)

    def diagnostics(self):
        return {"residual_scale": float(torch.sigmoid(self.residual_logit).detach().cpu())}
