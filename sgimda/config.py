from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclass(frozen=True)
class SGIMDAConfig:

    group_count: int = 2
    propagation_layers: int = 3
    hidden_dim: int = 32
    interaction_dim: int = 16
    attention_heads: int = 4
    dam_blocks: int = 2
    decoder_hidden_dim: int = 128
    dropout: float = 0.2
    topk_similarity: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 200
    patience: int = 20
    eval_every: int = 1
    eval_batch_size: int = 8192
    seed: int = 42

    def validate(self) -> "SGIMDAConfig":
        integer_fields = (
            self.group_count,
            self.propagation_layers,
            self.hidden_dim,
            self.interaction_dim,
            self.attention_heads,
            self.dam_blocks,
            self.decoder_hidden_dim,
            self.topk_similarity,
            self.epochs,
            self.patience,
            self.eval_every,
            self.eval_batch_size,
        )
        if min(integer_fields) < 1:
            raise ValueError("positive integer configuration values are required")
        if self.dam_blocks != 2:
            raise ValueError("the released SGI-MDA architecture uses two DAM blocks")
        if self.hidden_dim != 32:
            raise ValueError("the released SGI-MDA architecture uses hidden_dim=32")
        if self.group_count != 2 or self.propagation_layers != 3:
            raise ValueError("the released SGI-MDA architecture uses K=2 and L=3")
        if self.interaction_dim != 16 or self.attention_heads != 4:
            raise ValueError("the released SGI-MDA architecture uses i=16 and H=4")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.seed < 0:
            raise ValueError("invalid optimization configuration")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "SGIMDAConfig":
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError("configuration file must contain a mapping")
        return cls(**value).validate()
