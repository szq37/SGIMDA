from typing import Dict, Tuple

import torch
from torch.nn import functional as F

def sgi_mda_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:

    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("matching one-dimensional logits and labels are required")
    if not torch.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("nonfinite logits or nonbinary labels")
    positive = logits[labels == 1]
    negative = logits[labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("both positive and negative samples are required")
    if negative.numel() >= positive.numel():
        index = torch.randperm(negative.numel(), device="cpu")[: positive.numel()]
    else:
        index = torch.randint(negative.numel(), (positive.numel(),), device="cpu")
    index = index.to(negative.device)
    bpr = F.softplus(negative[index] - positive).mean()
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    total = 0.5 * bce + 0.5 * bpr
    return total, {
        "bce": float(bce.detach()),
        "bpr": float(bpr.detach()),
        "total": float(total.detach()),
    }
