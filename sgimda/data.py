from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PairPartition = Tuple[np.ndarray, np.ndarray]

@dataclass
class FoldInputs:
    association: np.ndarray
    train_association: np.ndarray
    mirna_sources: Tuple[np.ndarray, np.ndarray, np.ndarray]
    disease_sources: Tuple[np.ndarray, np.ndarray, np.ndarray]
    partitions: Dict[str, PairPartition]

def gaussian_interaction_profile(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    squared = np.sum(matrix * matrix, axis=1)
    gamma = 1.0 / max(float(squared.mean()), 1e-12)
    distance = np.maximum(
        squared[:, None] + squared[None, :] - 2.0 * (matrix @ matrix.T), 0.0
    )
    return np.exp(-gamma * distance).astype(np.float32)

def validate_split(
    association: np.ndarray, split: np.lib.npyio.NpzFile, protocol: str
) -> Dict[str, PairPartition]:
    association = np.asarray(association)
    if association.ndim != 2 or not np.isin(association, [0, 1]).all():
        raise ValueError("association matrix must be binary and rank-2")
    if protocol not in {"random", "cold_disease", "cold_mirna"}:
        raise ValueError("unknown evaluation protocol: " + protocol)

    pair_sets, entity_sets, partitions = {}, {}, {}
    for part in ("train", "val", "test"):
        arrays = []
        for suffix, label in (("pos", 1), ("neg", 0)):
            name = f"{part}_{suffix}"
            pairs = np.asarray(split[name])
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f"{name} must have shape [N, 2]")
            if not np.issubdtype(pairs.dtype, np.integer) or len(pairs) == 0:
                raise ValueError(f"{name} must be a nonempty integer array")
            if (
                (pairs < 0).any()
                or (pairs[:, 0] >= association.shape[0]).any()
                or (pairs[:, 1] >= association.shape[1]).any()
            ):
                raise ValueError(f"out-of-range indices in {name}")
            if not (association[pairs[:, 0], pairs[:, 1]] == label).all():
                raise ValueError(f"labels disagree with association matrix in {name}")
            arrays.append(pairs.astype(np.int64, copy=True))

        positive, negative = arrays
        if len(positive) != len(negative):
            raise ValueError("SGI-MDA expects a 1:1 positive/negative ratio")
        pairs = np.concatenate((positive, negative), axis=0)
        flat = pairs[:, 0] * association.shape[1] + pairs[:, 1]
        if len(set(flat.tolist())) != len(flat):
            raise ValueError(f"duplicate pairs in {part}")
        pair_sets[part] = set(flat.tolist())
        axis = 0 if protocol == "cold_mirna" else 1
        entity_sets[part] = set(pairs[:, axis].tolist())
        labels = np.concatenate(
            (np.ones(len(positive), dtype=np.int64),
             np.zeros(len(negative), dtype=np.int64))
        )
        partitions[part] = (pairs, labels)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if pair_sets[left] & pair_sets[right]:
            raise ValueError(f"pair leakage between {left} and {right}")
        if protocol != "random" and entity_sets[left] & entity_sets[right]:
            raise ValueError(f"cold-entity leakage between {left} and {right}")
    return partitions

def load_fold_inputs(
    data_root: str,
    splits_root: str,
    dataset: str,
    protocol: str,
    fold: int,
) -> FoldInputs:
    data_dir = Path(data_root) / dataset
    association = np.loadtxt(data_dir / "m_d.csv", delimiter=",", dtype=np.uint8)
    mirna_ss = np.loadtxt(data_dir / "m_ss.csv", delimiter=",", dtype=np.float32)
    disease_ss = np.loadtxt(data_dir / "d_ss.csv", delimiter=",", dtype=np.float32)
    if mirna_ss.shape != (association.shape[0], association.shape[0]):
        raise ValueError("miRNA SS shape mismatch")
    if disease_ss.shape != (association.shape[1], association.shape[1]):
        raise ValueError("disease SS shape mismatch")
    if not np.isfinite(mirna_ss).all() or not np.isfinite(disease_ss).all():
        raise ValueError("similarity matrices contain nonfinite values")

    split_path = Path(splits_root) / dataset / protocol / f"fold_{fold}.npz"
    with np.load(split_path, allow_pickle=False) as split:
        partitions = validate_split(association, split, protocol)
        train_positive = np.asarray(split["train_pos"], dtype=np.int64).copy()

    train_association = np.zeros_like(association)
    train_association[train_positive[:, 0], train_positive[:, 1]] = 1
    mirna_fs = cosine_similarity(train_association.astype(np.float32)).astype(np.float32)
    disease_fs = cosine_similarity(train_association.T.astype(np.float32)).astype(np.float32)
    mirna_gip = gaussian_interaction_profile(train_association)
    disease_gip = gaussian_interaction_profile(train_association.T)
    return FoldInputs(
        association=association,
        train_association=train_association,
        mirna_sources=(mirna_ss, mirna_fs, mirna_gip),
        disease_sources=(disease_ss, disease_fs, disease_gip),
        partitions=partitions,
    )
