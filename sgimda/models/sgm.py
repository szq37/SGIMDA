from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SOURCE_NAMES: Tuple[str, str, str] = ("SS", "FS", "GIP")

def _partition(matrix: np.ndarray, groups: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity source must be a square matrix")
    if not np.isfinite(matrix).all() or groups < 1:
        raise ValueError("invalid similarity source or group count")
    standardized = StandardScaler().fit_transform(matrix)
    effective = min(groups, len(standardized), len(np.unique(standardized, axis=0)))
    if effective < 1:
        raise ValueError("similarity source does not contain a clusterable entity")
    fitted = KMeans(n_clusters=effective, n_init=100, random_state=seed).fit(standardized)
    distance = np.linalg.norm(
        standardized - fitted.cluster_centers_[fitted.labels_], axis=1
    )
    return fitted.labels_.astype(np.int64), distance.astype(np.float64)

def build_source_groups(
    association: np.ndarray,
    mirna_sources: Sequence[np.ndarray],
    disease_sources: Sequence[np.ndarray],
    group_count: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:

    association = np.asarray(association)
    if association.ndim != 2 or not np.isin(association, [0, 1]).all():
        raise ValueError("association must be a binary matrix")
    n_mirna, n_disease = association.shape
    records: Dict[str, np.ndarray] = {}
    memberships = []
    offsets = {
        "miRNA": n_mirna + n_disease,
        "disease": n_mirna + n_disease + 3 * group_count,
    }
    for side, sources, node_offset, size in (
        ("miRNA", mirna_sources, 0, n_mirna),
        ("disease", disease_sources, n_mirna, n_disease),
    ):
        if len(sources) != 3 or any(np.asarray(x).shape != (size, size) for x in sources):
            raise ValueError("SGM expects SS, FS and GIP sources in this order")
        labels_by_source, distances_by_source = [], []
        for source_index, matrix in enumerate(sources):
            labels, distances = _partition(matrix, group_count, seed + source_index)
            for entity_index, label in enumerate(labels):
                memberships.append(
                    (node_offset + entity_index,
                     offsets[side] + source_index * group_count + int(label))
                )
            labels_by_source.append(labels)
            distances_by_source.append(distances)
        records[f"{side}_labels"] = np.stack(labels_by_source, axis=1)
        records[f"{side}_distance"] = np.stack(distances_by_source, axis=1)
        degree = association.sum(1 if side == "miRNA" else 0)
        records[f"{side}_train_degree"] = degree.astype(np.int64)

    membership = np.asarray(memberships, dtype=np.int64).reshape(-1, 2)
    positive = np.argwhere(association > 0)
    association_edges = np.column_stack(
        (positive[:, 0], positive[:, 1] + n_mirna)
    ).astype(np.int64)
    total_nodes = n_mirna + n_disease + 6 * group_count + 2
    loops = np.column_stack((np.arange(total_nodes), np.arange(total_nodes)))
    edges = np.concatenate(
        (association_edges, association_edges[:, ::-1], membership,
         membership[:, ::-1], loops)
    )
    edge_types = np.concatenate(
        (np.zeros(len(association_edges), dtype=np.int64),
         np.ones(len(association_edges), dtype=np.int64),
         np.full(len(membership), 2, dtype=np.int64),
         np.full(len(membership), 3, dtype=np.int64),
         np.full(total_nodes, 4, dtype=np.int64))
    )
    records.update(
        membership_edges=membership,
        association_edges=association_edges,
        source_names=np.asarray(SOURCE_NAMES),
        cluster_budget=np.asarray([group_count, group_count]),
        total_nodes=np.asarray(total_nodes),
    )
    return edges, edge_types, records

class SourceGroupModeling:

    def __init__(self, group_count: int = 2, seed: int = 42):
        self.group_count = int(group_count)
        self.seed = int(seed)

    def apply(self, bio, mirna_sources, disease_sources):
        import dgl
        import torch

        edges, edge_types, records = build_source_groups(
            bio.adj_matrix, mirna_sources, disease_sources,
            group_count=self.group_count, seed=self.seed
        )
        n_mirna, n_disease = bio.m_numb, bio.d_numb
        bio.cm_numb = 3 * self.group_count
        bio.cd_numb = 3 * self.group_count
        bio.bcm = n_mirna + n_disease
        bio.bcd = bio.bcm + bio.cm_numb
        bio.num_nodes = int(records["total_nodes"])
        bio.num_rels = 5
        bio.src = edges[:, 0].tolist()
        bio.dst = edges[:, 1].tolist()
        bio.etype = edge_types.tolist()
        bio.node = {
            0: ["miRNA", 0, n_mirna, n_mirna],
            1: ["disease", n_mirna, n_mirna + n_disease, n_disease],
            2: ["DimiRNA", bio.bcm, bio.bcd, bio.cm_numb],
            3: ["Didisease", bio.bcd, bio.num_nodes - 2, bio.cd_numb],
        }
        bio.graph = dgl.graph(
            (bio.src, bio.dst), num_nodes=bio.num_nodes
        ).to(bio.device)
        bio.graph.edata["type"] = torch.as_tensor(
            edge_types, dtype=torch.long, device=bio.device
        )
        bio.graph.edata["old_type"] = bio.graph.edata["type"].clone()

        def membership_matrix(side: str) -> torch.Tensor:
            labels = records[f"{side}_labels"]
            output = np.zeros((len(labels), 3 * self.group_count), dtype=np.float32)
            for source_index in range(3):
                rows = np.arange(len(labels))
                output[rows, source_index * self.group_count + labels[:, source_index]] = 1
            return torch.from_numpy(output)

        bio.adj_matrix_mc = membership_matrix("miRNA")
        bio.adj_matrix_dc = membership_matrix("disease")
        bio.mss_Di = None
        bio.dss_Di = None
        bio.config.get_data_params(bio)
        audit = {
            "module": "SGM",
            "sources": list(SOURCE_NAMES),
            "groups_per_source_and_side": self.group_count,
            "mirna_group_nodes": bio.cm_numb,
            "disease_group_nodes": bio.cd_numb,
            "membership_edges_one_direction": int(len(records["membership_edges"])),
        }
        return records, audit
