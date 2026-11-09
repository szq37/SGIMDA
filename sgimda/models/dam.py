import torch
from dgl.nn.pytorch import GATv2Conv
from torch import nn

class DynamicAttentionModule(nn.Module):

    def __init__(self, in_features: int, out_features: int, layers: int, heads: int):
        super().__init__()
        self.attentions = nn.ModuleList(
            GATv2Conv(
                in_features,
                out_features,
                num_heads=heads,
                allow_zero_in_degree=True,
                share_weights=True,
            )
            for _ in range(layers)
        )
        self.num_heads = heads

    def forward(self, features, graph):
        with graph.local_scope():
            hidden = features
            for attention in self.attentions:
                hidden = torch.sum(attention(graph, hidden), dim=1)
                graph.ndata["h"] = hidden
            return hidden

def install_dam(model):

    blocks = list(model.layers)
    if not blocks:
        raise ValueError("model has no propagation blocks")
    previous = blocks[0].GNN_embedding_Model
    if not all(block.GNN_embedding_Model is previous for block in blocks):
        raise ValueError("expected one encoder shared by all DAM blocks")
    first = previous.attentions[0]
    dam = DynamicAttentionModule(
        first._in_src_feats,
        first._out_feats,
        layers=len(previous.attentions),
        heads=previous.num_heads,
    ).to(model.emb.device)
    for block in blocks:
        block.GNN_embedding_Model = dam
    return {
        "module": "DAM",
        "blocks": len(blocks),
        "propagation_layers_per_block": len(dam.attentions),
        "attention_heads": previous.num_heads,
        "shared_across_blocks": True,
        "head_reduction": "sum",
    }
