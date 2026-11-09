import hashlib

import torch

from ..backbone.model import AdaptiveWeightLoss, GPASS
from ..backbone.preprocess import BioData, Config as BackboneConfig
from ..config import SGIMDAConfig
from ..data import FoldInputs
from ..utils import set_runtime_seed
from .dam import install_dam
from .perception import arm_linear_perception_after_reset
from .sgm import SourceGroupModeling
from .sim import StructuredInteractionModule

class SGIMDA(GPASS):
    pass

def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def _build_decoder(model, config: SGIMDAConfig, device):
    decoder = StructuredInteractionModule(
        input_dim=model.emb_dim,
        hidden_dim=config.decoder_hidden_dim,
        interaction_dim=config.interaction_dim,
    ).to(device)
    for layer in decoder.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
    decoder.stabilize_residual_start()
    return decoder

def build_sgimda(
    inputs: FoldInputs,
    dataset: str,
    device: str,
    seed: int,
    config: SGIMDAConfig,
):

    config.validate()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch_device = torch.device(device)
    set_runtime_seed(seed)

    backbone = BackboneConfig(dataset)
    backbone.gnn_layer = config.propagation_layers
    backbone.n_hidden = config.hidden_dim
    backbone.edge_emb_layer = config.dam_blocks
    backbone.ncf_hidden = config.decoder_hidden_dim
    backbone.dropout = config.dropout
    backbone.TopK_ss = config.topk_similarity
    backbone.lr = config.learning_rate
    backbone.weight_decay = config.weight_decay
    backbone.epochs = config.epochs
    backbone.patience = config.patience
    backbone.Dynamic_train = False
    backbone.device = torch_device

    bio = BioData(
        inputs.train_association,
        mss_list=list(inputs.mirna_sources),
        dss_list=list(inputs.disease_sources),
        device=torch_device,
        data_name=dataset,
        config=backbone,
    )
    records, sgm_audit = SourceGroupModeling(
        group_count=config.group_count, seed=42
    ).apply(bio, inputs.mirna_sources, inputs.disease_sources)

    model = SGIMDA(bio.config).to(torch_device)
    dam_audit = install_dam(model)
    arm_linear_perception_after_reset(model)

    with torch.no_grad():
        representation = model(bio.graph, bio.mss_graph, bio.dss_graph)
        model.predict(
            representation[:1],
            representation[bio.m_numb:bio.m_numb + 1],
        )
    model.reset_parameters()
    model.decoder = _build_decoder(model, config, torch_device)

    if any(isinstance(module, AdaptiveWeightLoss) for module in model.modules()):
        raise RuntimeError("adaptive loss is not part of the released SGI-MDA model")
    audit = {
        "model": "SGI-MDA",
        "sgm": sgm_audit,
        "dam": dam_audit,
        "perception": "linear_only",
        "sim": {
            "module": "SIM",
            "input_layout": "H0||H1||H2",
            "representation_dim": model.emb_dim,
            "interaction_dim": config.interaction_dim,
            "same_stage": "mean(product)||mean(abs_difference)",
            "cross_stage": "mean(six_ordered_products)",
        },
        "loss": "0.5*BCEWithLogits+0.5*BPR",
        "main_graph_nodes": int(bio.graph.num_nodes()),
        "main_graph_edges": int(bio.graph.num_edges()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_state_sha256": state_sha256(model),
    }
    return model, bio, records, audit
