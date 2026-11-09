import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import SGIMDAConfig
from .data import load_fold_inputs
from .losses import sgi_mda_loss
from .metrics import classification_metrics, select_f1_threshold
from .models.sgimda import build_sgimda
from .utils import write_json

def score_pairs(model, bio, pairs, device, batch_size=8192, return_logits=False):
    model.eval()
    probabilities, logits = [], []
    with torch.no_grad():
        representation = model(bio.graph, bio.mss_graph, bio.dss_graph)
        if not torch.isfinite(representation).all():
            raise RuntimeError("encoder produced nonfinite values")
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            mirna = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
            disease = torch.as_tensor(
                batch[:, 1] + bio.m_numb, dtype=torch.long, device=device
            )
            raw = model.decoder(representation[mirna], representation[disease])
            if not torch.isfinite(raw).all():
                raise RuntimeError("decoder produced nonfinite values")
            logits.append(raw.cpu().numpy())
            probabilities.append(torch.sigmoid(raw).cpu().numpy())
    scores = np.concatenate(probabilities)
    raw_scores = np.concatenate(logits)
    return (scores, raw_scores) if return_logits else scores

def train_fold(data_root, splits_root, output_root, dataset, protocol, fold,
               device, config: SGIMDAConfig):
    output = Path(output_root) / dataset / protocol / f"fold_{fold}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output is not empty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "RUNNING.json", {"started": time.time()})
    started = time.time()
    try:
        inputs = load_fold_inputs(data_root, splits_root, dataset, protocol, fold)
        seed = config.seed + fold
        model, bio, groups, audit = build_sgimda(
            inputs, dataset, device, seed, config
        )
        np.savez_compressed(output / "groups.npz", **groups)
        write_json(output / "module_audit.json", audit)
        write_json(output / "config.json", {
            **config.to_dict(), "dataset": dataset, "protocol": protocol,
            "fold": fold, "seed_for_fold": seed,
        })

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        train_pairs, train_labels = inputs.partitions["train"]
        val_pairs, val_labels = inputs.partitions["val"]
        test_pairs, test_labels = inputs.partitions["test"]
        train_m = torch.as_tensor(train_pairs[:, 0], dtype=torch.long, device=device)
        train_d = torch.as_tensor(
            train_pairs[:, 1] + bio.m_numb, dtype=torch.long, device=device
        )
        targets = torch.as_tensor(train_labels, dtype=torch.float32, device=device)

        history, best_auc, best_epoch, stale = [], -1.0, -1, 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            representation = model(bio.graph, bio.mss_graph, bio.dss_graph)
            logits = model.decoder(representation[train_m], representation[train_d])
            loss, details = sgi_mda_loss(logits, targets)
            loss.backward()
            optimizer.step()
            if epoch % config.eval_every == 0 or epoch == config.epochs:
                val_scores = score_pairs(
                    model, bio, val_pairs, device, config.eval_batch_size
                )
                val_auc = float(roc_auc_score(val_labels, val_scores))
                history.append({
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_aupr": float(average_precision_score(val_labels, val_scores)),
                    **details,
                })
                if val_auc > best_auc + 1e-8:
                    best_auc, best_epoch, stale = val_auc, epoch, 0
                    torch.save(model.state_dict(), output / "best_model.pt")
                else:
                    stale += config.eval_every
                print(
                    f"epoch={epoch:03d} loss={float(loss):.6f} "
                    f"val_auc={val_auc:.6f} best_epoch={best_epoch}", flush=True
                )
                if stale >= config.patience:
                    break

        model.load_state_dict(torch.load(
            output / "best_model.pt", map_location=device, weights_only=True
        ))
        val_scores, val_logits = score_pairs(
            model, bio, val_pairs, device, config.eval_batch_size, True
        )
        test_scores, test_logits = score_pairs(
            model, bio, test_pairs, device, config.eval_batch_size, True
        )
        threshold = select_f1_threshold(val_labels, val_scores)
        np.savez_compressed(
            output / "val_predictions.npz", pairs=val_pairs,
            labels=val_labels, scores=val_scores, logits=val_logits
        )
        np.savez_compressed(
            output / "test_predictions.npz", pairs=test_pairs,
            labels=test_labels, scores=test_scores, logits=test_logits
        )
        write_json(output / "history.json", history)
        result = {
            "model": "SGI-MDA", "dataset": dataset, "protocol": protocol,
            "fold": fold, "seed": seed, "best_epoch": best_epoch,
            "selection_val_auc": best_auc,
            "restored_val_auc": float(roc_auc_score(val_labels, val_scores)),
            "threshold": threshold,
            "metrics": classification_metrics(test_labels, test_scores, threshold),
            "module_audit": audit,
            "seconds": round(time.time() - started, 2),
        }
        write_json(output / "result.json", result)
        (output / "RUNNING.json").unlink()
        print("FINAL", json.dumps(result["metrics"]), flush=True)
        return result
    except Exception as error:
        write_json(output / "FAILED.json", {"error": repr(error), "time": time.time()})
        raise
