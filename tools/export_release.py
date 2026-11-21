"""Maintainer tool: turn the E07-A ARC-Challenge training checkpoint into the Hugging Face model folder.

The training checkpoint is one fp32 safetensors file: the fine-tuned encoder under ``encoder.`` (canonical
ModernBERT parameter names) and the success head under ``head.``. This tool splits it, without changing any tensor,
into a standard transformers ModernBERT checkpoint (``config.json`` + ``model.safetensors``) and ``head.safetensors``,
copies Laya's tokenizer files, and writes ``router_config.json`` with the candidate order, input view, selection
policy and SHA-256 of every exported file. It needs the private training checkpoint and its run record, and the
pinned Laya snapshot (for the encoder configuration and the tokenizer).
"""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "e333cb7c823cd1787276228260f56d02fbf5180b"
LAYA_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
HEAD_KEYS = {f"layers.{i}.{part}" for i in (0, 1, 4) for part in ("weight", "bias")}
EXPORTED = ("config.json", "model.safetensors", "head.safetensors", "tokenizer.json", "tokenizer_config.json")


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def read_checkpoint(path, record):
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    meta = json.loads(metadata["e07"])
    checks = {
        "variant": metadata.get("variant") == "full" == meta["deep_variant"] == record["deep_variant"],
        "arm": meta["arm"] == "D423" == record["arm"],
        "held_out": meta["split_id"] == "arcc" == record["split_id"],
        "run_hash": meta["run_hash"] == record["run_hash"],
        "models": meta["models"] == record["config"]["models"] and len(set(meta["models"])) == 20,
        "threshold": meta["threshold"] == record["threshold"],
        "fallback": meta["best_model_index"] == record["best_model_index"],
        "base": meta["encoder"]["revision"] == LAYA_REVISION and meta["encoder"]["weights_sha256"] == LAYA_SHA256,
        "file_sha256": sha256(path) == record["checkpoint"]["sha256"],
        "file_bytes": Path(path).stat().st_size == record["checkpoint"]["bytes"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Checkpoint does not match the run record: {failed}")
    return meta


def export(checkpoint, record_path, laya_snapshot, destination):
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    meta = read_checkpoint(checkpoint, record)
    tensors = load_file(str(checkpoint), device="cpu")
    encoder = {k.removeprefix("encoder."): v.contiguous() for k, v in tensors.items() if k.startswith("encoder.")}
    head = {k.removeprefix("head."): v.contiguous() for k, v in tensors.items() if k.startswith("head.")}
    if len(encoder) + len(head) != len(tensors) or set(head) != HEAD_KEYS:
        raise ValueError("Unexpected checkpoint layout")
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Expected finite fp32 tensors")
    n_encoder = sum(t.numel() for t in encoder.values())
    n_head = sum(t.numel() for t in head.values())
    if (n_encoder, n_head) != (record["parameters"]["encoder"], record["parameters"]["head"]):
        raise ValueError("Parameter counts differ from the run record")

    snapshot = Path(laya_snapshot)
    if snapshot.name != LAYA_REVISION:
        raise ValueError(f"Expected the Laya snapshot folder of revision {LAYA_REVISION}")
    if (snapshot / "model.safetensors").exists() and sha256(snapshot / "model.safetensors") != LAYA_SHA256:
        raise ValueError("Laya base checkpoint checksum mismatch")
    config = json.loads((snapshot / "encoder/config.json").read_text(encoding="utf-8"))
    if config["model_type"] != "modernbert" or config.get("dtype") != "float32":
        raise ValueError("Unexpected Laya encoder configuration")
    config["architectures"] = ["ModernBertModel"]

    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    save_file(encoder, str(out / "model.safetensors"), metadata={"format": "pt"})
    save_file(head, str(out / "head.safetensors"), metadata={"format": "pt"})
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(snapshot / "tokenizer" / name, out / name)

    threshold = float(meta["threshold"])
    settings = record["config"]["settings"]
    cfg = {
        "format_version": 2,
        "release": "2.0.0",
        "models": meta["models"],
        "input": {"head_tokens": 211, "tail_tokens": 212, "max_content_tokens": 423,
                  "template": "[CLS] content [SEP]; the mask token is replaced by a space; no other special tokens"},
        "representation": "CLS vector after the encoder's final_norm",
        "head": {"layers": "LayerNorm(1024) -> Linear(1024, 128) -> GELU -> Dropout(0.1) -> Linear(128, 20)",
                 "width": 128, "dropout": 0.1, "output": "sigmoid"},
        "policy": {"default": "raw", "fallback_model_index": int(meta["best_model_index"]),
                   "fallback_model": meta["models"][int(meta["best_model_index"])],
                   "fallback_margin": None if math.isinf(threshold) else threshold,
                   "rule": "gated: select the fallback model when p(top) - p(fallback) < fallback_margin",
                   "margin_grid": ["always_fallback", 0.2, 0.1, 0.05, 0.025, 0.01, 0.0],
                   "margin_selection": "highest routed accuracy on the validation rows"},
        "training": {"held_out_dataset": "arcc", "fit_queries": record["n_fit"],
                     "validation_queries": record["n_validation"], "selected_epoch": record["best_epoch"],
                     "epochs_run": record["epochs_run"], "encoder_lr": settings["encoder_lr"],
                     "head_lr": settings["head_lr"], "weight_decay": settings["weight_decay"],
                     "logical_batch": settings["logical_batch"], "max_epochs": settings["max_epochs"],
                     "precision": settings["encoder_precision"], "gpu": record["gpu"],
                     "trainable_parameters": record["parameters"]["trainable"],
                     "min_validation_bce": record["min_val_bce"]},
        "base_model": LAYA_REPO,
        "base_revision": LAYA_REVISION,
        "base_sha256": LAYA_SHA256,
        "training_checkpoint_sha256": record["checkpoint"]["sha256"],
        "files": {name: sha256(out / name) for name in EXPORTED},
    }
    (out / "router_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                            encoding="utf-8", newline="\n")
    print(f"Exported {n_encoder:,} encoder and {n_head:,} head parameters to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="E07-A training checkpoint (D423, ARC-Challenge held out)")
    parser.add_argument("--run-record", required=True, type=Path, help="JSON run record of that checkpoint")
    parser.add_argument("--laya-snapshot", required=True, type=Path,
                        help=f"local snapshot folder of {LAYA_REPO} at revision {LAYA_REVISION}")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.run_record, args.laya_snapshot, args.output_dir)
