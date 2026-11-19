"""Maintainer tool: convert the E06 B ARC-Challenge training checkpoint into public, tensors-only weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def export(checkpoint, destination):
    contents = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if contents["split_id"] != "arcc" or contents["arm"] != "B" or contents["input_view"] != "headtail":
        raise ValueError("Expected the preregistered arm B checkpoint trained with ARC-Challenge held out")
    names = list(contents["models"])
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("Expected the fixed pool of 20 distinct models")
    state = contents["state_dict"]
    selected = {k: v.contiguous() for k, v in state.items() if k.startswith(("head.layers.", "adapter_"))}
    expected = {"adapter_down.weight", "adapter_up.weight",
                *(f"head.layers.{i}.{part}" for i in (0, 1, 4) for part in ("weight", "bias"))}
    if set(selected) != expected or set(state) - set(selected) != {"final_norm.weight"}:
        raise ValueError("Unexpected checkpoint layout")
    if any(not torch.isfinite(tensor).all() for tensor in selected.values()):
        raise ValueError("Nonfinite adapter or head")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "adapter.safetensors"
    save_file(selected, str(output), metadata={"format": "pt", "arm": "B", "held_out": "arcc"})
    margin = float(contents["threshold"])
    cfg = dict(format_version=1, base_model="convaiinnovations/laya",
               base_revision=contents["laya_model_revision"], base_sha256=contents["encoder_weights_sha256"],
               laya_version=contents["laya_package"]["version"],
               laya_git=contents["laya_package"]["protocol_commit"], lora_rank=8,
               lora_alpha=16, head_width=128, head_dropout=0.1,
               fallback_margin=None if math.isinf(margin) else margin,
               fallback_model_index=int(contents["best_model_index"]), models=names,
               training_arm="B", held_out_dataset="arcc", selected_epoch=int(contents["epoch"]),
               input_tokens=423, head_tokens=211, tail_tokens=212,
               training_checkpoint_sha256=sha256(checkpoint), adapter_sha256=sha256(output))
    (destination / "router_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Exported {sum(v.numel() for v in selected.values()):,} learned parameters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "mini_intelligence/weights")
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)
