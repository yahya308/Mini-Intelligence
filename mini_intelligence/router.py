"""Mini Intelligence: apply the released low-rank adapter and predict success for its fixed candidate pool."""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import laya
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AdaptedMLP(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.original = original
        self.down = nn.Linear(original.Wo.in_features, rank, bias=False)
        self.up = nn.Linear(rank, original.Wo.out_features, bias=False)
        self.scale = alpha / rank

    def forward(self, normalized):
        # The final MLP is position-wise. The router only reads the final CLS.
        with torch.autocast(normalized.device.type, enabled=False):
            value, gate = self.original.Wi(normalized.float()).chunk(2, dim=-1)
            z = self.original.drop(self.original.act(value) * gate)
            out = self.original.Wo(z)
            out[:, 0] += self.up(self.down(z[:, 0])) * self.scale
            return out


class MiniIntelligence:
    """Load once, then call predict for one prompt or predict_batch for multiple prompts.

    policy="raw" (default) selects the candidate with the highest predicted success.
    policy="gated" keeps the fallback model unless the top candidate beats it by the saved validation margin.
    """

    def __init__(self, device=None, *, local_files_only=False, weights_dir=None):
        folder = Path(weights_dir) if weights_dir else Path(__file__).parent / "weights"
        self.config = json.loads((folder / "router_config.json").read_text(encoding="utf-8"))
        cfg = self.config
        if cfg["format_version"] != 1 or laya.__version__ != cfg["laya_version"]:
            raise ValueError("Unsupported adapter format or Laya runtime version")
        weights = folder / "adapter.safetensors"
        if file_hash(weights) != cfg["adapter_sha256"]:
            raise ValueError("Adapter checksum mismatch")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are CPU and CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device unavailable")
        snapshot = Path(snapshot_download(
            cfg["base_model"], revision=cfg["base_revision"], local_files_only=local_files_only,
            allow_patterns=["model.safetensors", "rl_agent_config.json", "tokenizer/*", "encoder/*"],
        ))
        if file_hash(snapshot / "model.safetensors") != cfg["base_sha256"]:
            raise ValueError("Base Laya checkpoint checksum mismatch")
        agent = laya.load(str(snapshot), device=str(self.device), fast=False)
        self.tokenizer, self.encoder = agent.tok, agent.model.encoder
        self.encoder.requires_grad_(False)
        self.models = tuple(cfg["models"])
        layer = self.encoder.layers[-1]
        adapted = AdaptedMLP(layer.mlp, cfg["lora_rank"], cfg["lora_alpha"])
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, cfg["head_width"]), nn.GELU(),
                                  nn.Dropout(cfg["head_dropout"]), nn.Linear(cfg["head_width"], len(self.models)))
        state = load_file(str(weights))
        adapted.down.load_state_dict({"weight": state.pop("adapter_down.weight")})
        adapted.up.load_state_dict({"weight": state.pop("adapter_up.weight")})
        if any(not key.startswith("head.layers.") for key in state):
            raise ValueError("Unexpected tensor in released adapter")
        self.head.load_state_dict({k.removeprefix("head.layers."): v for k, v in state.items()})
        layer.mlp = adapted
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.head.to(self.device).eval().requires_grad_(False)
        self.bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    def predict(self, prompt, *, policy="raw"):
        return self.predict_batch([prompt], policy=policy)[0]

    @torch.inference_mode()
    def predict_batch(self, prompts, batch_size=8, *, policy="raw"):
        if policy not in ("gated", "raw"):
            raise ValueError("Supported policies are 'raw' and 'gated'")
        if isinstance(prompts, str) or batch_size < 1:
            raise ValueError("Pass an iterable of prompt strings and a positive batch_size")
        prompts = list(prompts)
        if any(not isinstance(text, str) for text in prompts):
            raise TypeError("Each prompt must be text")
        if not prompts:
            return []
        tok = self.tokenizer
        raw = tok([text.replace(tok.mask_token, " ") for text in prompts],
                  add_special_tokens=False)["input_ids"]
        states = [[tok.cls_token_id, *(ids if len(ids) <= 423 else ids[:211] + ids[-212:]), tok.sep_token_id]
                  for ids in raw]
        order = sorted(range(len(states)), key=lambda i: len(states[i]))
        output = [None] * len(states)
        default = self.config["fallback_model_index"]
        threshold = self.config["fallback_margin"]
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            width = max(len(states[i]) for i in indices)
            ids = torch.full((len(indices), width), tok.pad_token_id, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for row, i in enumerate(indices):
                n = len(states[i])
                ids[row, :n] = torch.tensor(states[i], device=self.device)
                mask[row, :n] = 1
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else nullcontext()
            with amp:
                cls = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            probs = self.head(cls.float()).sigmoid().cpu()
            for i, scores in zip(indices, probs):
                top = int(scores.argmax())
                fallback = policy == "gated" and (threshold is None or float(scores[top] - scores[default]) < threshold)
                selected = default if fallback else top
                output[i] = {"selected_model": self.models[selected], "top_model": self.models[top],
                             "used_fallback": fallback, "probabilities": dict(zip(self.models, scores.tolist()))}
        return output


# Backward-compatible alias for the earlier name of this router.
E06Router = MiniIntelligence
