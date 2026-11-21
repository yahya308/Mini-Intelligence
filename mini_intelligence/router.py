"""Mini Intelligence v2: a fully fine-tuned ModernBERT encoder and a success head over a fixed pool of 20 models."""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

# Hugging Face model repository and the exact commit this package release was verified against.
HF_REPO = "Yahya308/Mini-Intelligence"
HF_REVISION = None  # set to the 40-character commit hash of the v2.0.0 upload
CONFIG_PATH = Path(__file__).resolve().parent / "router_config.json"
FILES = ("router_config.json", "config.json", "model.safetensors", "head.safetensors",
         "tokenizer.json", "tokenizer_config.json")
PRECISIONS = ("float32", "bfloat16")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SuccessHead(nn.Module):
    """LayerNorm -> Linear(hidden, 128) -> GELU -> Dropout -> Linear(128, n_models); one logit per candidate."""

    def __init__(self, hidden_size, n_models, width=128, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width, n_models))

    def forward(self, pooled):
        return self.layers(pooled)


class FP32MLP(nn.Module):
    """Run a wrapped MLP in float32 with autocast disabled (bfloat16 mode keeps the final MLP in float32,
    as during training)."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden):
        with torch.autocast(hidden.device.type, enabled=False):
            return self.mlp(hidden.float())


def head_tail(ids, head, tail):
    return ids if len(ids) <= head + tail else ids[:head] + ids[-tail:]


def resolve_model_dir(model_dir, revision, local_files_only):
    if model_dir is not None:
        return Path(model_dir)
    revision = revision or HF_REVISION
    if revision is None:
        raise ValueError("No pinned Hugging Face revision in this build; pass revision=... or model_dir=...")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, revision=revision, local_files_only=local_files_only,
                                  allow_patterns=list(FILES)))


class MiniIntelligence:
    """Load once, then call predict (or route) for one prompt, or predict_batch for several prompts.

    policy="raw" (default) selects the candidate with the highest predicted success.
    policy="gated" keeps the fallback model unless the top candidate beats it by the saved validation margin.
    precision="float32" (default) runs on CPU or CUDA; precision="bfloat16" (CUDA only) uses bfloat16 autocast
    with float32 weights and a float32 final MLP, the precision used during training.
    """

    def __init__(self, device=None, *, local_files_only=False, model_dir=None, revision=None,
                 precision="float32"):
        from transformers import AutoTokenizer, ModernBertModel

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are CPU and CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device unavailable")
        if precision == "bfloat16" and not (self.device.type == "cuda" and torch.cuda.is_bf16_supported()):
            raise ValueError("bfloat16 needs a CUDA GPU with bfloat16 support")
        self.precision = precision

        expected = CONFIG_PATH.read_bytes()
        self.config = cfg = json.loads(expected)
        if cfg["format_version"] != 2:
            raise ValueError("Unsupported router configuration format")
        folder = resolve_model_dir(model_dir, revision, local_files_only)
        if (folder / "router_config.json").read_bytes().replace(b"\r\n", b"\n") != expected.replace(b"\r\n", b"\n"):
            raise ValueError("router_config.json differs from the configuration bundled with this package")
        for name, digest in cfg["files"].items():
            if file_hash(folder / name) != digest:
                raise ValueError(f"Checksum mismatch for {name}")
        self.model_dir = folder

        self.models = tuple(cfg["models"])
        view = cfg["input"]
        self.head_tokens, self.tail_tokens = view["head_tokens"], view["tail_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        self.encoder, info = ModernBertModel.from_pretrained(folder, dtype=torch.float32, attn_implementation="sdpa",
                                                             local_files_only=True, output_loading_info=True)
        if any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Encoder weights do not match the ModernBERT layout: {info}")
        if precision == "bfloat16":
            last = self.encoder.layers[-1]
            last.mlp = FP32MLP(last.mlp)
        self.head = SuccessHead(self.encoder.config.hidden_size, len(self.models), cfg["head"]["width"],
                                cfg["head"]["dropout"])
        self.head.load_state_dict(load_file(str(folder / "head.safetensors")), strict=True)
        self.encoder.to(self.device).eval().requires_grad_(False)
        self.head.to(self.device).eval().requires_grad_(False)

    def predict(self, prompt, *, policy="raw"):
        return self.predict_batch([prompt], policy=policy)[0]

    route = predict

    def encode(self, prompts):
        """Encoder inputs: [CLS] + head+tail view of the content tokens + [SEP], one list of token IDs per prompt."""
        tok = self.tokenizer
        raw = tok([text.replace(tok.mask_token, " ") for text in prompts], add_special_tokens=False)["input_ids"]
        return [[tok.cls_token_id, *head_tail(ids, self.head_tokens, self.tail_tokens), tok.sep_token_id]
                for ids in raw]

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
        states = self.encode(prompts)
        pad = self.tokenizer.pad_token_id
        order = sorted(range(len(states)), key=lambda i: len(states[i]))
        output = [None] * len(states)
        rule = self.config["policy"]
        default, threshold = rule["fallback_model_index"], rule["fallback_margin"]
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            width = max(len(states[i]) for i in indices)
            ids = torch.full((len(indices), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, i in enumerate(indices):
                ids[row, :len(states[i])] = torch.tensor(states[i])
                mask[row, :len(states[i])] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bfloat16" else nullcontext()
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


# Backward-compatible alias for the v1 name of this router.
E06Router = MiniIntelligence
