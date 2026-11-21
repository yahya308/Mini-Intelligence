"""Check a Mini Intelligence v2 installation against the published ARC-Challenge reference probabilities.

The router downloads the pinned model revision from Hugging Face (or reads --model-dir) and checks the SHA-256 of
every file before loading. This tool then scores the 1,172 ARC-Challenge test prompts of LLMRouterBench and compares
the 20 probabilities per prompt with tools/reference/arc_challenge_v2.npz:

* ``fp32``: the package in float32 on an NVIDIA RTX 4070 Laptop GPU (the release reference);
* ``training_run_bf16``: the probabilities recorded by the training run (NVIDIA L40S, bfloat16 autocast).

The reference is keyed by the SHA-256 of each prompt's UTF-8 text; no prompt text is distributed. Provide the
prompts with --bench-dir (the LLMRouterBench result folder holding ``arcc/test/<model>/*.json``; if all 20 candidates
are present, routed accuracy is also computed from the recorded outcomes) or with --prompts (JSONL, one
{"prompt": ...} object per line). The check passes when every probability is within --tolerance of the fp32
reference (float32 runs) and every raw-choice difference is a near tie of the reference.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parent / "reference" / "arc_challenge_v2.npz"
ARC = "arcc"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_bench(folder, models):
    """Prompts (and outcomes, when all candidates are present) of the LLMRouterBench ARC-Challenge test split.

    For each candidate, the newest non-demo result file is used; every candidate must hold the same prompts."""
    base = Path(folder) / ARC / "test"
    prompts, outcomes = None, {}
    for model in models:
        files = sorted((base / model).glob("*.json"), reverse=True)
        doc = None
        for path in files:
            with path.open(encoding="utf-8") as handle:
                doc = json.load(handle)
            if not doc.get("demo"):
                break
        else:
            continue
        if (doc["dataset_name"], doc["split"], doc["model_name"]) != (ARC, "test", model):
            raise ValueError(f"Unexpected result identity in {path}")
        rows = {int(r["index"]): r for r in doc["records"]}
        current = {i: r["prompt"] for i, r in rows.items()}
        if prompts is None:
            prompts = current
        elif current != prompts:
            raise ValueError(f"ARC-Challenge prompts differ for {model}")
        outcomes[model] = {i: float(r["score"]) for i, r in rows.items()}
    if prompts is None:
        raise FileNotFoundError(f"No ARC-Challenge result files under {base}")
    return prompts, outcomes if len(outcomes) == len(models) else None


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line)["prompt"] for line in handle if line.strip()]


def choices(p, default, margin):
    raw = p.argmax(axis=1)
    gain = p[np.arange(len(raw)), raw] - p[:, default]
    fallback = np.ones(len(raw), bool) if margin is None else gain < margin
    return raw, np.where(fallback, default, raw)


def compare(actual, reference):
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    flips = np.flatnonzero(actual.argmax(1) != reference.argmax(1))
    ordered = np.sort(reference.astype(np.float64), axis=1)
    gap = (ordered[:, -1] - ordered[:, -2])[flips]
    return dict(max_abs_delta=float(delta.max()), mean_abs_delta=float(delta.mean()),
                raw_choices_matched=int(len(actual) - len(flips)),
                raw_choice_differences_reference_gap=[float(g) for g in gap])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bench-dir", type=Path, help="LLMRouterBench result folder containing arcc/test/<model>/")
    source.add_argument("--prompts", type=Path, help="JSONL file, one {\"prompt\": ...} object per line")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--model-dir", type=Path, help="local copy of the model folder instead of the pinned download")
    parser.add_argument("--revision", help="Hugging Face revision; default: the revision pinned in the package")
    parser.add_argument("--offline", action="store_true", help="use the local Hugging Face cache only")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="maximum |delta p| against the fp32 reference")
    args = parser.parse_args()

    from mini_intelligence import MiniIntelligence, router as package

    with np.load(REFERENCE, allow_pickle=False) as arrays:
        ref = {name: arrays[name] for name in arrays.files}
    model = MiniIntelligence(args.device, local_files_only=args.offline, model_dir=args.model_dir,
                             revision=args.revision, precision=args.precision)
    if ref["model_names"].tolist() != list(model.models):
        raise ValueError("Candidate order differs from the reference")

    outcomes = None
    if args.bench_dir:
        by_index, scores = read_bench(args.bench_dir, model.models)
        texts = [by_index.get(int(i)) for i in ref["record_index"]]
        if scores is not None:
            outcomes = np.array([[scores[m][int(i)] for m in model.models] for i in ref["record_index"]])
    else:
        by_hash = {sha256_text(t): t for t in read_jsonl(args.prompts)}
        texts = [by_hash.get(h) for h in ref["prompt_sha256"]]
    found = [t is not None and sha256_text(t) == h for t, h in zip(texts, ref["prompt_sha256"])]
    if not all(found):
        raise ValueError(f"{len(found) - sum(found)} of {len(found)} reference prompts are missing or differ")

    decisions = model.predict_batch(texts, batch_size=8, policy="gated")
    actual = np.array([[d["probabilities"][m] for m in model.models] for d in decisions], dtype=np.float32)
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite probability")
    rule = model.config["policy"]
    summary = {
        "model_dir": str(model.model_dir), "precision": args.precision, "device": str(model.device),
        "pinned_revision": package.HF_REVISION, "files_sha256_checked": model.config["files"],
        "queries": len(texts),
        "vs_fp32_reference": compare(actual, ref["fp32"]),
        "vs_training_run_bf16": compare(actual, ref["training_run_bf16"]),
    }
    if outcomes is not None:
        rows = np.arange(len(texts))
        for name, p in (("this_run", actual), ("fp32_reference", ref["fp32"]),
                        ("training_run_bf16", ref["training_run_bf16"])):
            raw, gated = choices(p, rule["fallback_model_index"], rule["fallback_margin"])
            summary[f"arc_accuracy_{name}"] = {"raw": float(outcomes[rows, raw].mean()),
                                               "gated": float(outcomes[rows, gated].mean())}
        summary["arc_accuracy_best_single"] = float(outcomes[:, rule["fallback_model_index"]].mean())
    check = summary["vs_fp32_reference"]
    near_ties = all(gap <= 2 * args.tolerance for gap in check["raw_choice_differences_reference_gap"])
    summary["passed"] = (args.precision == "float32" and check["max_abs_delta"] <= args.tolerance and near_ties)
    print(json.dumps(summary, indent=2))
    if args.precision == "float32" and not summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
