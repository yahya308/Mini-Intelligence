"""Maintainer tool: compare the release to E06 test probabilities without printing any prompt text."""
import argparse
import json
from pathlib import Path

import numpy as np

from mini_intelligence import MiniIntelligence


def verify(router, records, reference, mode="reference"):
    # Records contain the original complete feature population, in its original order.
    # Reference probabilities are keyed by (dataset_id, split, record_index) and split_id.
    choose = reference["split_id"] == "arcc"
    if not np.array_equal(reference["model_names"], router.models):
        raise ValueError("Candidate order differs from the reference")
    truth = reference["B"][choose]
    key_fields = ["dataset_id", "split", "record_index"]
    ref_keys = list(zip(*(reference[name][choose] for name in key_fields)))
    by_key = {tuple(record[k] for k in key_fields): record["prompt"] for record in records}
    if len(by_key) != len(records) or len(set(ref_keys)) != len(truth) or not len(truth):
        raise ValueError("Empty reference or duplicate record keys")

    if mode == "reference":
        # bf16 numerics depend on padded batch shape. Restore the original order of
        # distinct, length-sorted visible token IDs before selecting reference batches.
        prompt_view = {}
        first_prompt = {}
        wanted = {key: i for i, key in enumerate(ref_keys)}
        test_positions = {}
        tok = router.tokenizer
        for start in range(0, len(records), 128):
            group = records[start:start + 128]
            inputs = tok([row["prompt"].replace(tok.mask_token, " ") for row in group],
                         add_special_tokens=False)["input_ids"]
            for row, ids in zip(group, inputs):
                view = tuple(ids if len(ids) <= 423 else ids[:211] + ids[-212:])
                first_prompt.setdefault(view, row["prompt"])
                key = tuple(row[k] for k in key_fields)
                if key in wanted:
                    test_positions.setdefault(view, []).append(wanted[key])
        ordered = sorted(first_prompt, key=len)
        outputs = [None] * len(truth)
        for start in range(0, len(ordered), 8):
            batch = ordered[start:start + 8]
            if not any(view in test_positions for view in batch):
                continue
            predicted = router.predict_batch([first_prompt[view] for view in batch], policy="gated")
            for view, item in zip(batch, predicted):
                for i in test_positions.get(view, ()):
                    outputs[i] = item
    elif mode == "standalone":
        # policy="gated": selected_model is checked against the gated reference choice below.
        outputs = router.predict_batch([by_key[key] for key in ref_keys], policy="gated")
    else:
        raise ValueError("Unsupported batch layout")

    actual = np.array([[row["probabilities"][name] for name in router.models] for row in outputs])
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite score from release")
    delta = np.abs(actual - truth)
    top = truth.argmax(axis=1)
    default = router.config["fallback_model_index"]
    threshold = router.config["fallback_margin"]
    margin = truth[np.arange(len(top)), top] - truth[:, default]
    gated = np.full_like(top, default) if threshold is None else np.where(margin < threshold, default, top)
    raw_matches = sum(item["top_model"] == router.models[i] for item, i in zip(outputs, top))
    gate_matches = sum(item["selected_model"] == router.models[i] for item, i in zip(outputs, gated))
    summary = dict(batch_layout=mode, queries=len(top), probabilities=int(truth.size),
                   mean_absolute_error=float(delta.mean()), maximum_absolute_error=float(delta.max()),
                   raw_choices_matched=int(raw_matches), gated_choices_matched=int(gate_matches))
    if mode == "reference" and (delta.max() > 2e-4 or raw_matches != len(top) or gate_matches != len(top)):
        raise AssertionError(summary)
    return summary


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--queries", required=True, type=Path, help="JSONL in full original benchmark row order; "
                     "one object per query with dataset_id, split, record_index and prompt.")
    arg.add_argument("--reference", required=True, type=Path, help="E06 e06_probabilities.npz")
    arg.add_argument("--mode", choices=["reference", "standalone"], default="reference")
    args = arg.parse_args()
    with args.queries.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    with np.load(args.reference, allow_pickle=False) as reference:
        print(json.dumps(verify(MiniIntelligence("cuda", local_files_only=True), records, reference, args.mode), indent=2))
