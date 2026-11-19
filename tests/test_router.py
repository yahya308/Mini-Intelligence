"""Lightweight algebra, policy and distribution checks. No base download is needed."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from safetensors import safe_open
from torch import nn

import mini_intelligence
from mini_intelligence import E06Router, MiniIntelligence
from mini_intelligence.__main__ import build_parser
from mini_intelligence.router import AdaptedMLP

WEIGHTS = Path(mini_intelligence.__file__).resolve().parent / "weights"
CONFIG = json.loads((WEIGHTS / "router_config.json").read_text(encoding="utf-8"))


class FakeTokenizer:
    """Each whitespace-separated integer n becomes token ID n + 10."""
    mask_token = "[MASK]"
    pad_token_id, cls_token_id, sep_token_id = 0, 1, 2

    def __call__(self, texts, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [[int(word) + 10 for word in text.split()] for text in texts]}


class FakeEncoder:
    """Returns a CLS state equal to a logit row chosen by the first query token."""

    def __init__(self, table):
        self.table, self.seen = table, []

    def __call__(self, input_ids, attention_mask):
        self.seen.append(input_ids.clone())
        hidden = torch.zeros(*input_ids.shape, self.table.shape[1])
        hidden[:, 0] = self.table[input_ids[:, 1] % len(self.table)]
        return SimpleNamespace(last_hidden_state=hidden)


def fake_router():
    """Run the real selection code with stub encoder outputs."""
    models = CONFIG["models"]
    default, top = CONFIG["fallback_model_index"], models.index("gemma-2-9b-it")
    probs = torch.full((2, len(models)), 0.1)
    probs[:, default] = 0.60
    probs[0, top], probs[1, top] = 0.61, 0.70  # margin below / above the saved 0.025 threshold
    router = MiniIntelligence.__new__(MiniIntelligence)
    router.config, router.models = CONFIG, tuple(models)
    router.tokenizer, router.encoder, router.head = FakeTokenizer(), FakeEncoder(torch.logit(probs)), nn.Identity()
    router.device, router.bf16 = torch.device("cpu"), False
    return router, models[default], models[top]


class RouterTests(unittest.TestCase):
    def test_bundled_tensor_manifest(self):
        path = WEIGHTS / "adapter.safetensors"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CONFIG["adapter_sha256"])
        self.assertEqual(len(set(CONFIG["models"])), 20)
        self.assertEqual(CONFIG["held_out_dataset"], "arcc")
        self.assertEqual(CONFIG["models"][CONFIG["fallback_model_index"]], "Qwen3-8B")
        self.assertEqual(CONFIG["fallback_margin"], 0.025)
        with safe_open(path, framework="pt") as tensors:
            self.assertEqual(tensors.get_slice("adapter_down.weight").get_shape(), [8, 2624])
            self.assertTrue(all(key.startswith(("adapter_", "head.layers.")) for key in tensors.keys()))

    def test_cls_update_matches_merged_weight(self):
        torch.manual_seed(11)
        mlp = nn.Module()
        mlp.Wi, mlp.Wo = nn.Linear(7, 18, bias=False), nn.Linear(9, 7, bias=False)
        mlp.act, mlp.drop = nn.GELU(), nn.Identity()
        adapted = AdaptedMLP(mlp, rank=2, alpha=4)
        x = torch.randn(3, 5, 7)
        v, g = mlp.Wi(x).chunk(2, dim=-1)
        z = mlp.act(v) * g
        merged = mlp.Wo.weight + 2 * (adapted.up.weight @ adapted.down.weight)
        actual = adapted(x)
        torch.testing.assert_close(actual[:, 0], nn.functional.linear(z[:, 0], merged))
        torch.testing.assert_close(actual[:, 1:], mlp.Wo(z[:, 1:]))

    def test_default_policy_is_raw(self):
        router, _, top = fake_router()
        close, clear = router.predict_batch(["0", "1"])
        self.assertEqual(router.predict("0"), close)
        for decision in (close, clear):
            self.assertEqual((decision["selected_model"], decision["top_model"]), (top, top))
            self.assertFalse(decision["used_fallback"])
        self.assertEqual(build_parser().parse_args(["query"]).policy, "raw")

    def test_gated_policy_is_opt_in(self):
        router, default, top = fake_router()
        close, clear = router.predict_batch(["0", "1"], policy="gated")
        self.assertEqual((close["selected_model"], close["top_model"], close["used_fallback"]), (default, top, True))
        self.assertEqual((clear["selected_model"], clear["used_fallback"]), (top, False))
        self.assertEqual(router.predict("0", policy="gated"), close)
        self.assertEqual(build_parser().parse_args(["query", "--policy", "gated"]).policy, "gated")
        with self.assertRaises(ValueError):
            router.predict("0", policy="other")

    def test_long_prompt_keeps_head_and_tail(self):
        router, _, _ = fake_router()
        router.predict(" ".join(str(i) for i in range(500)))
        ids = router.encoder.seen[-1][0].tolist()
        expected = [1, *(i + 10 for i in range(211)), *(i + 10 for i in range(288, 500)), 2]
        self.assertEqual(len(ids), CONFIG["input_tokens"] + 2)
        self.assertEqual(ids, expected)

    def test_backward_compatible_name(self):
        self.assertIs(E06Router, MiniIntelligence)
        self.assertEqual(set(mini_intelligence.__all__), {"MiniIntelligence", "E06Router"})


if __name__ == "__main__":
    unittest.main()
