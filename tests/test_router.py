"""Lightweight configuration, policy, input-view and tooling checks. No model download is needed."""
import copy
import importlib.util
import json
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

import mini_intelligence
from mini_intelligence import E06Router, MiniIntelligence
from mini_intelligence import router as package
from mini_intelligence.__main__ import build_parser
from mini_intelligence.router import FP32MLP, SuccessHead

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(package.CONFIG_PATH.read_text(encoding="utf-8"))
REFERENCE = ROOT / "tools" / "reference" / "arc_challenge_v2.npz"


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


def fake_router(margin=None):
    """Run the real selection code with stub encoder outputs; optionally override the saved margin."""
    config = copy.deepcopy(CONFIG)
    if margin != "saved":
        config["policy"]["fallback_margin"] = margin
    models = config["models"]
    default, top = config["policy"]["fallback_model_index"], models.index("gemma-2-9b-it")
    probs = torch.full((2, len(models)), 0.1)
    probs[:, default] = 0.60
    probs[0, top], probs[1, top] = 0.61, 0.70  # gain 0.01 and 0.10 over the fallback model
    router = MiniIntelligence.__new__(MiniIntelligence)
    router.config, router.models = config, tuple(models)
    router.head_tokens, router.tail_tokens = config["input"]["head_tokens"], config["input"]["tail_tokens"]
    router.tokenizer, router.encoder, router.head = FakeTokenizer(), FakeEncoder(torch.logit(probs)), nn.Identity()
    router.device, router.precision = torch.device("cpu"), "float32"
    return router, models[default], models[top]


class ConfigTests(unittest.TestCase):
    def test_bundled_configuration(self):
        self.assertEqual(CONFIG["format_version"], 2)
        self.assertEqual(CONFIG["release"], mini_intelligence.__version__)
        self.assertEqual(len(CONFIG["models"]), 20)
        self.assertEqual(len(set(CONFIG["models"])), 20)
        self.assertEqual(CONFIG["models"], sorted(CONFIG["models"]))
        policy = CONFIG["policy"]
        self.assertEqual(CONFIG["models"][policy["fallback_model_index"]], "Qwen3-8B")
        self.assertEqual(policy["fallback_model"], "Qwen3-8B")
        self.assertEqual(policy["fallback_margin"], 0.0)
        self.assertEqual(policy["default"], "raw")
        self.assertEqual((CONFIG["input"]["head_tokens"], CONFIG["input"]["tail_tokens"]), (211, 212))
        self.assertEqual(CONFIG["input"]["max_content_tokens"], 423)
        self.assertEqual(CONFIG["training"]["held_out_dataset"], "arcc")
        self.assertEqual(set(CONFIG["files"]), set(package.FILES) - {"router_config.json"})
        for digest in (*CONFIG["files"].values(), CONFIG["base_sha256"], CONFIG["training_checkpoint_sha256"]):
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_revision_pin(self):
        if package.HF_REVISION is None:
            self.skipTest("release pin not set yet (filled in after the Hugging Face upload)")
        self.assertRegex(package.HF_REVISION, r"^[0-9a-f]{40}$")

    def test_checksum_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            shutil.copyfile(package.CONFIG_PATH, folder / "router_config.json")
            for name in CONFIG["files"]:
                (folder / name).write_bytes(b"not the released file")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                MiniIntelligence("cpu", model_dir=folder)
            (folder / "router_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "router_config.json differs"):
                MiniIntelligence("cpu", model_dir=folder)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            MiniIntelligence("cpu", precision="float16")
        with self.assertRaises(ValueError):
            MiniIntelligence("cpu", precision="bfloat16")


class PolicyTests(unittest.TestCase):
    def test_default_policy_is_raw(self):
        router, _, top = fake_router("saved")
        close, clear = router.predict_batch(["0", "1"])
        self.assertEqual(router.predict("0"), close)
        for decision in (close, clear):
            self.assertEqual((decision["selected_model"], decision["top_model"]), (top, top))
            self.assertFalse(decision["used_fallback"])
        self.assertEqual(build_parser().parse_args(["query"]).policy, "raw")

    def test_saved_margin_zero_makes_gated_equal_raw(self):
        router, _, _ = fake_router("saved")
        self.assertEqual(router.predict_batch(["0", "1"], policy="gated"), router.predict_batch(["0", "1"]))

    def test_gated_rule_with_positive_margin(self):
        router, default, top = fake_router(0.05)
        close, clear = router.predict_batch(["0", "1"], policy="gated")
        self.assertEqual((close["selected_model"], close["top_model"], close["used_fallback"]), (default, top, True))
        self.assertEqual((clear["selected_model"], clear["used_fallback"]), (top, False))
        self.assertEqual(build_parser().parse_args(["query", "--policy", "gated"]).policy, "gated")
        with self.assertRaises(ValueError):
            router.predict("0", policy="other")

    def test_always_fallback_margin(self):
        router, default, _ = fake_router(None)
        for decision in router.predict_batch(["0", "1"], policy="gated"):
            self.assertEqual((decision["selected_model"], decision["used_fallback"]), (default, True))

    def test_probabilities_cover_all_candidates(self):
        router, _, _ = fake_router("saved")
        decision = router.route("1")
        self.assertEqual(list(decision["probabilities"]), CONFIG["models"])
        self.assertAlmostEqual(decision["probabilities"][decision["top_model"]], 0.70, places=5)

    def test_batch_order_and_inputs(self):
        router, _, _ = fake_router("saved")
        decisions = router.predict_batch(["1 5 5 5", "0"], batch_size=1)
        self.assertEqual([d["top_model"] for d in decisions], ["gemma-2-9b-it"] * 2)
        self.assertAlmostEqual(decisions[0]["probabilities"]["gemma-2-9b-it"], 0.70, places=5)
        with self.assertRaises(ValueError):
            router.predict_batch("0")
        with self.assertRaises(TypeError):
            router.predict_batch([0])
        self.assertEqual(router.predict_batch([]), [])


class InputTests(unittest.TestCase):
    def test_long_prompt_keeps_head_and_tail(self):
        router, _, _ = fake_router("saved")
        router.predict(" ".join(str(i) for i in range(500)))
        ids = router.encoder.seen[-1][0].tolist()
        expected = [1, *(i + 10 for i in range(211)), *(i + 10 for i in range(288, 500)), 2]
        self.assertEqual(len(ids), CONFIG["input"]["max_content_tokens"] + 2)
        self.assertEqual(ids, expected)

    def test_short_prompt_is_unchanged(self):
        router, _, _ = fake_router("saved")
        self.assertEqual(router.encode([" ".join(map(str, range(423)))])[0], [1, *range(10, 433), 2])

    def test_mask_token_is_replaced(self):
        router, _, _ = fake_router("saved")
        self.assertEqual(router.encode(["3[MASK]4"])[0], [1, 13, 14, 2])


class ModuleTests(unittest.TestCase):
    def test_head_layout_matches_released_tensor_names(self):
        head = SuccessHead(1024, 20)
        names = {f"layers.{i}.{part}" for i in (0, 1, 4) for part in ("weight", "bias")}
        self.assertEqual(set(head.state_dict()), names)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 135828)

    def test_fp32_mlp_wrapper_is_exact_in_float32(self):
        torch.manual_seed(3)
        mlp = nn.Sequential(nn.Linear(6, 10), nn.GELU(), nn.Linear(10, 6))
        x = torch.randn(2, 5, 6)
        torch.testing.assert_close(FP32MLP(mlp)(x), mlp(x), rtol=0, atol=0)

    def test_public_names(self):
        self.assertIs(E06Router, MiniIntelligence)
        self.assertEqual(MiniIntelligence.route, MiniIntelligence.predict)
        self.assertEqual(set(mini_intelligence.__all__), {"MiniIntelligence", "E06Router"})
        args = build_parser().parse_args(["q", "--precision", "bfloat16", "--model-dir", "x", "--offline"])
        self.assertEqual((args.precision, args.model_dir, args.offline), ("bfloat16", "x", True))
        self.assertEqual(build_parser().parse_args(["q"]).precision, "float32")


class ToolTests(unittest.TestCase):
    def test_reference_file(self):
        with np.load(REFERENCE, allow_pickle=False) as ref:
            self.assertEqual(ref["model_names"].tolist(), CONFIG["models"])
            self.assertEqual(ref["fp32"].shape, (1172, 20))
            self.assertEqual(ref["training_run_bf16"].shape, (1172, 20))
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in ref["prompt_sha256"]))
            self.assertEqual(len(ref["record_index"]), 1172)
            for name in ("fp32", "training_run_bf16"):
                self.assertTrue(((ref[name] > 0) & (ref[name] < 1)).all())
            self.assertLess(np.abs(ref["fp32"] - ref["training_run_bf16"]).mean(), 0.05)

    def test_bench_reader(self):
        spec = importlib.util.spec_from_file_location("verify_release", ROOT / "tools" / "verify_release.py")
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        models = CONFIG["models"][:2]
        with tempfile.TemporaryDirectory() as folder:
            for k, model in enumerate(models):
                path = Path(folder) / "arcc" / "test" / model
                path.mkdir(parents=True)
                doc = {"dataset_name": "arcc", "split": "test", "model_name": model, "demo": False,
                       "records": [{"index": 5, "prompt": "q five", "score": k}, {"index": 2, "prompt": "q two", "score": 1}]}
                (path / "arcc-test-a.json").write_text(json.dumps(doc), encoding="utf-8")
                (path / "arcc-test-b.json").write_text(json.dumps({**doc, "demo": True, "records": []}), encoding="utf-8")
            prompts, outcomes = tool.read_bench(folder, models)
            self.assertEqual(prompts, {5: "q five", 2: "q two"})
            self.assertEqual(outcomes[models[1]][5], 1.0)
            prompts, outcomes = tool.read_bench(folder, CONFIG["models"])
            self.assertIsNone(outcomes)
        self.assertEqual(tool.sha256_text("a"), "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb")


if __name__ == "__main__":
    unittest.main()
