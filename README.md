# Mini Intelligence

Mini Intelligence is a small query router. Given an English query, it estimates, for each of 20 open language models of roughly 7–9B parameters, the probability that the model answers the query correctly. It returns the selected model, the highest-scoring model and all 20 scores. The caller can then send the query to the selected model.

Version 2.0.0 is a full fine-tune of the English [Laya](https://huggingface.co/convaiinnovations/laya) encoder (ModernBERT-large: 28 layers, about 395M parameters) with a 20-output prediction head. It was trained on outcomes recorded by [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench), as part of a master's thesis on LLM routing.

The model weights are on Hugging Face: [Yahya308/Mini-Intelligence](https://huggingface.co/Yahya308/Mini-Intelligence). This repository holds the inference package, the command-line tool, the maintainer tools and the tests.

## Install and use

Python 3.10 or later and Git are required. On first use, the router downloads the model (about 1.6 GB) from Hugging Face. It uses the exact commit pinned in this package and checks the SHA-256 of every file before loading.

```shell
git clone --branch v2.0.0 https://github.com/yahya308/Mini-Intelligence.git
cd Mini-Intelligence
python -m pip install .
mini-intelligence "Explain why seasons differ between the northern and southern hemispheres."
```

**GPU note.** For GPU inference, install a CUDA build of PyTorch before running `pip install .` (see [pytorch.org](https://pytorch.org/get-started/locally/)). The PyTorch wheels on PyPI for Windows are CPU-only. The release was verified with `torch 2.14.0+cu130` and `transformers 5.17.0`.

Command-line options:

- `--policy raw` (default) or `--policy gated`. See [Selection policies](#selection-policies).
- `--device cpu` or `--device cuda` selects the device. By default, CUDA is used when available.
- `--precision float32` (default) or `--precision bfloat16`. bfloat16 needs a CUDA GPU with bfloat16 support. See [Numerical precision](#numerical-precision).
- `--model-dir PATH` uses a local copy of the Hugging Face model folder instead of downloading it. The checksums are still checked.
- `--offline` uses an existing local Hugging Face cache.
- Omit the query argument to read the query from standard input.

`laya-e06` is kept as an alias of the `mini-intelligence` command.

```python
from mini_intelligence import MiniIntelligence

router = MiniIntelligence()  # MiniIntelligence(device="cpu"), MiniIntelligence(local_files_only=True)
decision = router.route("Write a Python function that merges two sorted lists.")
print(decision["selected_model"])
print(decision["probabilities"][decision["selected_model"]])

decisions = router.predict_batch(["Compare two sorting algorithms.", "Explain a lunar eclipse."])
gated = router.predict("Explain a lunar eclipse.", policy="gated")
```

`route` and `predict` are the same method. Each decision is a dictionary with these keys:

- `selected_model`: the model chosen by the policy.
- `top_model`: the model with the highest predicted success.
- `used_fallback`: `true` when the gated policy kept the fallback model.
- `probabilities`: the predicted success of all 20 candidates.

Other constructor options:

- `model_dir="path"` loads a local copy of the model folder.
- `revision="..."` selects another Hugging Face revision. The files must still match the checksums in `mini_intelligence/router_config.json`.
- `precision="bfloat16"` selects bfloat16 autocast on CUDA.

`E06Router` remains available as an alias of `MiniIntelligence`.

### Selection policies

- `raw` (default): select the candidate with the highest predicted success.
- `gated` (opt-in): select the fallback model, Qwen3-8B, unless the top candidate's predicted success exceeds Qwen3-8B's by at least a margin chosen on validation data. Qwen3-8B was the best single candidate on the fitting data. For the released checkpoint, the chosen margin is 0.0, so the gated policy never falls back and gives the same choices as `raw`. The policy is kept for compatibility with v1.

### Using v1

v1.0.0 (a rank-8 LoRA adapter on the Laya encoder) remains available through its tag. Its README documents its own installation and results:

```shell
git clone --branch v1.0.0 https://github.com/yahya308/Mini-Intelligence.git
```

v1 needs the `laya` package and downloads the base Laya model; its adapter weights are inside that tag. The two versions use different weights, so their probabilities differ.

## What changed from v1

| | v1.0.0 | v2.0.0 |
|---|---|---|
| Adaptation | Rank-8 LoRA on one matrix (the output projection of the final encoder layer's MLP); rest of the encoder frozen | Full fine-tune of every encoder parameter, embeddings included |
| Learned parameters | 165,012 | 394,917,524 (encoder 394,781,696; head 135,828) |
| Weights | Adapter and head in this repository; base Laya downloaded at load time | Complete encoder (standard ModernBERT checkpoint) and head on Hugging Face |
| Runtime dependency | `laya` package | transformers only |
| Default precision | bfloat16 encoder on GPU | float32 on CPU or GPU (bfloat16 optional) |
| Official splits, mean Sample-Avg / Dataset-Avg | 70.10% / 71.35% | 70.78% / 72.40% |
| ARC-Challenge (held out, released checkpoint) | 93.69% | 94.03% |

The input (423-token head and tail view), representation (CLS), head architecture, candidate pool and selection policies are unchanged.

## Model card

The full model card is on [Hugging Face](https://huggingface.co/Yahya308/Mini-Intelligence). Summary:

### What it is

- **Base:** English Laya, a ModernBERT-large encoder (`convaiinnovations/laya`, revision `e333cb7`).
- **Input:** the query text only. The tokenizer's mask token is replaced by a space. Up to 423 query tokens are read, between `[CLS]` and `[SEP]`. Longer queries are represented by their first 211 and last 212 tokens.
- **Adaptation:** full fine-tuning of the encoder and the head.
- **Representation:** the CLS vector after the encoder's final normalization.
- **Prediction head:** LayerNorm, then a 128-unit linear layer with GELU and dropout 0.1, then 20 sigmoid outputs. There is one output per candidate.

### Intended use

The router is meant for research and experimentation on routing English queries among this fixed pool of 20 candidates. It works best on queries that resemble the LLMRouterBench tasks. It cannot score other models, other versions of these models or a different candidate set.

The candidates, in output order, are:

1. DeepHermes-3-Llama-3-8B-Preview
2. DeepSeek-R1-0528-Qwen3-8B
3. DeepSeek-R1-Distill-Qwen-7B
4. Fin-R1
5. GLM-Z1-9B-0414
6. Intern-S1-mini
7. Llama-3.1-8B-Instruct
8. Llama-3.1-8B-UltraMedical
9. Llama-3.1-Nemotron-Nano-8B-v1
10. MiMo-7B-RL-0530
11. MiniCPM4.1-8B
12. NVIDIA-Nemotron-Nano-9B-v2
13. OpenThinker3-7B
14. Qwen2.5-Coder-7B-Instruct
15. Qwen3-8B (fallback model of the gated policy)
16. cogito-v1-preview-llama-8B
17. gemma-2-9b-it
18. glm-4-9b-chat
19. granite-3.3-8b-instruct
20. internlm3-8b-instruct

### Training data and procedure

- **Data:** queries from the 15 datasets of LLMRouterBench's 20-model setting. They cover mathematics, code generation, logical and general reasoning, knowledge and domain question answering (including medicine and finance), and emotion recognition in dialogue.
- **Targets:** LLMRouterBench records, for each query, whether each of the 20 candidates answered it correctly. These 20 binary outcomes are the multi-label targets.
- **Released checkpoint:** trained on all 11,481 queries of the 15 datasets: 9,759 for fitting and 1,722 for validation, split by prompt text. ARC-Challenge was kept out entirely.
- **Optimization:**
  - AdamW, weight decay 0.01. The encoder's LayerNorm weights are not decayed.
  - Learning rate 5e-5 for the encoder and 1e-3 for the head. The encoder rate was chosen from {1e-5, 2e-5, 5e-5} on validation data.
  - 6% linear warm-up, then linear decay. Gradient clipping at 1.0.
  - Batches of 16 queries of similar length.
  - Up to 5 epochs, with early stopping. The epoch with the lowest validation binary cross-entropy is kept; for this checkpoint, epoch 3.
- **Precision:** float32 weights with bfloat16 autocast. The final encoder layer's MLP runs in float32.
- **Hardware:** one NVIDIA L40S GPU, about 12.5 minutes of training.
- **Fallback for the gated policy:** Qwen3-8B. The margin (0.0) was selected on validation routed accuracy.

### Evaluation results

The training recipe was evaluated with LLMRouterBench's official protocol: the `split_by_dataset_then_prompt` split, 70% training and 30% test within each dataset, 5 seeds (42, 999, 2024, 2025 and 3407), the 20-model pool and the 15 datasets. Each split uses its own model, trained with the same recipe on that split's training part. These numbers therefore describe the recipe, not this exact released checkpoint.

The metrics measure how often the selected model's recorded answer was correct:

- **Sample-Avg:** accuracy over all test queries.
- **Dataset-Avg:** the mean of the per-dataset accuracies.

Mean ± standard deviation over the 5 splits:

| Method | Sample-Avg | Dataset-Avg |
|---|---:|---:|
| Mini Intelligence v2, raw (this recipe) | 70.78 ± 0.58% | 72.40 ± 1.06% |
| Mini Intelligence v2, gated | 70.40 ± 0.83% | 72.08 ± 0.93% |
| Mini Intelligence v1, raw | 70.10 ± 0.92% | 71.35 ± 1.11% |
| Best single model (Qwen3-8B) | 66.72 ± 0.87% | 68.01 ± 1.08% |

Compared with v1, the mean gain is +0.68 points Sample-Avg and +1.05 points Dataset-Avg. In paired bootstrap tests per split (2,000 resamples, 95% intervals), the Sample-Avg gain was above zero in 3 of the 5 splits. The Dataset-Avg gain was not distinguishable from zero in any split. v1 and v2 were trained on different GPUs.

For context only, the [LLMRouterBench paper](https://arxiv.org/abs/2601.07206) (Table 13) reports these Dataset-Avg values for published routers: Avengers 71.94%, MODEL-SAT 71.88%, EmbedLLM 71.24%, GraphRouter 70.29% and RouterDC 61.33%. We did not re-run them. Our numbers come from different hardware and runs, so we make no claim of a significant difference.

**Held-out ARC-Challenge** (this exact released checkpoint; 1,172 queries never used for training or selection):

- Mini Intelligence v2: 94.03% (raw and gated are identical).
- Best single model (Qwen3-8B): 93.77%.
- Mini Intelligence v1: 93.69%.

The difference from the best single model (+0.26 points; 95% paired bootstrap interval −0.77 to +1.28) is not significant. ARC-Challenge is near the ceiling for this pool and barely separates routers. On these queries, the Brier score averaged over the 20 candidates is 0.139 and the log loss 0.445 (v1: 0.196 and 0.575).

### Limitations

- **Measured on benchmark task types only.** Routing quality was measured on LLMRouterBench queries.
- **Unfamiliar task types.** On queries unlike the training tasks, the router can over-rely on patterns learned from similar-looking training tasks. It can then route poorly, for example by sending a whole family of queries to one model that is weak on them. Evaluate it on your own data, and compare it with simply using the best single model, before relying on it.
- **Fixed pool.** It only routes among these 20 specific models, as run by LLMRouterBench.
- **Estimates, not guarantees.** The probabilities are estimates of benchmark correctness learned from about 11,500 queries. They are not calibrated confidences. Validate them before using them as thresholds.
- **English only.** Long queries are seen only through their first 211 and last 212 tokens.
- **No cost or latency.** Cost, latency, answer length and user preferences are not modeled.

### Numerical precision

All figures are on the 1,172 ARC-Challenge queries, scored in batches of 8 unless noted. The default is float32. In float32, batch size and hardware change the probabilities only at the level of floating-point rounding:

- **Batch size.** One prompt at a time (as `predict` and the command line run) and batches of 8 agree to within 6e-6 on the GPU, with identical choices.
- **Same machine.** The package and the original training code, both in float32 on an NVIDIA RTX 4070 Laptop GPU, agree to within 1.2e-7, with identical choices.
- **CPU versus GPU.** Float32 on CPU and on the GPU agree to within 2e-5, with identical choices.
- **Float32 versus the training run.** The training run stored bfloat16 probabilities (NVIDIA L40S). Float32 probabilities differ from them by at most 0.042 (mean 0.0013). 1,168 of 1,172 choices are identical. The 4 differences are near ties (stored top-two gap below 0.001) and do not change routed accuracy (94.03%).
- **bfloat16 on the RTX 4070.** It matches the original code's bfloat16 output to within 1.2e-7. It differs from the stored L40S values by at most 0.021, with 1,169 of 1,172 identical choices.

## Provenance and reproducibility

| Item | Value |
|---|---|
| Model repository | [`Yahya308/Mini-Intelligence`](https://huggingface.co/Yahya308/Mini-Intelligence), commit `HF_COMMIT_PENDING` (tag `v2.0.0`) |
| `model.safetensors` SHA-256 | `7e29152f140bc4dadaf78152bc99210bda242a91a4ae66723aebe797a245b41e` |
| `head.safetensors` SHA-256 | `cf24e4e656b54404f06c72513eb2e03ca1929fe55f58965e397286fe2b495155` |
| Training checkpoint SHA-256 | `49acc7067cad04b3ecf944ede735080082bc7fc598970226d1cdc674d7ad7f37` |
| Base model | `convaiinnovations/laya`, revision `e333cb7c823cd1787276228260f56d02fbf5180b` |
| Base `model.safetensors` SHA-256 | `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` |

`mini_intelligence/router_config.json` records these values together with the candidate order, input view, selection policy and the checksum of every model file. It is identical to `router_config.json` in the model repository. At load time, the router checks every checksum and refuses to run on a mismatch.

The encoder and head on Hugging Face are an exact split of the training checkpoint: every tensor is bit-identical. Neither repository contains training prompts, benchmark questions, model responses, outcome labels or API credentials.

Tools:

- `tools/verify_release.py` checks an installation against `tools/reference/arc_challenge_v2.npz`. That file holds the reference probabilities for the 1,172 ARC-Challenge test prompts, keyed by a SHA-256 hash of each prompt. It needs the ARC-Challenge prompts from LLMRouterBench (not distributed here):

  ```shell
  python tools/verify_release.py --bench-dir <LLMRouterBench results folder containing arcc/test/>
  ```

  With all 20 candidates' result files present, it also reports routed accuracy.
- `tools/export_release.py` is a maintainer tool. It builds the Hugging Face folder from the private training checkpoint.

The training code and full experiment records are in the author's thesis repository, which is not public at the moment.

The unit tests need no model download:

```shell
python -m unittest discover -s tests -v
```

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff); GitHub shows it under "Cite this repository". Please also credit the upstream [Laya model](https://huggingface.co/convaiinnovations/laya) and [LLMRouterBench](https://arxiv.org/abs/2601.07206).

## License

The code in this repository and the model weights on Hugging Face are released under the [Apache License 2.0](LICENSE). Laya and ModernBERT are separate upstream components, licensed Apache-2.0 by their authors. See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for attribution.
