# Mini Intelligence

Mini Intelligence is a small query router. Given an English query, it estimates, for each of 20 open language models of roughly 7–9B parameters, the probability that the model answers the query correctly. It returns the selected model, the highest-scoring model and all 20 scores. The caller can then send the query to the selected model.

It is a fine-tuned version of the English [Laya](https://huggingface.co/convaiinnovations/laya) model (a ModernBERT-large encoder) trained on [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) outcomes. It was built as arm B of experiment E06 in a master's thesis on LLM routing.

## Install and use

Python 3.10 or later and Git are required. On first use, the router downloads the pinned English Laya checkpoint from Hugging Face.

```shell
git clone --branch v1.0.0 https://github.com/yahya308/Mini-Intelligence.git
cd Mini-Intelligence
python -m pip install .
mini-intelligence "Explain why seasons differ between the northern and southern hemispheres."
```

**GPU note.** For GPU inference, install a CUDA build of PyTorch first, before `pip install .` (see [pytorch.org](https://pytorch.org/get-started/locally/)); the PyTorch wheels on PyPI for Windows are CPU-only. The experiment used `torch 2.14.0+cu130` and `transformers 5.17.0`. On a CUDA GPU with bfloat16 support, the router uses the precision of the experiment (bfloat16 encoder, float32 final MLP and head). On CPU, it runs in float32.

Command-line options:

- `--policy raw` (default) or `--policy gated`. See [Selection policies](#selection-policies).
- `--device cpu` or `--device cuda` selects the device. By default, CUDA is used when available.
- `--offline` uses an existing local Hugging Face cache.
- Omit the query argument to read the query from standard input.

`laya-e06` is kept as an alias of the `mini-intelligence` command.

```python
from mini_intelligence import MiniIntelligence

router = MiniIntelligence()  # MiniIntelligence(device="cpu"), MiniIntelligence(local_files_only=True)
decision = router.predict("Write a Python function that merges two sorted lists.")
print(decision["selected_model"])
print(decision["probabilities"][decision["selected_model"]])

decisions = router.predict_batch(["Compare two sorting algorithms.", "Explain a lunar eclipse."])
gated = router.predict("Explain a lunar eclipse.", policy="gated")
```

Each decision is a dictionary with these keys:

- `selected_model`: the model chosen by the policy.
- `top_model`: the model with the highest predicted success.
- `used_fallback`: `true` when the gated policy kept the fallback model.
- `probabilities`: the predicted success of all 20 candidates.

`E06Router` remains available as an alias of `MiniIntelligence`.

### Selection policies

- `raw` (default): select the candidate with the highest predicted success. This was the preregistered primary policy of the experiment.
- `gated` (opt-in): select the fallback model, Qwen3-8B, unless the top candidate's predicted success exceeds Qwen3-8B's by at least 0.025. Qwen3-8B was the best single candidate on the fitting data, and the 0.025 margin was chosen on validation data. This margin is within the numerical noise described under [Limitations](#limitations).

## Model card

### What it is

- **Base:** English Laya, a ModernBERT-large encoder (`convaiinnovations/laya`).
- **Input:** the query text only. Up to 423 query tokens are read. Longer prompts are represented by their first 211 and last 212 tokens.
- **Adaptation:** the encoder is frozen except for a rank-8 LoRA update (alpha 16) on the output projection (`Wo`) of the final encoder layer's MLP.
- **Representation:** the CLS vector after the encoder's final normalization.
- **Prediction head:** LayerNorm, then a 128-unit linear layer with GELU and dropout 0.1, then 20 sigmoid outputs. There is one output per candidate.
- **Released learned parameters:** 165,012 (LoRA matrices and head, float32) in `mini_intelligence/weights/adapter.safetensors`. The base encoder weights are not redistributed.

### Intended use

The router is meant for research and experimentation on routing English queries among this fixed pool of 20 candidates. It works best on queries that resemble the LLMRouterBench tasks. It cannot score other models, other versions of these models, or a different candidate set.

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

- **Data:** queries from the 15 original LLMRouterBench datasets: aime, bbh, emorynlp, finqa, gpqa, humaneval, kandk, korbench, livecodebench, math500, mathbench, mbpp, medqa, meld and mmlupro.
- **Targets:** LLMRouterBench records, for each query, whether each of the 20 candidates answered it correctly. These 20 binary outcomes are the multi-label targets.
- **Released checkpoint:** trained on all 15 datasets, 11,481 queries. ARC-Challenge was kept out entirely.
- **Validation:** about 85/15 split of the training queries, grouped by prompt text (9,759 fitting, 1,722 validation).
- **Optimization:**
  - AdamW, batch size 128, weight decay 0.01.
  - Learning rate 1e-3 for the head and 3e-4 for the LoRA matrices.
  - Up to 16 epochs. The epoch with the lowest validation multi-label binary cross-entropy is kept; for this checkpoint, epoch 15.
- **Precision:** bfloat16 encoder, float32 final MLP (with LoRA) and head.
- **Fallback for the gated policy:** Qwen3-8B, the best single candidate on the fitting data. The margin (0.025) was selected on validation routed accuracy.

### Evaluation results

All results are from the thesis experiment E06, averaged over the five official LLMRouterBench 70/30 test splits. Each split uses its own checkpoint, trained on that split's training part. These numbers therefore describe the training recipe, not this exact released checkpoint.

The metrics measure how often the selected model's recorded answer was correct:

- **Sample-Avg:** accuracy over all test queries.
- **Dataset-Avg:** the mean of the per-dataset accuracies.

| Method | Sample-Avg | Dataset-Avg |
|---|---:|---:|
| E06-B raw (this recipe) | 70.10% | 71.35% |
| E06-A raw (frozen encoder, head only) | 70.08% | 71.27% |
| E06-B gated | 69.91% | 71.32% |
| TF-IDF kNN (k=32) | 69.31% | 70.57% |
| Best single model (Qwen3-8B) | 66.72% | 68.01% |

Fine-tuning (B) gave no meaningful gain over the frozen-encoder head (A). The mean paired difference was +0.02 points, and no split showed a significant difference.

**Held-out checks:**

- **ARC-Challenge** (this exact released checkpoint; 1,172 queries never used for training or selection):
  - raw: 93.69%
  - gated: 93.52%
  - best single model (Qwen3-8B): 93.77%

  ARC-Challenge is near ceiling for this pool and barely discriminates between routers.
- **LiveCodeBench** (a separate checkpoint trained without LiveCodeBench; 1,055 queries):
  - B raw: 36.49%
  - B gated: 38.10%
  - best single model: 67.68%

  The router does **not** transfer to unseen dataset families. On LiveCodeBench, B was also 2.09 points below A.

### Limitations

- **Fixed pool, benchmark prompts.** The scores are tied to the fixed candidate pool and to benchmark-style prompts. Other task distributions, prompt formats or model versions may not transfer, as the LiveCodeBench result shows.
- **English only.**
- **Not calibrated.** Scores estimate benchmark correctness; they are not calibrated confidences. On held-out LiveCodeBench, the Brier score was no better than that of a constant prior. Validate calibration before using the scores as confidence thresholds.
- **No cost or latency.** Cost, latency, answer length and user preferences are not modeled.
- **Numerical sensitivity.** Batch size and precision can change the probabilities by up to about 0.05:
  - Batch 1 versus batch 8 changed the raw choice on 4 of 269 ARC-Challenge prompts.
  - CPU float32 agreed with the GPU reference on 47 of 48 prompts checked.

  The 0.025 gated margin is within this noise. Raw choices are more stable.

## Provenance and reproducibility

| Item | Value |
|---|---|
| Base model | `convaiinnovations/laya`, revision `e333cb7c823cd1787276228260f56d02fbf5180b` |
| Base `model.safetensors` SHA-256 | `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` |
| Laya software | 0.4.37, GitHub commit `d1282e791d71e92f24f816ff3836e4fd1c2ded4a` |
| Training checkpoint SHA-256 | `8de1d26dcaca185099dc778cf56112447e42774e597d7da8ba031e1adfa8e5d7` |
| `adapter.safetensors` SHA-256 | `6273d4787e7c3c96ef9f7ae4a736300efe44e7677bede8322f5cf42b389e5e1b` |

`mini_intelligence/weights/router_config.json` records these values together with the candidate order, selected epoch and fallback rule. At load time, the router checks the Laya version and both checksums, and refuses to run on a mismatch.

The release contains no base encoder weights, training prompts, model responses or API credentials.

`tools/export_adapter.py` and `tools/verify_release.py` are maintainer tools. They need the private training checkpoint, and the benchmark prompt file with reference probabilities. These are not distributed, so the tools cannot be run from this repository alone. The training code and full experiment records are in the author's thesis repository, which is not public at the moment.

The unit tests need no model download:

```shell
python -m unittest discover -s tests -v
```

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff); GitHub shows it under "Cite this repository". Please also credit the upstream [Laya model](https://huggingface.co/convaiinnovations/laya) and [LLMRouterBench](https://aclanthology.org/2026.findings-acl.1881/).

## License

The code and adapter weights in this repository are released under the [Apache License 2.0](LICENSE). The Laya base model and software are separate upstream components, licensed Apache-2.0 by their authors. See [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md) for attribution.
