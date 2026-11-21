# Third-party attribution

**Laya.** Convai Innovations created the Laya model and software, licensed under the Apache License, Version 2.0. The Mini Intelligence v2 encoder was fine-tuned from the Laya encoder at revision `e333cb7c823cd1787276228260f56d02fbf5180b`. The model repository on Hugging Face redistributes the fine-tuned encoder, Laya's encoder configuration (with the architecture name changed to `ModernBertModel`) and Laya's tokenizer files, unchanged. This code repository contains no model weights. [Laya model card](https://huggingface.co/convaiinnovations/laya)

**ModernBERT.** Laya's encoder is based on ModernBERT-large by Answer.AI and LightOn, licensed under the Apache License, Version 2.0. [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)

**LLMRouterBench.** The training queries and per-model outcome labels come from LLMRouterBench, created by Hao Li and coauthors. No benchmark questions, candidate model outputs or outcome labels are redistributed. `tools/reference/arc_challenge_v2.npz` holds only this router's own probabilities for the ARC-Challenge test prompts, keyed by a SHA-256 hash of each prompt. [LLMRouterBench paper](https://arxiv.org/abs/2601.07206) · [Benchmark repository](https://github.com/ynulihao/LLMRouterBench)
