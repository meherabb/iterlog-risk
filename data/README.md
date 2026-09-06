# `data/` — no data is vendored here

Consistent with standard practice (and to keep this repository small and license-clean), no
dataset files are committed. Everything is fetched at run time through the Hugging Face
`datasets` library or `torchvision`, from the following public, unmodified sources:

| Dataset | Source | Role in the paper |
|---|---|---|
| ARC-Easy / ARC-Challenge | `allenai/ai2_arc` | LLM pool constituents |
| OpenBookQA | `allenai/openbookqa` | LLM pool constituent |
| CommonsenseQA | `tau/commonsense_qa` | LLM pool constituent |
| MMLU | `cais/mmlu` | Held-out group; group-conditional study (Appendix F) |
| TriviaQA | `mandarjoshi/trivia_qa` | Tier B (genuine decoding) |
| CIFAR-10 / CIFAR-10-C | `torchvision.datasets.CIFAR10`; CIFAR-10-C via `tensorflow-datasets`-style corruption files | Vision matched control; exchangeability stress test |

Run `python -m src.data.fetch_all` to download and cache everything under `data/cache/`
(gitignored). No dataset is modified from its public release; the only repository-side logic
is pool construction (capping the four-dataset pool at 24,000 items per replication draw,
matching Appendix E) and the score-family feature extraction described in
`docs/ARCHITECTURE.md`.

No model weights are vendored either — Pythia-410M, Qwen2.5-1.5B-Instruct, and
Phi-3.5-mini-instruct are pulled from the Hugging Face Hub on first use and cached locally.
