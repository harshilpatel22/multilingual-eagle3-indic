# Multilingual EAGLE-3 for Indic Languages

> **Cross-lingual acceptance-length (τ) degradation in EAGLE-3 speculative decoding, its tokenization-level mechanism, and a documented Indic EAGLE-3 draft head that recovers it.**

Companion code and data for *"Lost in Speculation: Cross-Lingual Acceptance-Length Degradation in EAGLE-3 and a Recovered Indic Draft Head"* (Harshil Patel), under review at TMLR.
🤗 Released head: [`SwitchXDDD/multilingual-eagle3-qwen3-8b`](https://huggingface.co/SwitchXDDD/multilingual-eagle3-qwen3-8b) (Apache-2.0).

## TL;DR

EAGLE-3 makes English LLM generation 2–5× faster, *losslessly* — but the draft heads in common use are English-trained, and acceptance length **τ collapses on Indic languages** (Qwen3-8B English head: τ 2.37 en → 1.36 hi → **1.07 gu**, i.e. ≈ no speedup). The cause is concrete: EAGLE-3 heads predict over a frequency-built **32k "draft vocabulary"** that **excludes ~half of all Hindi/Gujarati tokens**, so the draft can never propose them. Training a small Indic head on the target's *own* multilingual outputs **recovers** most of the lost τ — and it holds out-of-domain and at 32B.

| Qwen3-8B (FLORES, n=50) | English | Hindi | Gujarati |
|---|---|---|---|
| public English head | 2.37 | 1.36 | 1.07 |
| **our Indic head (v2, 3-seed)** | 1.40 | **1.86 ± 0.20** | **2.16 ± 0.29** |

We also report a controlled **negative result** (paper §5.5): token-balancing diverse English *trades away* Indic acceptance, because EAGLE-3 learning tracks **gradient mass**, not vocabulary **coverage**.

## Repository layout

| path | what |
|---|---|
| `src/eagle3/` | analysis package — tokenization inflation, draft-vocab coverage, data builders (+ unit tests in `tests/`) |
| `scripts/` | eval-set builder, tokenization analysis, the offline draft-vocab-coverage sweep, bootstrap CIs, figure generation |
| `scripts/gpu/` | GPU-phase scripts — serve + measure τ, regenerate training data, build the corpus, train seeds |
| `configs/` | EAGLE-3 draft-model configs (Qwen3-8B / 32B) |
| `results/` | all raw per-request τ CSVs + publication figures |
| `docs/RUNBOOK.md` | H100 provisioning + command runbook |

## Reproduce (local, CPU — the mechanism)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
hf auth login                          # FLORES-200 is gated — accept terms on its HF page
python scripts/build_eval_set.py        # FLORES+ -> data/eval/{en,hi,gu}.jsonl  (not redistributed)
python scripts/analyze_tokenization.py  # tokenization inflation -> results/ + figure
python scripts/vocab_sweep.py           # offline draft-vocab coverage — the $0 mechanism check
pytest                                  # unit tests (offline)
```

## Use the released head (GPU, SGLang)

```bash
python -m sglang.launch_server --model Qwen/Qwen3-8B --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path SwitchXDDD/multilingual-eagle3-qwen3-8b \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --dtype bfloat16
```
Training and τ measurement run on a rented H100 — see [`docs/RUNBOOK.md`](docs/RUNBOOK.md) and `scripts/gpu/`.

## Honest scope

A **research preview**: small (~2,100-conversation) FLORES-derived corpus, Hindi + Gujarati only, with a characterized **English tradeoff** (the head is Indic-specialized). Total compute: **≈ $60** of RunPod GPU credits, self-funded.

## Citation

```bibtex
@article{patel2026lostinspeculation,
  title  = {Lost in Speculation: Cross-Lingual Acceptance-Length Degradation in EAGLE-3 and a Recovered Indic Draft Head},
  author = {Patel, Harshil},
  year   = {2026},
  note   = {Under review at TMLR},
  url    = {https://github.com/harshilpatel22/multilingual-eagle3-indic}
}
```

## License & data provenance

Code released under **Apache-2.0** ([`LICENSE`](LICENSE)); the released head is Apache-2.0. **FLORES-200 evaluation data is NOT redistributed here** (CC BY-SA 4.0, gated) — `scripts/build_eval_set.py` regenerates it from the official source. Training prompts derive from FLORES-200; responses are Qwen3-8B's own generations. Please also cite EAGLE-3, SpecForge, Qwen3, FLORES-200, and Aya.
