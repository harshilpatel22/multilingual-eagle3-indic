"""Offline draft-vocab coverage sweep: pick draft_vocab_size + English:Indic ratio for v3.

Builds a frequency draft-vocab from representative text (ShareGPT English + FLORES Indic),
then reports what % of the FLORES eval tokens (data/eval/*.jsonl) each language keeps,
across K in {32k,48k,64k} x English:Indic token ratios. CPU-only; no GPU, no model weights.

The Indic distribution is estimated from the FLORES `dev` split (disjoint from the
`devtest` eval set, so no leakage). The training responses are Qwen3's, generated on GPU
— but the per-language token *distribution* that drives top-K selection is well-
approximated by representative in-language text; the trained head's true coverage is
re-checked post-training (Task D4).
"""
import argparse
from pathlib import Path

import pandas as pd

from eagle3.common.io import read_jsonl, ensure_dir
from eagle3.analysis.tokenization import load_tokenizer
from eagle3.data.vocab_coverage import sweep
from eagle3.data.flores import load_parallel, LANG_CODES

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS = ROOT / "results"
SHAREGPT = ROOT / "data" / "regen" / "sharegpt_sample.jsonl"

THRESHOLDS = {"en": 0.96, "hi": 0.98, "gu": 0.98}


def tok_seqs(tokenizer, texts):
    return tokenizer(list(texts), add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    ap.add_argument("--ks", type=int, nargs="+", default=[32000, 48000, 64000])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0],
                    help="English:Indic TOKEN ratio (en_budget = ratio * indic_tokens)")
    ap.add_argument("--flores-n", type=int, default=500,
                    help="FLORES sentences/lang (excluding eval ids) for the Indic distribution")
    args = ap.parse_args()

    tok = load_tokenizer(args.model_id)

    # English corpus sample (diverse English from ShareGPT/etc.)
    en_texts = [r["text"] for r in read_jsonl(SHAREGPT)]

    # Indic corpus sample: FLORES `dev` split, DISJOINT-by-construction from the `devtest`
    # eval set (don't exclude by id — dev and devtest reuse ids 0..N for *different*
    # sentences; the splits share no content). data/eval/*.jsonl is the full devtest.
    flores = load_parallel({k: LANG_CODES[k] for k in ("hi", "gu")}, split="dev")[: args.flores_n]
    hi_texts = [r["hi"] for r in flores]
    gu_texts = [r["gu"] for r in flores]
    if not hi_texts or not gu_texts:
        raise SystemExit("no FLORES dev rows loaded — check the dataset/split")

    seqs_by_lang = {
        "en": tok_seqs(tok, en_texts),
        "hi": tok_seqs(tok, hi_texts),
        "gu": tok_seqs(tok, gu_texts),
    }
    # Eval tokens = the canonical FLORES eval set (matches the paper's coverage metric)
    eval_by_lang = {
        lang: tok_seqs(tok, [r["text"] for r in read_jsonl(EVAL_DIR / f"{lang}.jsonl")])
        for lang in ("en", "hi", "gu")
    }

    hi_tok = sum(len(s) for s in seqs_by_lang["hi"])
    gu_tok = sum(len(s) for s in seqs_by_lang["gu"])
    indic_tokens = hi_tok + gu_tok

    all_rows = []
    for ratio in args.ratios:
        budgets = {"hi": hi_tok, "gu": gu_tok, "en": int(ratio * indic_tokens)}
        rows = sweep(seqs_by_lang, eval_by_lang, budgets, args.ks)
        for r in rows:
            r["ratio"] = ratio
        all_rows.extend(rows)

    def cov(ratio, K):
        return {r["lang"]: r["coverage"] for r in all_rows if r["ratio"] == ratio and r["K"] == K}

    # A K is "safe" if it clears every threshold at EVERY swept ratio (so the ratio can be
    # chosen freely on training grounds). Once Indic is saturated, English coverage is the
    # only lever K moves -> recommend the smallest K attaining the best English coverage.
    safe_ks = [K for K in args.ks
               if all(cov(r, K)[l] >= THRESHOLDS[l] for r in args.ratios for l in THRESHOLDS)]
    rec_K = None
    if safe_ks:
        best_en = max(min(cov(r, K)["en"] for r in args.ratios) for K in safe_ks)
        rec_K = min(K for K in safe_ks if min(cov(r, K)["en"] for r in args.ratios) >= best_en - 1e-9)
    # Headline: is Indic coverage ~invariant to the English ratio? (=> ratio not coverage-bound)
    indic_invariant = all(abs(cov(args.ratios[0], K)[l] - cov(r, K)[l]) < 1e-3
                          for K in args.ks for r in args.ratios for l in ("hi", "gu"))

    df = pd.DataFrame(all_rows)
    ensure_dir(RESULTS)
    out = RESULTS / "vocab_sweep.csv"
    df.to_csv(out, index=False)
    print(df.pivot_table(index=["ratio", "K"], columns="lang", values="coverage").round(4).to_string())
    print(f"\nEnglish sample: {len(en_texts)} convs | Indic sample: {len(flores)} sents/lang "
          f"(hi {hi_tok} tok, gu {gu_tok} tok)")
    print(f"thresholds: {THRESHOLDS}")
    print(f"Indic coverage ~invariant to English ratio: {indic_invariant}  "
          f"(=> en:indic ratio is NOT coverage-constrained; set it on training grounds)")
    if rec_K is not None:
        en_lo = min(cov(r, rec_K)["en"] for r in args.ratios)
        print(f"RECOMMENDATION: draft_vocab_size={rec_K}  "
              f"(Indic ~100% at all ratios; English coverage {en_lo:.4f})")
    else:
        print("NO K clears all thresholds at all ratios — widen --ks (e.g. add 80000) or lower top --ratios.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
