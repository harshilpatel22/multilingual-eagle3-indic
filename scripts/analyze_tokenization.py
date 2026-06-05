"""Compute per-language token-inflation (Qwen3 tokenizer) over the FLORES eval set.

Outputs results/phase2_tokenization_corr.csv (tau_drop left empty — filled in Phase 1)
and results/figures/tokenization.png. CPU-only; downloads the Qwen3 tokenizer (no weights).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eagle3.common.io import read_jsonl, ensure_dir
from eagle3.analysis.tokenization import build_tokenization_table, load_tokenizer, count_tokens

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "gu"])
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    args = ap.parse_args()

    texts_by_lang = {
        lang: [r["text"] for r in read_jsonl(EVAL_DIR / f"{lang}.jsonl")]
        for lang in args.langs
    }
    tok = load_tokenizer(args.model_id)
    rows = build_tokenization_table(texts_by_lang, lambda xs: count_tokens(xs, tok), base="en")

    df = pd.DataFrame(rows)
    ensure_dir(RESULTS)
    csv_path = RESULTS / "phase2_tokenization_corr.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")

    ensure_dir(RESULTS / "figures")
    ax = df.plot.bar(x="lang", y="inflation_vs_en", legend=False)
    ax.set_ylabel("token inflation vs. English")
    ax.set_title(f"Token inflation by language ({args.model_id} tokenizer)")
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    fig_path = RESULTS / "figures" / "tokenization.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
