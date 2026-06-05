"""Download a diverse-English conversation sample to data/regen/sharegpt_sample.jsonl
(local, for the vocab sweep + English token-per-conversation mean). Best-effort English.

The vocab sweep only needs a representative diverse-English text sample to estimate the
English token-frequency distribution + mean tokens/conversation; the ACTUAL training
English corpus is SpecForge's ShareGPT, assembled on the pod (Task C2). Source is a CLI
arg so we can swap mirrors; handles ShareGPT-style ({conversations:[{value}]}) and
messages-style ({messages:[{content}]}) schemas. CPU-only; no model weights.
"""
import argparse
from pathlib import Path

from eagle3.common.io import write_jsonl, ensure_dir
from eagle3.analysis.tokenization import load_tokenizer, count_tokens

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "regen" / "sharegpt_sample.jsonl"


def conv_to_text(example):
    """Flatten a conversation's turns into one text blob (schema-agnostic)."""
    turns = (example.get("conversations") or example.get("conversation")
             or example.get("messages") or [])
    parts = []
    for t in turns:
        if isinstance(t, dict):
            parts.append(t.get("value") or t.get("content") or "")
        elif isinstance(t, str):
            parts.append(t)
    return "\n".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="anon8231489123/ShareGPT_Vicuna_unfiltered")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3000, help="conversations to sample")
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset_id, split=args.split, streaming=True)
    rows = []
    for ex in ds:
        text = conv_to_text(ex).strip()
        if len(text) > 200:           # skip near-empty conversations
            rows.append({"text": text})
        if len(rows) >= args.n:
            break

    if not rows:
        raise SystemExit("no conversations extracted — check --dataset-id/--split and the "
                         "schema handled in conv_to_text()")

    ensure_dir(OUT.parent)
    write_jsonl(OUT, rows)

    tok = load_tokenizer(args.model_id)
    counts = count_tokens([r["text"] for r in rows], tok)
    mean_tok = sum(counts) / len(counts)
    print(f"wrote {OUT}  ({len(rows)} conversations)")
    print(f"mean English tokens/conversation = {mean_tok:.1f}")
    print("(feed the mean to balance.english_convs_for_token_ratio in Task C2)")


if __name__ == "__main__":
    main()
