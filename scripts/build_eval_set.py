"""Build the multilingual eval set: FLORES+ -> data/eval/{en,hi,gu}.jsonl (aligned by id).

Requires HF auth (FLORES+ is gated). Idempotent: skips if all outputs already exist.
"""
import argparse
from pathlib import Path

from eagle3.common.io import write_jsonl
from eagle3.data.flores import LANG_CODES, load_parallel
from eagle3.data.build import to_per_language_rows

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--langs", nargs="+", default=list(LANG_CODES))
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = ap.parse_args()

    targets = {lang: EVAL_DIR / f"{lang}.jsonl" for lang in args.langs}
    if not args.force and all(p.exists() for p in targets.values()):
        print(f"All eval files already exist in {EVAL_DIR} (use --force to rebuild).")
        return

    codes = {lang: LANG_CODES[lang] for lang in args.langs}
    aligned = load_parallel(lang_codes=codes, split=args.split)
    per_lang = to_per_language_rows(aligned, args.langs)
    for lang, path in targets.items():
        write_jsonl(path, per_lang[lang])
        print(f"wrote {len(per_lang[lang])} rows -> {path}")


if __name__ == "__main__":
    main()
