"""Build fresh Indic regen-input prompts from FLORES `dev` (disjoint from the devtest
eval) for the v3 Indic scale-up. Output = SpecForge regen schema:
  {"id": "hi-dev-0", "conversations": [{"role": "user", "content": <sentence>}]}
These get fed to the target (Qwen3-8B) on the pod to regenerate assistant responses.
"""
import argparse
from pathlib import Path

from eagle3.data.flores import load_parallel, LANG_CODES
from eagle3.common.io import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "gu"])
    ap.add_argument("--n", type=int, default=800, help="prompts per language")
    ap.add_argument("--split", default="dev", help="FLORES split (dev = disjoint from devtest eval)")
    ap.add_argument("--out", default=str(ROOT / "data" / "regen" / "regen_input_v3_indic.jsonl"))
    args = ap.parse_args()

    fl = load_parallel({k: LANG_CODES[k] for k in args.langs}, split=args.split)[: args.n]
    rows = []
    for lang in args.langs:
        for r in fl:
            rid = "{}-{}-{}".format(lang, args.split, r["id"])
            rows.append({"id": rid, "conversations": [{"role": "user", "content": r[lang]}]})
    write_jsonl(args.out, rows)
    print("wrote {} prompts ({} langs x {}) to {}".format(len(rows), len(args.langs), args.n, args.out))


if __name__ == "__main__":
    main()
