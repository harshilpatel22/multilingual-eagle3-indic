"""Build the v3 training corpus ON THE POD and verify real-corpus draft-vocab coverage.

Corpus = existing 700hi+700gu (extracted from balanced_train by id prefix)
       + fresh regenerated Indic (regen_v3_indic.jsonl)
       + N UltraChat English convs, sized so english_tokens ~= RATIO * indic_tokens.
Then builds the top-`draft_vocab` frequency vocab over the REAL corpus and reports
en/hi/gu coverage on the FLORES eval set (sanity vs the offline sweep; escalate to 48k
only if 32k English coverage is materially below ~96.5%).

Token budget + coverage both use raw content tokens (add_special_tokens=False), matching
scripts/vocab_sweep.py so the numbers are comparable.
"""
import argparse, json, random
from collections import Counter
from transformers import AutoTokenizer


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def content_ids(tok, conv):
    out = []
    for t in conv["conversations"]:
        out += tok(t.get("content", ""), add_special_tokens=False)["input_ids"]
    return out


def lang_of(row):
    p = row["id"].split("-")[0]
    return "en" if p.startswith("en") else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", default="/workspace/data/balanced_train.jsonl")
    ap.add_argument("--regen-indic", default="/workspace/data/regen_v3_indic.jsonl")
    ap.add_argument("--english", default="/workspace/data/ultrachat_en.jsonl/ultrachat_train.jsonl")
    ap.add_argument("--eval-dir", default="/workspace/eval")
    ap.add_argument("--out", default="/workspace/data/multilingual_v3_train.jsonl")
    ap.add_argument("--ratio", type=float, default=1.0, help="english:indic TOKEN ratio")
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draft-vocab", type=int, default=32000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)

    # ---- Indic: existing 700hi+700gu + fresh regen ----
    bal = read_jsonl(args.balanced)
    indic_existing = [r for r in bal if r["id"].split("-")[0] in ("hi", "gu")]
    indic_new = read_jsonl(args.regen_indic)
    indic = indic_existing + indic_new
    indic_ids = [content_ids(tok, r) for r in indic]
    indic_tok = sum(len(x) for x in indic_ids)

    # ---- English: take UltraChat convs until english_tokens ~= ratio*indic_tok ----
    english_all = read_jsonl(args.english)
    random.seed(args.seed)
    random.shuffle(english_all)
    target_en = args.ratio * indic_tok
    english_sel, eng_ids, en_tok = [], [], 0
    for r in english_all:
        if en_tok >= target_en:
            break
        ids = content_ids(tok, r)
        english_sel.append({"id": "en-" + str(r["id"])[:16], "conversations": r["conversations"]})
        eng_ids.append(ids)
        en_tok += len(ids)

    corpus = indic + english_sel
    random.seed(args.seed)
    random.shuffle(corpus)
    write_jsonl(args.out, corpus)

    comp = Counter(lang_of(r) for r in corpus)
    print("composition (convs):", dict(comp))
    print("  indic=%d (existing %d + new %d) | english=%d | total=%d"
          % (len(indic), len(indic_existing), len(indic_new), len(english_sel), len(corpus)))
    print("  tokens: indic=%d english=%d  en:indic ratio=%.3f" % (indic_tok, en_tok, en_tok / indic_tok))
    print("wrote", args.out)

    # ---- real-corpus coverage check ----
    freq = Counter()
    for ids in indic_ids + eng_ids:
        freq.update(ids)
    top = set(t for t, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:args.draft_vocab])
    print("--- real-corpus %dk draft-vocab coverage on FLORES eval[:50] ---" % (args.draft_vocab // 1000))
    for lang in ("en", "hi", "gu"):
        rows = read_jsonl(args.eval_dir + "/" + lang + ".jsonl")[:50]
        tot = ins = 0
        for r in rows:
            ids = tok(r["text"], add_special_tokens=False)["input_ids"]
            tot += len(ids)
            ins += sum(1 for i in ids if i in top)
        print("  %s: %.4f" % (lang, ins / tot if tot else 0.0))


if __name__ == "__main__":
    main()
