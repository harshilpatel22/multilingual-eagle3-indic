"""Generic acceptance-length (tau) measurement client (runs on the pod).

Hits a live SGLang /generate (server already launched with a specific head + spec config),
applies the Qwen3 chat template (enable_thinking=False), temperature 0, and records
meta_info.spec_accept_length per request. Works for FLORES (eval/) or Aya (heldout/) by
pointing --eval-dir. The spec config (num-steps/topk/draft-tokens) is fixed at SERVER
launch, so re-launch the server to change it; pass --tag to label the output rows.
"""
import argparse, json, csv, statistics, requests
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="/workspace/eval")
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "gu"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="", help="label written into each row (e.g. seed0_cfg314_flores)")
    ap.add_argument("--url", default="http://127.0.0.1:30000/generate")
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_id)

    def gen(text):
        p = tok.apply_chat_template([{"role": "user", "content": text}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        err = None
        for _ in range(3):
            try:
                return requests.post(a.url, json={"text": p, "sampling_params":
                                     {"temperature": 0, "max_new_tokens": a.max_new}}, timeout=240).json()
            except Exception as ex:
                err = ex
        raise err

    rows = []
    for l in a.langs:
        data = [json.loads(x) for x in open(a.eval_dir + "/" + l + ".jsonl")][:a.n]
        accs = []
        for r in data:
            resp = gen(r["text"])
            acc = resp["meta_info"]["spec_accept_length"]
            rows.append({"id": r["id"], "lang": l, "accept_length": acc, "tag": a.tag})
            accs.append(acc)
        print("%s: tau=%.3f (n=%d)" % (l, statistics.mean(accs), len(accs)), flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "accept_length", "tag"])
        for z in rows:
            w.writerow([z["id"], z["lang"], z["accept_length"], z["tag"]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
