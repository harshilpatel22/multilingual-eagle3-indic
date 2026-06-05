import json, requests, statistics, csv
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
U="http://127.0.0.1:30000/generate"; L=["en","hi","gu"]; N=50
ENG={"en":2.367,"hi":1.356,"gu":1.067}; V1={"en":1.289,"hi":1.678,"gu":1.968}
def g(x):
    p=tok.apply_chat_template([{"role":"user","content":x}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    e=None
    for _ in range(3):
        try: return requests.post(U,json={"text":p,"sampling_params":{"temperature":0,"max_new_tokens":128}},timeout=180).json()
        except Exception as ex: e=ex
    raise e
rows_out=[]; v2={}
for l in L:
    rows=[json.loads(x) for x in open("/workspace/eval/"+l+".jsonl")][:N]
    rr=[{"id":r["id"],"lang":l,"accept_length":g(r["text"])["meta_info"]["spec_accept_length"]} for r in rows]
    v2[l]=statistics.mean([z["accept_length"] for z in rr]); rows_out+=rr
    print(l+": v2_tau="+format(v2[l],".3f"),flush=True)
with open("/workspace/results/phase3_tau_v2.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","lang","accept_length"])
    [w.writerow([z["id"],z["lang"],z["accept_length"]]) for z in rows_out]
print("=== en-head | v1 (indic-heavy 5ep) | v2 (balanced 5ep) ===")
for l in L: print("  "+l+": eng="+format(ENG[l],".3f")+"  v1="+format(V1[l],".3f")+"  v2="+format(v2[l],".3f"))
print("saved /workspace/results/phase3_tau_v2.csv")
