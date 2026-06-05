import json, requests, statistics, csv
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
URL="http://127.0.0.1:30000/generate"
LANGS=["en","hi","gu"]; N=50
def gen(text):
    msgs=[{"role":"user","content":text}]
    prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    err=None
    for _ in range(3):
        try:
            return requests.post(URL,json={"text":prompt,"sampling_params":{"temperature":0,"max_new_tokens":128}},timeout=180).json()
        except Exception as e: err=e
    raise err
all_recs=[]; summary={}; samples={}
for lang in LANGS:
    rows=[json.loads(l) for l in open("/workspace/eval/"+lang+".jsonl")][:N]
    recs=[]
    for i,r in enumerate(rows):
        d=gen(r["text"]); mi=d["meta_info"]
        if i==0: samples[lang]=d["text"][:110]
        recs.append({"id":r["id"],"lang":lang,"accept_length":mi["spec_accept_length"],"accept_rate":mi["spec_accept_rate"],"completion_tokens":mi["completion_tokens"],"verify_ct":mi["spec_verify_ct"]})
    taus=[x["accept_length"] for x in recs]
    summary[lang]=(len(recs),statistics.mean(taus),statistics.median(taus)); all_recs+=recs
    print(lang+": n="+str(len(recs))+" MEAN_TAU="+format(statistics.mean(taus),".3f")+" median="+format(statistics.median(taus),".3f"), flush=True)
for lang in LANGS:
    print("["+lang+"] sample gen: "+repr(samples[lang]), flush=True)
with open("/workspace/results/phase1_tau_by_language.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["id","lang","accept_length","accept_rate","completion_tokens","verify_ct"]); w.writeheader(); w.writerows(all_recs)
print("=== SUMMARY: tau by language (Qwen3-8B + Tengyunw EAGLE3 English head) ===")
en=summary["en"][1]
for lang in LANGS:
    n,m,md=summary[lang]; print("  "+lang+": mean_tau="+format(m,".3f")+"  retention_vs_en="+format(m/en*100,".1f")+"%")
print("saved /workspace/results/phase1_tau_by_language.csv")
