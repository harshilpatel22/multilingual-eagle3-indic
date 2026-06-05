#!/bin/bash
# EXP-0 (decisive, no retrain): re-measure the v3 seed-0 head AND the v2 head under BOTH
# fa3 and flashinfer on FLORES[:50] en/hi/gu, spec 3/1/4, temp 0. Single GPU, sequential.
# Isolates the backend confound on identical weights + gives the first apples-to-apples
# v3-vs-v2 on a common backend. ~1 hr.
set +e
V3=/workspace/outputs/draft-v3-seed0/epoch_4_step_23765
V2=$(ls -dt /workspace/outputs/draft-multilingual-8b-v2/epoch_4_step_* 2>/dev/null | head -1)
echo "V3 head: $V3"; echo "V2 head: $V2"
bash /workspace/serve_measure.sh "$V3" fa3        v3_fa3
bash /workspace/serve_measure.sh "$V3" flashinfer v3_flashinfer
bash /workspace/serve_measure.sh "$V2" fa3        v2_fa3
bash /workspace/serve_measure.sh "$V2" flashinfer v2_flashinfer
python3 - <<'PY'
import csv, statistics, os
def m(tag):
    p=f"/workspace/results/exp0_backend/{tag}.csv"
    if not os.path.exists(p): return None
    d={}
    for r in csv.DictReader(open(p)): d.setdefault(r["lang"],[]).append(float(r["accept_length"]))
    return {k:statistics.mean(v) for k,v in d.items()}
rows={t:m(t) for t in ["v2_fa3","v2_flashinfer","v3_fa3","v3_flashinfer"]}
print("\n=== EXP-0: tau (FLORES[:50], cfg 3/1/4, temp 0) ===")
print(f"{'config':16}{'en':>8}{'hi':>8}{'gu':>8}")
for t,mm in rows.items():
    if mm: print(f"{t:16}{mm.get('en',0):8.3f}{mm.get('hi',0):8.3f}{mm.get('gu',0):8.3f}")
    else: print(f"{t:16}  (missing)")
def d(a,b,l):
    return (rows[a][l]-rows[b][l]) if rows.get(a) and rows.get(b) else float('nan')
print("\nBackend delta within each head (fa3 - flashinfer)  [near 0 => backend is NOT the cause]:")
for h in ["v3","v2"]:
    print(f"  {h}: en {d(h+'_fa3',h+'_flashinfer','en'):+.3f}  hi {d(h+'_fa3',h+'_flashinfer','hi'):+.3f}  gu {d(h+'_fa3',h+'_flashinfer','gu'):+.3f}")
print("\nApples-to-apples v3 - v2 on the SAME backend  [the real question]:")
for bk in ["fa3","flashinfer"]:
    print(f"  {bk}: en {d('v3_'+bk,'v2_'+bk,'en'):+.3f}  hi {d('v3_'+bk,'v2_'+bk,'hi'):+.3f}  gu {d('v3_'+bk,'v2_'+bk,'gu'):+.3f}")
print("\n(v2 3-seed fa3 band for context: en 1.31-1.47, hi 1.66-2.06, gu 1.86-2.44)")
PY
echo "EXP0_DONE $(date)"
