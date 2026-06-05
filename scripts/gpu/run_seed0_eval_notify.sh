#!/bin/bash
# Autonomous (pod-side, nohup): wait for seed-0 training (PID passed as $1) to finish,
# serve the trained head, measure FLORES + Aya tau (config 3/1/4), then notify via
# ntfy.sh (push + email) with the results + "safe to stop the pod". Session-independent.
set +e
TRAIN_PID="${1:?need training PID}"
TOPIC="eagle3-harshil-v3done-x7k2"
EMAIL="harshilgami26@gmail.com"
LOG=/workspace/seed0_eval.log
echo "=== autonomous eval+notify started $(date) (watching PID $TRAIN_PID) ===" > "$LOG"

notify() {  # $1 = message  (topic push only; ntfy blocks anonymous email)
  curl -s -H "Title: EAGLE3 v3 seed0" -H "Priority: high" -H "Tags: white_check_mark" \
       -d "$1" "https://ntfy.sh/$TOPIC" >> "$LOG" 2>&1
  echo "$1" > /workspace/seed0_RESULTS.txt
}

# 1. wait for training to finish
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "training exited $(date)" >> "$LOG"
sleep 20  # let the final checkpoint flush to the network FS

# 2. locate the final checkpoint
CKPT=$(ls -d /workspace/outputs/draft-v3-seed0/epoch_4_step_* 2>/dev/null | sort -V | tail -1)
[ -z "$CKPT" ] && CKPT=$(ls -d /workspace/outputs/draft-v3-seed0/epoch_* 2>/dev/null | sort -V | tail -1)
echo "CKPT=$CKPT" >> "$LOG"
if [ -z "$CKPT" ]; then
  notify "v3 seed0 training finished but NO checkpoint was found - eval skipped. Keep the pod up and check."
  echo "NOCKPT $(date)" > /workspace/seed0_ALLDONE.txt; exit 1
fi

# 3. serve the trained head (flashinfer for H100/A100, config 3/1/4)
cd /workspace/SpecForge && source .venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
pkill -9 -f "[s]glang"; sleep 5
nohup python -m sglang.launch_server --model-path Qwen/Qwen3-8B --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$CKPT" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.8 --cuda-graph-max-bs 1 --tp-size 1 --dtype bfloat16 --attention-backend flashinfer \
  --host 127.0.0.1 --port 30000 > /workspace/serve_v3_eval.log 2>&1 &
SVPID=$!

# 4. wait for server health (slow FS cold start; up to ~25 min)
HEALTHY=no
for i in $(seq 1 100); do
  curl -s -o /dev/null -w "%{http_code}" localhost:30000/health 2>/dev/null | grep -q 200 && { HEALTHY=yes; break; }
  kill -0 "$SVPID" 2>/dev/null || break
  sleep 15
done
echo "server healthy=$HEALTHY $(date)" >> "$LOG"
if [ "$HEALTHY" != "yes" ]; then
  notify "v3 seed0 trained OK but the eval server failed to start - tau NOT measured. Keep the pod up and check."
  pkill -9 -f "[s]glang"; echo "SERVERFAIL $(date)" > /workspace/seed0_ALLDONE.txt; exit 1
fi

# 5. measure FLORES + held-out Aya (config 3/1/4)
python /workspace/measure_tau.py --eval-dir /workspace/eval    --langs en hi gu --n 50 --tag seed0_flores_314 \
  --out /workspace/results/phase3_tau_v3_seed0_flores.csv >> "$LOG" 2>&1
python /workspace/measure_tau.py --eval-dir /workspace/heldout --langs en hi gu --n 50 --tag seed0_aya_314 \
  --out /workspace/results/heldout_v3_seed0.csv >> "$LOG" 2>&1

# 6. tear the server down (free GPU)
pkill -9 -f "[s]glang"; sleep 3

# 7. compute per-language means and build the message
python - <<'PY' >> "$LOG" 2>&1
import csv, statistics
def m(path):
    d={}
    try:
        for r in csv.DictReader(open(path)): d.setdefault(r["lang"],[]).append(float(r["accept_length"]))
    except Exception: return {}
    return {k:statistics.mean(v) for k,v in d.items()}
fl=m("/workspace/results/phase3_tau_v3_seed0_flores.csv"); ay=m("/workspace/results/heldout_v3_seed0.csv")
def g(d,k): return d.get(k,float("nan"))
msg=("v3 seed0 DONE. FLORES tau en=%.2f hi=%.2f gu=%.2f | Aya en=%.2f hi=%.2f gu=%.2f "
     "(eng-head baseline en2.37/hi1.36/gu1.07). Safe to STOP the H100 pod."
     %(g(fl,"en"),g(fl,"hi"),g(fl,"gu"),g(ay,"en"),g(ay,"hi"),g(ay,"gu")))
open("/workspace/seed0_msg.txt","w").write(msg)
PY
MSG=$(cat /workspace/seed0_msg.txt 2>/dev/null)
[ -z "$MSG" ] && MSG="v3 seed0 eval finished but results failed to parse - check the pod."
notify "$MSG"
echo "DONE $(date)" > /workspace/seed0_ALLDONE.txt
echo "=== all done $(date) ===" >> "$LOG"
