#!/bin/bash
# Atom: serve ONE (head, backend) with EAGLE3 spec 3/1/4, measure FLORES tau, kill server.
# Args: $1=head_path  $2=backend(fa3|flashinfer)  $3=tag  [$4=eval_dir(default /workspace/eval)]
# Writes /workspace/results/exp0_backend/<tag>.csv. HF_HUB_OFFLINE=1 (avoids the tokenizer hang).
set +e
HEAD="${1:?head path}"; BACKEND="${2:?backend}"; TAG="${3:?tag}"; EVALDIR="${4:-/workspace/eval}"
OUT=/workspace/results/exp0_backend; mkdir -p "$OUT"
LOG=/workspace/sm_${TAG}.log
if [ ! -e "$HEAD" ]; then echo "[$TAG] HEAD NOT FOUND: $HEAD" | tee -a "$LOG"; exit 2; fi
pkill -9 -f "[s]glang.launch_server"; sleep 5
cd /workspace/SpecForge && source .venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "[$TAG] serving $HEAD under $BACKEND $(date)" | tee -a "$LOG"
nohup python -m sglang.launch_server --model-path Qwen/Qwen3-8B --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$HEAD" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.8 --cuda-graph-max-bs 1 --tp-size 1 --dtype bfloat16 --attention-backend "$BACKEND" \
  --host 127.0.0.1 --port 30000 >> "$LOG" 2>&1 &
SV=$!
HEALTHY=no
for i in $(seq 1 150); do
  curl -s -o /dev/null -w "%{http_code}" localhost:30000/health 2>/dev/null | grep -q 200 && { HEALTHY=yes; break; }
  kill -0 $SV 2>/dev/null || break
  sleep 10
done
RC=1
if [ "$HEALTHY" = yes ]; then
  python /workspace/measure_tau.py --eval-dir "$EVALDIR" --langs en hi gu --n 50 --tag "$TAG" --out "$OUT/${TAG}.csv"
  RC=$?
else
  echo "[$TAG] SERVER FAILED TO START — tail:" | tee -a "$LOG"; tail -6 "$LOG"
fi
pkill -9 -f "[s]glang.launch_server"; sleep 4
echo "[$TAG] done rc=$RC $(date)"
exit $RC
