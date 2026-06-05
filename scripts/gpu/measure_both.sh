#!/bin/bash
# Measure FLORES + held-out Aya tau (config set at server launch) against the live SGLang
# server on :30000. Offline tokenizer (HF_HUB_OFFLINE) to avoid the online-check hang.
cd /workspace/SpecForge && source .venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "=== FLORES (config from server) ==="
python /workspace/measure_tau.py --eval-dir /workspace/eval --langs en hi gu --n 50 \
  --tag s0_flores --out /workspace/results/phase3_tau_v3_seed0_flores.csv
echo "===AYA==="
python /workspace/measure_tau.py --eval-dir /workspace/heldout --langs en hi gu --n 50 \
  --tag s0_aya --out /workspace/results/heldout_v3_seed0.csv
echo "===MEASURE_DONE==="
