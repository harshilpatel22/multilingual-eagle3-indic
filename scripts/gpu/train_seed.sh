#!/bin/bash
# Train ONE v3 seed UNINTERRUPTED (no --resume), 5 epochs, same recipe as v3 seed-0, then
# free disk (keep only the final head's config+safetensors; drop the 3.2GB training_state.pt
# and any intermediate checkpoints — avoids the volume-quota blowup). Arg: $1 = seed number.
set +e
SEED="${1:?seed number}"
cd /workspace/SpecForge && source .venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=/workspace/cache/compiled
OUT=/workspace/outputs/draft-v3-seed$SEED
rm -rf "$OUT"   # clean start — explicitly NO --resume (the whole point of this experiment)
echo "=== train v3 seed=$SEED START $(date) ==="
torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen3-8B \
  --draft-model-config /workspace/SpecForge/configs/qwen3-8b-eagle3.json \
  --train-data-path /workspace/data/multilingual_v3_train.jsonl \
  --output-dir "$OUT" \
  --num-epochs 5 --learning-rate 1e-4 --max-length 4096 --batch-size 1 \
  --chat-template qwen --cache-dir /workspace/cache --tp-size 1 \
  --embedding-key model.embed_tokens.weight --attention-backend sdpa --seed "$SEED" --save-interval 100000
echo "=== train v3 seed=$SEED exit=$? $(date) ==="
# disk hygiene: keep only the final epoch dir; drop training_state.pt (resume-only) + intermediates
FINAL=$(ls -dt "$OUT"/epoch_* 2>/dev/null | head -1)
for dd in "$OUT"/epoch_*; do [ "$dd" != "$FINAL" ] && rm -rf "$dd"; done
[ -n "$FINAL" ] && rm -f "$FINAL/training_state.pt"
echo "final head: $FINAL"; ls -la "$FINAL" 2>/dev/null
df -h /workspace 2>/dev/null | tail -1
