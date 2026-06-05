#!/bin/bash
cd /workspace/SpecForge && source .venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.8 --cuda-graph-max-bs 1 --tp-size 1 --dtype bfloat16 \
  --attention-backend fa3 --host 127.0.0.1 --port 30000
