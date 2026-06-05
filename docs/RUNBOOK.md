# H100 RUNBOOK (GPU phase — verified against live sources 2026-05-31)

Online mode is the default (frozen target resident on GPU, no big disk cache). Run long jobs in tmux.
Before training: pass the §4 fluency gate (prompt the target in hi/gu, confirm fluent).

## 0. Provisioning the box (run on the Mac; verified prices 2026-06-01)
On-demand H100 80GB: RunPod ~$1.99/hr community, ~$2.39-2.79 secure; Vast.ai ~$1.87-4.00 (marketplace);
Lambda ~$2.99-3.29 (CUDA/PyTorch/SSH preinstalled). Whole GPU phase ~ a few H100-hrs (~$20-50).
STOP/terminate the pod when idle; a network volume costs ~cents/GB/mo even when stopped.

Operating model: SSH-drive from the local Claude session (keeps repo + context).
Requirement: key-based, NON-interactive SSH from the Mac. Test before handing over:
    ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 "nvidia-smi"   # must print GPU, no password prompt

RunPod steps:
  1. Add the Mac SSH public key (~/.ssh/id_ed25519.pub) to RunPod > Settings > SSH Keys.
  2. Deploy Pod: 1x H100 80GB; template = recent PyTorch/CUDA image (avoids flash-attn/CUDA mismatch);
     attach a Network Volume (~150GB) mounted at /workspace; enable SSH.
  3. Copy the ssh command; confirm the nvidia-smi test above runs with no password prompt.

First actions once connected: verify GPU/CUDA/driver -> push this repo up -> clone + install
SpecForge + SGLang (reconcile live READMEs, §7) -> Phase-0 repro -> §4 fluency gate -> Phase 1.

## Phase 0 reproduction (Llama-3.1-8B) — examples/run_llama3.1_8b_eagle3_online.sh
torchrun --standalone --nproc_per_node 1 SpecForge/scripts/train_eagle3.py \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config SpecForge/configs/llama3-8B-eagle3.json \
  --train-data-path cache/dataset/sharegpt_train.jsonl \
  --output-dir outputs/llama3-8b-eagle3-sharegpt --num-epochs 10 --batch-size 1 \
  --tp-size 1 --learning-rate 1e-4 --max-length 4096 --chat-template llama3 \
  --cache-dir cache --attention-backend sdpa --target-model-backend sglang \
  --log-interval 10 --sglang-mem-fraction-static 0.25

## Phase 3 training (Qwen3-8B) — examples/run_qwen3_8b_eagle3_online.sh
#   swap --train-data-path to our multilingual regen; keep --chat-template qwen
torchrun --standalone --nproc_per_node 1 SpecForge/scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen3-8B \
  --draft-model-config SpecForge/configs/qwen3-8b-eagle3.json \
  --train-data-path data/regen/multilingual_train.jsonl \
  --output-dir outputs/draft-multilingual-8b --num-epochs 10 --batch-size 1 \
  --tp-size 1 --learning-rate 1e-4 --max-length 4096 --chat-template qwen \
  --cache-dir cache --embedding-key model.embed_tokens.weight

## tau benchmark — benchmarks/bench_eagle3.py (launches its OWN sglang server)
#   config tuple = batch,num_steps,eagle_topk,num_draft_tokens ; "1,0,0,0"=baseline
python SpecForge/benchmarks/bench_eagle3.py --model <TARGET> --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path <HEAD> --tp-size 1 --dtype bfloat16 \
  --config-list 1,0,0,0 1,3,1,4 --benchmark-list mtbench gsm8k --output-dir results/

## Serving / lossless check — sglang.launch_server
python -m sglang.launch_server --model <TARGET> --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path <HEAD> --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp 1 --dtype bfloat16

## OPEN ITEMS to resolve on the box
- Checkpoint/resume flags: not in the example scripts — check `train_eagle3.py --help`.
- No FLORES/multilingual benchmarker upstream. Phase-1 cross-lingual tau needs either
  (a) a custom benchmarker added to benchmarks/benchmarker/ + registry.py, or
  (b) our own client reading per-request acceptance from SGLang meta_info (gives raw per-request data).
- Training-data JSONL schema: read docs basic_usage/data_preparation.html before Phase 3 regen.
