#!/usr/bin/env bash
set -euo pipefail

# InterHuman second-stage evaluation: text-to-motion transformer + VQ decoder.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/media/nas/jiaqi/t2m/t2m_data/InterHuman}"
IMAGE="${IMAGE:-interdist-jittor:cuda11.1}"
GPU_ID="${GPU_ID:-0}"
DOCKER_GPUS="${DOCKER_GPUS:-device=$GPU_ID}"
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0}"

cd "$ROOT/.."

docker run --rm --gpus "$DOCKER_GPUS" --network host -e use_mkl=0 \
  -v "$ROOT":/workspace \
  -v "$ROOT/.jittor_cache_docker":/root/.cache/jittor \
  -v "$DATA_ROOT":/data/InterHuman:ro \
  -w /workspace "$IMAGE" \
  bash -lc "python3 eval_t2m.py --dataname InterHuman --gpu_id $CONTAINER_GPU_ID \
    --data-root /data/InterHuman \
    --eval-model-pth ./checkpoints_eval/interhuman/interclip.ckpt \
    --eval-batch-size 96 --replication-times 20 \
    --vq_model_pth ./checkpoints/interh_vq_model.pth \
    --resume_trans ./checkpoints/interh_t2m_model.pth"
