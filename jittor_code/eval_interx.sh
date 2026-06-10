#!/usr/bin/env bash
set -euo pipefail

# Inter-X second-stage evaluation. DATA_ROOT must contain processed/, splits/,
# and text2motion/checkpoints/ from the Inter-X dataset release.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/media/nas/jiaqi/t2m/t2m_data/InterX}"
IMAGE="${IMAGE:-interdist-jittor:cuda11.1}"
GPU_ID="${GPU_ID:-2}"
DOCKER_GPUS="${DOCKER_GPUS:-device=$GPU_ID}"
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
REPLICATION_TIMES="${REPLICATION_TIMES:-20}"
EVAL_MODEL_PTH="${EVAL_MODEL_PTH:-./checkpoints_eval/interx/text_mot_match/model/finest.pkl}"
EVAL_MODEL_ARG=""
if [[ -n "$EVAL_MODEL_PTH" ]]; then
  EVAL_MODEL_ARG="--eval-model-pth $EVAL_MODEL_PTH"
fi

cd "$ROOT/.."

docker run --rm --gpus "$DOCKER_GPUS" --network host -e use_mkl=0 \
  -v "$ROOT":/workspace \
  -v "$ROOT/.jittor_cache_docker":/root/.cache/jittor \
  -v "$DATA_ROOT":/data/InterX:ro \
  -w /workspace "$IMAGE" \
  bash -lc "python3 eval_t2m.py --dataname InterX --gpu_id $CONTAINER_GPU_ID \
    --data-root /data/InterX \
    --eval-batch-size $EVAL_BATCH_SIZE --replication-times $REPLICATION_TIMES $EVAL_MODEL_ARG \
    --vq_model_pth ./checkpoints/inter_x_vq_model.pth \
    --resume_trans ./checkpoints/inter_x_t2m_model.pth"
