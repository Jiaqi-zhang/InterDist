#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-all}"

if [[ "$STAGE" != "all" && "$STAGE" != "vq" && "$STAGE" != "t2m" ]]; then
  echo "Usage: DATA_ROOT=/absolute/path/to/InterHuman bash $0 [all|vq|t2m]" >&2
  exit 2
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "DATA_ROOT must point to the InterHuman dataset root." >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 2
fi

DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
IMAGE="${IMAGE:-interdist-jittor:cuda11.1}"
GPU_ID="${GPU_ID:-0}"
DOCKER_GPUS="${DOCKER_GPUS:-device=$GPU_ID}"
CONTAINER_GPU_ID="${CONTAINER_GPU_ID:-0}"
EVAL_MODEL_PTH="${EVAL_MODEL_PTH:-./checkpoints_eval/interhuman/interclip.ckpt}"

VQ_EXP_NAME="${VQ_EXP_NAME:-interh_vq_model}"
VQ_BATCH_SIZE="${VQ_BATCH_SIZE:-128}"
VQ_TOTAL_ITER="${VQ_TOTAL_ITER:-100000}"
VQ_LR="${VQ_LR:-2e-4}"
VQ_LR_SCHEDULER="${VQ_LR_SCHEDULER:-70000}"

T2M_EXP_NAME="${T2M_EXP_NAME:-interh_t2m_model}"
T2M_BATCH_SIZE="${T2M_BATCH_SIZE:-48}"
T2M_TOTAL_ITER="${T2M_TOTAL_ITER:-200000}"
T2M_LR="${T2M_LR:-2e-4}"
T2M_LR_SCHEDULER="${T2M_LR_SCHEDULER:-60000}"
VQ_MODEL_PTH="${VQ_MODEL_PTH:-./results/InterHuman/$VQ_EXP_NAME/net_best_fid.pth}"

mkdir -p "$ROOT/.jittor_cache_docker"

run_in_docker() {
  docker run --rm --gpus "$DOCKER_GPUS" --network host \
    -e use_mkl=0 \
    -v "$ROOT":/workspace \
    -v "$ROOT/.jittor_cache_docker":/root/.cache/jittor \
    -v "$DATA_ROOT":/data/InterHuman:ro \
    -w /workspace "$IMAGE" "$@"
}

if [[ "$STAGE" == "all" || "$STAGE" == "vq" ]]; then
  run_in_docker python3 train_vq.py \
    --gpu_id "$CONTAINER_GPU_ID" \
    --dataname InterHuman \
    --data-root /data/InterHuman \
    --eval-model-pth "$EVAL_MODEL_PTH" \
    --batch-size "$VQ_BATCH_SIZE" \
    --lr "$VQ_LR" \
    --total-iter "$VQ_TOTAL_ITER" \
    --lr-scheduler "$VQ_LR_SCHEDULER" \
    --ex_loss \
    --exp_name "$VQ_EXP_NAME"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "t2m" ]]; then
  run_in_docker python3 train_t2m.py \
    --gpu_id "$CONTAINER_GPU_ID" \
    --dataname InterHuman \
    --data-root /data/InterHuman \
    --eval-model-pth "$EVAL_MODEL_PTH" \
    --batch-size "$T2M_BATCH_SIZE" \
    --lr "$T2M_LR" \
    --total-iter "$T2M_TOTAL_ITER" \
    --lr-scheduler "$T2M_LR_SCHEDULER" \
    --exp_name "$T2M_EXP_NAME" \
    --vq_model_pth "$VQ_MODEL_PTH"
fi
