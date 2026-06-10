# InterDist Jittor Implementation

[Back to the project homepage](../README.md)

This directory contains the Jittor implementation of InterDist. It supports training the VQ-VAE and InterDist Transformer on the InterHuman and Inter-X datasets. Data preprocessing, training, and evaluation all run inside Docker containers, so the host only needs Docker, GPU drivers, datasets, and model files.

All commands below are assumed to run from the `jittor_code/` directory.

## 1. Prepare the Docker Environment

### 1.1 Prerequisites

- Linux
- NVIDIA GPU with a working NVIDIA driver
- Docker
- NVIDIA Container Toolkit

First, verify that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:11.1.1-base-ubuntu20.04 nvidia-smi
```

### 1.2 Build the Image

```bash
docker build -t interdist-jittor:cuda11.1 -f docker/Dockerfile .
mkdir -p .jittor_cache_docker
```

The image is based on `jittor/jittor-cuda-11-1:latest`. Training and evaluation scripts use the image name `interdist-jittor:cuda11.1` by default. Override it with the `IMAGE` environment variable.

## 2. Prepare the Datasets

### 2.1 InterHuman

Download InterHuman according to the [InterGen instructions](https://github.com/tr3e/InterGen/tree/master?tab=readme-ov-file#2-get-data), then extract `motions_processed.zip`. The dataset root must contain at least:

```text
InterHuman/
├── annots/
├── motions_processed/
│   ├── person1/
│   └── person2/
└── split/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── ignore_list.txt
```

If the downloaded dataset does not contain `split/ignore_list.txt`, use the container to copy the file provided by this repository:

```bash
export DATA_ROOT=/absolute/path/to/InterHuman

docker run --rm \
  -v "$PWD":/workspace \
  -v "$DATA_ROOT":/data/InterHuman \
  -w /workspace interdist-jittor:cuda11.1 \
  cp data/ignore_list.txt /data/InterHuman/split/ignore_list.txt
```

InterHuman does not require any additional motion preprocessing.

### 2.2 Inter-X

Download the dataset according to the [Inter-X instructions](https://github.com/liangxuy/Inter-X?tab=readme-ov-file#dataset-download). Before preprocessing, the dataset root must contain at least:

```text
InterX/
├── motions/
│   └── <sequence_id>/
│       ├── P1.npz
│       └── P2.npz
├── processed/
│   ├── glove/
│   ├── motions/
│   │   ├── train.h5
│   │   ├── val.h5
│   │   └── test.h5
│   └── texts_processed/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Download the neutral body model from [SMPL-X](https://smpl-x.is.tue.mpg.de/) and place it at:

```text
data/body_model/smplx/SMPLX_NEUTRAL.npz
```

Use Docker to generate `processed/motions_norm/`:

```bash
export DATA_ROOT=/absolute/path/to/InterX

docker run --rm --gpus all --network host -e use_mkl=0 \
  -v "$PWD":/workspace \
  -v "$PWD/.jittor_cache_docker":/root/.cache/jittor \
  -v "$DATA_ROOT":/workspace/data/InterX \
  -w /workspace interdist-jittor:cuda11.1 \
  python3 data/prepare_dataset_interx.py
```

When preprocessing is complete, the following directory should be generated:

```text
InterX/processed/motions_norm/
├── train/
├── val/
└── test/
```

### 2.3 Prepare Training Statistics

Before training the VQ-VAE with `--ex_loss`, copy the repository statistics to the locations read by the code and datasets.

InterHuman:

```bash
export DATA_ROOT=/absolute/path/to/InterHuman

docker run --rm \
  -v "$PWD":/workspace \
  -v "$DATA_ROOT":/data/InterHuman \
  -w /workspace interdist-jittor:cuda11.1 \
  bash -lc '
    cp data/stats/interh_global_mean.npy data/stats/global_mean.npy
    cp data/stats/interh_global_std.npy data/stats/global_std.npy
    cp data/stats/interh_mean_dist.npy /data/InterHuman/motions_processed/mean_dist.npy
    cp data/stats/interh_std_dist.npy /data/InterHuman/motions_processed/std_dist.npy
  '
```

Inter-X:

```bash
export DATA_ROOT=/absolute/path/to/InterX

docker run --rm \
  -v "$PWD":/workspace \
  -v "$DATA_ROOT":/data/InterX \
  -w /workspace interdist-jittor:cuda11.1 \
  bash -lc '
    mkdir -p /data/InterX/processed/meta
    cp data/stats/interx_mean_reset.npy /data/InterX/processed/meta/mean_reset.npy
    cp data/stats/interx_std_reset.npy /data/InterX/processed/meta/std_reset.npy
    cp data/stats/interx_mean_dist.npy /data/InterX/processed/meta/mean_dist.npy
    cp data/stats/interx_std_dist.npy /data/InterX/processed/meta/std_dist.npy
  '
```

## 3. Prepare Model Files

Training periodically invokes the evaluation model for the corresponding dataset. Transformer training also requires CLIP text encoder parameters in a format readable by Jittor. Prepare the following files before training:

```text
checkpoints/
└── clip_vit_l_14_336px_text.pth
checkpoints_eval/
├── interhuman/
│   └── interclip.ckpt
└── interx/
    └── text_mot_match/
        └── model/
            └── finest.pkl
```

The VQ-VAE and Transformer weights used for formal evaluation must use the following fixed filenames:

```text
checkpoints/
├── interh_vq_model.pth
├── interh_t2m_model.pth
├── inter_x_vq_model.pth
└── inter_x_t2m_model.pth
```

Run the following script to download the InterHuman and Inter-X VQ-VAE and InterDist Transformer weights for the Jittor implementation from Google Drive:

```bash
bash data/download_models.sh
```

Use the Jittor weights released with this project or weights converted to a Jittor-readable format. Do not directly mix them with the original PyTorch weights.

## 4. Train with Docker

The repository provides two training scripts:

- `train_interhuman.sh`
- `train_interx.sh`

The scripts mount the current code directory, the Jittor compilation cache, and the host dataset, then run training inside the container. Training results are saved to:

```text
results/<dataset>/<experiment_name>/
├── net_best_fid.pth
├── net_last.pth
└── opt.txt
```

### 4.1 Train on InterHuman

Train the VQ-VAE and Transformer in sequence:

```bash
DATA_ROOT=/absolute/path/to/InterHuman \
GPU_ID=0 \
bash train_interhuman.sh all
```

Train only the VQ-VAE:

```bash
DATA_ROOT=/absolute/path/to/InterHuman \
GPU_ID=0 \
bash train_interhuman.sh vq
```

Train only the Transformer:

```bash
DATA_ROOT=/absolute/path/to/InterHuman \
GPU_ID=0 \
bash train_interhuman.sh t2m
```

By default, the Transformer loads:

```text
results/InterHuman/interh_vq_model/net_best_fid.pth
```

### 4.2 Train on Inter-X

Train the VQ-VAE and Transformer in sequence:

```bash
DATA_ROOT=/absolute/path/to/InterX \
GPU_ID=0 \
bash train_interx.sh all
```

You can also run the stages separately:

```bash
DATA_ROOT=/absolute/path/to/InterX GPU_ID=0 bash train_interx.sh vq
DATA_ROOT=/absolute/path/to/InterX GPU_ID=0 bash train_interx.sh t2m
```

By default, the Transformer loads:

```text
results/InterX/inter_x_vq_model/net_best_fid.pth
```

### 4.3 Training Script Parameters

| Environment Variable | Description | Default |
| --- | --- | --- |
| `DATA_ROOT` | Dataset root on the host; required | None |
| `IMAGE` | Docker image | `interdist-jittor:cuda11.1` |
| `GPU_ID` | Host GPU index exposed to the container | `0` |
| `DOCKER_GPUS` | Value passed to `docker run --gpus` | `device=$GPU_ID` |
| `CONTAINER_GPU_ID` | GPU index passed to Jittor inside the container | `0` |
| `EVAL_MODEL_PTH` | Evaluation model path inside the container | Dataset-specific |
| `VQ_EXP_NAME` | VQ-VAE experiment name | Dataset-specific |
| `VQ_BATCH_SIZE` | VQ-VAE training batch size | `128` |
| `VQ_TOTAL_ITER` | Total VQ-VAE training iterations | InterHuman `100000`, Inter-X `150000` |
| `VQ_LR` | VQ-VAE learning rate | InterHuman `2e-4`, Inter-X `1e-4` |
| `VQ_LR_SCHEDULER` | VQ-VAE learning-rate decay iteration | InterHuman `70000`, Inter-X `120000` |
| `T2M_EXP_NAME` | Transformer experiment name | Dataset-specific |
| `T2M_BATCH_SIZE` | Transformer training batch size | `48` |
| `T2M_TOTAL_ITER` | Total Transformer training iterations | `200000` |
| `T2M_LR` | Transformer learning rate | `2e-4` |
| `T2M_LR_SCHEDULER` | Transformer learning-rate decay iteration | InterHuman `60000`, Inter-X `50000` |
| `VQ_MODEL_PTH` | Path to the VQ-VAE weights used by the Transformer | `net_best_fid.pth` from the corresponding experiment |

For example, reduce the batch sizes for both InterHuman training stages:

```bash
DATA_ROOT=/absolute/path/to/InterHuman \
GPU_ID=0 \
VQ_BATCH_SIZE=64 \
T2M_BATCH_SIZE=24 \
bash train_interhuman.sh all
```

Train only the Transformer using existing VQ-VAE weights:

```bash
DATA_ROOT=/absolute/path/to/InterX \
GPU_ID=0 \
VQ_MODEL_PTH=./checkpoints/inter_x_vq_model.pth \
bash train_interx.sh t2m
```

## 5. Model Evaluation

Formal text-to-interaction motion generation evaluation also runs through Docker scripts:

- `eval_interhuman.sh`
- `eval_interx.sh`

To evaluate models produced by the current training run, first copy the best weights inside the container:

```bash
docker run --rm \
  -v "$PWD":/workspace \
  -w /workspace interdist-jittor:cuda11.1 \
  bash -lc '
    mkdir -p checkpoints
    cp results/InterHuman/interh_vq_model/net_best_fid.pth checkpoints/interh_vq_model.pth
    cp results/InterHuman/interh_t2m_model/net_best_fid.pth checkpoints/interh_t2m_model.pth
    cp results/InterX/inter_x_vq_model/net_best_fid.pth checkpoints/inter_x_vq_model.pth
    cp results/InterX/inter_x_t2m_model/net_best_fid.pth checkpoints/inter_x_t2m_model.pth
  '
```

Evaluate InterHuman:

```bash
DATA_ROOT=/absolute/path/to/InterHuman \
GPU_ID=0 \
bash eval_interhuman.sh
```

Evaluate Inter-X:

```bash
DATA_ROOT=/absolute/path/to/InterX \
GPU_ID=0 \
bash eval_interx.sh
```

For Inter-X, you can override the evaluation batch size and number of replications:

```bash
DATA_ROOT=/absolute/path/to/InterX \
GPU_ID=0 \
EVAL_BATCH_SIZE=16 \
REPLICATION_TIMES=20 \
bash eval_interx.sh
```

The evaluation reports FID, Diversity, R-Precision, Matching Score, and Multimodality.

