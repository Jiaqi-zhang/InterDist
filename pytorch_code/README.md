# InterDist PyTorch Implementation

[Back to the project homepage](../README.md)

This directory contains the PyTorch implementation of InterDist. It supports VQ-VAE training, InterDist Transformer training, and model evaluation on the InterHuman and Inter-X datasets.


## Environment Setup

```bash
conda env create -f environment.yml
conda activate t2m
pip install gdown
```

This environment is based on Python 3.9.19, PyTorch 2.3.1, and CUDA 12.1. Adjust the relevant dependencies to match your local CUDA and driver versions.

## Data Preparation

All commands are assumed to run from the `pytorch_code/` directory.

### InterHuman

Download the InterHuman dataset according to the [InterGen instructions](https://github.com/tr3e/InterGen/tree/master?tab=readme-ov-file#2-get-data) and arrange it as follows:

```text
data/InterHuman/
├── annots/
├── checkpoints/
│   └── ViT-L-14-336px.pt
├── eval_model/
│   └── interclip.ckpt
├── motions/
├── motions_processed/
└── split/
```

If `data/InterHuman/split/` does not contain `ignore_list.txt`, copy `data/ignore_list.txt` into that directory.

### Inter-X

Download the dataset according to the [Inter-X instructions](https://github.com/liangxuy/Inter-X?tab=readme-ov-file#dataset-download) and arrange it as follows:

```text
data/InterX/
├── processed/
│   ├── glove/
│   ├── motions/
│   ├── texts_processed/
│   └── inter-x.h5
├── splits/
└── text2motion/
    └── checkpoints/
```

Download the model file from [SMPL-X](https://smpl-x.is.tue.mpg.de/) and place it according to the instructions in `data/body_model/smplx/Put_SMPLX_NEUTRAL_npz_file_to_here.txt`. Then run:

```bash
python data/prepare_dataset_interx.py
```

The processed motion data will be saved to `data/InterX/processed/motions_norm/`.

## Download Pretrained Models

The PyTorch and Jittor pretrained models are independent. Run the following script to download the InterHuman and Inter-X VQ-VAE and InterDist Transformer weights for the PyTorch implementation from Google Drive:

```bash
bash data/download_models.sh
```

After the download, the following files should be available:

```text
checkpoints/
├── interh_vq_model.pth
├── interh_t2m_model.pth
├── inter_x_vq_model.pth
└── inter_x_t2m_model.pth
```

Dataset evaluation models are not included in this script. Place the InterHuman `interclip.ckpt` file in `data/InterHuman/eval_model/` with the dataset. The Inter-X evaluation model must be located in `data/InterX/text2motion/checkpoints/`.

These weights are only compatible with the PyTorch implementation. Do not mix them with pretrained models from the Jittor directory.

## Model Training

Training results are saved to `results/<dataset>/<experiment_name>/` by default.

### 1. Train the VQ-VAE

InterHuman:

```bash
python train_vq.py --gpu_id 0 --dataname InterHuman \
    --lr 2e-4 --total-iter 100000 --lr-scheduler 70000 --ex_loss \
    --exp_name interh_vq_model
```

Inter-X:

```bash
python train_vq.py --gpu_id 0 --dataname InterX \
    --lr 1e-4 --total-iter 150000 --lr-scheduler 120000 --ex_loss \
    --exp_name inter_x_vq_model
```

### 2. Train the InterDist Transformer

The second training stage requires the VQ-VAE weights produced in the first stage. The examples below use the downloaded pretrained VQ-VAE.

InterHuman:

```bash
python train_t2m.py --gpu_id 0 --dataname InterHuman \
    --lr-scheduler 60000 --exp_name interh_t2m_model \
    --vq_model_pth ./checkpoints/interh_vq_model.pth
```

Inter-X:

```bash
python train_t2m.py --gpu_id 0 --dataname InterX \
    --lr-scheduler 50000 --exp_name inter_x_t2m_model \
    --vq_model_pth ./checkpoints/inter_x_vq_model.pth
```

To use a VQ-VAE that you trained yourself, replace `--vq_model_pth` with `net_best_fid.pth` or `net_last.pth` from the corresponding experiment directory.

## Model Testing

### 1. VQ-VAE Reconstruction Evaluation

InterHuman:

```bash
python eval_vq.py --gpu_id 0 --dataname InterHuman \
    --vq_model_pth ./checkpoints/interh_vq_model.pth
```

Inter-X:

```bash
python eval_vq.py --gpu_id 0 --dataname InterX \
    --vq_model_pth ./checkpoints/inter_x_vq_model.pth
```

### 2. Text-to-Interaction Motion Generation Evaluation

InterHuman:

```bash
python eval_t2m.py --gpu_id 0 --dataname InterHuman \
    --vq_model_pth ./checkpoints/interh_vq_model.pth \
    --resume_trans ./checkpoints/interh_t2m_model.pth
```

Inter-X:

```bash
python eval_t2m.py --gpu_id 0 --dataname InterX \
    --vq_model_pth ./checkpoints/inter_x_vq_model.pth \
    --resume_trans ./checkpoints/inter_x_t2m_model.pth
```

The evaluation reports metrics including FID, Diversity, R-Precision, Matching Score, and Multimodality.
