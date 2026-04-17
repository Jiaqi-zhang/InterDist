#!/bin/bash

TARGET_DIR="./checkpoints/"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "The pretrained models will be stored in the '$TARGET_DIR' folder."
echo "--------------------------------------------------"
echo "Starting to download models..."

# Download inter_x_vq_model.pth
echo "Downloading inter_x_vq_model.pth ..."
gdown "https://drive.google.com/uc?id=1jImtPhXTT25LksosvXhOtgP0v1G6KXm1" -O inter_x_vq_model.pth

# Download interh_t2m_model.pth
echo "Downloading interh_t2m_model.pth ..."
gdown "https://drive.google.com/uc?id=1UMS141cCRhXQv49fslVXpbQQLvsv_QB2" -O interh_t2m_model.pth

# Download interh_vq_model.pth
echo "Downloading interh_vq_model.pth ..."
gdown "https://drive.google.com/uc?id=1O7Oor6FfVEToqHBCyG3PUXys8l4pirLE" -O interh_vq_model.pth

# Download inter_x_t2m_model.pth
echo "Downloading inter_x_t2m_model.pth ..."
gdown "https://drive.google.com/uc?id=1SRi60oTK4WMw5Wal7OZmiEtfx4pI943i" -O inter_x_t2m_model.pth

echo "--------------------------------------------------"
echo "All models downloaded successfully!"
