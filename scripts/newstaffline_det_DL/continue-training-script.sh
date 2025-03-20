#!/bin/bash
# Run script for continuing staff line detection training for another 50 epochs

# Load required modules
module load miniconda/4.7.12
module load cuda/12.1

# Activate conda environment
source activate omr_benchmark

# Set paths (adjust these for your environment)
BASE_DIR="/homes/es314/omr-objdet-benchmark"
DATA_DIR="$BASE_DIR/data"
SCRIPT_DIR="$BASE_DIR/scripts/staffline_det_DL"
OUTPUT_DIR="$BASE_DIR/staffline_det_DL/staff_line_output"

# Image and annotation directories
IMAGE_DIR="$DATA_DIR/images"
XML_DIR="$DATA_DIR/annotations"

# Checkpoint path for continued training
CHECKPOINT_PATH="$BASE_DIR/staffline_det_DL/staff_line_output/models/checkpoint_epoch_50.pth"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR/logs"

# Run the pipeline with continue_from parameter
# Set epochs to 100 (50 previous + 50 new)
nohup python "$SCRIPT_DIR/pipeline.py" \
    --image_dir "$IMAGE_DIR" \
    --xml_dir "$XML_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 0.001 \
    --gpu_id 1 \
    --num_workers 4 \
    --target_size 512 512 \
    --subset_fraction 0.7 \
    --verify \
    --visualize \
    --skip_preprocess \
    --continue_from "$CHECKPOINT_PATH" > "$OUTPUT_DIR/logs/continued_training_epochs_50_to_100.txt" 2>&1

echo "Continued training completed!"