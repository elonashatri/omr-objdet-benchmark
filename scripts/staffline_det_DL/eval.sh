#!/bin/bash
# Memory-efficient inference script for staff line detection

# Load required modules
module load miniconda/4.7.12
module load cuda/12.1

# Activate conda environment
source activate omr_benchmark

# Set paths (adjust these for your environment)
BASE_DIR="/homes/es314/omr-objdet-benchmark"
DATA_DIR="$BASE_DIR/data"
SCRIPT_DIR="$BASE_DIR/scripts/staffline_det_DL"
OUTPUT_DIR="$BASE_DIR/staffline_det_DL/inference_output"

# Image and model paths
IMAGE_DIR="$DATA_DIR/images"
MODEL_PATH="/homes/es314/omr-objdet-benchmark/staffline_det_DL/staff_line_output/models/best_model.pth"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process a subset of 5 images for testing
# Limit maximum image dimension to 512 pixels to save memory
python "$SCRIPT_DIR/inference_.py" \
    --weights "$MODEL_PATH" \
    --input '/homes/es314/omr-objdet-benchmark/scripts/staffline_det_DL/inference_img' \
    --output "$OUTPUT_DIR" \
    --batch \
    --gpu_id 1 \
    --max_size 512 \
    --subset 5

echo "Memory-efficient inference execution completed!"
