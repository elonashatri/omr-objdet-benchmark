#!/bin/bash
# Run script for staff line detection system

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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the pipeline
nohup python "$SCRIPT_DIR/pipeline.py" \
    --image_dir "$IMAGE_DIR" \
    --xml_dir "$XML_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --epochs 50 \
    --learning_rate 0.001 \
    --gpu_id 1 \
    --num_workers 4 \
    --target_size 512 512 \
    --subset_fraction 0.7 \
    --verify \
    --visualize > 4-50-001lr-4w-512-07-pt2.txt

echo "Pipeline execution completed!"