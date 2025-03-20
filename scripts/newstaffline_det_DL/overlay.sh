#!/bin/bash
# Overlay staff line detection results on original images

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
JSON_DIR="$OUTPUT_DIR/json"
OVERLAY_DIR="$OUTPUT_DIR/overlays"

# Create overlay directory if it doesn't exist
mkdir -p "$OVERLAY_DIR"

# Visualize the JSON results overlaid on the original images
python "$SCRIPT_DIR/overlay_batch.py" \
    --images '/homes/es314/omr-objdet-benchmark/scripts/staffline_det_DL/inference_img' \
    --json "$JSON_DIR" \
    --output "$OVERLAY_DIR" \
    --thickness 2 \
    --batch

echo "Staff line visualization completed! Results saved to $OVERLAY_DIR"