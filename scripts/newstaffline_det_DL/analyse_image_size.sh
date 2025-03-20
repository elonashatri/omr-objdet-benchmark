#!/bin/bash

# Load required modules
module load miniconda/4.7.12

# Activate conda environment
source activate omr_benchmark

# Set paths
BASE_DIR="/homes/es314/omr-objdet-benchmark"
SCRIPTS_DIR="$BASE_DIR/scripts/staffline_det_DL"
IMAGE_DIR="$BASE_DIR/data/images"

# Save the analyzer script
cp image_size_analyzer.py $SCRIPTS_DIR/

# Run the analyzer
cd $SCRIPTS_DIR
python image_size_analyzer.py $IMAGE_DIR