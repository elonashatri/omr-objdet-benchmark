#!/bin/bash
# Script to run the evaluation on all four Faster R-CNN models
# using each model's specific test dataset and mapping file from args.json

# Set variables
DEFAULT_TEST_DIR="/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset/test"
DEFAULT_MAPPING_FILE="/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset/mapping.txt"
NUM_IMAGES=1449
IOU_THRESHOLD=0.4
CONF_THRESHOLD=0.3
MAP_IOU_THRESHOLDS="0.5,0.75,0.8,0.9,0.95"
GPU_ID=2

# List of model directories to evaluate
MODEL_DIRS=(
    "/import/c4dm-05/elona/faster-rcnn-models-march-2025/staff-half-older_config_faster_rcnn_omr_output"
    "/import/c4dm-05/elona/faster-rcnn-models-march-2025/half-older_config_faster_rcnn_omr_output"
    "/import/c4dm-05/elona/faster-rcnn-models-march-2025/full-with-staff-output"
    "/import/c4dm-05/elona/faster-rcnn-models-march-2025/full-no-staff-output"
)

# Copy the OMRDataset module to make it easily importable
# This ensures the evaluation script can find and use your dataset implementation
cp /homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/omr_dataset.py ./

# Run the evaluation script with the fixed dataset loading
python evaluate_models.py \
    --model_dirs "${MODEL_DIRS[@]}" \
    --default_test_dir "$DEFAULT_TEST_DIR" \
    --default_mapping_file "$DEFAULT_MAPPING_FILE" \
    --num_images "$NUM_IMAGES" \
    --iou_threshold "$IOU_THRESHOLD" \
    --conf_threshold "$CONF_THRESHOLD" \
    --map_iou_thresholds "$MAP_IOU_THRESHOLDS" \
    --gpu_id "$GPU_ID"

echo "Evaluation complete!"