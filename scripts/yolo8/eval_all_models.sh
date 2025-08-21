#!/bin/bash
# Script to run the evaluation on all YOLOv8 models

# Set variables
DEFAULT_DATA_YAML="/homes/es314/omr-objdet-benchmark/data/202-24classes-yolo-9654-data-splits/dataset.yaml"
SPECIAL_DATA_YAML="/homes/es314/omr-objdet-benchmark/data/202-24classes-yolo-9654-data-splits/dataset.yaml"
DOREMI_DATA_DIR="/homes/es314/omr-objdet-benchmark/DOREMI_v3"
DOREMI_MAPPING="/homes/es314/omr-objdet-benchmark/data/202-24classes-yolo-9654-data-splits/filtered_class_mapping.json"
SPECIAL_MODELS="train3-yolo-9654-data-splits train-202-24classes-yolo-9654-data-splits"
DOREMI_MODEL="doremiv1-94classes"
NUM_IMAGES=100  # Set to 0 to use all test images
IOU_THRESHOLD=0.4
CONF_THRESHOLD=0.3
MAP_IOU_THRESHOLDS="0.5,0.75"  # Reduced thresholds for stability
USE_CLASS_CONF_THRESHOLDS="--use_class_conf_thresholds"  # Use class-specific thresholds
DEVICE=0
OUTPUT_DIR="/homes/es314/runs/detect/train"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# # Evaluate each model individually to prevent one failure from stopping the entire process
# for MODEL_PATH in \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train3-yolo-9654-data-splits/weights/best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/staffline_extreme-2/weights/best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/staffline_extreme/weights/best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/experiment-1-staffline-enhacment-april/weights/best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/doremiv1-94classes/weights/best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/detect/train11/weights/best.pt" \
#     # "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/detect/train8/weights/best.pt" \
#     "/homes/es314/runs/detect/train/weights/best.pt"

MODEL_PATHS=(
    "/homes/es314/runs/detect/train/weights/best.pt"
    # Add more model paths here as needed
)

for MODEL_PATH in "${MODEL_PATHS[@]}"

do
    echo "===== Evaluating model: $MODEL_PATH ====="
    
    # Special handling for DOREMI model
    if [[ "$MODEL_PATH" == *"$DOREMI_MODEL"* ]]; then
        echo "Using DOREMI dataset and mapping for $MODEL_PATH"
        EXTRA_ARGS="--doremi_data_dir $DOREMI_DATA_DIR --doremi_mapping $DOREMI_MAPPING"
    else
        EXTRA_ARGS=""
    fi
    
    # Run the evaluation script for a single model
    python /homes/es314/omr-objdet-benchmark/scripts/yolo8/eval_all_models.py \
        --model_dirs "$MODEL_PATH" \
        --default_data_yaml "$DEFAULT_DATA_YAML" \
        --special_data_yaml "$SPECIAL_DATA_YAML" \
        --special_models $SPECIAL_MODELS \
        --num_images "$NUM_IMAGES" \
        --iou_threshold "$IOU_THRESHOLD" \
        --conf_threshold "$CONF_THRESHOLD" \
        --map_iou_thresholds "$MAP_IOU_THRESHOLDS" \
        $USE_CLASS_CONF_THRESHOLDS \
        --device "$DEVICE" \
        $EXTRA_ARGS
    
    echo "===== Finished evaluating: $MODEL_PATH ====="
    echo ""
done

echo "Evaluation complete! Results saved to $OUTPUT_DIR"