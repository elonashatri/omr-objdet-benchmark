#!/bin/bash
# Script to run the evaluation on YOLOv8 models with MUSCIMA++ dataset

# Set variables - update these paths for your environment
DEFAULT_DATA_YAML="/homes/es314/omr-objdet-benchmark/data/yolo-9654-data-splits/dataset.yaml"
SPECIAL_DATA_YAML="/homes/es314/omr-objdet-benchmark/data/202-24classes-yolo-9654-data-splits/dataset.yaml"
MUSCIMA_DATA_DIR="/path/to/your/muscima/dataset"  # Update this path
MUSCIMA_ANNOTATION_DIR="/path/to/your/muscima/xml_annotations"  # Update this path
MUSCIMA_MAPPING="/path/to/your/muscima/class_mapping.json"  # Optional, can be omitted

# Keep these for backward compatibility
DOREMI_DATA_DIR="/homes/es314/omr-objdet-benchmark/DOREMI_v3"
DOREMI_MAPPING="/homes/es314/DOREMI_version_2/DOREMI_v3/doremiv1-94-class_map.json"

# Model settings
SPECIAL_MODELS="train3-yolo-9654-data-splits train-202-24classes-yolo-9654-data-splits"
MUSCIMA_MODELS="muscima muscima-pp"  # Model names that should use MUSCIMA++ dataset
DOREMI_MODEL="doremiv1-94classes"

# Evaluation settings
NUM_IMAGES=100  # Set to 0 to use all test images
IOU_THRESHOLD=0.4
CONF_THRESHOLD=0.3
MAP_IOU_THRESHOLDS="0.5,0.75"
USE_CLASS_CONF_THRESHOLDS="--use_class_conf_thresholds"  # Use class-specific thresholds
DEVICE=0
OUTPUT_DIR="/path/to/your/output/directory"  # Update this path

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of models to evaluate - update with your model paths
MODELS=(
    "/path/to/your/muscima/model/weights/best.pt"  # Update this with your MUSCIMA model path
    # You can add more models here
)

# Evaluate each model individually
for MODEL_PATH in "${MODELS[@]}"
do
    echo "===== Evaluating model: $MODEL_PATH ====="
    
    # Determine which dataset to use based on model name
    if [[ "$MODEL_PATH" == *"muscima"* ]]; then
        echo "Using MUSCIMA++ dataset for $MODEL_PATH"
        EXTRA_ARGS="--muscima_data_dir $MUSCIMA_DATA_DIR --muscima_annotation_dir $MUSCIMA_ANNOTATION_DIR"
        # Add mapping if available
        if [ -f "$MUSCIMA_MAPPING" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --muscima_mapping $MUSCIMA_MAPPING"
        fi
    elif [[ "$MODEL_PATH" == *"$DOREMI_MODEL"* ]]; then
        echo "Using DOREMI dataset for $MODEL_PATH"
        EXTRA_ARGS="--doremi_data_dir $DOREMI_DATA_DIR --doremi_mapping $DOREMI_MAPPING"
    else
        EXTRA_ARGS=""
    fi
    
    # Run the evaluation script for a single model
    python /path/to/your/eval_script.py \
        --model_dirs "$MODEL_PATH" \
        --default_data_yaml "$DEFAULT_DATA_YAML" \
        --special_data_yaml "$SPECIAL_DATA_YAML" \
        --special_models $SPECIAL_MODELS \
        --muscima_models $MUSCIMA_MODELS \
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