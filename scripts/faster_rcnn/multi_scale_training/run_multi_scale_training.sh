#!/bin/bash
# =========================================================================
# Multi-Scale Enhanced Faster R-CNN Training Script for OMR Dataset
# =========================================================================
# Created: April 2025
# Dataset: DoReMi Staff Detection Dataset
# Purpose: Train a Faster R-CNN model with advanced multi-scale techniques
#          for improved staff-line and music symbol detection
# 
# Multi-scale enhancements:
# - Image pyramid during inference using scales 0.5, 0.75, 1.0, 1.25, 1.5
# - Multi-scale training with variable input sizes (400-800px)
# - Test-time augmentation with flips, rotations, and brightness variations
# - Enhanced Feature Pyramid Network with attention mechanisms
# - Custom aspect ratios and sizes for better staff line detection
# =========================================================================

# Set timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting multi-scale enhanced training at: $(date)"

# Base directories
DATA_DIR="/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset"
OUTPUT_BASE="/import/c4dm-05/elona/faster-rcnn-models-march-2025"

# Create experiment-specific directories
MODEL_NAME="multi-scale-staff-faster-rcnn-resnet101"
OUTPUT_DIR="$OUTPUT_BASE/$MODEL_NAME"
LOG_DIR="$OUTPUT_BASE/$MODEL_NAME-logs"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Create a log file for this run
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
touch "$LOG_FILE"

# Echo system information
echo "==== System Information ====" | tee -a "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Python: $(python3 --version)" | tee -a "$LOG_FILE"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')" | tee -a "$LOG_FILE"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')" | tee -a "$LOG_FILE"
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')" | tee -a "$LOG_FILE"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  for i in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
    echo "GPU $i: $(python3 -c "import torch; print(torch.cuda.get_device_name($i))")" | tee -a "$LOG_FILE"
  done
fi

# Echo dataset information
echo -e "\n==== Dataset Information ====" | tee -a "$LOG_FILE"
echo "Dataset directory: $DATA_DIR" | tee -a "$LOG_FILE"
echo "Training images: $(ls $DATA_DIR/train/images | wc -l)" | tee -a "$LOG_FILE"
echo "Validation images: $(ls $DATA_DIR/val/images | wc -l)" | tee -a "$LOG_FILE"
if [ -d "$DATA_DIR/test/images" ]; then
  echo "Test images: $(ls $DATA_DIR/test/images | wc -l)" | tee -a "$LOG_FILE"
fi
echo "Mapping file: $DATA_DIR/mapping.txt" | tee -a "$LOG_FILE"
echo "Number of classes: $(wc -l < $DATA_DIR/mapping.txt)" | tee -a "$LOG_FILE"

# Echo experiment information
echo -e "\n==== Experiment Information ====" | tee -a "$LOG_FILE"
echo "Experiment: Multi-Scale Enhanced Faster R-CNN for Staff Detection with advanced image pyramid techniques" | tee -a "$LOG_FILE"
echo "Model name: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log directory: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script with multi-scale configurations
echo -e "\n==== Starting Training with Multi-Scale Enhancements ====" | tee -a "$LOG_FILE"
echo "Command executed:" | tee -a "$LOG_FILE"

# Construct the command with all parameters
CMD="python3 /homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/multi_scale_training/multi_scale_training.py \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --log_dir=$LOG_DIR \
  --backbone=resnet101-custom \
  --num_classes=218 \
  --batch_size=1 \
  --val_batch_size=1 \
  --num_epochs=5 \
  --learning_rate=0.0005 \
  --decay_factor=0.95 \
  --decay_steps=40000 \
  --momentum=0.9 \
  --weight_decay=0.0001 \
  --gradient_clipping_by_norm=10.0 \
  --image_size=1200,2400 \
  --min_size=1200 \
  --max_size=2400 \
  --multi_scale_train \
  --multi_scale_min_sizes=600,800,1000,1200,1400,1600 \
  --multi_scale_inference \
  --inference_scales=0.25,0.4,0.6,0.8,1.0,1.25,1.5,1.75 \
  --test_time_augmentation \
  --random_crop_prob=0.3 \
  --brightness_range=0.9,1.1 \
  --contrast_range=1.0,1.2 \
  --enable_sharpening \
  --sharpening_prob=0.25 \
  --anchor_sizes=2,4,8,16,32,64,128,256 \
  --aspect_ratios=0.03,0.05,0.1,0.25,0.5,1.0,2.0,4.0,10.0,20.0,30.0 \
  --height_stride=4 \
  --width_stride=4 \
  --features_stride=4 \
  --initial_crop_size=17 \
  --maxpool_kernel_size=1 \
  --maxpool_stride=1 \
  --atrous_rate=2 \
  --first_stage_nms_score_threshold=0.0 \
  --first_stage_nms_iou_threshold=0.6 \
  --first_stage_max_proposals=3000 \
  --second_stage_nms_score_threshold=0.04 \
  --second_stage_nms_iou_threshold=0.45 \
  --second_stage_max_detections_per_class=3000 \
  --second_stage_max_total_detections=3500 \
  --first_stage_localization_loss_weight=4.0 \
  --first_stage_objectness_loss_weight=2.0 \
  --save_freq=10 \
  --eval_freq=5 \
  --print_freq=10 \
  --num_workers=4 \
  --pin_memory \
  --gpu_id=2 \
  --data_subset=0.001 \
  --pretrained \
  --num_visualizations=5"

# Display the command
echo "$CMD" | tee -a "$LOG_FILE"

# Execute the command and capture output
eval $CMD 2>&1 | tee -a "$LOG_FILE"

# Print training completion information
echo -e "\n==== Training Complete ====" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Logs saved to: $LOG_DIR" | tee -a "$LOG_FILE"
echo "To visualize results: tensorboard --logdir=$LOG_DIR" | tee -a "$LOG_FILE"

# # Add a note about how to run inference with multi-scale
# echo -e "\n==== Multi-Scale Inference Instructions ====" | tee -a "$LOG_FILE"
# echo "To perform multi-scale inference on new images:" | tee -a "$LOG_FILE"
# echo "python3 multi_scale_inference.py \\" | tee -a "$LOG_FILE"
# echo "  --model_path=$OUTPUT_DIR/best.pt \\" | tee -a "$LOG_FILE"
# echo "  --input_dir=/path/to/images \\" | tee -a "$LOG_FILE"
# echo "  --output_dir=/path/to/save/results \\" | tee -a "$LOG_FILE"
# echo "  --mapping_file=$DATA_DIR/mapping.txt \\" | tee -a "$LOG_FILE"
# echo "  --scales=0.5,0.75,1.0,1.25,1.5 \\" | tee -a "$LOG_FILE"
# echo "  --test_time_augmentation \\" | tee -a "$LOG_FILE"
# echo "  --confidence_threshold=0.5" | tee -a "$LOG_FILE"

echo "Script execution complete"