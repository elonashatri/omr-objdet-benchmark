#!/bin/bash
# =========================================================================
# Faster R-CNN with Inception ResNet v2 Training Script for OMR Dataset
# =========================================================================
# Created: April 2025
# Dataset: DoReMi Staff Detection Dataset
# Purpose: Train a Faster R-CNN model with Inception ResNet v2 backbone
#          for staff-line detection in music scores
# 
# Optimization notes:
# - Modified anchor sizes and aspect ratios to handle both very thin
#   staff lines (width_cov: 0.07, height_cov: 0.15) and small objects 
#   like augmentation dots (size ratio ~7.5:1)
# - Using multi-scale training to better handle size variation
# - Feature stride adjusted for better small object detection
# - Increased proposals to handle dense object scenes (avg 14.5 symbols per staff)
# - NMS and IoU thresholds tuned for high-overlap scenarios (104,688 overlaps)
# - Optimized loss weights for better staff line detection
# =========================================================================
# module load miniconda/4.7.12
# module load cuda/12.1  # Using CUDA 12.1 for compatibility with recent PyTorch
# conda activate omr_benchmark
# cd /homes/es314/omr-objdet-benchmark/scripts/faster_rcnn

# Set timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting training at: $(date)"

# Base directories
DATA_DIR="/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset"
OUTPUT_BASE="/import/c4dm-05/elona/faster-rcnn-models-march-2025"

# Create experiment-specific directories
MODEL_NAME="full-staff-faster-rcnn-inception-resnet-v2"
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
echo "Experiment: Full Staff Detection with Faster R-CNN + Inception ResNet v2" | tee -a "$LOG_FILE"
echo "Model name: $MODEL_NAME" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log directory: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Run the training script with configurations adapted for Inception ResNet v2
echo -e "\n==== Starting Training ====" | tee -a "$LOG_FILE"
echo "Command executed:" | tee -a "$LOG_FILE"

# Construct the full command with parameters
python3 /homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/inception_resnetv2_training.py \
  --data_dir=/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset \
  --output_dir=/import/c4dm-05/elona/faster-rcnn-models-march-2025/full-staff-faster-rcnn-inception-resnet-v2 \
  --log_dir=/import/c4dm-05/elona/faster-rcnn-models-march-2025/full-staff-faster-rcnn-inception-resnet-v2-logs \
  --backbone=inception_resnet_v2 \
  --num_classes=218 \
  --batch_size=1 \
  --val_batch_size=1 \
  --num_epochs=100 \
  --learning_rate=0.001 \
  --decay_factor=0.95 \
  --decay_steps=40000 \
  --momentum=0.9 \
  --weight_decay=0.0001 \
  --gradient_clipping_by_norm=10.0 \
  --image_size=600,1200 \
  --min_size=600 \
  --max_size=1200 \
  --anchor_sizes=4,8,16,32,64,128 \
  --aspect_ratios=0.05,0.1,0.25,1.0,2.0,4.0,10.0,20.0 \
  --height_stride=4 \
  --width_stride=4 \
  --features_stride=4 \
  --initial_crop_size=17 \
  --maxpool_kernel_size=1 \
  --maxpool_stride=1 \
  --atrous_rate=2 \
  --first_stage_nms_score_threshold=0.0 \
  --first_stage_nms_iou_threshold=0.5 \
  --first_stage_max_proposals=2000 \
  --second_stage_nms_score_threshold=0.05 \
  --second_stage_nms_iou_threshold=0.4 \
  --second_stage_max_detections_per_class=2000 \
  --second_stage_max_total_detections=2500 \
  --first_stage_localization_loss_weight=4.0 \
  --first_stage_objectness_loss_weight=2.0 \
  --save_freq=10 \
  --eval_freq=5 \
  --print_freq=10 \
  --num_workers=4 \
  --pin_memory \
  --gpu_id=0 \
  --data_subset=0.5 \
  --pretrained

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

# Add a note about how to run inference
# echo -e "\n==== Inference Instructions ====" | tee -a "$LOG_FILE"
# echo "To run inference on new images:" | tee -a "$LOG_FILE"
# echo "python3 /homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/inference.py \\" | tee -a "$LOG_FILE"
# echo "  --model_path=$OUTPUT_DIR/best.pt \\" | tee -a "$LOG_FILE"
# echo "  --input_dir=/path/to/images \\" | tee -a "$LOG_FILE"
# echo "  --output_dir=/path/to/save/results \\" | tee -a "$LOG_FILE"
# echo "  --mapping_file=$DATA_DIR/mapping.txt \\" | tee -a "$LOG_FILE"
# echo "  --min_size=500 --max_size=1000 \\" | tee -a "$LOG_FILE"
# echo "  --confidence_threshold=0.5" | tee -a "$LOG_FILE"

echo "Script execution complete"