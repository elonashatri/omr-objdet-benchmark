#!/bin/bash

# Run the stem and barline inference script
python infer_stems_barlines_v2.py \
  --staff-lines "/homes/es314/omr-objdet-benchmark/scripts/encoding/pixe-enhancment/results/enhanced/bartokmvt5-062_standard.json" \
  --detections "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/object_detections/bartokmvt5-062_detections.csv" \
  --output-dir "/homes/es314/omr-objdet-benchmark/scripts/stem-barline-detections/visualizations"


# bartokmvt5-062

# To process all files in a directory with automatic image detection, uncomment and use this loop:
# for staff_file in /homes/es314/omr-objdet-benchmark/scripts/encoding/results/staff_lines/*_staff_lines.json; do
#   base_name=$(basename "$staff_file" _staff_lines.json)
#   detection_file="/homes/es314/omr-objdet-benchmark/scripts/encoding/results/object_detections/${base_name}_detections.json"
#   image_file="/homes/es314/omr-objdet-benchmark/data/images/${base_name}.png"
#   
#   if [ -f "$detection_file" ]; then
#     echo "Processing $base_name..."
#     
#     # Try to find the image file
#     image_arg=""
#     if [ -f "$image_file" ]; then
#       image_arg="--image $image_file"
#     else
#       # Try jpg extension
#       image_file="${image_file%.png}.jpg"
#       if [ -f "$image_file" ]; then
#         image_arg="--image $image_file"
#       fi
#     fi
#     
#     # Run inference
#     python infer_stems_barlines.py \
#       --staff-lines "$staff_file" \
#       --detections "$detection_file" \
#       $image_arg \
#       --output-dir "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/enhanced_detections"
#   else
#     echo "Skipping $base_name, detection file not found"
#   fi
# done

echo "Inference complete! Now running image overlay..."


# 
# After inference is done, create overlay images
IMAGE_PATH="/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/bartokmvt5-062.png"
DETECTION_PATH="/homes/es314/omr-objdet-benchmark/scripts/stem-barline-detections/visualizations/bartokmvt5-062_enhanced_detections.json"
OUTPUT_PATH="/homes/es314/omr-objdet-benchmark/scripts/stem-barline-detections/visualizations/bartokmvt5-062_visualizations1.png"

# Create the output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Run the overlay script
python image_overlay.py \
  --image "$IMAGE_PATH" \
  --detections "$DETECTION_PATH" \
  --output "$OUTPUT_PATH"

echo "Created overlay at $OUTPUT_PATH"

# Create additional visualizations showing only original or inferred elements
python image_overlay.py \
  --image "$IMAGE_PATH" \
  --detections "$DETECTION_PATH" \
  --output "${OUTPUT_PATH%.png}_original_only.png" \
  --no-inferred

python image_overlay.py \
  --image "$IMAGE_PATH" \
  --detections "$DETECTION_PATH" \
  --output "${OUTPUT_PATH%.png}_inferred_only.png" \
  --no-original

echo "Created additional visualizations"