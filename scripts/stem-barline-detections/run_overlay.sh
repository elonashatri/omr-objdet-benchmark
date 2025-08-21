
#!/bin/bash

# Path to the original score image - ADJUST THIS PATH TO YOUR IMAGE
IMAGE_PATH="/homes/es314/omr-objdet-benchmark/scripts/stem-barline-detections/Accidentals-004.png"

# Path to the enhanced detections
DETECTION_PATH="/homes/es314/omr-objdet-benchmark/scripts/stem-barline-detections/Accidentals-004_enhanced_detections.json"

# Output path for the overlay image
OUTPUT_PATH="/homes/es314/omr-objdet-benchmark/scripts/stem-barline-detections/Accidentals-004_overlay.png"

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