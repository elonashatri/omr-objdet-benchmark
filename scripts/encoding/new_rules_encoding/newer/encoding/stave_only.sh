#!/bin/bash

python main.py \
  --detection "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/object_detections/stave-only_detections.csv" \
  --staff-lines "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/staff_lines/stave-only_staff_lines.json" \
  --output-xml "stave-only.musicxml" \
  --output-image "stave-only_detections_score_visualization.png"