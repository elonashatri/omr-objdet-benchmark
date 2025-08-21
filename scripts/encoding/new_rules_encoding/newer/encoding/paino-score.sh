#!/bin/bash

python main.py \
  --detection "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/results/object_detections/Piano_Sonate_No.1_Op.2_No.1_in_F_Minor-006_detections.csv" \
  --staff-lines "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/results/staff_lines/Piano_Sonate_No.1_Op.2_No.1_in_F_Minor-006_staff_lines.json" \
  --output-xml "Piano_Sonate_No.1_Op.2_No.1_in_F_Minor-006.musicxml" \
  --output-image "Piano_Sonate_No.1_Op.2_No.1_in_F_Minor-006_detections_score_visualization.png"
  --staff-mode "piano" 