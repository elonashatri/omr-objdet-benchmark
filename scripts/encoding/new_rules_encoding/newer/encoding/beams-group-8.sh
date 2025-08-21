#!/bin/bash

python main.py \
  --detection "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/results/object_detections/beam_groups_8_semiquavers-001_detections.csv" \
  --staff-lines /homes/es314/omr-objdet-benchmark/scripts/encoding/new_rules_encoding/newer/beam_groups_8_semiquavers-001_pixel_perfect.json \
  --original-image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/beam_groups_8_semiquavers-001.png