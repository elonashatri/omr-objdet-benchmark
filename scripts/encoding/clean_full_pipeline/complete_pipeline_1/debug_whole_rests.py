#!/usr/bin/env python3
"""
Debug script to examine whole rest detection and merging issues
"""

import json
import os
import sys
from pathlib import Path

# Path to your combined detections file
combined_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/1-2-Kirschner_-_Chissà_che_cosa_pensa-001/combined_detections/1-2-Kirschner_-_Chissà_che_cosa_pensa-001_combined_detections.json"

# Path to your structure and symbol detections
structure_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/1-2-Kirschner_-_Chissà_che_cosa_pensa-001/structure_detections/1-2-Kirschner_-_Chissà_che_cosa_pensa-001_detections.json"
symbol_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/1-2-Kirschner_-_Chissà_che_cosa_pensa-001/symbol_detections/1-2-Kirschner_-_Chissà_che_cosa_pensa-001_detections.json"

def examine_whole_rests():
    """Find and examine whole rests in the detection files"""
    
    # Load detection files
    with open(combined_path, 'r') as f:
        combined_data = json.load(f)
    with open(structure_path, 'r') as f:
        structure_data = json.load(f)
    with open(symbol_path, 'r') as f:
        symbol_data = json.load(f)
        
    # Find whole rests in each file
    combined_whole_rests = []
    structure_whole_rests = []
    symbol_whole_rests = []
    
    for detection in combined_data.get("detections", []):
        class_name = detection.get("class_name", "").lower()
        if "restwhole" in class_name or "whole rest" in class_name:
            combined_whole_rests.append(detection)
            
    for detection in structure_data.get("detections", []):
        class_name = detection.get("class_name", "").lower()
        if "restwhole" in class_name or "whole rest" in class_name:
            structure_whole_rests.append(detection)
            
    for detection in symbol_data.get("detections", []):
        class_name = detection.get("class_name", "").lower()
        if "restwhole" in class_name or "whole rest" in class_name:
            symbol_whole_rests.append(detection)
    
    # Print findings
    print("\n=== WHOLE REST ANALYSIS ===")
    print(f"Structure model: {len(structure_whole_rests)} whole rests")
    print(f"Symbol model: {len(symbol_whole_rests)} whole rests")
    print(f"Combined file: {len(combined_whole_rests)} whole rests")
    
    # Print position details for each whole rest
    print("\nWhole rests from structure model:")
    for i, rest in enumerate(structure_whole_rests):
        bbox = rest.get("bbox", {})
        x = bbox.get("center_x", 0)
        y = bbox.get("center_y", 0)
        conf = rest.get("confidence", 0)
        print(f"  Rest {i+1}: pos=({x:.2f}, {y:.2f}), conf={conf:.3f}")
        
    print("\nWhole rests from symbol model:")
    for i, rest in enumerate(symbol_whole_rests):
        bbox = rest.get("bbox", {})
        x = bbox.get("center_x", 0)
        y = bbox.get("center_y", 0)
        conf = rest.get("confidence", 0)
        print(f"  Rest {i+1}: pos=({x:.2f}, {y:.2f}), conf={conf:.3f}")
        
    print("\nWhole rests in combined file:")
    for i, rest in enumerate(combined_whole_rests):
        bbox = rest.get("bbox", {})
        x = bbox.get("center_x", 0)
        y = bbox.get("center_y", 0)
        conf = rest.get("confidence", 0)
        print(f"  Rest {i+1}: pos=({x:.2f}, {y:.2f}), conf={conf:.3f}")
        
    # Group rests by approximate vertical position
    staff_groups = {}
    
    for rest in combined_whole_rests:
        bbox = rest.get("bbox", {})
        y_pos = bbox.get("center_y", 0)
        
        # Group by 5% of vertical position
        group_key = int(y_pos * 20)
        
        if group_key not in staff_groups:
            staff_groups[group_key] = []
        staff_groups[group_key].append(rest)
    
    print("\nGrouped by vertical position (staffs):")
    for key, group in staff_groups.items():
        print(f"  Staff group {key} (y ≈ {key/20:.2f}): {len(group)} rests")
        for i, rest in enumerate(group):
            bbox = rest.get("bbox", {})
            x = bbox.get("center_x", 0)
            y = bbox.get("center_y", 0)
            conf = rest.get("confidence", 0)
            print(f"    Rest {i+1}: pos=({x:.2f}, {y:.2f}), conf={conf:.3f}")

if __name__ == "__main__":
    examine_whole_rests()