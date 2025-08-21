#!/usr/bin/env python3
"""
Complete Music Notation Analysis Pipeline with Dual Models

This script integrates multiple stages of Optical Music Recognition (OMR) with two models:
1. Detect structural elements (barlines, stems, beams) using Faster R-CNN ONNX model
2. Detect other musical symbols using YOLOv8 model
3. Detect staff lines with pixel-perfect alignment
4. Enhance detections by adding missing stems
5. Merge all detections
6. Generate MusicXML with visualizations

Usage:
    python complete_pipeline.py --image example.png 
                               --structure_model path/to/faster_rcnn.onnx 
                               --structure_mapping path/to/structure_classes.txt
                               --symbol_model path/to/yolo.pt 
                               --symbol_mapping path/to/symbol_classes.json
"""

import os
import sys
import argparse
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
import traceback
import logging
from to_midi import add_to_pipeline as add_midi_generation
from custom_xml_generator import generate_custom_xml
from music_agents import MusicScoreAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('omr_pipeline.log')
    ]
)
logger = logging.getLogger('omr_pipeline')

# Import detection modules
try:
    # This is needed for the imported modules to find their imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        logger.info(f"Added {current_dir} to system path")
    
    # Import detector module
    from inference_detect_notation import detect_music_notation, load_class_names
    
    # Import staff line detector
    from staff_line_detector import EnhancedStaffDetector, detect_staff_lines
    
    # Import stem detection enhancement
    from stem_detector import main as enhance_stems_main
    
    # Import MusicXML generation
    from processor import OMRProcessor
    from encoding_visualization import visualize_score, visualize_overlay
    
    logger.info("Successfully imported all modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def create_output_directories(base_dir):
    """Create output directories for each stage of processing"""
    directories = {
        "structure_detections": os.path.join(base_dir, "structure_detections"),
        "symbol_detections": os.path.join(base_dir, "symbol_detections"),
        "combined_detections": os.path.join(base_dir, "combined_detections"),
        "staff_lines": os.path.join(base_dir, "staff_lines"),
        "enhanced": os.path.join(base_dir, "enhanced"),
        "merged": os.path.join(base_dir, "merged"),
        "musicxml": os.path.join(base_dir, "musicxml"),
        "visualization": os.path.join(base_dir, "visualization")
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info(f"Created output directories in {base_dir}")
    return directories

def detect_structure_elements(image_path, model_path, class_mapping_path, output_dir, 
                           conf_threshold=0.3, max_detections=1000, iou_threshold=0.35):
    """
    Detect structural elements (barlines, stems, beams, systemicBarline) in the image using Faster R-CNN ONNX
    
    Args:
        image_path: Path to the input image
        model_path: Path to the Faster R-CNN ONNX model
        class_mapping_path: Path to the class mapping file for structural elements
        output_dir: Directory to save detection results
        conf_threshold: Confidence threshold for detections
        max_detections: Maximum number of detections to consider
        iou_threshold: IoU threshold for non-maximum suppression
        
    Returns:
        Path to the JSON file with detection results
    """
    try:
        logger.info(f"Detecting structural elements in {image_path} using {model_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the image stem (filename without extension)
        img_stem = Path(image_path).stem
        
        # Run detection
        detect_music_notation(
            model_path, 
            image_path, 
            class_mapping_path, 
            output_dir,
            conf_threshold,
            max_detections,
            iou_threshold
        )
        
        # Get the path to the detection JSON file
        detection_json = os.path.join(output_dir, f"{img_stem}_detections.json")
        
        # Verify that the detection file was created
        if not os.path.exists(detection_json):
            logger.error(f"Structure detection file {detection_json} was not created")
            detection_csv = os.path.join(output_dir, f"{img_stem}_detections.csv")
            if os.path.exists(detection_csv):
                logger.info(f"Using CSV detection file instead: {detection_csv}")
                return detection_csv
            else:
                raise FileNotFoundError(f"No structure detection files were created for {image_path}")
        
        logger.info(f"Structural elements detected and saved to {detection_json}")
        return detection_json
    
    except Exception as e:
        logger.error(f"Error detecting structural elements: {e}")
        logger.error(traceback.format_exc())
        raise

def detect_music_symbols(image_path, model_path, class_mapping_path, output_dir, 
                       conf_threshold=0.3, max_detections=1000, iou_threshold=0.35):
    """
    Detect musical symbols in the image using YOLO model
    
    Args:
        image_path: Path to the input image
        model_path: Path to the YOLO model
        class_mapping_path: Path to the class mapping file for musical symbols
        output_dir: Directory to save detection results
        conf_threshold: Confidence threshold for detections
        max_detections: Maximum number of detections to consider
        iou_threshold: IoU threshold for non-maximum suppression
        
    Returns:
        Path to the JSON file with detection results
    """
    try:
        logger.info(f"Detecting musical symbols in {image_path} using {model_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the image stem (filename without extension)
        img_stem = Path(image_path).stem
        
        # Run detection
        detect_music_notation(
            model_path, 
            image_path, 
            class_mapping_path, 
            output_dir,
            conf_threshold,
            max_detections,
            iou_threshold
        )
        
        # Get the path to the detection JSON file
        detection_json = os.path.join(output_dir, f"{img_stem}_detections.json")
        
        # Verify that the detection file was created
        if not os.path.exists(detection_json):
            logger.error(f"Symbol detection file {detection_json} was not created")
            detection_csv = os.path.join(output_dir, f"{img_stem}_detections.csv")
            if os.path.exists(detection_csv):
                logger.info(f"Using CSV detection file instead: {detection_csv}")
                return detection_csv
            else:
                raise FileNotFoundError(f"No symbol detection files were created for {image_path}")
        
        logger.info(f"Musical symbols detected and saved to {detection_json}")
        return detection_json
    
    except Exception as e:
        logger.error(f"Error detecting musical symbols: {e}")
        logger.error(traceback.format_exc())
        raise

def correct_rest_types_by_position(detections, staff_data):
    """
    Correct rest types based on their position relative to staff lines,
    only when model confidence is low or position is very clear.
    Respects model source for certain rest types.
    """
    # First, extract all rests
    rests = []
    rest_indices = []
    other_objects = []
    
    for i, detection in enumerate(detections):
        class_name = detection.get('class_name', '').lower()
        if 'rest' in class_name:
            rests.append(detection)
            rest_indices.append(i)
        else:
            other_objects.append(detection)
    
    # If we have no rests or no staff data, return unchanged
    if not rests or not staff_data or "staff_systems" not in staff_data:
        return detections
    
    # Extract staff lines
    staff_systems = staff_data.get("staff_systems", [])
    staff_lines = []
    
    for system in staff_systems:
        system_lines = []
        for line_idx in system.get("lines", []):
            if line_idx < len(staff_data.get("detections", [])):
                line = staff_data["detections"][line_idx]
                system_lines.append(line)
        
        if len(system_lines) == 5:  # Only use complete 5-line staves
            # Sort lines by vertical position
            system_lines.sort(key=lambda x: x["bbox"]["center_y"])
            staff_lines.append(system_lines)
    
    # Now correct each rest based on its position
    corrected_rests = []
    for rest in rests:
        rest_y = rest["bbox"]["center_y"]
        rest_x = rest["bbox"]["center_x"]
        original_type = rest["class_name"].lower()
        model_confidence = rest.get("confidence", 0)
        source_model = rest.get("source_model", "unknown")
        
        # Find which staff this rest belongs to
        closest_staff = None
        min_distance = float('inf')
        
        for staff in staff_lines:
            staff_top = staff[0]["bbox"]["center_y"]
            staff_bottom = staff[4]["bbox"]["center_y"]
            
            # Check if rest is within staff vertical range (with some margin)
            staff_height = staff_bottom - staff_top
            margin = staff_height * 0.7  # Allow rests slightly outside staff
            
            if rest_y >= staff_top - margin and rest_y <= staff_bottom + margin:
                # Calculate distance to staff center
                staff_center = (staff_top + staff_bottom) / 2
                distance = abs(rest_y - staff_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_staff = staff
        
        # If we found a staff for this rest
        if closest_staff:
            middle_line_y = closest_staff[2]["bbox"]["center_y"]  # Line 3 (middle line)
            top_line_y = closest_staff[0]["bbox"]["center_y"]  # Line 1 (top line)
            bottom_line_y = closest_staff[4]["bbox"]["center_y"]  # Line 5 (bottom line)
            
            line_spacing = (bottom_line_y - top_line_y) / 4  # Distance between lines
            
            # Distance from middle line
            distance_from_middle = abs(rest_y - middle_line_y)
            
            # Only correct if confidence is low or position is very clear
            should_correct = False
            corrected_type = None
            
            # Check if it's a whole or half rest
            if distance_from_middle < line_spacing * 0.3:
                # Very close to middle line - could be whole or half rest
                
                # Determine position-based type
                if rest_y > middle_line_y - line_spacing * 0.1:
                    position_type = "restWhole"  # Sits on middle line
                    position_certainty = min(1.0, 1.0 - (middle_line_y - rest_y) / (line_spacing * 0.3))
                else:
                    position_type = "restHalf"  # Hangs from middle line
                    position_certainty = min(1.0, 1.0 - (rest_y - (middle_line_y - line_spacing * 0.2)) / (line_spacing * 0.3))
                
                # RESPECT MODEL SPECIALIZATION
                # If it's a high-confidence detection from the model that specializes in this rest type, trust it
                if (position_type == "restWhole" and source_model == "structure" and model_confidence > 0.85) or \
                   (position_type == "restHalf" and source_model == "symbol" and model_confidence > 0.7):
                    # Trust these model predictions completely
                    print(f"Trusting {source_model} model for {original_type} at ({rest_x:.1f}, {rest_y:.1f})")
                    should_correct = False
                # Otherwise only override if:
                # 1. Model confidence is low OR
                # 2. Position is very clear AND doesn't match the current type
                elif model_confidence < 0.75 or (position_certainty > 0.9 and position_type.lower() not in original_type):
                    corrected_type = position_type
                    should_correct = True
                    
                    if position_type == "restWhole":
                        print(f"Corrected rest at ({rest_x:.1f}, {rest_y:.1f}) to whole rest (sits on middle line)")
                    else:
                        print(f"Corrected rest at ({rest_x:.1f}, {rest_y:.1f}) to half rest (hangs from middle line)")
            
            # Apply correction if needed
            if should_correct and corrected_type:
                rest["class_name"] = corrected_type
                
                # Set new confidence based on position certainty
                new_confidence = max(model_confidence, 0.85)  # Don't lower confidence if it's already high
                rest["confidence"] = new_confidence
        
        corrected_rests.append(rest)
    
    # Merge the corrected rests back with other objects
    result = other_objects.copy()
    for i, rest in zip(rest_indices, corrected_rests):
        result.insert(i, rest)
    
    return result

def combine_detections(structure_detections_path, symbol_detections_path, 
                     structure_class_mapping_path, symbol_class_mapping_path, output_dir,
                     high_conf_threshold=0.5, overlap_threshold=0.8): 
    """
    Combine detections from structure and symbol models with targeted merging
    
    Args:
        structure_detections_path: Path to the structural elements detection file
        symbol_detections_path: Path to the musical symbols detection file
        structure_class_mapping_path: Path to class mapping for structural elements
        symbol_class_mapping_path: Path to class mapping for musical symbols
        output_dir: Directory to save combined results
        high_conf_threshold: Threshold for considering detections as high-confidence
        overlap_threshold: Threshold for overlap (not used in targeted approach)
        
    Returns:
        Path to the JSON file with combined detection results
    """
    try:
        logger.info(f"Combining detections from {structure_detections_path} and {symbol_detections_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the filename stem from either detection file
        img_stem = Path(structure_detections_path).stem.replace('_detections', '')
        
        # Load structure class names
        structure_class_names = load_class_names(structure_class_mapping_path)
        
        # Load symbol class names
        symbol_class_names = load_class_names(symbol_class_mapping_path)
        
        # Load structure detections
        try:
            with open(structure_detections_path, 'r') as f:
                structure_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from {structure_detections_path}, trying as CSV")
            # If JSON fails, try to read as CSV and convert to JSON format
            import csv
            structure_data = {"detections": []}
            try:
                with open(structure_detections_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        detection = {
                            "class_id": int(row["class_id"]),
                            "class_name": row["class_name"],
                            "confidence": float(row["confidence"]),
                            "bbox": {
                                "x1": float(row["x1"]),
                                "y1": float(row["y1"]),
                                "x2": float(row["x2"]),
                                "y2": float(row["y2"]),
                                "width": float(row["width"]),
                                "height": float(row["height"]),
                                "center_x": float(row["center_x"]),
                                "center_y": float(row["center_y"])
                            }
                        }
                        structure_data["detections"].append(detection)
            except Exception as e:
                logger.error(f"Error reading structure detections as CSV: {e}")
                structure_data = {"detections": []}
        
        # Load symbol detections
        try: 
            with open(symbol_detections_path, 'r') as f:
                symbol_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from {symbol_detections_path}, trying as CSV")
            # If JSON fails, try to read as CSV and convert to JSON format
            import csv
            symbol_data = {"detections": []}
            try:
                with open(symbol_detections_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        detection = {
                            "class_id": int(row["class_id"]),
                            "class_name": row["class_name"],
                            "confidence": float(row["confidence"]),
                            "bbox": {
                                "x1": float(row["x1"]),
                                "y1": float(row["y1"]),
                                "x2": float(row["x2"]),
                                "y2": float(row["y2"]),
                                "width": float(row["width"]),
                                "height": float(row["height"]),
                                "center_x": float(row["center_x"]),
                                "center_y": float(row["center_y"])
                            }
                        }
                        symbol_data["detections"].append(detection)
            except Exception as e:
                logger.error(f"Error reading symbol detections as CSV: {e}")
                symbol_data = {"detections": []}
        
        # Check structure detections for missing class names and add them if possible
        updated_structure_detections = []
        for detection in structure_data.get("detections", []):
            # If the detection has a class_id but no class_name, try to add it from the mapping
            if "class_id" in detection and ("class_name" not in detection or not detection["class_name"]):
                class_id = detection["class_id"]
                if class_id in structure_class_names:
                    detection["class_name"] = structure_class_names[class_id]
            updated_structure_detections.append(detection)
        structure_data["detections"] = updated_structure_detections
        
        # Try to get image dimensions from detections (if available)
        image_width = 1000  # Default width
        image_height = 1000  # Default height
        
        # Try to extract image dimensions from the first detection
        if structure_data.get("detections") and "bbox" in structure_data["detections"][0]:
            first_det = structure_data["detections"][0]
            if "image_width" in first_det:
                image_width = first_det["image_width"]
            if "image_height" in first_det:
                image_height = first_det["image_height"]
                
        # Also check if there's metadata with image size
        if "metadata" in structure_data and "image_size" in structure_data["metadata"]:
            img_size = structure_data["metadata"]["image_size"]
            if "width" in img_size:
                image_width = img_size["width"]
            if "height" in img_size:
                image_height = img_size["height"]
        
        # Import the targeted merger module
        try:
            # Try importing the targeted merger
            from targeted_merger import combine_detections_targeted
            
            logger.info(f"Using targeted merging with rest specialization for beams, barlines, stems, and rests...")

            combined_data = combine_detections_targeted(
                structure_data.get("detections", []),
                symbol_data.get("detections", []),
                high_conf_threshold,
                image_width,
                image_height
            )
                
        except ImportError as e:
            logger.warning(f"Targeted merger module not found: {e}. Using standard combination method")
            
            # Fallback to standard method - take structure elements from structure model
            # and everything else from symbol model
            combined_data = {"detections": []}
            
            # Define structural elements to keep from structure model
            structural_types = ['barline', 'stem', 'beam', 'systemicbarline', 'restwhole']
            
            # Add structural elements from structure model
            structure_count = 0
            for detection in structure_data.get("detections", []):
                class_name = detection.get("class_name", "").lower()
                # Check if this class is a structural element we want to keep
                is_structural = any(struct_type in class_name for struct_type in structural_types)
                if is_structural:
                    combined_data["detections"].append(detection)
                    structure_count += 1
            
            # Add all non-structural elements from symbol model
            symbol_count = 0
            for detection in symbol_data.get("detections", []):
                class_name = detection.get("class_name", "").lower()
                # Check if this class is NOT a structural element (to avoid duplicates)
                is_structural = any(struct_type in class_name for struct_type in structural_types)
                if not is_structural:
                    combined_data["detections"].append(detection)
                    symbol_count += 1
            
            logger.info(f"Using standard method: added {structure_count} structural elements and {symbol_count} non-structural elements")
        
        # Save combined detections
        combined_json = os.path.join(output_dir, f"{img_stem}_combined_detections.json")
        with open(combined_json, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        logger.info(f"Combined {len(structure_data.get('detections', []))} structure detections with {len(symbol_data.get('detections', []))} symbol detections")
        logger.info(f"Saved {len(combined_data['detections'])} combined detections to {combined_json}")
        
        return combined_json
    
    except Exception as e:
        logger.error(f"Error combining detections: {e}")
        logger.error(traceback.format_exc())
        raise
    
def detect_staves(image_path, detections_path, output_dir, pixel_perfect=True):
    """
    Detect staff lines in the image
    
    Args:
        image_path: Path to the input image
        detections_path: Path to the detection file
        output_dir: Directory to save staff line results
        pixel_perfect: Whether to use pixel-perfect alignment
        
    Returns:
        Path to the JSON file with staff line results
    """
    try:
        logger.info(f"Detecting staff lines in {image_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the image stem (filename without extension)
        img_stem = Path(image_path).stem
        
        # Detect staff lines
        staff_data = detect_staff_lines(
            image_path, 
            detections_path,
            output_dir,
            pixel_perfect
        )
        
        # Get the path to the staff line JSON file
        alignment_type = "pixel_perfect" if pixel_perfect else "standard"
        staff_json = os.path.join(output_dir, f"{img_stem}_{alignment_type}.json")
        
        # Verify that the staff line file was created
        if not os.path.exists(staff_json):
            # Try alternative filename
            staff_json = os.path.join(output_dir, f"{img_stem}_staff_lines.json")
            if not os.path.exists(staff_json):
                logger.error(f"Staff line file {staff_json} was not created")
                raise FileNotFoundError(f"No staff line files were created for {image_path}")
        
        logger.info(f"Staff lines detected and saved to {staff_json}")
        return staff_json
    
    except Exception as e:
        logger.error(f"Error detecting staff lines: {e}")
        logger.error(traceback.format_exc())
        raise

def enhance_detections(image_path, detections_path, staff_lines_path, output_dir):
    """
    Enhance detections by adding missing stems and merging closely related elements
    
    Args:
        image_path: Path to the input image
        detections_path: Path to the detection file
        staff_lines_path: Path to the staff lines file
        output_dir: Directory to save enhanced results
        
    Returns:
        Path to the JSON file with enhanced results
    """
    try:
        logger.info(f"Enhancing detections in {image_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the image stem (filename without extension)
        img_stem = Path(image_path).stem
        
        # Call main function from stem_detector module
        from stem_detector import main as enhance_stems_main
        
        # Call the stem detection main function
        enhanced_path = os.path.join(output_dir, f"{img_stem}_enhanced_detections.json")
        
        # Run the stem enhancer main function
        logger.info(f"Running stem detection enhancer...")
        enhance_stems_main(staff_lines_path, detections_path, output_dir, image_path)
        
        # Check if output file was created
        if not os.path.exists(enhanced_path):
            # Try alternative naming patterns that might be used by stem_detector
            base_name = os.path.basename(detections_path).replace("_detections.json", "")
            alt_path = os.path.join(output_dir, f"{base_name}_enhanced_detections.json")
            
            if os.path.exists(alt_path):
                enhanced_path = alt_path
                logger.info(f"Found enhanced detections at {enhanced_path}")
            else:
                logger.warning(f"Enhanced detections file not found at {enhanced_path} or {alt_path}")
                # Fall back to original file
                logger.info(f"Falling back to original detections")
                shutil.copy(detections_path, enhanced_path)
        
        logger.info(f"Stem detection enhancement completed")
        return enhanced_path
    
    except Exception as e:
        logger.error(f"Error enhancing detections: {e}")
        logger.error(traceback.format_exc())
        # If enhancement fails, return the original detections
        logger.info(f"Falling back to original detections: {detections_path}")
        
        # Copy the original detections to the output directory
        enhanced_json = os.path.join(output_dir, f"{Path(image_path).stem}_enhanced.json")
        shutil.copy(detections_path, enhanced_json)
        
        return enhanced_json
    
def merge_detections(detections_path, staff_lines_path, enhanced_path, output_dir, overlap_threshold=0.5):
   """
   Merge all detections into a single file with enhanced rest handling
   
   Args:
       detections_path: Path to the original detection file
       staff_lines_path: Path to the staff lines file
       enhanced_path: Path to the enhanced detections file
       output_dir: Directory to save merged results
       overlap_threshold: Default overlap threshold for detection merging
       
   Returns:
       Path to the JSON file with merged results
   """
   try:
       logger.info(f"Merging detections from {detections_path} and {staff_lines_path}")
       
       # Create output directory if it doesn't exist
       os.makedirs(output_dir, exist_ok=True)
       
       # Get the filename stem
       img_stem = Path(detections_path).stem.replace('_combined_detections', '').replace('_detections', '')
       
       # Merged detections path
       merged_json = os.path.join(output_dir, f"{img_stem}_merged.json")
       
       # Load the original detections
       with open(detections_path, 'r') as f:
           original_data = json.load(f)
       
       # Load the staff lines
       with open(staff_lines_path, 'r') as f:
           staff_data = json.load(f)
       
       # Load enhanced detections if they exist
       if enhanced_path and os.path.exists(enhanced_path):
           with open(enhanced_path, 'r') as f:
               enhanced_data = json.load(f)
       else:
           enhanced_data = {"detections": []}
       
       # New approach: track objects by source and use specialized model mapping
       structure_detections = {}
       symbol_detections = {}
       staff_line_elements = []
       inferred_elements = []
       
       # Define which model is better for which rest type
       rest_model_preference = {
           "restwhole": "structure",   # FR-CNN is 46.06% better
           "resthalf": "symbol",       # YOLO is 32.52% better
           "rest16th": "structure",    # FR-CNN is 1.11% better
           "rest8th": "structure",     # FR-CNN is 24.24% better
           "restquarter": "symbol", # Match with targeted_merger.py
           "rest32nd": "either"        # Both models equally good
       }
       
       # For non-rest elements that are structurally important
       structure_element_types = ["barline", "stem", "beam", "systemicbarline"]
       
       # Collect staff line elements
       for detection in staff_data.get("detections", []):
           staff_line_elements.append(detection)
       
       # Collect inferred elements from enhanced data
       for detection in enhanced_data.get("detections", []):
           if detection.get("inferred", False):
               inferred_elements.append(detection)
               continue
       
       # Process original data (structure model)
       for detection in original_data.get("detections", []):
           class_name = detection.get("class_name", "").lower()
           
           # Skip staff lines
           if "staff_line" in class_name:
               continue
               
           # Mark detection source
           detection["source_model"] = "structure"
           
           # Create grid-based position key for grouping
           bbox = detection.get("bbox", {})
           pos_x = int(bbox.get("center_x", 0) / 10) * 10
           pos_y = int(bbox.get("center_y", 0) / 10) * 10
           position_key = f"{pos_x}_{pos_y}"
           
           # For rests, preserve exact type
           if "rest" in class_name:
               element_type = class_name
           elif any(struct_type in class_name for struct_type in structure_element_types):
               element_type = next(struct_type for struct_type in structure_element_types 
                                  if struct_type in class_name)
           else:
               element_type = class_name
           
           # Store with compound key
           compound_key = f"{element_type}_{position_key}"
           
           if compound_key not in structure_detections:
               structure_detections[compound_key] = []
           structure_detections[compound_key].append(detection)
       
       # Process symbol model detections
       for detection in enhanced_data.get("detections", []):
           # Skip inferred elements and staff lines
           if detection.get("inferred", False) or "staff_line" in detection.get("class_name", "").lower():
               continue
               
           class_name = detection.get("class_name", "").lower()
           
           # Mark detection source
           detection["source_model"] = "symbol"
           
           # Create grid-based position key for grouping
           bbox = detection.get("bbox", {})
           pos_x = int(bbox.get("center_x", 0) / 10) * 10
           pos_y = int(bbox.get("center_y", 0) / 10) * 10
           position_key = f"{pos_x}_{pos_y}"
           
           # For rests, preserve exact type
           if "rest" in class_name:
               element_type = class_name
           elif any(struct_type in class_name for struct_type in structure_element_types):
               element_type = next(struct_type for struct_type in structure_element_types 
                                  if struct_type in class_name)
           else:
               element_type = class_name
           
           # Store with compound key
           compound_key = f"{element_type}_{position_key}"
           
           if compound_key not in symbol_detections:
               symbol_detections[compound_key] = []
           symbol_detections[compound_key].append(detection)
       
       # Create merged detections based on model preferences
       merged_elements = []
       
       # Process all unique elements from both models
       all_compound_keys = set(list(structure_detections.keys()) + list(symbol_detections.keys()))
       
       # Count rest types for reporting
       rest_counts = {}
       
       # SPECIAL HANDLING FOR QUARTER RESTS
       # Extract all quarter rests and process them first
       quarter_rests = []
       
       for compound_key in list(all_compound_keys):
           if "restquarter" in compound_key:
               # Get all quarter rests from this position
               structure_quarters = structure_detections.get(compound_key, [])
               symbol_quarters = symbol_detections.get(compound_key, [])
               
               # Combine quarters from both models for this position
               all_quarters = structure_quarters + symbol_quarters
               
               # If we have quarters at this position, decide which to keep
               if all_quarters:
                   # Keep only the highest confidence quarter rest
                   best_quarter = max(all_quarters, key=lambda x: x.get("confidence", 0))
                   quarter_rests.append(best_quarter)
               
               # Remove this key from further processing
               all_compound_keys.discard(compound_key)
       
       # Now do a global deduplication of quarter rests based on distance
       deduplicated_quarters = []
       quarter_rest_positions = set()  # Track positions we've already seen
       
       # Sort quarters by confidence
       quarter_rests.sort(key=lambda x: x.get("confidence", 0), reverse=True)
       
       # Go through quarters from highest to lowest confidence
       for rest in quarter_rests:
           x = rest["bbox"]["center_x"]
           y = rest["bbox"]["center_y"]
           
           # Check if this quarter is close to any we've already kept
           is_duplicate = False
           for existing_x, existing_y in quarter_rest_positions:
               # Use a larger distance threshold (40 pixels)
               distance = ((x - existing_x) ** 2 + (y - existing_y) ** 2) ** 0.5
               if distance < 40:  # Aggressive quarter rest deduplication
                   is_duplicate = True
                   break
           
           # If not a duplicate, keep it
           if not is_duplicate:
               deduplicated_quarters.append(rest)
               quarter_rest_positions.add((x, y))
               
               # Debug info
               logger.info(f"Keeping quarter rest at ({x:.1f}, {y:.1f}) with confidence {rest.get('confidence', 0):.2f}")
       
       # Add the deduplicated quarter rests to our final merged elements
       merged_elements.extend(deduplicated_quarters)
       
       # Process all other non-quarter-rest elements
       for compound_key in all_compound_keys:
           element_parts = compound_key.split('_', 1)
           element_type = element_parts[0] if len(element_parts) > 0 else ""
           position = element_parts[1] if len(element_parts) > 1 else ""
           
           # Skip quarter rests (we've already processed them)
           if "restquarter" in element_type:
               continue
           
           # Determine appropriate model for this element type
           preferred_model = None
           
           # For rests, use rest-specific preferences
           if "rest" in element_type:
               for rest_type, model in rest_model_preference.items():
                   if rest_type in element_type:
                       preferred_model = model
                       break
                       
               # Default if not found in preferences
               if not preferred_model:
                   preferred_model = "symbol"  # Default to symbol model for unlisted rest types
                   
               # Update rest counts
               if element_type not in rest_counts:
                   rest_counts[element_type] = 0
               rest_counts[element_type] += 1
           
           # For structural elements, prefer structure model
           elif any(struct_type in element_type for struct_type in structure_element_types):
               preferred_model = "structure"
           
           # For other elements, prefer symbol model
           else:
               preferred_model = "symbol"
           
           # Get detections from both models for this element at this position
           structure_elems = structure_detections.get(compound_key, [])
           symbol_elems = symbol_detections.get(compound_key, [])
           
           # No overlap case - just use what we have
           if not structure_elems and symbol_elems:
               # Only symbol model detected this element
               best_elem = max(symbol_elems, key=lambda x: x.get("confidence", 0))
               merged_elements.append(best_elem)
               continue
               
           if structure_elems and not symbol_elems:
               # Only structure model detected this element
               best_elem = max(structure_elems, key=lambda x: x.get("confidence", 0))
               merged_elements.append(best_elem)
               continue
           
           # Both models detected this element - use preferred model
           if preferred_model == "structure":
               best_elem = max(structure_elems, key=lambda x: x.get("confidence", 0))
               merged_elements.append(best_elem)
           elif preferred_model == "symbol":
               best_elem = max(symbol_elems, key=lambda x: x.get("confidence", 0))
               merged_elements.append(best_elem)
           else:  # "either" - use highest confidence
               struct_best = max(structure_elems, key=lambda x: x.get("confidence", 0))
               symbol_best = max(symbol_elems, key=lambda x: x.get("confidence", 0))
               
               best_elem = struct_best if struct_best.get("confidence", 0) >= symbol_best.get("confidence", 0) else symbol_best
               merged_elements.append(best_elem)
           
           # Debug rest selection
           if "rest" in element_type:
               x, y = position.split('_')
               selected_model = "structure" if best_elem.get("source_model") == "structure" else "symbol"
               logger.info(f"Selected {element_type} at ({x}, {y}) from {selected_model} model with confidence {best_elem.get('confidence', 0):.2f}")
       
       # Log rest statistics
       if rest_counts:
           logger.info(f"Final rest distribution: {rest_counts}")
       
       # Build final merged data
       merged_data = {
           "detections": []
       }
       
       # Add in proper order
       merged_data["detections"].extend(merged_elements)    # Regular elements first
       merged_data["detections"].extend(inferred_elements)  # Then inferred elements
       merged_data["detections"].extend(staff_line_elements) # Then staff lines
       
       # Add staff system information
       merged_data["staff_systems"] = staff_data.get("staff_systems", [])
       
       # Log status
       logger.info(f"Merged data contains {len(merged_elements)} regular elements, {len(inferred_elements)} inferred elements, and {len(staff_line_elements)} staff lines")
       logger.info("Using specialized model preferences for different rest types")
       
       # Save the merged detections
       with open(merged_json, 'w') as f:
           json.dump(merged_data, f, indent=2)
       
       logger.info(f"Detections merged and saved to {merged_json}")
       return merged_json
   
   except Exception as e:
       logger.error(f"Error merging detections: {e}")
       logger.error(traceback.format_exc())
       raise

def generate_musicxml(merged_path, staff_lines_path, image_path, output_dir, visualization_dir):
    """
    Generate MusicXML from the merged detections, and also generate custom XML format and MIDI
    
    Args:
        merged_path: Path to the merged detection file
        staff_lines_path: Path to the staff lines file
        image_path: Path to the original image
        output_dir: Directory to save MusicXML output
        visualization_dir: Directory to save visualizations
        
    Returns:
        Dictionary with paths to generated files
    """
    from debug_chord import debug_leftover_chord_attributes
    try:
        logger.info(f"Generating MusicXML from {merged_path}")
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Get the filename stem
        img_stem = Path(image_path).stem
        
        # Output paths
        xml_path = os.path.join(output_dir, f"{img_stem}.musicxml")
        custom_xml_path = os.path.join(output_dir, f"{img_stem}.custom.xml")
        viz_path = os.path.join(visualization_dir, f"{img_stem}_symbolic.png")
        overlay_path = os.path.join(visualization_dir, f"{img_stem}_overlay.png")
        midi_path = os.path.join(output_dir, f"{img_stem}.mid")
        
        # Create processor
        processor = OMRProcessor(merged_path, staff_lines_path)
        
        # Process and generate MusicXML
        musicxml = processor.process()
        
        # Save MusicXML
        if musicxml:
            with open(xml_path, 'w') as f:
                f.write(musicxml)
        else:
            logger.error("Failed to generate MusicXML")
            return None
        
        debug_leftover_chord_attributes(processor)
        
        # Generate visualizations
        visualize_score(processor, viz_path)
        visualize_overlay(processor, image_path, overlay_path)
        
        # Generate custom XML format
        custom_xml_path_result = None
        try:
            logger.info(f"Generating custom XML format to {custom_xml_path}")
            # Add this debug print in complete_pipeline.py before generating custom XML
            print(f"DEBUG: Merged detections path: {processor.json_path if hasattr(processor, 'json_path') else 'unknown'}")
            custom_xml_path_result = generate_custom_xml(processor, custom_xml_path)
            logger.info(f"Custom XML generated and saved to {custom_xml_path}")
        except Exception as e:
            logger.error(f"Error generating custom XML: {e}")
            logger.error(traceback.format_exc())
            logger.info("Continuing with MusicXML generation despite custom XML error")
        
        # Generate MIDI file
        midi_path_result = None
        try:
            logger.info(f"Generating MIDI file to {midi_path}")
            midi_path_result = add_midi_generation(processor, output_dir, xml_path)
            if midi_path_result:
                logger.info(f"MIDI file generated and saved to {midi_path_result}")
            else:
                logger.error("Failed to generate MIDI file")
        except Exception as e:
            logger.error(f"Error generating MIDI: {e}")
            logger.error(traceback.format_exc())
            logger.info("Continuing despite MIDI generation error")
        
        logger.info(f"MusicXML generated and saved to {xml_path}")
        logger.info(f"Symbolic visualization saved to {viz_path}")
        logger.info(f"Overlay visualization saved to {overlay_path}")
        
        return {
            "musicxml": xml_path,
            "custom_xml": custom_xml_path_result if custom_xml_path_result else None,
            "symbolic_viz": viz_path,
            "overlay_viz": overlay_path,
            "midi": midi_path_result if midi_path_result else None
        }
    
    except Exception as e:
        logger.error(f"Error generating MusicXML: {e}")
        logger.error(traceback.format_exc())
        raise

    
def process_image(image_path, structure_model_path, structure_mapping_path, 
                 symbol_model_path, symbol_mapping_path, output_base_dir, 
                 conf_threshold=0.3, max_detections=1000, iou_threshold=0.35,
                 pixel_perfect=True, skip_enhancement=False,
                 high_conf_threshold=0.5, overlap_threshold=0.5):
    """
    Process a single image through the complete pipeline using dual models
    
    Args:
        image_path: Path to the input image
        structure_model_path: Path to the structural element detection model
        structure_mapping_path: Path to the class mapping for structural elements
        symbol_model_path: Path to the musical symbol detection model
        symbol_mapping_path: Path to the class mapping for musical symbols
        output_base_dir: Base directory for output
        conf_threshold: Confidence threshold for detections
        max_detections: Maximum number of detections to consider
        iou_threshold: IoU threshold for non-maximum suppression
        pixel_perfect: Whether to use pixel-perfect alignment for staff lines
        skip_enhancement: Whether to skip the enhancement step
        
    Returns:
        Dictionary with paths to all generated files
    """
    try:
        # Get the filename stem
        img_stem = Path(image_path).stem
        
        # Create a specific output directory for this image
        image_output_dir = os.path.join(output_base_dir, img_stem)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Create output directories for each stage
        dirs = create_output_directories(image_output_dir)
        
        # Step 1a: Detect structural elements using Faster R-CNN
        logger.info("Step 1a: Detecting structural elements (barlines, stems, beams)...")
        structure_detection_path = detect_structure_elements(
            image_path, structure_model_path, structure_mapping_path, 
            dirs["structure_detections"], conf_threshold, max_detections, iou_threshold
        )
        
        # Step 1b: Detect musical symbols using YOLO
        logger.info("Step 1b: Detecting musical symbols...")
        symbol_detection_path = detect_music_symbols(
            image_path, symbol_model_path, symbol_mapping_path, 
            dirs["symbol_detections"], conf_threshold, max_detections, iou_threshold
        )
        
        # Step 1c: Combine detections from both models
        logger.info("Step 1c: Combining detections from both models...")
        combined_detection_path = combine_detections(
            structure_detection_path, symbol_detection_path,
            structure_mapping_path, symbol_mapping_path,
            dirs["combined_detections"],
            high_conf_threshold=high_conf_threshold,
            overlap_threshold=overlap_threshold
        )
        agent = MusicScoreAgent()
        agent_recommendations = agent.analyze_and_recommend(structure_detection_path, symbol_detection_path, image_path)
        if agent_recommendations['density'] > 15:
            high_conf_threshold = agent_recommendations['high_conf_threshold']
            overlap_threshold = agent_recommendations['overlap_threshold']
            logger.info(f"Agent recommendations: {agent_recommendations['processing_notes']}")
    
        # Step 2: Detect staff lines
        logger.info("Step 2: Detecting staff lines...")
        staff_lines_path = detect_staves(
            image_path, combined_detection_path, dirs["staff_lines"], pixel_perfect
        )
        
        # Step 3: Enhance detections (add missing stems, etc.)
        if not skip_enhancement:
            logger.info("Step 3: Enhancing detections...")
            enhanced_path = enhance_detections(
                image_path, combined_detection_path, staff_lines_path, dirs["enhanced"]
            )
        else:
            logger.info("Step 3: Skipping enhancement...")
            enhanced_path = None
        
        # Step 4: Merge all detections
        logger.info("Step 4: Merging detections...")
        merged_path = merge_detections(
            combined_detection_path, staff_lines_path, enhanced_path, dirs["merged"]
        )
        
        # Step 5: Generate MusicXML, custom XML, and MIDI
        logger.info("Step 5: Generating MusicXML, custom XML, and MIDI...")
        xml_results = generate_musicxml(
            merged_path, staff_lines_path, image_path, 
            dirs["musicxml"], dirs["visualization"]
        )
        
        # Return all generated paths
        result = {
            "structure_detection": structure_detection_path,
            "symbol_detection": symbol_detection_path,
            "combined_detection": combined_detection_path,
            "staff_lines": staff_lines_path,
            "enhanced": enhanced_path,
            "merged": merged_path,
            "musicxml": xml_results["musicxml"],
            "custom_xml": xml_results["custom_xml"],
            "symbolic_viz": xml_results["symbolic_viz"],
            "overlay_viz": xml_results["overlay_viz"]
            # "midi": xml_results.get("midi")  # Include MIDI path if available
        }
        
        logger.info(f"Successfully processed {image_path}")
        return result
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        logger.error(traceback.format_exc())
        raise
    
    
def main():
    """Main entry point for the complete pipeline"""
    parser = argparse.ArgumentParser(description="Complete Music Notation Analysis Pipeline with Dual Models")
    
    # Required arguments
    parser.add_argument("--image", required=True, help="Path to input image or directory of images")
    
    # Structure model arguments (Faster R-CNN for barlines, stems, beams)
    parser.add_argument("--structure_model", required=True, help="Path to Faster R-CNN ONNX model for structural elements")
    parser.add_argument("--structure_mapping", required=True, help="Path to class mapping file for structural elements")
    
    # Symbol model arguments (YOLO for other musical symbols)
    parser.add_argument("--symbol_model", required=True, help="Path to YOLOv8 model for musical symbols")
    parser.add_argument("--symbol_mapping", required=True, help="Path to class mapping file for musical symbols")
    
    # Optional arguments
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--max_detections", type=int, default=1000, help="Maximum detections")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for NMS")
    parser.add_argument("--no_pixel_perfect", action="store_true", help="Disable pixel-perfect staff alignment")
    parser.add_argument("--skip_enhancement", action="store_true", help="Skip detection enhancement step")
    parser.add_argument("--staff_mode", choices=["auto", "piano"], default="piano", help="Staff mode for MusicXML")
    
    parser.add_argument("--high_conf", type=float, default=0.5, 
                   help="High confidence threshold for detection merging")
    parser.add_argument("--overlap", type=float, default=0.5, 
                   help="Overlap (IoU) threshold for detection merging")
    
    args = parser.parse_args()
    
    # Validate paths
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image path does not exist: {args.image}")
        sys.exit(1)
    
    structure_model_path = Path(args.structure_model)
    if not structure_model_path.exists():
        logger.error(f"Structure model path does not exist: {args.structure_model}")
        sys.exit(1)
    
    structure_mapping_path = Path(args.structure_mapping)
    if not structure_mapping_path.exists():
        logger.error(f"Structure mapping path does not exist: {args.structure_mapping}")
        sys.exit(1)
    
    symbol_model_path = Path(args.symbol_model)
    if not symbol_model_path.exists():
        logger.error(f"Symbol model path does not exist: {args.symbol_model}")
        sys.exit(1)
    
    symbol_mapping_path = Path(args.symbol_mapping)
    if not symbol_mapping_path.exists():
        logger.error(f"Symbol mapping path does not exist: {args.symbol_mapping}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process single image or directory
    if image_path.is_file():
        # Process single image
        logger.info(f"Processing single image: {image_path}")
        try:
            result = process_image(
                str(image_path), 
                str(structure_model_path), str(structure_mapping_path),
                str(symbol_model_path), str(symbol_mapping_path), 
                str(output_dir),
                args.conf, args.max_detections, args.iou,
                not args.no_pixel_perfect, args.skip_enhancement,
                high_conf_threshold=args.high_conf,
                overlap_threshold=args.overlap
            )
            
            # Print summary
            print("\nProcessing complete!")
            print(f"Input image: {image_path}")
            print(f"MusicXML output: {result['musicxml']}")
            print(f"MIDI output: {result.get('midi', 'Not generated')}")
            print(f"Symbolic visualization: {result['symbolic_viz']}")
            print(f"Overlay visualization: {result['overlay_viz']}")
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            sys.exit(1)
    
    elif image_path.is_dir():
        # Process directory of images
        logger.info(f"Processing directory of images: {image_path}")
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = [f for f in image_path.glob('*') if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.error(f"No image files found in {image_path}")
            sys.exit(1)
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process each image
        successful = []
        failed = []
        
        for img_file in image_files:
            logger.info(f"Processing {img_file}...")
            try:
                result = process_image(
                    str(img_file), 
                    str(structure_model_path), str(structure_mapping_path),
                    str(symbol_model_path), str(symbol_mapping_path), 
                    str(output_dir),
                    args.conf, args.max_detections, args.iou,
                    not args.no_pixel_perfect, args.skip_enhancement,
                    high_conf_threshold=args.high_conf,
                    overlap_threshold=args.overlap
                )
                successful.append((str(img_file), result['musicxml'], result.get('midi')))
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")
                failed.append(str(img_file))
        
        # Print summary
        print("\nProcessing complete!")
        print(f"Successfully processed {len(successful)} images")
        print(f"Failed to process {len(failed)} images")
        
        if successful:
            print("\nSuccessfully processed images:")
            for img, xml, midi in successful:
                midi_status = f"MIDI: {midi}" if midi else "MIDI: Not generated"
                print(f"  {img} -> XML: {xml}, {midi_status}")
        
        if failed:
            print("\nFailed to process images:")
            for img in failed:
                print(f"  {img}")
    
    else:
        logger.error(f"Invalid image path: {args.image}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    


# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/complete_pipeline.py  \
#         --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png \
#         --structure_model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/may_2023_ex003.onnx \
#         --structure_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/mapping.txt \
#         --symbol_model /import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt \
#         --symbol_mapping /homes/es314/omr-objdet-benchmark/data/class_mapping.json



# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/complete_pipeline.py \
#         --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png \
#         --structure_model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/may_2023_ex001.onnx \
#         --structure_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/mapping.txt \
#         --symbol_model /import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt \
#         --symbol_mapping /homes/es314/omr-objdet-benchmark/data/class_mapping.json



# best model is /homes/es314/runs/detect/train/weights/best.pt which is a continuation of training for /import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt for another 77 epochs so around 160 epochs

# there is an issue with image size when overlaying
# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/complete_pipeline.py  \
#         --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png \
#         --structure_model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/may_2023_ex001.onnx \
#         --structure_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/mapping.txt \
#         --symbol_model /homes/es314/runs/detect/train/weights/best.pt \
#         --symbol_mapping /homes/es314/omr-objdet-benchmark/data/class_mapping.json