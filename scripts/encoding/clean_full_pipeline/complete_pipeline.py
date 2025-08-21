#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List, Any, Optional

# Import all necessary modules
from inference_detect_notation import detect_music_notation
from link_obj import MusicSymbolLinker
from pitch_identification import PitchIdentifier
from visualisation import visualize_pitched_score

def detect_and_save_staff_lines(image_path, object_detections_path, output_dir, pixel_perfect=True, debug=True):
    """
    Detect staff lines in a music score image using object detection results and save the results
    
    Args:
        image_path: Path to the image file
        object_detections_path: Path to the JSON file containing object detections
        output_dir: Directory to save results
        pixel_perfect: Enable pixel-perfect alignment
        debug: Enable debug mode
        
    Returns:
        Path to the staff lines JSON file
    """
    from staff_line_detector import EnhancedStaffDetector  # Adjust import path as needed


    
    detector = EnhancedStaffDetector(debug=debug, pixel_perfect=pixel_perfect)

    img_stem = os.path.splitext(os.path.basename(image_path))[0]

    # Prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    staff_json_path = os.path.join(output_dir, f"{img_stem}_staff_lines.json")
    staff_viz_path = os.path.join(output_dir, f"{img_stem}_staff_lines.png")

    try:
        staff_data = detector.detect(image_path, object_detections_path)

        with open(staff_json_path, 'w') as f:
            json.dump(staff_data, f, indent=2)

        detector.visualize(image_path, staff_data, staff_viz_path)

        if debug:
            print(f"Staff line detection complete. Results saved to {staff_json_path}")
            print(f"Visualization saved to {staff_viz_path}")

        return staff_json_path

    except Exception as e:
        print(f"Error detecting staff lines: {e}")
        empty_result = {"staff_systems": [], "detections": []}
        with open(staff_json_path, 'w') as f:
            json.dump(empty_result, f)
        return staff_json_path

def enhance_stem_detections(staff_lines_path, object_detections_path, output_dir, image_path=None):
    """
    Enhance object detections by inferring missing stems
    
    Args:
        staff_lines_path: Path to staff lines JSON file
        object_detections_path: Path to object detections JSON file
        output_dir: Directory to save enhanced detections
        image_path: Optional path to the original score image
        
    Returns:
        Path to the enhanced detections JSON file
    """
    from stem_detector import OMREnhancer, load_json, save_json_file
    
    # Create output filename
    base_name = os.path.basename(object_detections_path).replace("_detections.json", "")
    output_path = os.path.join(output_dir, f"{base_name}_enhanced_detections.json")
    
    # Load staff line and object detection data
    print(f"Loading staff lines from {staff_lines_path}")
    staff_info = load_json(staff_lines_path)
    
    print(f"Loading object detections from {object_detections_path}")
    object_detections = load_json(object_detections_path)
    
    if not staff_info or not object_detections:
        print("Failed to load required input files.")
        # Create a placeholder enhanced file
        result = {"detections": []}
        save_json_file(result, output_path)
        return output_path
    
    # Extract detections from the object_detections file
    detections = object_detections.get("detections", [])
    print(f"Loaded {len(detections)} object detections")
    
    # Create enhancer
    enhancer = OMREnhancer(staff_info)
    
    # Enhance detections - focusing only on stem detection
    print("Inferring missing stems...")
    enhanced_detections = enhancer.enhance_detections(detections, image_path)
    
    # Count original and inferred objects
    original_count = len(detections)
    enhanced_count = len(enhanced_detections)
    inferred_count = sum(1 for det in enhanced_detections if det.get("inferred", False))
    inferred_stems_count = sum(1 for det in enhanced_detections 
                              if det.get("inferred", False) and det.get("class_name") == "stem")
    
    print(f"Original detections: {original_count}")
    print(f"Inferred stems: {inferred_stems_count}")
    print(f"Total enhanced detections: {enhanced_count}")
    
    # Create output structure mirroring the input
    result = object_detections.copy()
    result["detections"] = enhanced_detections
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["enhanced"] = True
    result["metadata"]["inferred_stems_count"] = inferred_stems_count
    
    # Save results
    print(f"Saving enhanced detections to {output_path}")
    save_json_file(result, output_path)
    
    # Create overlay visualization
    try:
        if image_path and os.path.exists(image_path):
            from stem_detector import overlay_detections_on_image
            viz_path = os.path.join(output_dir, f"{base_name}_stem_visualization.png")
            overlay_detections_on_image(
                image_path, 
                enhanced_detections,
                output_path=viz_path,
                show_original=True,
                show_inferred=True
            )
            print(f"Visualization saved to {viz_path}")
    except Exception as e:
        print(f"Error generating visualization: {e}")
    
    return output_path


def merge_detections(staff_json, detection_json, output_path):
    """
    Merge staff lines and object detections into a single file
    
    Args:
        staff_json: Path to staff lines JSON file
        detection_json: Path to object detections JSON file
        output_path: Path to save merged file
        
    Returns:
        None
    """
    print(f"Merging staff lines from {staff_json} and detections from {detection_json}")
    
    try:
        # Load staff lines
        with open(staff_json, 'r') as f:
            staff_data = json.load(f)
            
        # Load object detections
        with open(detection_json, 'r') as f:
            detection_data = json.load(f)
            
        # Create merged data structure
        merged_data = {
            "staffs": staff_data.get("staff_systems", []),
            "detections": detection_data.get("detections", [])
        }
        
        # Save merged data
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
            
        print(f"Merged data saved to {output_path}")
    except Exception as e:
        print(f"Error merging detections: {e}")
        # Create a simple merged file
        with open(output_path, 'w') as f:
            json.dump({"staffs": [], "detections": []}, f)


def run_full_pipeline(image_path, model_path, class_mapping_file, output_base_dir="results"):
    """
    Run the complete music notation analysis pipeline:
    1. Detect musical symbols using YOLOv8
    2. Detect staff lines
    3. Enhance detections by adding missing stems
    4. Merge detections
    5. Link related symbols
    6. Identify pitches
    7. Visualize results
    
    Args:
        image_path: Path to image or directory of images
        model_path: Path to YOLOv8 model
        class_mapping_file: Path to class mapping JSON file
        output_base_dir: Base directory for all outputs
        
    Returns:
        Dictionary with paths to all generated files
    """
    # Create output directories
    staff_dir = os.path.join(output_base_dir, "staff_lines")
    object_dir = os.path.join(output_base_dir, "object_detections")
    stem_dir = os.path.join(output_base_dir, "stem_detections")
    merged_dir = os.path.join(output_base_dir, "merged_detections")
    linked_dir = os.path.join(output_base_dir, "linked_data")
    pitched_dir = os.path.join(output_base_dir, "pitched_data")
    visualization_dir = os.path.join(output_base_dir, "visualizations")
    
    os.makedirs(staff_dir, exist_ok=True)
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(stem_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(linked_dir, exist_ok=True)
    os.makedirs(pitched_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Process single image or directory
    if os.path.isdir(image_path):
        # Find all images
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(list(Path(image_path).glob(f"*{ext}")))
        
        all_results = []
        for img_file in image_files:
            img_stem = img_file.stem
            print(f"\nProcessing image: {img_file}")
            
            try:
                result = process_single_image(
                    str(img_file),
                    img_stem,
                    model_path,
                    class_mapping_file,
                    staff_dir,
                    object_dir,
                    stem_dir,
                    merged_dir,
                    linked_dir,
                    pitched_dir,
                    visualization_dir
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        return all_results
    else:
        # Process single image
        img_name = os.path.basename(image_path)
        img_stem = os.path.splitext(img_name)[0]
        
        return process_single_image(
            image_path,
            img_stem,
            model_path,
            class_mapping_file,
            staff_dir,
            object_dir,
            stem_dir,
            merged_dir,
            linked_dir,
            pitched_dir,
            visualization_dir
        )


def process_single_image(image_path, img_stem, model_path, class_mapping_file,
                         staff_dir, object_dir, stem_dir, merged_dir, linked_dir, 
                         pitched_dir, visualization_dir):
    """Process a single image through the entire pipeline"""
    result_paths = {}
    
    # Determine if this is a Faster R-CNN model based on the path
    is_faster_rcnn = any([
        "/full-no-staff-output/" in model_path,
        "/full-with-staff-output/" in model_path,
        "/half-older_config_faster_rcnn_omr_output/" in model_path,
        "/staff-half-older_config_faster_rcnn_omr_output/" in model_path,
        # Alternatively, check if the model path doesn't contain "yolo8runs"
        not "/yolo8runs/" in model_path
    ])
    
    # Select the appropriate class mapping file for Faster R-CNN models
    if is_faster_rcnn:
        # Check if this is a staff-variant of Faster R-CNN
        is_staff_variant = any([
            "/full-with-staff-output/" in model_path,
            "/staff-half-older_config_faster_rcnn_omr_output/" in model_path
        ])
        
        if is_staff_variant:
            # Use staff variant mapping
            faster_rcnn_mapping = "/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset/mapping.txt"
        else:
            # Use regular variant mapping
            faster_rcnn_mapping = "/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/mapping.txt"
        
        print(f"Using Faster R-CNN mapping file: {faster_rcnn_mapping}")
        actual_class_mapping = faster_rcnn_mapping
    else:
        # Use the provided YOLO class mapping
        actual_class_mapping = class_mapping_file
    
    print(f"Model type: {'Faster R-CNN' if is_faster_rcnn else 'YOLOv8'} - {model_path}")
    
    # Step 1: Detect musical symbols
    print(f"Step 1: Detecting musical symbols using {'Faster R-CNN' if is_faster_rcnn else 'YOLOv8'} model...")
    object_json = os.path.join(object_dir, f"{img_stem}_detections.json")
    object_viz = os.path.join(object_dir, f"{img_stem}_detection.jpg")
    
    # Make sure the output directory exists
    os.makedirs(object_dir, exist_ok=True)
    
    # Use the existing detection function with the selected mapping
    detect_music_notation(model_path, image_path, actual_class_mapping, object_dir)
    
    result_paths["object_json"] = object_json
    result_paths["object_visualization"] = object_viz

    # Check if object detection succeeded
    if not os.path.exists(object_json):
        print(f"Warning: Object detection file not found: {object_json}")
        # Create an empty detections file
        with open(object_json, 'w') as f:
            json.dump({"detections": []}, f)
    
    # Step 2: Detect staff lines using the object detections
    print("Step 2: Detecting staff lines...")
    staff_json = os.path.join(staff_dir, f"{img_stem}_staff_lines.json")
    staff_viz = os.path.join(staff_dir, f"{img_stem}_staff_lines.png")
    
    # Ensure the output directory exists
    os.makedirs(staff_dir, exist_ok=True)
    
    # Now that we have object detections, we can detect staff lines
    try:
        staff_json_path = detect_and_save_staff_lines(
                    image_path=image_path,
                    object_detections_path=object_json,
                    output_dir=staff_dir,
                    pixel_perfect=True,
                    debug=True
                )

        print(f"Staff lines saved to: {staff_json_path}")
    except Exception as e:
        print(f"Error detecting staff lines: {e}")
        # Create an empty staff file if detection fails
        with open(staff_json, 'w') as f:
            json.dump({"staff_systems": [], "detections": []}, f)
    
    result_paths["staff_json"] = staff_json
    result_paths["staff_visualization"] = staff_viz
    
    # Step 3: Enhance detections with missing stems
    print("Step 3: Enhancing detections by inferring missing stems...")
    enhanced_json = os.path.join(stem_dir, f"{img_stem}_enhanced_detections.json")
    
    # Ensure the output directory exists
    os.makedirs(stem_dir, exist_ok=True)
    
    # Check if the staff file exists
    if not os.path.exists(staff_json):
        print(f"Warning: Staff line file not found: {staff_json}")
        # Create an empty staff file
        with open(staff_json, 'w') as f:
            json.dump({"staff_systems": [], "detections": []}, f)
    
    # Enhance detections with stems
    try:
        enhanced_json_path = enhance_stem_detections(staff_json, object_json, stem_dir, image_path)
        print(f"Enhanced detections saved to: {enhanced_json_path}")
    except Exception as e:
        print(f"Error enhancing detections with stems: {e}")
        # Copy the original object json as a fallback
        try:
            shutil.copy(object_json, enhanced_json)
        except Exception as copy_error:
            print(f"Error copying object detections: {copy_error}")
            # Create an empty enhanced file
            with open(enhanced_json, 'w') as f:
                json.dump({"detections": []}, f)
    
    result_paths["enhanced_json"] = enhanced_json
    
    # Step 4: Merge staff lines and enhanced object detections
    print("Step 4: Merging staff lines and enhanced object detections...")
    merged_json = os.path.join(merged_dir, f"{img_stem}_merged.json")
    
    # Ensure the output directory exists
    os.makedirs(merged_dir, exist_ok=True)
    
    # Now merge the files (even if they're empty)
    try:
        merge_detections(staff_json, enhanced_json, merged_json)
    except Exception as e:
        print(f"Error merging detections: {e}")
        # Create a simple merged file
        with open(merged_json, 'w') as f:
            json.dump({"staffs": [], "detections": []}, f)
    
    result_paths["merged_json"] = merged_json

    # Step 5: Link symbols
    print("Step 5: Linking musical symbols...")
    linked_json = os.path.join(linked_dir, f"{img_stem}_merged_linked_data.json")
    linked_viz = os.path.join(linked_dir, f"{img_stem}_linked_visualization.png")

    # Ensure the output directory exists
    os.makedirs(linked_dir, exist_ok=True)

    print(f"Will save linked data to: {linked_json}")
    try:
        linker = MusicSymbolLinker(merged_json)
        # Use list() when iterating over dictionaries to prevent the "dictionary changed size" error
        linked_result = linker.process(linked_dir)
    except Exception as e:
        print(f"Error in linking: {e}")
        # Create a simple linked file
        shutil.copy(merged_json, linked_json)

    # Verify the file exists and has content
    if os.path.exists(linked_json):
        print(f"Verified linked_json exists at: {linked_json}")
        try:
            with open(linked_json, 'r') as f:
                linked_content = json.load(f)
                print(f"File contains {len(linked_content.get('detections', []))} detections")
        except Exception as e:
            print(f"Error reading linked file: {e}")
            # Create a replacement file
            with open(linked_json, 'w') as f:
                json.dump({"detections": []}, f)
    else:
        print(f"WARNING: Expected linked data file not found: {linked_json}")
        # Create a placeholder file
        with open(linked_json, 'w') as f:
            json.dump({"detections": []}, f)

    # Step 6: Identify pitches
    print("Step 6: Identifying pitches for noteheads...")
    pitched_json = os.path.join(pitched_dir, f"{img_stem}_pitched.json")

    # Ensure the output directory exists
    os.makedirs(pitched_dir, exist_ok=True)

    # Try to process the linked file for pitch identification
    try:
        print(f"Using linked data for pitch identification")
        identifier = PitchIdentifier(debug=True)
        identifier.process_file(linked_json, pitched_json)
    except Exception as e:
        print(f"Error in pitch identification: {e}")
        # Create a placeholder pitched file
        with open(pitched_json, 'w') as f:
            json.dump({"detections": []}, f)
    
    # Step 7: Create visualization with pitch information
    print("Step 7: Creating visualization with pitch information...")
    pitch_viz = os.path.join(visualization_dir, f"{img_stem}_pitched_visualization.png")
    
    # Ensure the output directory exists
    os.makedirs(visualization_dir, exist_ok=True)
    
    try:
        visualize_pitched_score(image_path, pitched_json, pitch_viz)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        # Copy the original image as a fallback
        try:
            shutil.copy(image_path, pitch_viz)
        except Exception as copy_error:
            print(f"Error copying original image: {copy_error}")
    
    result_paths["pitch_visualization"] = pitch_viz
    
    print(f"Completed processing for {img_stem}")
    return result_paths 


def main():
    parser = argparse.ArgumentParser(description="Complete music score analysis pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Base output directory")
    
    args = parser.parse_args()
    
    run_full_pipeline(args.image, args.model, args.class_mapping, args.output_dir)
    
if __name__ == "__main__":
    main()