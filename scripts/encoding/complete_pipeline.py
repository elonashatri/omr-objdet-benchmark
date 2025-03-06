import os
import argparse
import json
from pathlib import Path
from staff_line_detector import detect_and_save_staff_lines, merge_detections
from inference_detect_notation import detect_music_notation
from link_obj import MusicSymbolLinker
from pitch_identification import PitchIdentifier
from visualisation import visualize_pitched_score

def run_full_pipeline(image_path, model_path, class_mapping_file, output_base_dir="results"):
    """
    Run the complete music notation analysis pipeline:
    1. Detect staff lines
    2. Detect musical symbols using YOLOv8
    3. Merge detections
    4. Link related symbols
    5. Identify pitches
    6. Visualize results
    
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
    merged_dir = os.path.join(output_base_dir, "merged_detections")
    linked_dir = os.path.join(output_base_dir, "linked_data")
    pitched_dir = os.path.join(output_base_dir, "pitched_data")
    visualization_dir = os.path.join(output_base_dir, "visualizations")
    
    os.makedirs(staff_dir, exist_ok=True)
    os.makedirs(object_dir, exist_ok=True)
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
            merged_dir,
            linked_dir,
            pitched_dir,
            visualization_dir
        )

def process_single_image(image_path, img_stem, model_path, class_mapping_file,
                         staff_dir, object_dir, merged_dir, linked_dir, pitched_dir, visualization_dir):
    """Process a single image through the entire pipeline"""
    result_paths = {}
    
    # Step 1: Detect staff lines
    print("Step 1: Detecting staff lines...")
    staff_json = os.path.join(staff_dir, f"{img_stem}_staff_lines.json")
    staff_viz = os.path.join(staff_dir, f"{img_stem}_staff_lines.png")
    
    staff_json_path = detect_and_save_staff_lines(image_path, staff_dir)
    
    result_paths["staff_json"] = staff_json
    result_paths["staff_visualization"] = staff_viz
    
    # Step 2: Detect musical symbols
    print("Step 2: Detecting musical symbols...")
    object_json = os.path.join(object_dir, f"{img_stem}_detections.json")
    object_viz = os.path.join(object_dir, f"{img_stem}_detection.jpg")
    
    detect_music_notation(model_path, image_path, class_mapping_file, object_dir)
    
    result_paths["object_json"] = object_json
    result_paths["object_visualization"] = object_viz
    
    # Step 3: Merge detections
    print("Step 3: Merging staff lines and object detections...")
    merged_json = os.path.join(merged_dir, f"{img_stem}_merged.json")
    
    merge_detections(staff_json, object_json, merged_json)
    
    result_paths["merged_json"] = merged_json

    # Step 4: Link symbols
    print("Step 4: Linking musical symbols...")
    linked_json = os.path.join(linked_dir, f"{img_stem}_merged_linked_data.json")
    linked_viz = os.path.join(linked_dir, f"{img_stem}_linked_visualization.png")

    print(f"Will save linked data to: {linked_json}")
    linker = MusicSymbolLinker(merged_json)
    linked_result = linker.process(linked_dir)
    # print(f"Actual linked data file: {linked_result}")

    # Verify the file exists and has content
    if os.path.exists(linked_json):
        print(f"Verified linked_json exists at: {linked_json}")
        with open(linked_json, 'r') as f:
            linked_content = json.load(f)
            print(f"File contains {len(linked_content.get('detections', []))} detections")
    else:
        print(f"WARNING: Expected linked data file not found: {linked_json}")
        # Try to find the actual file that was created
        actual_files = [f for f in os.listdir(linked_dir) if f.startswith(img_stem) and f.endswith('.json')]
        if actual_files:
            linked_json = os.path.join(linked_dir, actual_files[0])
            print(f"Found alternative linked data file: {linked_json}")
    
    # # Step 5: Identify pitches
    # print("Step 5: Identifying pitches for noteheads...")
    # pitched_json = os.path.join(pitched_dir, f"{img_stem}_pitched.json")

    # # Check if linked JSON exists and has content
    # if os.path.exists(linked_json):
    #     with open(linked_json, 'r') as f:
    #         linked_data = json.load(f)
    #         if "detections" in linked_data and linked_data["detections"]:
    #             print(f"Linked data has {len(linked_data['detections'])} detections")
                
    #             # Now proceed with pitch identification
    #             identifier = PitchIdentifier(debug=True)
    #             identifier.process_file(linked_json, pitched_json)
    #         else:
    #             print(f"WARNING: Linked data file has no detections")
    #             # Create an empty pitched file
    #             with open(pitched_json, 'w') as f:
    #                 json.dump({"detections": []}, f, indent=2)
    # else:
    #     print(f"ERROR: Linked data file not found: {linked_json}")
    
    # Step 5: Identify pitches
    print("Step 5: Identifying pitches for noteheads...")
    pitched_json = os.path.join(pitched_dir, f"{img_stem}_pitched.json")

    # Use the merged file directly since the linked file has issues
    print(f"Using merged data for pitch identification")
    identifier = PitchIdentifier(debug=True)
    identifier.process_file(merged_json, pitched_json)
    
    # Step 6: Create visualization with pitch information
    print("Step 6: Creating visualization with pitch information...")
    pitch_viz = os.path.join(visualization_dir, f"{img_stem}_pitched_visualization.png")
    
    visualize_pitched_score(image_path, pitched_json, pitch_viz)
    
    result_paths["pitch_visualization"] = pitch_viz
    
    print(f"Completed processing for {img_stem}")
    return result_paths

def main():
    parser = argparse.ArgumentParser(description="Complete music score analysis pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="/homes/es314/omr-objdet-benchmark/scripts/encoding/results", help="Base output directory")
    
    args = parser.parse_args()
    
    run_full_pipeline(args.image, args.model, args.class_mapping, args.output_dir)
    
if __name__ == "__main__":
    main()