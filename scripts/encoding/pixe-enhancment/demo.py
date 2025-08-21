#!/usr/bin/env python3
"""
Pixel-Perfect Staff Line Alignment Demo
---------------------------------------
This script demonstrates how to apply the pixel-perfect alignment
to fix slight offsets in staff line positions.
"""

import os
import json
import cv2
import matplotlib.pyplot as plt
from precise_pixel_alignment import fix_staff_alignment, visualize_alignment_improvement
from element_driven_staff_detection import ElementDrivenStaffDetector

def main():
    """Main function to demonstrate pixel-perfect staff line alignment"""
    # Configure paths
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/Accidentals-004.png"  # Replace with your image path
    detected_elements_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/pixe-enhancment/results/enhanced/Accidentals-004_pixel_perfect.json"  # Replace with your YOLO detections
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate initial staff lines using element-driven approach
    print("Step 1: Generating initial staff lines...")
    detector = ElementDrivenStaffDetector(debug=True)
    initial_staff_data = detector.detect(image_path, detected_elements_path)
    
    initial_json = os.path.join(output_dir, "initial_staff_lines.json")
    with open(initial_json, 'w') as f:
        json.dump(initial_staff_data, f, indent=2)
    
    detector.visualize(image_path, initial_staff_data, os.path.join(output_dir, "initial_visualization.png"))
    
    # Step 2: Apply pixel-perfect alignment
    print("\nStep 2: Applying pixel-perfect alignment...")
    aligned_staff_data = fix_staff_alignment(
        image_path, 
        initial_json,
        os.path.join(output_dir, "aligned_staff_lines.json")
    )
    
    print("\nComplete! Check the comparison visualization to see the improvement.")

def apply_to_existing_results(image_path, existing_staff_json, output_dir="results/pixel_perfect"):
    """
    Apply pixel-perfect alignment to existing staff detection results
    
    Args:
        image_path: Path to the music score image
        existing_staff_json: Path to existing staff line JSON file
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(existing_staff_json))[0]
    aligned_json = os.path.join(output_dir, f"{base_name}_aligned.json")
    comparison_viz = os.path.join(output_dir, f"{base_name}_comparison.png")
    
    print(f"Applying pixel-perfect alignment to: {existing_staff_json}")
    
    # Apply alignment
    aligned_data = fix_staff_alignment(image_path, existing_staff_json, aligned_json)
    
    # Create zoomed-in comparison for better visualization of small offsets
    create_zoomed_comparison(image_path, existing_staff_json, aligned_json, 
                           os.path.join(output_dir, f"{base_name}_zoomed.png"))
    
    print(f"Alignment complete. Results saved to {output_dir}")
    return aligned_json

def create_zoomed_comparison(image_path, original_json, aligned_json, output_path):
    """
    Create a zoomed-in comparison to better visualize the small alignment differences
    
    Args:
        image_path: Path to the music score image
        original_json: Path to original staff line JSON
        aligned_json: Path to aligned staff line JSON
        output_path: Path to save zoomed comparison
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load staff data
    with open(original_json, 'r') as f:
        original_staff = json.load(f)
    
    with open(aligned_json, 'r') as f:
        aligned_staff = json.load(f)
    
    # Get dimensions of the image
    height, width = img.shape[:2]
    
    # Find a good region to zoom in on
    # Ideally, focus on area with staff lines and visible original content
    # For simplicity, we'll use the center portion of the first staff system
    zoom_center_x = width // 2
    zoom_center_y = 0
    
    if original_staff.get("staff_systems"):
        first_system = original_staff["staff_systems"][0]
        line_indices = first_system.get("lines", [])
        
        if line_indices:
            # Calculate center of first staff system
            y_values = []
            for idx in line_indices:
                if idx < len(original_staff.get("detections", [])):
                    y_values.append(original_staff["detections"][idx]["bbox"]["center_y"])
            
            if y_values:
                zoom_center_y = int(sum(y_values) / len(y_values))
    
    # Define zoom window dimensions
    zoom_width = min(400, width // 2)
    zoom_height = min(200, height // 2)
    
    # Calculate zoom window boundaries
    zoom_x1 = max(0, zoom_center_x - zoom_width // 2)
    zoom_y1 = max(0, zoom_center_y - zoom_height // 2)
    zoom_x2 = min(width, zoom_center_x + zoom_width // 2)
    zoom_y2 = min(height, zoom_center_y + zoom_height // 2)
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 10))
    
    # Full image with both original and aligned staff lines
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.title("Full Image Comparison")
    
    # Colors for visualization
    original_color = (1, 0, 0, 0.7)    # Red with alpha
    aligned_color = (0, 0.7, 0, 0.8)   # Green with alpha
    
    # Draw original staff lines
    for system in original_staff.get("staff_systems", []):
        for line_idx in system.get("lines", []):
            if line_idx < len(original_staff.get("detections", [])):
                line = original_staff["detections"][line_idx]
                x1 = line["bbox"]["x1"]
                y1 = line["bbox"]["center_y"]
                x2 = line["bbox"]["x2"]
                y2 = line["bbox"]["center_y"]
                
                plt.plot([x1, x2], [y1, y2], color=original_color, linewidth=2,
                         label='Original' if line_idx == system["lines"][0] else "")
    
    # Draw aligned staff lines
    for system in aligned_staff.get("staff_systems", []):
        for line_idx in system.get("lines", []):
            if line_idx < len(aligned_staff.get("detections", [])):
                line = aligned_staff["detections"][line_idx]
                x1 = line["bbox"]["x1"]
                y1 = line["bbox"]["center_y"]
                x2 = line["bbox"]["x2"]
                y2 = line["bbox"]["center_y"]
                
                plt.plot([x1, x2], [y1, y2], color=aligned_color, linewidth=1.5,
                         label='Pixel-perfect' if line_idx == system["lines"][0] else "")
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    # Add rectangle to show zoom area
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (zoom_x1, zoom_y1), zoom_x2 - zoom_x1, zoom_y2 - zoom_y1,
        linewidth=2, edgecolor='blue', facecolor='none'
    )
    plt.gca().add_patch(rect)
    plt.axis('off')
    
    # Zoomed view
    plt.subplot(2, 1, 2)
    plt.imshow(img[zoom_y1:zoom_y2, zoom_x1:zoom_x2])
    plt.title("Zoomed View of Alignment Differences")
    
    # Draw original staff lines in zoomed view
    for system in original_staff.get("staff_systems", []):
        for line_idx in system.get("lines", []):
            if line_idx < len(original_staff.get("detections", [])):
                line = original_staff["detections"][line_idx]
                x1 = max(zoom_x1, line["bbox"]["x1"])
                y1 = line["bbox"]["center_y"]
                x2 = min(zoom_x2, line["bbox"]["x2"])
                y2 = line["bbox"]["center_y"]
                
                if zoom_x1 <= x2 and x1 <= zoom_x2 and zoom_y1 <= y1 <= zoom_y2:
                    plt.plot([x1 - zoom_x1, x2 - zoom_x1], [y1 - zoom_y1, y2 - zoom_y1], 
                             color=original_color, linewidth=3)
    
    # Draw aligned staff lines in zoomed view
    for system in aligned_staff.get("staff_systems", []):
        for line_idx in system.get("lines", []):
            if line_idx < len(aligned_staff.get("detections", [])):
                line = aligned_staff["detections"][line_idx]
                x1 = max(zoom_x1, line["bbox"]["x1"])
                y1 = line["bbox"]["center_y"]
                x2 = min(zoom_x2, line["bbox"]["x2"])
                y2 = line["bbox"]["center_y"]
                
                if zoom_x1 <= x2 and x1 <= zoom_x2 and zoom_y1 <= y1 <= zoom_y2:
                    plt.plot([x1 - zoom_x1, x2 - zoom_x1], [y1 - zoom_y1, y2 - zoom_y1], 
                             color=aligned_color, linewidth=2)
    
    # Add side ruler to measure pixel offsets
    y_ticks = range(0, zoom_y2 - zoom_y1, 5)
    plt.yticks(y_ticks, [str(y) for y in y_ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Zoomed comparison saved to: {output_path}")

def process_batch(image_dir, staff_json_dir, output_dir="results/pixel_perfect_batch"):
    """
    Process a batch of images and existing staff line detections
    
    Args:
        image_dir: Directory containing music score images
        staff_json_dir: Directory containing staff line JSON files
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in staff_json_dir
    import glob
    json_files = glob.glob(os.path.join(staff_json_dir, "*.json"))
    
    print(f"Found {len(json_files)} staff line JSON files to process")
    
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        img_base = base_name.replace("_staff_lines", "").replace("_detections", "")
        
        # Look for matching image
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            test_path = os.path.join(image_dir, f"{img_base}{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if not image_path:
            print(f"Could not find image for {json_file}, skipping...")
            continue
        
        # Process this pair
        print(f"Processing: {os.path.basename(image_path)} + {os.path.basename(json_file)}")
        try:
            apply_to_existing_results(image_path, json_file, output_dir)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"Batch processing complete! Results saved to {output_dir}")

if __name__ == "__main__":
    # Run the demo
    main()
    
    # Uncomment to apply to existing results
    apply_to_existing_results("/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/Accidentals-004.png", "/homes/es314/omr-objdet-benchmark/scripts/encoding/pixe-enhancment/results/enhanced/Accidentals-004_pixel_perfect.json")
    
    # Uncomment to process a batch of files
    # process_batch("images", "staff_detections")