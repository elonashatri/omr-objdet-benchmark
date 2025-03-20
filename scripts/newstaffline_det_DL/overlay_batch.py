#!/usr/bin/env python
"""
Overlay Staff Line Detection JSON Results on Original Images

This script visualizes the staff line detection results by overlaying the
JSON output on the original images, with improved coordinate handling.
"""

import os
import argparse
import sys
from pathlib import Path


def visualize_json_results(image_path, json_path, output_path, line_thickness=2, debug=False):
    """
    Overlay JSON detection results on the original image.
    
    Args:
        image_path (str): Path to the original image
        json_path (str): Path to the JSON detection results
        output_path (str): Path to save the visualization
        line_thickness (int): Thickness of the staff lines to draw
        debug (bool): Whether to print debug information
    """
    import cv2
    import json
    import numpy as np
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Print image dimensions for debugging
    img_height, img_width = image.shape[:2]
    if debug:
        print(f"Image dimensions: {img_width}x{img_height}")
    
    # Load the JSON results
    with open(json_path, 'r') as f:
        detection_data = json.load(f)
    
    # Check if there are any staff systems detected
    if len(detection_data["staff_systems"]) == 0:
        print("No staff systems found in JSON results")
        cv2.imwrite(output_path, image)
        return
    
    # Convert image to RGB (from BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a custom colormap for different staff systems
    num_systems = len(detection_data["staff_systems"])
    distinct_colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (255, 165, 0),   # Orange
        (128, 0, 128),   # Purple
        (165, 42, 42),   # Brown
        (0, 128, 0)      # Dark green
    ]
    
    # Draw staff systems and lines
    overlay = image_rgb.copy()
    
    # Draw a debugging grid if requested
    if debug:
        # Draw vertical lines every 500 pixels
        for x in range(0, img_width, 500):
            cv2.line(overlay, (x, 0), (x, img_height), (100, 100, 100), 1)
            cv2.putText(overlay, str(x), (x+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        
        # Draw horizontal lines every 500 pixels
        for y in range(0, img_height, 500):
            cv2.line(overlay, (0, y), (img_width, y), (100, 100, 100), 1)
            cv2.putText(overlay, str(y), (5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Print some detected coordinates for debugging
    if debug and len(detection_data["detections"]) > 0:
        first_line = detection_data["detections"][0]["bbox"]
        print(f"First staff line coordinates: ({first_line['x1']}, {first_line['y1']}) to ({first_line['x2']}, {first_line['y2']})")
        if len(detection_data["detections"]) > 1:
            last_line = detection_data["detections"][-1]["bbox"]
            print(f"Last staff line coordinates: ({last_line['x1']}, {last_line['y1']}) to ({last_line['x2']}, {last_line['y2']})")
    
    # Draw each detection
    for detection in detection_data["detections"]:
        bbox = detection["bbox"]
        system_id = detection["staff_system"]
        line_number = detection["line_number"]
        
        # Get color for this staff system
        color = distinct_colors[system_id % len(distinct_colors)]
        
        # Extract coordinates
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
        
        # Clamp coordinates to image bounds (to prevent errors if coordinates are outside image)
        x1 = max(0, min(x1, img_width-1))
        x2 = max(0, min(x2, img_width-1))
        y1 = max(0, min(y1, img_height-1))
        y2 = max(0, min(y2, img_height-1))
        
        # Draw a line for this staff line
        cv2.line(overlay, (x1, (y1+y2)//2), (x2, (y1+y2)//2), color, line_thickness)
        
        # Add line number at the beginning of the line
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{line_number}"
        cv2.putText(overlay, text, (max(0, x1-20), (y1+y2)//2+5), font, 0.5, color, 1, cv2.LINE_AA)
    
    # Add legend for staff systems
    legend_height = 30
    legend_margin = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for idx, system in enumerate(detection_data["staff_systems"]):
        system_id = system["id"]
        color = distinct_colors[system_id % len(distinct_colors)]
        
        # Calculate position for this legend entry
        legend_y = legend_margin + idx * legend_height
        
        # Draw color swatch
        cv2.rectangle(overlay, 
                     (legend_margin, legend_y), 
                     (legend_margin + 20, legend_y + 20), 
                     color, -1)
        
        # Add system label
        text = f"System {system_id}"
        cv2.putText(overlay, text, 
                   (legend_margin + 30, legend_y + 15), 
                   font, 0.5, color, 1, cv2.LINE_AA)
    
    # Blend the visualization with the original image
    alpha = 0.7
    output_image = cv2.addWeighted(overlay, alpha, image_rgb, 1-alpha, 0)
    
    # Convert back to BGR for OpenCV
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    # Save the visualization
    cv2.imwrite(output_path, output_image_bgr)
    print(f"Visualization saved to: {output_path}")
    
    return output_image


def batch_visualize_json_results(image_dir, json_dir, output_dir, line_thickness=2, debug=False):
    """
    Overlay JSON detection results on all images in a directory.
    
    Args:
        image_dir (str): Directory containing original images
        json_dir (str): Directory containing JSON detection results
        output_dir (str): Directory to save visualizations
        line_thickness (int): Thickness of the staff lines to draw
        debug (bool): Whether to print debug information
    """
    import os
    import glob
    from tqdm import tqdm
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(json_dir, "*_results.json"))
    
    if debug:
        print(f"Found {len(json_files)} JSON files in {json_dir}")
    
    for json_file in tqdm(json_files, desc="Visualizing results"):
        try:
            # Extract base name
            base_name = os.path.basename(json_file).replace("_results.json", "")
            
            # Find corresponding image
            image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
            image_path = None
            for ext in image_extensions:
                potential_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"Could not find image for {base_name}")
                continue
            
            if debug:
                print(f"Processing: {base_name}")
                print(f"Image path: {image_path}")
                print(f"JSON path: {json_file}")
            
            # Set output path
            output_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            
            # Visualize results
            visualize_json_results(image_path, json_file, output_path, line_thickness, debug)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Staff Line Detection Results")
    
    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing original images or single image file")
    parser.add_argument("--json", type=str, required=True,
                        help="Directory containing JSON results or single JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save visualizations or output image path")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Thickness of the staff lines to draw")
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in the input directory")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information and draw grid")
    
    args = parser.parse_args()
    
    if args.batch:
        # Process all images in the directory
        batch_visualize_json_results(args.images, args.json, args.output, args.thickness, args.debug)
    else:
        # Process a single image
        visualize_json_results(args.images, args.json, args.output, args.thickness, args.debug)