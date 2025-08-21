# /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/visualisation.py
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from pathlib import Path
from collections import defaultdict

def create_color_map(class_names):
    """
    Create a color map for different symbol classes
    
    Args:
        class_names: List of unique class names
        
    Returns:
        Dictionary mapping class names to colors (RGB format for matplotlib)
    """
    # Set of colors for different classes (RGB format for matplotlib)
    colors = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 0.8, 0.0),    # Green
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 0.6, 0.0),    # Orange
        (0.5, 0.0, 0.5),    # Purple
        (0.0, 0.5, 0.5),    # Teal
        (1.0, 0.0, 0.5),    # Pink
        (0.5, 0.5, 0.0),    # Olive
        (0.0, 0.7, 1.0),    # Sky Blue
        (0.7, 0.0, 0.0),    # Dark Red
        (0.0, 0.5, 0.0),    # Dark Green
        (0.0, 0.0, 0.7),    # Dark Blue
        (0.7, 0.7, 0.0),    # Yellow
        (0.7, 0.0, 0.7),    # Magenta
        (0.0, 0.7, 0.7),    # Cyan
    ]
    
    # Create color map
    color_map = {}
    staff_line_color = (0.6, 0.6, 0.6)  # Gray for staff lines
    
    for i, class_name in enumerate(class_names):
        if "staff" in class_name.lower():
            color_map[class_name] = staff_line_color
        else:
            color_map[class_name] = colors[i % len(colors)]
    
    return color_map

def visualize_pitched_score(image_path, data_path, output_path=None, show_confidence=False, debug=True):
    """
    Visualize a music score with detected symbols and their pitches
    
    Args:
        image_path: Path to the original image
        data_path: Path to the JSON file with detection and pitch data
        output_path: Path to save the visualization (if None, display instead)
        show_confidence: Whether to show confidence scores
        debug: Whether to print debug information
    """
    if debug:
        print(f"Visualizing: {image_path}")
        print(f"Using data: {data_path}")
    
    # Load image
    try:
        img = plt.imread(str(image_path))
        if debug:
            print(f"Successfully loaded image: {img.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        # Try using OpenCV instead
        try:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if debug:
                print(f"Loaded image with OpenCV: {img.shape}")
        except Exception as e:
            print(f"Failed to load image with OpenCV: {e}")
            return
    
    # Load detection data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        if debug:
            print(f"Successfully loaded data")
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    # Get unique class names
    class_names = set()
    for detection in data.get("detections", []):
        class_names.add(detection.get("class_name", "unknown"))
    
    # Create color map
    color_map = create_color_map(list(class_names))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, img.shape[0] / img.shape[1] * 20))
    ax.imshow(img)
    
    # Count noteheads and pitch information
    noteheads = []
    noteheads_with_pitch = []
    
    for detection in data.get("detections", []):
        class_name = detection.get("class_name", "").lower()
        if "notehead" in class_name:
            noteheads.append(detection)
            if "pitch" in detection:
                noteheads_with_pitch.append(detection)
    
    if debug:
        print(f"Found {len(noteheads)} noteheads, {len(noteheads_with_pitch)} with pitch information")
    
    # Draw staff lines first
    for detection in data.get("detections", []):
        class_name = detection.get("class_name", "")
        if "staff" in class_name.lower():
            box = detection["bbox"]
            rect = patches.Rectangle(
                (box["x1"], box["y1"]),
                box["width"],
                box["height"],
                linewidth=1,
                edgecolor=color_map.get(class_name, (0.6, 0.6, 0.6)),
                facecolor=color_map.get(class_name, (0.6, 0.6, 0.6)),
                alpha=0.7
            )
            ax.add_patch(rect)
    
    # Draw all other symbols
    for detection in data.get("detections", []):
        class_name = detection.get("class_name", "")
        if "staff" in class_name.lower():
            continue  # Skip staff lines as we already drew them
            
        # Get bounding box
        box = detection["bbox"]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (box["x1"], box["y1"]),
            box["width"],
            box["height"],
            linewidth=2,
            edgecolor=color_map.get(class_name, (1.0, 0.0, 0.0)),
            facecolor="none"
        )
        ax.add_patch(rect)
        
        # Prepare label text
        label_parts = [class_name]
        
        # Add confidence if requested
        if show_confidence and "confidence" in detection:
            conf = detection["confidence"]
            label_parts.append(f"{conf:.2f}")
        
        # Add pitch information for noteheads
        if "pitch" in detection:
            pitch_info = detection["pitch"]
            note_name = pitch_info.get("note_name", "")
            if note_name:
                label_parts.append(note_name)
        
        # Create the label text
        label = "\n".join(label_parts)
        
        # Calculate label position
        y_offset = box["y1"] - 5
        if y_offset < 10:
            y_offset = box["y2"] + 5  # Place below if not enough space above
        
        # Draw label with background box
        ax.text(
            box["x1"], 
            y_offset,
            label,
            color="black",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            ha="left",
            va="bottom" if y_offset == box["y1"] - 5 else "top"
        )
    
    # Add title and adjust layout
    plt.title("Music Score with Detected Symbols and Pitches")
    plt.axis("off")
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        try:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            if debug:
                print(f"Saved visualization to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False
    else:
        plt.show()
        return True

def main():
    parser = argparse.ArgumentParser(description="Visualize music score with pitch information")
    parser.add_argument("--image", type=str, required=True, help="Path to original image")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON file with detection and pitch data")
    parser.add_argument("--output", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--show-confidence", action="store_true", help="Show confidence scores")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    visualize_pitched_score(args.image, args.data, args.output, args.show_confidence, args.debug)

if __name__ == "__main__":
    main()