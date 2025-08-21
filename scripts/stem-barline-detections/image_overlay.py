#!/usr/bin/env python3
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def overlay_detections_on_image(image_path, detections, output_path=None, show_original=False, 
                               show_inferred=True, confidence_threshold=0.5):
    """
    Overlay detection bounding boxes on the original image.
    
    Args:
        image_path: Path to the original image
        detections: List of detection dictionaries
        output_path: Path to save the output image
        show_original: Whether to show original detections
        show_inferred: Whether to show inferred detections
        confidence_threshold: Minimum confidence for showing detections
    """
    # Load the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a blank image if original can't be loaded
        image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(image)
    
    # Define colors for different element types
    color_map = {
        'stem': 'red',
        'barline': 'blue',
        'systemicBarline': 'purple',
        'noteheadBlack': 'green',
        'noteheadHalf': 'lime',
        'noteheadWhole': 'teal',
        'gClef': 'orange',
        'fClef': 'orange',
        'accidentalSharp': 'magenta',
        'accidentalFlat': 'magenta',
        'accidentalNatural': 'magenta',
        'rest': 'brown',
        'flag': 'cyan'
    }
    
    # Default color for unknown types
    default_color = 'gray'
    
    # Count of detections by type
    shown_counts = {'original': 0, 'inferred': 0}
    
    # Draw bounding boxes
    for det in detections:
        # Skip detections below confidence threshold
        confidence = det.get('confidence', 1.0)
        if confidence < confidence_threshold:
            continue
        
        # Skip based on inference status
        is_inferred = det.get('inferred', False)
        if is_inferred and not show_inferred:
            continue
        if not is_inferred and not show_original:
            continue
        
        # Get bounding box
        bbox = det.get('bbox', {})
        if not bbox:
            continue
        
        x = bbox.get('x1', 0)
        y = bbox.get('y1', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        
        # Get class name
        class_name = det.get('class_name', '')
        
        # Set color and style based on element type and inference status
        color = color_map.get(class_name, default_color)
        
        # Use dashed line for inferred elements
        linestyle = '--' if is_inferred else '-'
        linewidth = 1 if is_inferred else 2
        
        # Create rectangle patch
        rect = Rectangle((x, y), width, height, 
                         linewidth=linewidth, 
                         edgecolor=color, 
                         linestyle=linestyle,
                         facecolor='none')
        
        # Add the patch to the axis
        ax.add_patch(rect)
        
        # Update counts
        if is_inferred:
            shown_counts['inferred'] += 1
        else:
            shown_counts['original'] += 1
    
    # Add title with counts
    total_shown = shown_counts['original'] + shown_counts['inferred']
    ax.set_title(f"Music Notation Elements: {total_shown} elements shown\n"
                f"({shown_counts['original']} original, {shown_counts['inferred']} inferred)")
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend
    legend_elements = []
    for class_name, color in color_map.items():
        # Check if this class exists in the detections
        if any(det.get('class_name') == class_name for det in detections):
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=class_name))
    
    # Add inferred vs original to legend
    if show_original and show_inferred:
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, label='Original'))
        legend_elements.append(plt.Line2D([0], [0], color='black', lw=1, linestyle='--', label='Inferred'))
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=min(5, len(legend_elements)), frameon=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Overlay image saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def create_overlay_from_files(image_path, detection_path, output_path=None, 
                              show_original=True, show_inferred=True):
    """Create overlay visualization from files."""
    # Load detections
    try:
        detection_data = load_json(detection_path)
        detections = detection_data.get('detections', [])
        print(f"Loaded {len(detections)} detections from {detection_path}")
    except Exception as e:
        print(f"Error loading detections: {e}")
        return None
    
    # Create and save overlay
    return overlay_detections_on_image(
        image_path, 
        detections, 
        output_path=output_path,
        show_original=show_original,
        show_inferred=show_inferred
    )

def main():
    parser = argparse.ArgumentParser(description='Create image overlay of music notation detections')
    parser.add_argument('--image', required=True, help='Path to original score image')
    parser.add_argument('--detections', required=True, help='Path to detections JSON file')
    parser.add_argument('--output', help='Path to save overlay image')
    parser.add_argument('--original', action='store_true', default=True, help='Show original detections')
    parser.add_argument('--no-original', action='store_false', dest='original', help='Hide original detections')
    parser.add_argument('--inferred', action='store_true', default=True, help='Show inferred elements')
    parser.add_argument('--no-inferred', action='store_false', dest='inferred', help='Hide inferred elements')
    
    args = parser.parse_args()
    
    create_overlay_from_files(
        args.image,
        args.detections,
        args.output,
        show_original=args.original,
        show_inferred=args.inferred
    )

if __name__ == "__main__":
    main()