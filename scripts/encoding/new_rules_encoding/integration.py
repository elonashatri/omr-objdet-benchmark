import sys
import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import main OMR processor class from our enhanced pipeline
# In a real implementation, you would import this from your module
# For this example, we'll assume it's in a file called omr_processor.py
try:
    from omr_processor import OMRProcessor
except ImportError:
    # Include a minimal version here for standalone operation
    class OMRProcessor:
        def __init__(self, detection_data=None, staff_lines_data=None, class_mapping_data=None):
            self.detections = detection_data
            self.staff_lines = staff_lines_data
            self.class_mapping = class_mapping_data
        
        def process(self):
            print("Processing detections...")
            # In the real implementation, this would do the full conversion
            return "<score-partwise><!-- Placeholder MusicXML --></score-partwise>"
        
        def visualize(self, output_path=None):
            print(f"Generating visualization at {output_path}")
            # In the real implementation, this would create a detailed visualization

def load_csv_detections(filepath):
    """Load detections from a CSV file and convert to standardized format."""
    detections = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            
            # Create standardized detection object
            detection = {
                'class_id': int(row['class_id']),
                'class_name': row['class_name'],
                'confidence': float(row['confidence']),
                'bbox': {
                    'x1': float(row['x1']),
                    'y1': float(row['y1']),
                    'x2': float(row['x2']),
                    'y2': float(row['y2']),
                    'width': float(row['width']),
                    'height': float(row['height']),
                    'center_x': float(row['center_x']),
                    'center_y': float(row['center_y'])
                }
            }
            detections.append(detection)
    
    return detections

def load_json_detections(filepath):
    """Load detections from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if the JSON data follows the expected structure
    if 'detections' in data:
        return data['detections']
    
    # If not, assume it's already a list of detections
    return data

def load_staff_lines(filepath):
    """Load staff line data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading staff lines from {filepath}: {e}")
        return None

def load_class_mapping(filepath):
    """Load class mapping data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading class mapping from {filepath}: {e}")
        return None

def create_simple_visualization(detections, staff_lines, output_path):
    """
    Create a simple visualization of the detections and staff lines.
    This is a fallback if the main visualization method isn't available.
    """
    plt.figure(figsize=(15, 10))
    
    # Define colors for different object types
    colors = {
        'noteheadBlack': 'green',
        'accidentalSharp': 'blue',
        'accidentalFlat': 'purple',
        'accidentalNatural': 'cyan',
        'gClef': 'red',
        'fClef': 'brown',
        'barline': 'black',
        'staff_line': 'gray'
    }
    
    # Plot staff lines if available
    if staff_lines and 'detections' in staff_lines:
        for line in staff_lines['detections']:
            if line['class_name'] == 'staff_line':
                plt.plot(
                    [line['bbox']['x1'], line['bbox']['x2']],
                    [line['bbox']['center_y'], line['bbox']['center_y']],
                    color='lightgray', linestyle='-', linewidth=1
                )
    
    # Plot detected objects
    for det in detections:
        class_name = det['class_name']
        bbox = det['bbox'] 
        
        color = colors.get(class_name, 'gray')
        
        # Draw rectangle for the object
        rect = plt.Rectangle(
            (bbox['x1'], bbox['y1']),
            bbox['width'], bbox['height'],
            linewidth=1, edgecolor=color, facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        # Add label for selected object types
        if class_name in ['gClef', 'fClef', 'accidentalSharp', 'accidentalFlat']:
            plt.text(
                bbox['center_x'], bbox['y1'] - 5,
                class_name,
                fontsize=8, ha='center', color=color
            )
    
    # Set axis limits with some margin
    all_x = []
    all_y = []
    
    for det in detections:
        bbox = det['bbox']
        all_x.extend([bbox['x1'], bbox['x2']])
        all_y.extend([bbox['y1'], bbox['y2']])
    
    margin = 50
    plt.xlim(min(all_x) - margin, max(all_x) + margin)
    plt.ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Add title and labels
    plt.title('Music Score Detection Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(alpha=0.3)
    
    # Add legend
    legend_patches = [plt.Line2D([0], [0], color=color, label=name, marker='s', linestyle='None') 
                     for name, color in colors.items()]
    plt.legend(handles=legend_patches, loc='upper right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Simple visualization saved to {output_path}")

def process_file(detection_path, staff_lines_path, class_mapping_path=None, 
                output_dir=None, prefix=None):
    """
    Process a single file with the OMR pipeline.
    
    Args:
        detection_path: Path to the detection file (CSV or JSON)
        staff_lines_path: Path to the staff lines file (JSON)
        class_mapping_path: Path to the class mapping file (JSON)
        output_dir: Directory to save output files
        prefix: Prefix for output filenames
    
    Returns:
        Dictionary with paths to output files
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set defaults
    if not output_dir:
        output_dir = '.'
    
    if not prefix:
        # Use the detection filename as prefix
        prefix = os.path.splitext(os.path.basename(detection_path))[0]
    
    # Paths for output files
    xml_path = os.path.join(output_dir, f"{prefix}.musicxml")
    viz_path = os.path.join(output_dir, f"{prefix}_visualization.png")
    debug_path = os.path.join(output_dir, f"{prefix}_debug.json")
    
    # Load data
    print(f"Loading detection data from {detection_path}")
    if detection_path.endswith('.csv'):
        detections = load_csv_detections(detection_path)
    else:
        detections = load_json_detections(detection_path)
    
    print(f"Loading staff lines data from {staff_lines_path}")
    staff_lines = load_staff_lines(staff_lines_path)
    
    class_mapping = None
    if class_mapping_path:
        print(f"Loading class mapping from {class_mapping_path}")
        class_mapping = load_class_mapping(class_mapping_path)
    
    # Process with OMR pipeline
    print("\nProcessing music score...")
    processor = OMRProcessor(detection_data=detections, 
                            staff_lines_data=staff_lines,
                            class_mapping_data=class_mapping)
    
    # Generate MusicXML
    musicxml = processor.process()
    
    # Save MusicXML
    with open(xml_path, 'w') as f:
        f.write(musicxml)
    print(f"MusicXML saved to {xml_path}")
    
    # Generate visualization
    try:
        processor.visualize(viz_path)
    except Exception as e:
        print(f"Error in main visualization: {e}")
        print("Falling back to simple visualization")
        create_simple_visualization(detections, staff_lines, viz_path)
    
    # Save debug info
    try:
        debug_info = {
            'num_detections': len(detections),
            'num_staff_systems': len(staff_lines.get('staff_systems', [])),
            'detection_types': defaultdict(int)
        }
        
        for det in detections:
            debug_info['detection_types'][det['class_name']] += 1
        
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
        print(f"Debug info saved to {debug_path}")
    except Exception as e:
        print(f"Error saving debug info: {e}")
    
    return {
        'musicxml': xml_path,
        'visualization': viz_path,
        'debug': debug_path
    }

def main():
    parser = argparse.ArgumentParser(description='Process music score detections into MusicXML')
    parser.add_argument('detection_path', help='Path to detection file (CSV or JSON)')
    parser.add_argument('staff_lines_path', help='Path to staff lines file (JSON)')
    parser.add_argument('--class_mapping', help='Path to class mapping file (JSON)')
    parser.add_argument('--output_dir', default='.', help='Directory to save output files')
    parser.add_argument('--prefix', help='Prefix for output filenames')
    
    args = parser.parse_args()
    
    # Process the file
    output_paths = process_file(
        args.detection_path,
        args.staff_lines_path,
        args.class_mapping,
        args.output_dir,
        args.prefix
    )
    
    print("\nProcessing complete!")
    print(f"MusicXML: {output_paths['musicxml']}")
    print(f"Visualization: {output_paths['visualization']}")
    print(f"Debug info: {output_paths['debug']}")

if __name__ == "__main__":
    main()