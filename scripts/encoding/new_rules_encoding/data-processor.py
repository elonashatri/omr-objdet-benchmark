import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

def load_csv_detections(filepath):
    """Load detections from a CSV file."""
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
            
            # Create bbox dictionary for consistency
            bbox = {
                'x1': row['x1'],
                'y1': row['y1'],
                'x2': row['x2'],
                'y2': row['y2'],
                'width': row['width'],
                'height': row['height'],
                'center_x': row['center_x'],
                'center_y': row['center_y']
            }
            
            detection = {
                'class_id': int(row['class_id']),
                'class_name': row['class_name'],
                'confidence': row['confidence'],
                'bbox': bbox
            }
            detections.append(detection)
    
    return detections

def load_json_detections(filepath):
    """Load detections from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'detections' in data:
        return data['detections']
    return data

def load_staff_lines(filepath):
    """Load staff line data from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def visualize_detections(detections, staff_lines=None, output_path=None):
    """
    Visualize the detected objects and staff lines.
    
    Args:
        detections: List of detection dictionaries
        staff_lines: Staff line data (optional)
        output_path: Path to save the visualization (optional)
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define colors for different object types
    colors = {
        'noteheadBlack': 'green',
        'accidentalSharp': 'blue',
        'accidentalFlat': 'purple',
        'accidentalNatural': 'cyan',
        'gClef': 'red',
        'fClef': 'orange',
        'barline': 'brown',
        'staff_line': 'black'
    }
    
    # Draw detected objects
    for det in detections:
        class_name = det['class_name']
        bbox = det['bbox'] if 'bbox' in det else det
        
        color = colors.get(class_name, 'gray')
        
        rect = patches.Rectangle(
            (bbox['x1'], bbox['y1']),
            bbox['width'], bbox['height'],
            linewidth=1, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add a label for the object
        if class_name in ['gClef', 'fClef', 'accidentalSharp', 'accidentalFlat', 'accidentalNatural']:
            ax.text(bbox['center_x'], bbox['center_y'] - 10, class_name,
                  fontsize=8, ha='center', color=color)
    
    # Draw staff lines if provided
    if staff_lines and 'detections' in staff_lines:
        for line in staff_lines['detections']:
            if line['class_name'] == 'staff_line':
                bbox = line['bbox']
                plt.plot([bbox['x1'], bbox['x2']], [bbox['center_y'], bbox['center_y']],
                       color='black', linestyle='-', alpha=0.5)
                
                # Add staff system and line number labels
                if 'staff_system' in line and 'line_number' in line:
                    ax.text(bbox['x1'] - 30, bbox['center_y'],
                          f"Sys {line['staff_system']}, Line {line['line_number']}",
                          fontsize=6, ha='right', va='center')
    
    # Add a legend
    legend_elements = [patches.Patch(facecolor='none', edgecolor=color, label=name)
                      for name, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set axis limits
    all_x1 = [det['bbox']['x1'] if 'bbox' in det else det['x1'] for det in detections]
    all_y1 = [det['bbox']['y1'] if 'bbox' in det else det['y1'] for det in detections]
    all_x2 = [det['bbox']['x2'] if 'bbox' in det else det['x2'] for det in detections]
    all_y2 = [det['bbox']['y2'] if 'bbox' in det else det['y2'] for det in detections]
    
    margin = 50
    ax.set_xlim(min(all_x1) - margin, max(all_x2) + margin)
    ax.set_ylim(min(all_y1) - margin, max(all_y2) + margin)
    
    # Add title and labels
    ax.set_title('Music Score Object Detection Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def analyze_relationships(detections, staff_lines):
    """
    Analyze relationships between detected objects and staff lines.
    
    Args:
        detections: List of detection dictionaries
        staff_lines: Staff line data
    
    Returns:
        Dictionary with analysis results
    """
    results = {
        'object_counts': defaultdict(int),
        'staff_systems': len(staff_lines.get('staff_systems', [])),
        'notes_per_staff': defaultdict(int),
        'accidentals_per_staff': defaultdict(int)
    }
    
    # Count objects by type
    for det in detections:
        results['object_counts'][det['class_name']] += 1
    
    # Calculate average staff line distance
    if 'detections' in staff_lines:
        line_distances = []
        
        # Group staff lines by system
        staff_systems = {}
        for line in staff_lines['detections']:
            if 'staff_system' in line and 'line_number' in line:
                sys_id = line['staff_system']
                if sys_id not in staff_systems:
                    staff_systems[sys_id] = {}
                staff_systems[sys_id][line['line_number']] = line['bbox']['center_y']
        
        # Calculate distances between adjacent lines
        for sys_id, lines in staff_systems.items():
            line_nums = sorted(lines.keys())
            for i in range(1, len(line_nums)):
                curr_line = line_nums[i]
                prev_line = line_nums[i-1]
                distance = abs(lines[curr_line] - lines[prev_line])
                line_distances.append(distance)
        
        if line_distances:
            results['avg_staff_line_distance'] = sum(line_distances) / len(line_distances)
            results['staff_line_distances'] = line_distances
    
    # Assign notes and accidentals to staff systems
    if 'detections' in staff_lines:
        # Create staff system bounding boxes
        staff_bounds = {}
        for sys_id, lines in staff_systems.items():
            if not lines:
                continue
            
            # Find top and bottom y-coordinates
            y_values = list(lines.values())
            top_y = min(y_values) - results.get('avg_staff_line_distance', 20)
            bottom_y = max(y_values) + results.get('avg_staff_line_distance', 20)
            
            # Use x-coordinates from any staff line in this system
            for line in staff_lines['detections']:
                if line.get('staff_system') == sys_id:
                    left_x = line['bbox']['x1']
                    right_x = line['bbox']['x2']
                    staff_bounds[sys_id] = (left_x, right_x, top_y, bottom_y)
                    break
        
        # Assign objects to staff systems
        for det in detections:
            bbox = det['bbox'] if 'bbox' in det else det
            center_x, center_y = bbox['center_x'], bbox['center_y']
            
            # Find the closest staff system
            closest_sys = None
            min_distance = float('inf')
            
            for sys_id, (left_x, right_x, top_y, bottom_y) in staff_bounds.items():
                # Check if within horizontal bounds
                if left_x <= center_x <= right_x:
                    # Check vertical distance
                    if top_y <= center_y <= bottom_y:
                        # Object is within staff bounds
                        distance = 0
                    else:
                        # Calculate vertical distance to staff
                        distance = min(abs(center_y - top_y), abs(center_y - bottom_y))
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_sys = sys_id
            
            if closest_sys is not None:
                # Count by object type
                if det['class_name'] == 'noteheadBlack':
                    results['notes_per_staff'][closest_sys] += 1
                elif 'accidental' in det['class_name'].lower():
                    results['accidentals_per_staff'][closest_sys] += 1
    
    return results

def generate_musicxml_skeleton(detections, staff_lines):
    """
    Generate a basic MusicXML skeleton based on the detected objects.
    
    Args:
        detections: List of detection dictionaries
        staff_lines: Staff line data
    
    Returns:
        MusicXML string
    """
    # Create a basic MusicXML structure
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    xml += '<score-partwise version="3.1">\n'
    
    # Add part list
    xml += '  <part-list>\n'
    
    # Create one part per staff system
    num_systems = len(staff_lines.get('staff_systems', []))
    for i in range(num_systems):
        xml += f'    <score-part id="P{i+1}">\n'
        xml += f'      <part-name>Part {i+1}</part-name>\n'
        xml += '    </score-part>\n'
    
    xml += '  </part-list>\n'
    
    # Add parts
    for i in range(num_systems):
        xml += f'  <part id="P{i+1}">\n'
        xml += '    <measure number="1">\n'
        xml += '      <attributes>\n'
        xml += '        <divisions>4</divisions>\n'
        xml += '        <key>\n'
        xml += '          <fifths>0</fifths>\n'
        xml += '        </key>\n'
        xml += '        <time>\n'
        xml += '          <beats>4</beats>\n'
        xml += '          <beat-type>4</beat-type>\n'
        xml += '        </time>\n'
        xml += '        <clef>\n'
        xml += '          <sign>G</sign>\n'
        xml += '          <line>2</line>\n'
        xml += '        </clef>\n'
        xml += '      </attributes>\n'
        
        # Placeholder for notes (would be filled in with actual data in a complete implementation)
        xml += '      <!-- Notes would be added here based on detected noteheads -->\n'
        
        xml += '    </measure>\n'
        xml += '  </part>\n'
    
    xml += '</score-partwise>\n'
    
    return xml

def main():
    # Example usage
    detection_file = "detections.csv"  # or .json
    staff_lines_file = "staff_lines.json"
    
    # Load detections
    if detection_file.endswith('.csv'):
        detections = load_csv_detections(detection_file)
    else:
        detections = load_json_detections(detection_file)
    
    # Load staff lines
    staff_lines = load_staff_lines(staff_lines_file)
    
    # Visualize
    visualize_detections(detections, staff_lines, "visualization.png")
    
    # Analyze relationships
    analysis = analyze_relationships(detections, staff_lines)
    print("Analysis Results:")
    print(json.dumps(analysis, indent=2))
    
    # Generate MusicXML skeleton
    musicxml = generate_musicxml_skeleton(detections, staff_lines)
    
    with open("output.musicxml", "w") as f:
        f.write(musicxml)
    
    print("MusicXML skeleton written to output.musicxml")

if __name__ == "__main__":
    main()