import math
import json
import os
import argparse
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks

# Constants for staff line spacing and musical elements
STAFF_LINE_SPACING = 27  # Approximate spacing between staff lines (can be calculated dynamically)
MIN_STEM_LENGTH = 3.0 * STAFF_LINE_SPACING  # Minimum stem length in pixels

# Constants derived from statistical analysis
STEM_WIDTH = 3.0  # Median stem width
STEM_HEIGHT_MEAN = 73.4  
STEM_HEIGHT_MEDIAN = 69.0
STEM_HEIGHT_STD = 25.7

BARLINE_THICKNESS = 4.0  # Median barline width
BARLINE_HEIGHT_MEDIAN = 378.0
BARLINE_HEIGHT_MEAN = 474.5
BARLINE_HEIGHT_STD = 395.3

SYSTEMIC_BARLINE_THICKNESS = 4.0  # Median systemic barline width
SYSTEMIC_BARLINE_HEIGHT_MEDIAN = 2782.0

# Class IDs (adjust these according to your model's class mapping)
CLASS_NOTEHEAD_BLACK = 0
CLASS_NOTEHEAD_HALF = 1
CLASS_NOTEHEAD_WHOLE = 2
CLASS_STEM = 5
CLASS_BARLINE = 8
CLASS_GCLEF = 14

def analyze_chord_structure(noteheads, staff_info):
    """
    Analyze the structure of a chord to determine stem configurations
    
    Args:
        noteheads: List of notehead detections
        staff_info: Staff information dictionary
    
    Returns:
        Dictionary with chord analysis results
    """
    if not noteheads:
        return {'needs_stem': False}
    
    # Get staff system from first notehead
    staff_system = noteheads[0].get('staff_system')
    if staff_system is None:
        return {'needs_stem': False}
    
    # Get staff lines for this system
    staff_lines = [line for line in staff_info.get('detections', []) 
                  if line.get('staff_system') == staff_system]
    
    if not staff_lines:
        return {'needs_stem': False}
    
    # Get middle line position
    y_positions = [line['bbox']['center_y'] for line in staff_lines]
    middle_line_y = y_positions[len(y_positions) // 2] if y_positions else None
    
    if middle_line_y is None:
        return {'needs_stem': False}
    
    # Sort noteheads by vertical position (y-coordinate)
    noteheads_sorted = sorted(noteheads, key=lambda x: x['bbox']['center_y'])
    
    # Calculate pitch range
    pitch_range = 0
    if len(noteheads_sorted) > 1:
        pitch_range = noteheads_sorted[-1]['bbox']['center_y'] - noteheads_sorted[0]['bbox']['center_y']
    
    # Count noteheads above and below middle line
    above_middle = sum(1 for nh in noteheads if nh['bbox']['center_y'] < middle_line_y)
    below_middle = len(noteheads) - above_middle
    
    # Determine if there are potentially multiple voices
    # Criteria: more than 4 noteheads with significant pitch range
    multiple_voices = (len(noteheads) > 4 and 
                      pitch_range > 3 * (y_positions[1] - y_positions[0]))
    
    # Determine default stem direction
    # General rule: stems up if below middle line, down if above
    default_stem_direction = 'up' if below_middle > above_middle else 'down'
    
    # Define voice groups for complex chords
    voice_groups = {}
    
    if multiple_voices:
        # Attempt to separate into voices based on spacing
        clusters = []
        current_cluster = [noteheads_sorted[0]]
        
        # Simple clustering by vertical distance
        staff_spacing = (y_positions[1] - y_positions[0]) if len(y_positions) > 1 else 20
        threshold = staff_spacing * 1.5
        
        for i in range(1, len(noteheads_sorted)):
            current = noteheads_sorted[i]
            previous = noteheads_sorted[i-1]
            
            if current['bbox']['center_y'] - previous['bbox']['center_y'] > threshold:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [current]
            else:
                current_cluster.append(current)
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Assign stem directions to clusters
        for i, cluster in enumerate(clusters):
            avg_y = sum(nh['bbox']['center_y'] for nh in cluster) / len(cluster)
            direction = 'up' if avg_y > middle_line_y else 'down'
            voice_groups[f'voice_{i}'] = {
                'noteheads': cluster,
                'stem_direction': direction
            }
    else:
        # Single voice - all noteheads use the default stem direction
        voice_groups['main_voice'] = {
            'noteheads': noteheads,
            'stem_direction': default_stem_direction
        }
    
    return {
        'needs_stem': True,
        'default_stem_direction': default_stem_direction,
        'multiple_voices': multiple_voices,
        'voice_groups': voice_groups
    }

def infer_stems_for_chord(chord_analysis, staff_info):
    """
    Infer stems for a chord based on its structure analysis
    
    Args:
        chord_analysis: Dictionary with chord analysis results
        staff_info: Staff information dictionary
    
    Returns:
        List of inferred stem detections
    """
    if not chord_analysis.get('needs_stem', False):
        return []
    
    inferred_stems = []
    
    # Process each voice group
    for voice_id, voice_data in chord_analysis.get('voice_groups', {}).items():
        noteheads = voice_data.get('noteheads', [])
        if not noteheads:
            continue
        
        stem_direction = voice_data.get('stem_direction', 'up')
        
        # Get staff system from first notehead
        staff_id = noteheads[0].get('staff_system')
        
        # Calculate stem position
        if stem_direction == 'up':
            # Right side of rightmost notehead for up stems
            x_pos = max(nh['bbox']['x2'] - 3 for nh in noteheads)
            y_bottom = min(nh['bbox']['center_y'] for nh in noteheads)
            
            # Calculate appropriate stem height
            stem_height = STEM_HEIGHT_MEDIAN
            stem_height = min(STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD, 
                            max(STEM_HEIGHT_MEDIAN, stem_height))
            
            y_top = y_bottom - stem_height
        else:
            # Left side of leftmost notehead for down stems
            x_pos = min(nh['bbox']['x1'] + 3 for nh in noteheads)
            y_top = max(nh['bbox']['center_y'] for nh in noteheads)
            
            # Calculate appropriate stem height
            stem_height = STEM_HEIGHT_MEDIAN
            stem_height = min(STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD, 
                            max(STEM_HEIGHT_MEDIAN, stem_height))
            
            y_bottom = y_top + stem_height
        
        # Create the inferred stem
        stem = {
            'class_id': CLASS_STEM,
            'class_name': 'stem',
            'confidence': 0.85,
            'bbox': {
                'x1': x_pos - STEM_WIDTH/2,
                'y1': y_top,
                'x2': x_pos + STEM_WIDTH/2,
                'y2': y_bottom,
                'width': STEM_WIDTH,
                'height': y_bottom - y_top,
                'center_x': x_pos,
                'center_y': (y_top + y_bottom) / 2
            },
            'staff_system': staff_id,
            'inferred': True,
            'voice_id': voice_id
        }
        
        inferred_stems.append(stem)
    
    return inferred_stems

def detect_barlines_computer_vision(image_path, staff_info):
    """
    Detect barlines using computer vision techniques
    
    Args:
        image_path: Path to score image
        staff_info: Dictionary with staff line information
    
    Returns:
        List of detected barline dictionaries
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot load image from {image_path}")
        return []
    
    # Get image dimensions
    height, width = img.shape
    
    # Preprocess image to enhance vertical lines
    # Threshold the image to get binary
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Create kernels for morphological operations
    kernel_vertical = np.ones((31, 1), np.uint8)  # Adjust size based on image resolution
    
    # Enhance vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
    
    # Get all staff systems
    staff_systems = staff_info.get('staff_systems', [])
    
    detected_barlines = []
    
    # Process each staff system
    for system in staff_systems:
        system_id = system.get('id')
        
        # Get staff lines for this system
        staff_lines = [line for line in staff_info.get('detections', []) 
                      if line.get('staff_system') == system_id]
        
        if not staff_lines:
            continue
        
        # Calculate staff boundaries
        top_line = min(staff_lines, key=lambda x: x['bbox']['y1'])
        bottom_line = max(staff_lines, key=lambda x: x['bbox']['y2'])
        
        staff_top = top_line['bbox']['y1'] - 10  # Add margin
        staff_bottom = bottom_line['bbox']['y2'] + 10  # Add margin
        staff_left = min(line['bbox']['x1'] for line in staff_lines)
        staff_right = max(line['bbox']['x2'] for line in staff_lines)
        
        # Calculate staff height
        staff_height = staff_bottom - staff_top
        
        # Crop to staff region with margins
        margin = staff_height * 0.5  # Adjust as needed
        roi_top = max(0, int(staff_top - margin))
        roi_bottom = min(height, int(staff_bottom + margin))
        roi_left = max(0, int(staff_left))
        roi_right = min(width, int(staff_right))
        
        # Extract ROI
        staff_roi = vertical_lines[roi_top:roi_bottom, roi_left:roi_right]
        
        # Create vertical projection profile
        vertical_projection = np.sum(staff_roi, axis=0)
        
        # Normalize projection
        if np.max(vertical_projection) > 0:
            vertical_projection = vertical_projection / np.max(vertical_projection)
        
        # Detect peaks in the projection (potential barlines)
        peaks, _ = find_peaks(vertical_projection, 
                             height=0.5,         # Minimum height of the peak
                             distance=30,        # Minimum distance between peaks
                             prominence=0.3)     # Minimum prominence of peak
        
        # Filter peaks by checking the vertical extent of the lines
        for peak_idx in peaks:
            # Convert peak index to original image coordinates
            x_pos = roi_left + peak_idx
            
            # Check vertical extent
            column = staff_roi[:, peak_idx]
            non_zero = np.nonzero(column)[0]
            
            if len(non_zero) > staff_height * 0.7:  # At least 70% of staff height
                # Get the actual vertical extent
                y_start = roi_top + non_zero[0]
                y_end = roi_top + non_zero[-1]
                
                # Create barline detection
                barline = {
                    'class_id': CLASS_BARLINE,
                    'class_name': 'barline',
                    'confidence': 0.9,
                    'bbox': {
                        'x1': x_pos - BARLINE_THICKNESS/2,
                        'y1': y_start,
                        'x2': x_pos + BARLINE_THICKNESS/2,
                        'y2': y_end,
                        'width': BARLINE_THICKNESS,
                        'height': y_end - y_start,
                        'center_x': x_pos,
                        'center_y': (y_start + y_end) / 2
                    },
                    'staff_system': system_id,
                    'cv_detected': True
                }
                
                detected_barlines.append(barline)
    
    return detected_barlines


def infer_stems_for_beamed_group(beamed_noteheads, beam, staff_info):
    inferred_stems = []

    beam_x1 = beam['bbox']['x1']
    beam_x2 = beam['bbox']['x2']
    beam_y1 = beam['bbox']['y1']
    beam_y2 = beam['bbox']['y2']
    beam_direction = 'up' if beam_y1 > beam_y2 else 'down'

    slope = (beam_y2 - beam_y1) / (beam_x2 - beam_x1 + 1e-6)

    for nh in beamed_noteheads:
        nh_box = nh['bbox']
        x_center = nh_box['center_x']
        y_note = nh_box['center_y']
        staff_id = nh.get('staff_system')

        # Project y position on beam line
        y_beam = beam_y1 + slope * (x_center - beam_x1)

        # Ensure minimal height from notehead to beam
        desired_min_height = STAFF_LINE_SPACING * 2.0
        current_height = abs(y_beam - y_note)

        if current_height < desired_min_height:
            if beam_direction == 'up':
                y_beam = y_note - desired_min_height
            else:
                y_beam = y_note + desired_min_height

        # Determine x-position of the stem (right side if up, left side if down)
        if beam_direction == 'up':
            x_stem = nh_box['x2'] - STEM_WIDTH / 2
            y_top = y_beam
            y_bottom = y_note
        else:
            x_stem = nh_box['x1'] + STEM_WIDTH / 2
            y_top = y_note
            y_bottom = y_beam

        stem_height = abs(y_bottom - y_top)
        stem = {
            'class_id': CLASS_STEM,
            'class_name': 'stem',
            'confidence': 0.95,
            'bbox': {
                'x1': x_stem - STEM_WIDTH / 2,
                'x2': x_stem + STEM_WIDTH / 2,
                'y1': y_top,
                'y2': y_bottom,
                'width': STEM_WIDTH,
                'height': stem_height,
                'center_x': x_stem,
                'center_y': (y_top + y_bottom) / 2
            },
            'staff_system': staff_id,
            'inferred': True,
            'beam_id': beam.get('id', None),
            'stem_inferred_via_beam': True
        }

        inferred_stems.append(stem)

    return inferred_stems

class OMREnhancer:
    def __init__(self, staff_info):
        self.staff_info = staff_info
        self.staff_systems = staff_info['staff_systems']
        self.staff_lines = staff_info['detections']
        self.calculate_staff_parameters()
    
    def calculate_staff_parameters(self):
        self.staff_parameters = {}
        for system in self.staff_systems:
            system_id = system['id']
            lines = [line for line in self.staff_lines if line['staff_system'] == system_id]
            lines_sorted = sorted(lines, key=lambda x: x['bbox']['center_y'])
            if len(lines_sorted) >= 5:
                y_positions = [line['bbox']['center_y'] for line in lines_sorted]
                spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
                avg_spacing = sum(spacings) / len(spacings)
                top = y_positions[0] - avg_spacing
                bottom = y_positions[-1] + avg_spacing
                left = min(line['bbox']['x1'] for line in lines_sorted)
                right = max(line['bbox']['x2'] for line in lines_sorted)
                self.staff_parameters[system_id] = {
                    'y_positions': y_positions,
                    'avg_spacing': avg_spacing,
                    'top': top,
                    'bottom': bottom,
                    'left': left,
                    'right': right,
                    'middle_line_y': y_positions[2],
                }
    def assign_elements_to_staff(self, detections):
        enhanced_detections = []
        for detection in detections:
            if detection.get('staff_system') is not None:
                enhanced_detections.append(detection)
                continue
            center_y = detection['bbox']['center_y']
            assigned_system = None
            min_distance = float('inf')
            for system_id, params in self.staff_parameters.items():
                if params['top'] - params['avg_spacing'] <= center_y <= params['bottom'] + params['avg_spacing']:
                    distance = abs(center_y - params['middle_line_y'])
                    if distance < min_distance:
                        min_distance = distance
                        assigned_system = system_id
            if assigned_system is not None:
                detection['staff_system'] = assigned_system
            enhanced_detections.append(detection)
        return enhanced_detections
    

    
    def enhance_detections(self, detections_dict, image_path=None):
        detections = self.assign_elements_to_staff(detections_dict)
        beamed_groups = self.group_noteheads_by_beam(detections)
        beamed_notehead_ids = {id(nh) for group, _ in beamed_groups for nh in group}
        inferred_beam_stems = []
        for group, beam in beamed_groups:
            inferred_beam_stems.extend(infer_stems_for_beamed_group(group, beam, self.staff_info))
        enhanced_detections = detections + inferred_beam_stems
        return enhanced_detections

    def group_noteheads_by_beam(self, detections):
        beams = [d for d in detections if d.get('class_name') == 'beam']
        noteheads = [d for d in detections if d.get('class_id') in [CLASS_NOTEHEAD_BLACK, CLASS_NOTEHEAD_HALF, CLASS_NOTEHEAD_WHOLE]]
        beam_groups = []
        for beam in beams:
            bx1, bx2 = beam['bbox']['x1'], beam['bbox']['x2']
            by1, by2 = beam['bbox']['y1'], beam['bbox']['y2']
            in_beam = []
            for nh in noteheads:
                nx, ny = nh['bbox']['center_x'], nh['bbox']['center_y']
                if bx1 <= nx <= bx2 and min(by1, by2) - 50 <= ny <= max(by1, by2) + 50:
                    in_beam.append(nh)
            if in_beam:
                beam_groups.append((in_beam, beam))
        return beam_groups

    def group_noteheads_into_chords(self, detections, x_threshold=40, y_overlap_threshold=0.5):
        """
        Group noteheads that likely belong to the same chord, accounting for both
        horizontal proximity and vertical overlap (for cases like offset noteheads).
        
        Args:
            detections: List of detected musical elements
            x_threshold: Maximum horizontal distance between noteheads in same chord
            y_overlap_threshold: Minimum vertical overlap ratio for noteheads to be considered together
        
        Returns:
            List of chord groups, where each group is a list of noteheads
        """
        def noteheads_form_chord(nh1, nh2):
            # Horizontal distance between centers
            close_x = abs(nh1['bbox']['center_x'] - nh2['bbox']['center_x']) <= x_threshold

            # Vertical overlap
            y1_top, y1_bot = nh1['bbox']['y1'], nh1['bbox']['y2']
            y2_top, y2_bot = nh2['bbox']['y1'], nh2['bbox']['y2']
            overlap = max(0, min(y1_bot, y2_bot) - max(y1_top, y2_top))
            min_height = min(nh1['bbox']['height'], nh2['bbox']['height'])
            sufficient_y_overlap = (overlap / min_height) >= y_overlap_threshold

            return close_x or sufficient_y_overlap

        # Filter for noteheads
        noteheads = [d for d in detections 
                     if d.get('class_id') in [CLASS_NOTEHEAD_BLACK, CLASS_NOTEHEAD_HALF, CLASS_NOTEHEAD_WHOLE]]

        # Sort noteheads by x-position
        noteheads.sort(key=lambda d: d['bbox']['center_x'])

        # Group noteheads by proximity and overlap
        chord_groups = []
        current_group = []

        for notehead in noteheads:
            if not current_group:
                current_group = [notehead]
                continue

            last_notehead = current_group[-1]

            if (notehead.get('staff_system') == last_notehead.get('staff_system') and
                any(noteheads_form_chord(notehead, nh) for nh in current_group)):
                current_group.append(notehead)
            else:
                chord_groups.append(current_group)
                current_group = [notehead]

        if current_group:
            chord_groups.append(current_group)

        return chord_groups

    
    def infer_stems_for_chord_groups(self, chord_groups):
        """
        Infer stems for chord groups
        
        Args:
            chord_groups: List of chord groups (lists of noteheads)
        
        Returns:
            List of inferred stem detections
        """
        inferred_stems = []
        
        for chord_noteheads in chord_groups:
            # Analyze chord structure
            chord_analysis = analyze_chord_structure(chord_noteheads, self.staff_info)
            
            # Infer stems based on chord analysis
            chord_stems = infer_stems_for_chord(chord_analysis, self.staff_info)
            
            inferred_stems.extend(chord_stems)
        
        return inferred_stems
    
    def enhance_detections(self, detections_dict, image_path=None):
        """
        Main method to enhance detections by inferring missing elements
        
        Args:
            detections_dict: List of detection dictionaries
            image_path: Optional path to the score image for computer vision analysis
        
        Returns:
            Enhanced list of detections including inferred elements
        """
        # Assign elements to staff systems
        detections = self.assign_elements_to_staff(detections_dict)
        
        # Group noteheads into chords
        chord_groups = self.group_noteheads_into_chords(detections)
        
        # Infer stems for chords
        inferred_stems = self.infer_stems_for_chord_groups(chord_groups)
        
        # Detect barlines using computer vision if image path is provided
        inferred_barlines = []
        if image_path and os.path.exists(image_path):
            inferred_barlines = detect_barlines_computer_vision(image_path, self.staff_info)
        
        # Combine original detections with inferred elements
        enhanced_detections = detections + inferred_stems + inferred_barlines
        
        return enhanced_detections
    
    def visualize_results(self, original_detections, enhanced_detections, image_path=None):
        """
        Visualize the original and enhanced detections
        
        Args:
            original_detections: List of original detection dictionaries
            enhanced_detections: List of enhanced detection dictionaries
            image_path: Optional path to the background image
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Function to plot detections on an axis
        def plot_detections(ax, detections, title):
            ax.set_title(title)
            
            # Plot staff lines
            for line in self.staff_lines:
                bbox = line['bbox']
                ax.plot([bbox['x1'], bbox['x2']], [bbox['center_y'], bbox['center_y']], 'k-', linewidth=1)
            
            # Colors for different element types
            colors = {
                CLASS_NOTEHEAD_BLACK: 'black',
                CLASS_NOTEHEAD_HALF: 'gray',
                CLASS_NOTEHEAD_WHOLE: 'lightgray',
                CLASS_STEM: 'red',
                CLASS_BARLINE: 'blue',
                CLASS_GCLEF: 'green'
            }
            
            # Plot detections
            for detection in detections:
                bbox = detection['bbox']
                class_id = detection['class_id']
                is_inferred = detection.get('inferred', False)
                
                color = colors.get(class_id, 'purple')
                linestyle = '--' if is_inferred else '-'
                linewidth = 1 if is_inferred else 2
                
                # Draw bounding box
                rect = plt.Rectangle(
                    (bbox['x1'], bbox['y1']),
                    bbox['width'],
                    bbox['height'],
                    linewidth=linewidth,
                    edgecolor=color,
                    linestyle=linestyle,
                    facecolor='none'
                )
                ax.add_patch(rect)
            
            # Set axis limits
            all_x = [det['bbox']['x1'] for det in detections]
            all_x += [det['bbox']['x2'] for det in detections]
            all_y = [det['bbox']['y1'] for det in detections]
            all_y += [det['bbox']['y2'] for det in detections]
            
            if all_x and all_y:
                margin = 50
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(max(all_y) + margin, min(all_y) - margin)  # Inverted y-axis for image coordinates
            
            ax.set_aspect('equal')
            
            # If image path is provided, try to load and display the image
            if image_path and os.path.exists(image_path):
                try:
                    img = plt.imread(image_path)
                    ax.imshow(img, aspect='equal', alpha=0.3, extent=[min(all_x) - margin, max(all_x) + margin, 
                                                                     min(all_y) - margin, max(all_y) + margin])
                except Exception as e:
                    print(f"Could not load image: {e}")
        
        # Plot original detections
        plot_detections(ax1, original_detections, 'Original Detections')
        
        # Plot enhanced detections
        plot_detections(ax2, enhanced_detections, 'Enhanced Detections')
        
        # Add legend
        handles = [plt.Line2D([0], [0], color=c, label=f"Class {id}") 
                  for id, c in {
                      CLASS_NOTEHEAD_BLACK: 'black',
                      CLASS_STEM: 'red',
                      CLASS_BARLINE: 'blue',
                      CLASS_GCLEF: 'green'
                  }.items()]
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', label='Inferred'))
        
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(handles))
        
        plt.tight_layout()
        return fig

def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving to {file_path}: {e}")

def main(staff_lines_path, object_detections_path, output_dir=None, image_path=None):
    """
    Main function to process staff line and object detection data
    
    Args:
        staff_lines_path: Path to staff lines JSON file
        object_detections_path: Path to object detections JSON file
        output_dir: Directory to save results (defaults to same directory as input)
        image_path: Optional path to the original score image
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(object_detections_path)
    
    # Create output filename
    base_name = os.path.basename(object_detections_path).replace("_detections.json", "")
    output_path = os.path.join(output_dir, f"{base_name}_enhanced_detections.json")
    
    # Load staff line and object detection data
    print(f"Loading staff lines from {staff_lines_path}")
    staff_info = load_json_file(staff_lines_path)
    
    print(f"Loading object detections from {object_detections_path}")
    object_detections = load_json_file(object_detections_path)
    
    if not staff_info or not object_detections:
        print("Failed to load required input files.")
        return
    
    # Extract detections from the object_detections file
    detections = object_detections.get("detections", [])
    print(f"Loaded {len(detections)} object detections")
    
    # Create enhancer
    enhancer = OMREnhancer(staff_info)
    
    # Enhance detections
    print("Inferring missing stems and barlines...")
    enhanced_detections = enhancer.enhance_detections(detections, image_path)
    
    # Count original and inferred objects
    original_count = len(detections)
    enhanced_count = len(enhanced_detections)
    inferred_count = sum(1 for det in enhanced_detections if det.get("inferred", False))
    
    print(f"Original detections: {original_count}")
    print(f"Inferred elements: {inferred_count}")
    print(f"Total enhanced detections: {enhanced_count}")
    
    # Create output structure mirroring the input
    result = object_detections.copy()
    result["detections"] = enhanced_detections
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["enhanced"] = True
    result["metadata"]["inferred_count"] = inferred_count
    
    # Save results
    print(f"Saving enhanced detections to {output_path}")
    save_json_file(result, output_path)
    
    # Generate visualization if matplotlib is available
    try:
        print("Generating visualization...")
        fig = enhancer.visualize_results(detections, enhanced_detections, image_path)
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"{base_name}_enhancement_visualization.png")
        fig.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {viz_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error generating visualization: {e}")
    
    print("Enhancement complete!")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance OMR detections by inferring missing stems and barlines")
    parser.add_argument("--staff-lines", required=True, help="Path to staff lines JSON file")
    parser.add_argument("--detections", required=True, help="Path to object detections JSON file")
    parser.add_argument("--output-dir", help="Directory to save enhanced detections (defaults to same as input)")
    parser.add_argument("--image", help="Path to original score image for computer vision analysis")
    
    args = parser.parse_args()
    
    main(args.staff_lines, args.detections, args.output_dir, args.image)