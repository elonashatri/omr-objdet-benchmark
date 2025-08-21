#/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/stem_detector.py
import math
import json
import os
import argparse
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle

# Constants for staff line spacing and musical elements
STAFF_LINE_SPACING = 27  # Approximate spacing between staff lines (can be calculated dynamically)
MIN_STEM_LENGTH = 3.0 * STAFF_LINE_SPACING  # Minimum stem length in pixels

# Constants derived from statistical analysis
STEM_WIDTH = 3.0  # Median stem width
STEM_HEIGHT_MEAN = 73.4  
STEM_HEIGHT_MEDIAN = 69.0
STEM_HEIGHT_STD = 25.7

# Class IDs (adjust these according to your model's class mapping)
CLASS_NOTEHEAD_BLACK = 0
CLASS_NOTEHEAD_HALF = 1
CLASS_NOTEHEAD_WHOLE = 2
CLASS_STEM = 5
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
    
    # Calculate the average position of all noteheads in the chord
    avg_y = sum(nh['bbox']['center_y'] for nh in noteheads) / len(noteheads)
    
    # Determine stem direction based on average position relative to middle line
    # Following standard notation practice: stems up when below middle line, down when above
    default_stem_direction = 'up' if avg_y > middle_line_y else 'down'
    
    # Count noteheads above and below middle line for more complex decisions
    above_middle = sum(1 for nh in noteheads if nh['bbox']['center_y'] < middle_line_y)
    below_middle = len(noteheads) - above_middle
    
    # Special case: if noteheads equally distributed, use position of the extreme notes
    if above_middle == below_middle and len(noteheads) > 1:
        # Calculate distance from middle line to highest and lowest notes
        top_distance = middle_line_y - noteheads_sorted[0]['bbox']['center_y']
        bottom_distance = noteheads_sorted[-1]['bbox']['center_y'] - middle_line_y
        
        # Use the direction that gives more balanced appearance
        default_stem_direction = 'down' if top_distance > bottom_distance else 'up'
    
    # Determine if there are potentially multiple voices
    # Criteria: more than 4 noteheads with significant pitch range
    multiple_voices = (len(noteheads) > 4 and 
                      pitch_range > 3 * (y_positions[1] - y_positions[0]))
    
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
            cluster_avg_y = sum(nh['bbox']['center_y'] for nh in cluster) / len(cluster)
            direction = 'up' if cluster_avg_y > middle_line_y else 'down'
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

def infer_stems_for_beamed_group(beamed_noteheads, beam, staff_info):
    """
    Create stems that properly connect noteheads to beams
    
    Args:
        beamed_noteheads: List of noteheads associated with the beam
        beam: The beam detection
        staff_info: Staff information dictionary
    
    Returns:
        List of inferred stem detections
    """
    inferred_stems = []
    processed_notehead_ids = set()

    beam_x1, beam_y1 = beam['bbox']['x1'], beam['bbox']['y1']
    beam_x2, beam_y2 = beam['bbox']['x2'], beam['bbox']['y2']

    # Calculate slope of beam
    beam_dx = beam_x2 - beam_x1
    beam_dy = beam_y2 - beam_y1
    beam_slope = beam_dy / (beam_dx + 1e-6)
    
    # Determine beam direction clearly
    beam_direction = beam.get('beam_direction', 'up' if beam_slope < 0 else 'down')

    # Sort noteheads by x-coordinate for consistent processing
    sorted_noteheads = sorted(beamed_noteheads, key=lambda nh: nh['bbox']['center_x'])

    for nh in sorted_noteheads:
        # Skip noteheads we've already processed
        notehead_id = nh.get('id')
        if notehead_id in processed_notehead_ids:
            continue
        
        # Mark this notehead as processed
        processed_notehead_ids.add(notehead_id)
        
        nh_x = nh['bbox']['center_x']
        nh_y = nh['bbox']['center_y']

        # Use stored projected beam y if available, otherwise calculate it
        if 'projected_beam_y' in nh:
            proj_beam_y = nh['projected_beam_y']
        else:
            # Project the notehead horizontally onto the beam line
            proj_beam_y = beam_y1 + beam_slope * (nh_x - beam_x1)

        # Ensure stem connects directly to the beam
        if beam_direction == 'up':
            # For up stems, position at right side of notehead
            x_stem = nh['bbox']['x2'] - STEM_WIDTH / 2
            # Top of stem is at the beam, bottom at notehead
            y_top = proj_beam_y
            y_bottom = nh_y
        else:
            # For down stems, position at left side of notehead
            x_stem = nh['bbox']['x1'] + STEM_WIDTH / 2
            # Top of stem is at notehead, bottom at the beam
            y_top = nh_y
            y_bottom = proj_beam_y

        stem_height = abs(y_bottom - y_top)
        
        # Ensure minimum stem length (but don't override connection to beam)
        min_stem_length = STAFF_LINE_SPACING * 2.5
        if stem_height < min_stem_length:
            if beam_direction == 'up':
                y_top = y_bottom - min_stem_length
            else:
                y_bottom = y_top + min_stem_length

        # Create a unique ID for this stem
        stem_id = f"stem_beam_{notehead_id}_{beam.get('id', 'unknown')}"

        stem = {
            'id': stem_id,
            'class_id': CLASS_STEM,
            'class_name': 'stem',
            'confidence': 0.95,
            'bbox': {
                'x1': x_stem - STEM_WIDTH / 2,
                'x2': x_stem + STEM_WIDTH / 2,
                'y1': y_top,
                'y2': y_bottom,
                'width': STEM_WIDTH,
                'height': abs(y_bottom - y_top),
                'center_x': x_stem,
                'center_y': (y_top + y_bottom) / 2
            },
            'staff_system': nh.get('staff_system'),
            'inferred': True,
            'beam_id': beam.get('id'),
            'notehead_id': notehead_id,
            'stem_direction': beam_direction,
            'stem_inferred_via_beam': True
        }

        inferred_stems.append(stem)

    return inferred_stems

def validate_stem_beam_connections(stems, beams):
    """
    Validate and fix stem connections to beams
    
    Args:
        stems: List of stem detections
        beams: List of beam detections
    
    Returns:
        List of validated/fixed stem detections
    """
    validated_stems = []
    
    for stem in stems:
        # Skip stems not associated with beams
        if not stem.get('beam_id'):
            validated_stems.append(stem)
            continue
            
        # Find associated beam
        beam = next((b for b in beams if b.get('id') == stem['beam_id']), None)
        if not beam:
            validated_stems.append(stem)
            continue
            
        # Get beam and stem properties
        beam_x1, beam_y1 = beam['bbox']['x1'], beam['bbox']['y1']
        beam_x2, beam_y2 = beam['bbox']['x2'], beam['bbox']['y2']
        stem_x = stem['bbox']['center_x']
        
        # Calculate where the stem should meet the beam
        beam_slope = (beam_y2 - beam_y1) / (beam_x2 - beam_x1 + 1e-6)
        target_y = beam_y1 + beam_slope * (stem_x - beam_x1)
        
        # Get stem direction and adjust the appropriate end
        stem_direction = stem.get('stem_direction', 'up')
        bbox = stem['bbox'].copy()
        
        if stem_direction == 'up':
            # Adjust top of stem to meet beam
            bbox['y1'] = target_y
            bbox['height'] = bbox['y2'] - bbox['y1']
        else:
            # Adjust bottom of stem to meet beam
            bbox['y2'] = target_y
            bbox['height'] = bbox['y2'] - bbox['y1']
        
        # Update center_y
        bbox['center_y'] = (bbox['y1'] + bbox['y2']) / 2
        
        # Create validated stem
        validated_stem = stem.copy()
        validated_stem['bbox'] = bbox
        validated_stems.append(validated_stem)
    
    return validated_stems

class OMREnhancer:
    def __init__(self, staff_info):
        self.staff_info = staff_info
        self.staff_systems = staff_info['staff_systems']
        self.staff_lines = staff_info['detections']
        self.calculate_staff_parameters()
    
    def calculate_staff_parameters(self):
        """Calculate staff parameters for each staff system"""
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
        """Assign detected elements to their closest staff system"""
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

    def group_noteheads_by_beam(self, detections, vertical_threshold_ratio=1.5, horizontal_margin_ratio=0.5):
        """
        Group noteheads with beams they should connect to, with improved matching criteria
        
        Args:
            detections: List of detection dictionaries
            vertical_threshold_ratio: Ratio of staff spacing for vertical matching
            horizontal_margin_ratio: Ratio of staff spacing for horizontal margin
        
        Returns:
            List of tuples (noteheads, beam) for each beamed group
        """
        beams = [d for d in detections if d.get('class_name') == 'beam']
        noteheads = [d for d in detections if d.get('class_id') in [
            CLASS_NOTEHEAD_BLACK, CLASS_NOTEHEAD_HALF, CLASS_NOTEHEAD_WHOLE
        ]]
        beam_groups = []

        # Ensure IDs for all elements for tracking relationships
        for i, element in enumerate(detections):
            if 'id' not in element:
                element['id'] = f"{element.get('class_name', 'element')}_{i}"
        
        for beam in beams:
            bx1, bx2 = beam['bbox']['x1'], beam['bbox']['x2']
            by1, by2 = beam['bbox']['y1'], beam['bbox']['y2']
            slope = (by2 - by1) / (bx2 - bx1 + 1e-6)

            # Determine beam direction more reliably
            beam_direction = 'up' if slope < 0 else 'down'
            beam['beam_direction'] = beam_direction
            
            in_beam = []
            for nh in noteheads:
                nx, ny = nh['bbox']['center_x'], nh['bbox']['center_y']
                
                # Get staff parameters for this notehead
                staff_system_id = nh.get('staff_system')
                avg_spacing = self.staff_parameters.get(staff_system_id, {}).get('avg_spacing', STAFF_LINE_SPACING)
                
                # Add horizontal margin based on staff spacing
                horizontal_margin = avg_spacing * horizontal_margin_ratio
                
                # Check horizontal alignment with margin
                if (bx1 - horizontal_margin) <= nx <= (bx2 + horizontal_margin):
                    # Find projected y position of beam at notehead's x position
                    projected_beam_y = by1 + slope * (nx - bx1)
                    vertical_distance = abs(ny - projected_beam_y)

                    # Dynamically compute threshold based on staff spacing
                    vertical_threshold = avg_spacing * vertical_threshold_ratio

                    # Consider stem length in matching - noteheads should be within a reasonable stem length
                    max_stem_length = STAFF_LINE_SPACING * 5  # Typical max stem length
                    
                    # Check if the notehead is in the right direction from the beam
                    direction_match = True
                    if beam_direction == 'up' and ny > projected_beam_y:
                        direction_match = vertical_distance <= max_stem_length
                    elif beam_direction == 'down' and ny < projected_beam_y:
                        direction_match = vertical_distance <= max_stem_length
                    
                    if direction_match and vertical_distance <= max_stem_length:
                        # Store projected beam y with the notehead for stem creation
                        nh_copy = nh.copy()
                        nh_copy['projected_beam_y'] = projected_beam_y
                        nh_copy['beam_id'] = beam.get('id')
                        in_beam.append(nh_copy)

            if in_beam:
                beam_groups.append((in_beam, beam))

        return beam_groups


    def group_noteheads_into_chords(self, detections, x_threshold=40, y_overlap_threshold=0.5):
        """
        Group noteheads that likely belong to the same chord based on
        horizontal proximity and vertical overlap
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
        Main method to enhance detections by inferring missing stems
        
        Args:
            detections_dict: List of detection dictionaries
            image_path: Optional path to the score image (not used in this version)
        
        Returns:
            Enhanced list of detections including inferred stems
        """
        # Assign unique IDs to all elements if not already present
        for i, detection in enumerate(detections_dict):
            if 'id' not in detection:
                detection['id'] = f"{detection.get('class_name', 'element')}_{i}"
        
        # Assign elements to staff systems
        detections = self.assign_elements_to_staff(detections_dict)
        
        # Keep track of which noteheads already have stems assigned
        notehead_has_stem = {}  # Map notehead ID to stem ID
        
        # First, identify existing stems in the detections
        existing_stems = [d for d in detections if d.get('class_name') == 'stem']
        # We'd need logic to associate these with noteheads, but since the data structure
        # doesn't explicitly link them, we'll skip this for now and just infer new ones
        
        # Process beamed groups first (they take priority)
        beamed_groups = self.group_noteheads_by_beam(detections, 
                                                    vertical_threshold_ratio=2.0,
                                                    horizontal_margin_ratio=0.8)
        
        beams = [d for d in detections if d.get('class_name') == 'beam']
        
        # Infer stems for beamed groups
        inferred_beam_stems = []
        for group, beam in beamed_groups:
            # Create stems for this beam group
            beam_stems = infer_stems_for_beamed_group(group, beam, self.staff_info)
            
            # Mark these noteheads as having stems
            for stem in beam_stems:
                notehead_id = stem.get('notehead_id')
                if notehead_id:
                    notehead_has_stem[notehead_id] = stem.get('id', f"stem_{len(inferred_beam_stems) + len(inferred_beam_stems)}")
            
            inferred_beam_stems.extend(beam_stems)
        
        # Validate and fix stem connections to beams
        inferred_beam_stems = validate_stem_beam_connections(inferred_beam_stems, beams)
        
        # Group noteheads into chords
        chord_groups = self.group_noteheads_into_chords(detections)
        
        # Filter chord groups to exclude noteheads that already have stems
        filtered_chord_groups = []
        for group in chord_groups:
            # Only keep noteheads that don't already have stems
            filtered_group = [nh for nh in group if nh.get('id') not in notehead_has_stem]
            if filtered_group:  # Only keep non-empty groups
                filtered_chord_groups.append(filtered_group)
        
        # Infer stems for the remaining chord groups
        inferred_chord_stems = []
        for chord_noteheads in filtered_chord_groups:
            # Analyze chord structure
            chord_analysis = analyze_chord_structure(chord_noteheads, self.staff_info)
            
            # Infer stems based on chord analysis
            chord_stems = infer_stems_for_chord(chord_analysis, self.staff_info)
            
            # Mark these noteheads as having stems
            for stem in chord_stems:
                notehead_id = stem.get('notehead_id')
                if notehead_id:
                    notehead_has_stem[notehead_id] = stem.get('id', f"stem_{len(inferred_beam_stems) + len(inferred_chord_stems)}")
            
            inferred_chord_stems.extend(chord_stems)
        
        # Combine original detections with inferred stems
        enhanced_detections = detections + inferred_beam_stems + inferred_chord_stems
        
        # Perform final deduplication to ensure no notehead has multiple stems
        enhanced_detections = deduplicate_stems(enhanced_detections)
        
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
                                                                     max(all_y) + margin, min(all_y) - margin])
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
                      CLASS_GCLEF: 'green'
                  }.items()]
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', label='Inferred'))
        
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(handles))
        
        plt.tight_layout()
        return fig


def overlay_detections_on_image(image_path, detections, output_path=None, show_original=True, 
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


def load_json(file_path):
    """Load a JSON file."""
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
    staff_info = load_json(staff_lines_path)
    
    print(f"Loading object detections from {object_detections_path}")
    object_detections = load_json(object_detections_path)
    
    if not staff_info or not object_detections:
        print("Failed to load required input files.")
        return
    
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
    
    # Generate side-by-side visualization
    try:
        print("Generating side-by-side visualization...")
        fig = enhancer.visualize_results(detections, enhanced_detections, image_path)
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"{base_name}_stem_comparison.png")
        fig.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to {viz_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Error generating comparison visualization: {e}")
    
    print("Stem detection complete!")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance OMR detections by inferring missing stems")
    parser.add_argument("--staff-lines", required=True, help="Path to staff lines JSON file")
    parser.add_argument("--detections", required=True, help="Path to object detections JSON file")
    parser.add_argument("--output-dir", help="Directory to save enhanced detections (defaults to same as input)")
    parser.add_argument("--image", help="Path to original score image for visualization")
    
    args = parser.parse_args()
    
    main(args.staff_lines, args.detections, args.output_dir, args.image)

def infer_stems_for_chord(chord_analysis, staff_info):
    """
    Infer stems for a chord based on its structure analysis with improved positioning
    
    Args:
        chord_analysis: Dictionary with chord analysis results
        staff_info: Staff information dictionary
    
    Returns:
        List of inferred stem detections
    """
    if not chord_analysis.get('needs_stem', False):
        return []
    
    inferred_stems = []
    processed_notehead_ids = set()
    
    # Process each voice group
    for voice_id, voice_data in chord_analysis.get('voice_groups', {}).items():
        noteheads = voice_data.get('noteheads', [])
        if not noteheads:
            continue
        
        stem_direction = voice_data.get('stem_direction', 'up')
        
        # Get staff system from first notehead
        staff_id = noteheads[0].get('staff_system')
        
        # For simple chords (single voice), just create one stem for the entire chord
        if voice_id == 'main_voice' and len(noteheads) > 1:
            # Choose the notehead for stem attachment
            sorted_by_y = sorted(noteheads, key=lambda nh: nh['bbox']['center_y'])
            if stem_direction == 'up':
                # For up stems, use the highest notehead (smallest y)
                primary_notehead = sorted_by_y[0]
            else:
                # For down stems, use the lowest notehead (largest y)
                primary_notehead = sorted_by_y[-1]
            
            # Calculate stem position
            if stem_direction == 'up':
                # Right side of notehead for up stems
                x_pos = primary_notehead['bbox']['x2'] - 3
                y_bottom = primary_notehead['bbox']['center_y']
                
                # Calculate stem top position extending past the entire chord
                highest_y = min(nh['bbox']['y1'] for nh in noteheads)
                y_top = highest_y - (STAFF_LINE_SPACING * 1.5)
            else:
                # Left side of notehead for down stems
                x_pos = primary_notehead['bbox']['x1'] + 3
                y_top = primary_notehead['bbox']['center_y']
                
                # Calculate stem bottom position extending past the entire chord
                lowest_y = max(nh['bbox']['y2'] for nh in noteheads)
                y_bottom = lowest_y + (STAFF_LINE_SPACING * 1.5)
            
            # Create a unique ID for this stem
            stem_id = f"stem_chord_{primary_notehead.get('id', 'unknown')}_{voice_id}"
            
            # Create the inferred stem
            stem = {
                'id': stem_id,
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
                'notehead_id': primary_notehead.get('id'),
                'associated_notehead_ids': [nh.get('id') for nh in noteheads],
                'voice_id': voice_id,
                'stem_direction': stem_direction
            }
            
            inferred_stems.append(stem)
            
            # Mark all noteheads in this chord as processed
            for nh in noteheads:
                processed_notehead_ids.add(nh.get('id'))
                
        else:
            # For multiple voices or single notes, create individual stems
            for nh in noteheads:
                # Skip if already processed or whole notes (which don't need stems)
                notehead_id = nh.get('id')
                if notehead_id in processed_notehead_ids or nh.get('class_id') == CLASS_NOTEHEAD_WHOLE:
                    continue
                
                # Mark as processed
                processed_notehead_ids.add(notehead_id)
                
                # Calculate stem position
                if stem_direction == 'up':
                    # Right side of notehead for up stems
                    x_pos = nh['bbox']['x2'] - 3
                    y_bottom = nh['bbox']['center_y']
                    
                    # Calculate appropriate stem height
                    stem_height = STEM_HEIGHT_MEDIAN
                    stem_height = min(STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD, 
                                    max(STEM_HEIGHT_MEDIAN, stem_height))
                    
                    y_top = y_bottom - stem_height
                else:
                    # Left side of notehead for down stems
                    x_pos = nh['bbox']['x1'] + 3
                    y_top = nh['bbox']['center_y']
                    
                    # Calculate appropriate stem height
                    stem_height = STEM_HEIGHT_MEDIAN
                    stem_height = min(STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD, 
                                    max(STEM_HEIGHT_MEDIAN, stem_height))
                    
                    y_bottom = y_top + stem_height
                
                # Create a unique ID for this stem
                stem_id = f"stem_indiv_{notehead_id}_{voice_id}"
                
                # Create the inferred stem
                stem = {
                    'id': stem_id,
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
                    'notehead_id': notehead_id,
                    'voice_id': voice_id,
                    'stem_direction': stem_direction
                }
                
                inferred_stems.append(stem)
    
    return inferred_stems

def deduplicate_stems(detections):
    """
    Ensure there are no duplicate stems for the same notehead
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        List of deduplicated detections
    """
    # Separate stems from other detections
    stems = [d for d in detections if d.get('class_name') == 'stem']
    other_elements = [d for d in detections if d.get('class_name') != 'stem']
    
    # Group stems by their associated notehead_id
    stems_by_notehead = {}
    for stem in stems:
        notehead_id = stem.get('notehead_id')
        if notehead_id:
            if notehead_id not in stems_by_notehead:
                stems_by_notehead[notehead_id] = []
            stems_by_notehead[notehead_id].append(stem)
    
    # For each notehead with multiple stems, keep only the highest priority one
    deduplicated_stems = []
    for notehead_id, related_stems in stems_by_notehead.items():
        if len(related_stems) == 1:
            # Only one stem, no deduplication needed
            deduplicated_stems.append(related_stems[0])
        else:
            # Multiple stems for the same notehead, prioritize
            # 1. Stems connected to beams
            # 2. Higher confidence stems
            # 3. Explicitly created stems vs inferred ones
            beam_stems = [s for s in related_stems if s.get('beam_id') is not None]
            if beam_stems:
                # Prioritize beam stems, and among those, the one with highest confidence
                best_stem = max(beam_stems, key=lambda s: s.get('confidence', 0))
            else:
                # No beam stems, use confidence as tiebreaker
                best_stem = max(related_stems, key=lambda s: s.get('confidence', 0))
            
            deduplicated_stems.append(best_stem)
    
    # Add stems that don't have an associated notehead_id
    deduplicated_stems.extend([s for s in stems if not s.get('notehead_id')])
    
    # Recombine other elements with deduplicated stems
    return other_elements + deduplicated_stems