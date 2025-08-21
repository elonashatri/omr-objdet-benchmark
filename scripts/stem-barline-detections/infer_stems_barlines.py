import numpy as np
import math
import json
import os
import argparse
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

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

# File paths for input data
STAFF_LINES_PATH = "/homes/es314/omr-objdet-benchmark/scripts/encoding/pixe-enhancment/results/enhanced/Accidentals-004_standard.json"
OBJECT_DETECTIONS_PATH = "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/object_detections/Accidentals-004_detections.json"

# Class IDs (adjust these according to your model's class mapping)
CLASS_NOTEHEAD_BLACK = 0
CLASS_NOTEHEAD_HALF = 1
CLASS_NOTEHEAD_WHOLE = 2
CLASS_STEM = 5
CLASS_BARLINE = 8
CLASS_GCLEF = 14

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float
    
    @staticmethod
    def from_dict(bbox_dict):
        return BoundingBox(
            x1=bbox_dict['x1'],
            y1=bbox_dict['y1'],
            x2=bbox_dict['x2'],
            y2=bbox_dict['y2'],
            width=bbox_dict['width'],
            height=bbox_dict['height'],
            center_x=bbox_dict['center_x'],
            center_y=bbox_dict['center_y']
        )
    
    def to_dict(self):
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'width': self.width,
            'height': self.height,
            'center_x': self.center_x,
            'center_y': self.center_y
        }

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    staff_system: Optional[int] = None
    inferred: bool = False
    
    @staticmethod
    def from_dict(detection_dict):
        return Detection(
            class_id=detection_dict['class_id'],
            class_name=detection_dict['class_name'],
            confidence=detection_dict['confidence'],
            bbox=BoundingBox.from_dict(detection_dict['bbox']),
            staff_system=detection_dict.get('staff_system'),
            inferred=detection_dict.get('inferred', False)
        )
    
    def to_dict(self):
        result = {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox.to_dict(),
            'inferred': self.inferred
        }
        if self.staff_system is not None:
            result['staff_system'] = self.staff_system
        return result


class OMREnhancer:
    def __init__(self, staff_info):
        """
        Initialize the enhancer with staff line information
        
        Args:
            staff_info: Dictionary containing staff systems and their line positions
        """
        self.staff_info = staff_info
        self.staff_systems = staff_info['staff_systems']
        self.staff_lines = staff_info['detections']
        self.calculate_staff_parameters()
    
    def calculate_staff_parameters(self):
        """Calculate key parameters for each staff system"""
        self.staff_parameters = {}
        
        for system in self.staff_systems:
            system_id = system['id']
            lines = [line for line in self.staff_lines if line['staff_system'] == system_id]
            # lines_sorted = sorted(lines, key=lambda x: x['line_number'])
            lines_sorted = sorted(lines, key=lambda x: x['bbox']['center_y'])
            print(f"System {system_id}: y_positions = {[l['bbox']['center_y'] for l in lines_sorted]}")


            
            if len(lines_sorted) >= 5:  # Standard 5-line staff
                y_positions = [line['bbox']['center_y'] for line in lines_sorted]
                
                # Calculate average spacing between staff lines
                spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
                avg_spacing = sum(spacings) / len(spacings)
                
                # Calculate staff boundaries
                top = y_positions[0] - avg_spacing  # Above top line
                bottom = y_positions[-1] + avg_spacing  # Below bottom line
                left = min(line['bbox']['x1'] for line in lines_sorted)
                right = max(line['bbox']['x2'] for line in lines_sorted)
                
                self.staff_parameters[system_id] = {
                    'y_positions': y_positions,
                    'avg_spacing': avg_spacing,
                    'top': top,
                    'bottom': bottom,
                    'left': left,
                    'right': right,
                    'middle_line_y': y_positions[2],  # Middle staff line
                }
    
    def assign_elements_to_staff(self, detections):
        """
        Assign each detected element to a staff system
        
        Args:
            detections: List of detected musical elements
        
        Returns:
            List of detections with assigned staff_system property
        """
        enhanced_detections = []
        
        for detection in detections:
            det = Detection.from_dict(detection) if not isinstance(detection, Detection) else detection
            
            # Skip if already assigned
            if det.staff_system is not None:
                enhanced_detections.append(det)
                continue
            
            # Find the closest staff system
            center_y = det.bbox.center_y
            assigned_system = None
            min_distance = float('inf')
            
            for system_id, params in self.staff_parameters.items():
                # Check if element is within or close to staff vertical boundaries
                if params['top'] - params['avg_spacing'] <= center_y <= params['bottom'] + params['avg_spacing']:
                    # Calculate vertical distance to the middle line of the staff
                    distance = abs(center_y - params['middle_line_y'])
                    if distance < min_distance:
                        min_distance = distance
                        assigned_system = system_id
            
            det.staff_system = assigned_system
            enhanced_detections.append(det)
        
        return enhanced_detections
    
    def group_noteheads_into_chords(self, detections, x_threshold=15, stem_threshold=5):
        """
        Group noteheads that likely belong to the same chord
        
        Args:
            detections: List of detected musical elements
            x_threshold: Maximum horizontal distance between noteheads in the same chord
            stem_threshold: Distance threshold for associating stems with noteheads
        
        Returns:
            List of chord groups, where each group is a list of noteheads
        """
        # Extract notehead detections
        noteheads = [d for d in detections if d.class_id in [CLASS_NOTEHEAD_BLACK, CLASS_NOTEHEAD_HALF, CLASS_NOTEHEAD_WHOLE]]
        
        # Extract existing stems for later association
        stems = [d for d in detections if d.class_id == CLASS_STEM]
        
        # Sort noteheads by x-position
        noteheads.sort(key=lambda d: d.bbox.center_x)
        
        # Group noteheads by horizontal proximity and staff system
        chord_groups = []
        current_group = []
        
        for notehead in noteheads:
            # Start a new group if current one is empty
            if not current_group:
                current_group = [notehead]
                continue
            
            # Check if this notehead should be added to current group
            last_notehead = current_group[-1]
            
            if (notehead.staff_system == last_notehead.staff_system and
                    abs(notehead.bbox.center_x - last_notehead.bbox.center_x) <= x_threshold):
                current_group.append(notehead)
            else:
                # Start a new group
                chord_groups.append(current_group)
                current_group = [notehead]
        
        # Add the last group if not empty
        if current_group:
            chord_groups.append(current_group)
        
        # Associate existing stems with chords
        for stem in stems:
            for group in chord_groups:
                # Check horizontal alignment
                group_x_min = min(notehead.bbox.x1 for notehead in group)
                group_x_max = max(notehead.bbox.x2 for notehead in group)
                
                # Extend slightly to account for proper stem positioning
                extended_x_min = group_x_min - stem_threshold
                extended_x_max = group_x_max + stem_threshold
                
                if extended_x_min <= stem.bbox.center_x <= extended_x_max:
                    # Check vertical overlap
                    group_y_min = min(notehead.bbox.y1 for notehead in group)
                    group_y_max = max(notehead.bbox.y2 for notehead in group)
                    
                    if (stem.bbox.y1 <= group_y_max and stem.bbox.y2 >= group_y_min):
                        # This stem is associated with this chord group
                        group.append(stem)
                        break
        
        return chord_groups
    
    def infer_stems_for_chords(self, chord_groups):
        """
        Infer stems for chord groups that don't have them
        
        Args:
            chord_groups: List of chord groups (lists of noteheads)
        
        Returns:
            List of inferred stem detections
        """
        inferred_stems = []
        
        for group in chord_groups:
            # Skip groups that already have stems
            if any(d.class_id == CLASS_STEM for d in group):
                continue
            
            # Skip if not a valid chord (no noteheads)
            noteheads = [d for d in group if d.class_id in [CLASS_NOTEHEAD_BLACK, CLASS_NOTEHEAD_HALF, CLASS_NOTEHEAD_WHOLE]]
            if not noteheads:
                continue
            
            # Get staff parameters for this group
            staff_id = noteheads[0].staff_system
            if staff_id is None or staff_id not in self.staff_parameters:
                continue
                
            staff_params = self.staff_parameters[staff_id]
            
            # Determine stem direction based on music notation rules
            avg_y = sum(nh.bbox.center_y for nh in noteheads) / len(noteheads)
            stem_direction = 'up' if avg_y > staff_params['middle_line_y'] else 'down'
            
            # For chord groups with wide pitch range, use majority rule
            if len(noteheads) > 1:
                pitch_range = max(nh.bbox.center_y for nh in noteheads) - min(nh.bbox.center_y for nh in noteheads)
                if pitch_range > staff_params['avg_spacing'] * 2:
                    notes_above = sum(1 for nh in noteheads if nh.bbox.center_y < staff_params['middle_line_y'])
                    notes_below = len(noteheads) - notes_above
                    stem_direction = 'down' if notes_above > notes_below else 'up'
            
            # Create stem object with appropriate position and size based on statistical analysis
            if stem_direction == 'up':
                # Right side of rightmost notehead for up stems
                x_pos = max(nh.bbox.x2 - STEM_WIDTH for nh in noteheads)
                
                # Calculate stem height based on statistics (mean ± some standard deviation based on context)
                # For multi-note chords, use longer stems
                stem_height = STEM_HEIGHT_MEDIAN
                if len(noteheads) > 1:
                    # Longer stem for chords with wider pitch range
                    pitch_range = max(nh.bbox.center_y for nh in noteheads) - min(nh.bbox.center_y for nh in noteheads)
                    stem_height = min(STEM_HEIGHT_MEDIAN + pitch_range/2, STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD)
                
                y_top = min(nh.bbox.y1 for nh in noteheads) - stem_height
                y_bottom = min(nh.bbox.y1 for nh in noteheads)
            else:
                # Left side of leftmost notehead for down stems
                x_pos = min(nh.bbox.x1 for nh in noteheads) + STEM_WIDTH
                
                # Calculate stem height based on statistics
                stem_height = STEM_HEIGHT_MEDIAN
                if len(noteheads) > 1:
                    # Longer stem for chords with wider pitch range
                    pitch_range = max(nh.bbox.center_y for nh in noteheads) - min(nh.bbox.center_y for nh in noteheads)
                    stem_height = min(STEM_HEIGHT_MEDIAN + pitch_range/2, STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD)
                
                y_top = max(nh.bbox.y2 for nh in noteheads)
                y_bottom = max(nh.bbox.y2 for nh in noteheads) + stem_height
            
            # Create the inferred stem with dimensions from statistical analysis
            stem = Detection(
                class_id=CLASS_STEM,
                class_name='stem',
                confidence=0.85,  # Inferred confidence
                bbox=BoundingBox(
                    x1=x_pos - STEM_WIDTH/2,
                    y1=y_top,
                    x2=x_pos + STEM_WIDTH/2,
                    y2=y_bottom,
                    width=STEM_WIDTH,
                    height=y_bottom - y_top,
                    center_x=x_pos,
                    center_y=(y_top + y_bottom) / 2
                ),
                staff_system=staff_id,
                inferred=True
            )
            
            inferred_stems.append(stem)
        
        return inferred_stems
    
    def infer_barlines(self, detections):
        """
        Infer barlines based on staff systems and existing elements
        
        Args:
            detections: List of detected musical elements
        
        Returns:
            List of inferred barline detections
        """
        inferred_barlines = []
        existing_barlines = [d for d in detections if d.class_id == CLASS_BARLINE]
        
        # Determine if we should use regular barlines or systemic barlines
        # If we have multiple staff systems, check their separation
        staff_system_count = len(self.staff_parameters)
        use_systemic_barlines = False
        
        if staff_system_count > 1:
            # Calculate distance between staff systems
            system_distances = []
            staff_middles = []
            
            for system_id, params in self.staff_parameters.items():
                staff_middles.append((system_id, params['middle_line_y']))
            
            staff_middles.sort(key=lambda x: x[1])  # Sort by vertical position
            
            for i in range(len(staff_middles) - 1):
                system_distances.append(staff_middles[i+1][1] - staff_middles[i][1])
            
            # If staff systems are close enough, use systemic barlines
            if system_distances and max(system_distances) < SYSTEMIC_BARLINE_HEIGHT_MEDIAN / 2:
                use_systemic_barlines = True
        
        # Process each staff system
        for system_id, params in self.staff_parameters.items():
            # Get existing barlines for this staff
            staff_barlines = [b for b in existing_barlines if b.staff_system == system_id]
            
            # Get clefs for this staff (to avoid placing barlines on clefs)
            staff_clefs = [d for d in detections if d.class_id == CLASS_GCLEF and d.staff_system == system_id]
            clef_positions = [c.bbox.center_x for c in staff_clefs]
            
            # Get noteheads for this staff (to estimate measure boundaries)
            staff_noteheads = [d for d in detections 
                              if d.class_id in [CLASS_NOTEHEAD_BLACK, CLASS_NOTEHEAD_HALF, CLASS_NOTEHEAD_WHOLE] 
                              and d.staff_system == system_id]
            
            # Staff dimensions
            staff_height = params['bottom'] - params['top']
            staff_left = params['left']
            staff_right = params['right']
            staff_middle_y = params['middle_line_y']
            
            # Add measure start barline if none exists near the start
            has_start_barline = any(b.bbox.center_x < staff_left + 100 for b in staff_barlines)
            if not has_start_barline and not any(abs(x - staff_left) < 80 for x in clef_positions):
                # Place barline at the start of the staff, leaving room for clef
                x_pos = staff_left + 10
                inferred_barlines.append(Detection(
                    class_id=CLASS_BARLINE,
                    class_name='barline',
                    confidence=0.80,
                    bbox=BoundingBox(
                        x1=x_pos - BARLINE_THICKNESS/2,
                        y1=params['top'],
                        x2=x_pos + BARLINE_THICKNESS/2,
                        y2=params['bottom'],
                        width=BARLINE_THICKNESS,
                        height=BARLINE_HEIGHT_MEDIAN,  # Use median height from statistics
                        center_x=x_pos,
                        center_y=staff_middle_y
                    ),
                    staff_system=system_id,
                    inferred=True
                ))
            
            # Add measure end barline if none exists near the end
            has_end_barline = any(b.bbox.center_x > staff_right - 100 for b in staff_barlines)
            if not has_end_barline:
                # Place barline at the end of the staff
                x_pos = staff_right - 10
                
                if use_systemic_barlines and system_id > 0:
                    # Skip creating individual barlines for lower staves when using systemic barlines
                    pass
                else:
                    # Create individual barline or first staff of systemic barline
                    barline_height = staff_height
                    barline_y1 = params['top'] 
                    barline_y2 = params['bottom']
                    
                    if use_systemic_barlines and system_id == 0:
                        # For systemic barlines, span from top of first staff to bottom of last staff
                        last_system_id = max(self.staff_parameters.keys())
                        last_params = self.staff_parameters[last_system_id]
                        barline_height = last_params['bottom'] - params['top']
                        barline_y2 = last_params['bottom']
                    
                    inferred_barlines.append(Detection(
                        class_id=CLASS_BARLINE,
                        class_name='barline' if not use_systemic_barlines else 'systemicBarline',
                        confidence=0.80,
                        bbox=BoundingBox(
                            x1=x_pos - BARLINE_THICKNESS/2,
                            y1=barline_y1,
                            x2=x_pos + BARLINE_THICKNESS/2,
                            y2=barline_y2,
                            width=BARLINE_THICKNESS,
                            height=barline_height,
                            center_x=x_pos,
                            center_y=(barline_y1 + barline_y2) / 2
                        ),
                        staff_system=system_id,
                        inferred=True
                    ))
            
            # Infer intermediate barlines if we have noteheads
            if staff_noteheads and len(staff_barlines) < 3:  # If there aren't many barlines detected
                # Sort noteheads by x-position
                staff_noteheads.sort(key=lambda d: d.bbox.center_x)
                
                # Find gaps in the horizontal distribution of noteheads
                # These gaps might indicate measure boundaries
                if len(staff_noteheads) > 5:  # Need enough noteheads to find patterns
                    x_positions = [nh.bbox.center_x for nh in staff_noteheads]
                    
                    # Calculate distances between consecutive noteheads
                    distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                    avg_distance = sum(distances) / len(distances)
                    std_distance = np.std(distances)
                    
                    # Identify gaps that are significantly larger than average
                    # These might be measure boundaries
                    measure_boundary_threshold = avg_distance + 1.5 * std_distance
                    potential_boundaries = [(x_positions[i] + x_positions[i+1])/2 
                                           for i in range(len(x_positions)-1) 
                                           if distances[i] > measure_boundary_threshold]
                    
                    # Filter boundaries to avoid placing too close to existing barlines
                    min_barline_separation = 100  # Minimum x-distance between barlines
                    existing_x = [b.bbox.center_x for b in staff_barlines]
                    
                    for x_boundary in potential_boundaries:
                        # Skip if too close to existing barlines or clefs
                        if (any(abs(x_boundary - x) < min_barline_separation for x in existing_x) or
                            any(abs(x_boundary - x) < 80 for x in clef_positions)):
                            continue
                        
                        # Create inferred barline
                        inferred_barlines.append(Detection(
                            class_id=CLASS_BARLINE,
                            class_name='barline',
                            confidence=0.70,  # Lower confidence for inferred intermediate barlines
                            bbox=BoundingBox(
                                x1=x_boundary - BARLINE_THICKNESS/2,
                                y1=params['top'],
                                x2=x_boundary + BARLINE_THICKNESS/2,
                                y2=params['bottom'],
                                width=BARLINE_THICKNESS,
                                height=staff_height,
                                center_x=x_boundary,
                                center_y=staff_middle_y
                            ),
                            staff_system=system_id,
                            inferred=True
                        ))
        
        return inferred_barlines
    
    def enhance_detections(self, detections_dict):
        """
        Main method to enhance detections by inferring missing elements
        
        Args:
            detections_dict: List of detection dictionaries
        
        Returns:
            Enhanced list of detections including inferred elements
        """
        # Convert dictionary detections to Detection objects
        detections = [Detection.from_dict(d) for d in detections_dict]
        
        # Assign elements to staff systems
        detections = self.assign_elements_to_staff(detections)
        
        # Group noteheads into chords
        chord_groups = self.group_noteheads_into_chords(detections)
        
        # Infer stems for chords that don't have them
        inferred_stems = self.infer_stems_for_chords(chord_groups)
        
        # Infer barlines
        inferred_barlines = self.infer_barlines(detections)
        
        # Combine original detections with inferred elements
        enhanced_detections = detections + inferred_stems + inferred_barlines
        
        # Convert back to dictionaries
        return [d.to_dict() for d in enhanced_detections]
    
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
                bbox = detection['bbox'] if isinstance(detection, dict) else detection.bbox.to_dict()
                class_id = detection['class_id'] if isinstance(detection, dict) else detection.class_id
                is_inferred = detection.get('inferred', False) if isinstance(detection, dict) else detection.inferred
                
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
            all_x = [det['bbox']['x1'] if isinstance(det, dict) else det.bbox.x1 for det in detections]
            all_x += [det['bbox']['x2'] if isinstance(det, dict) else det.bbox.x2 for det in detections]
            all_y = [det['bbox']['y1'] if isinstance(det, dict) else det.bbox.y1 for det in detections]
            all_y += [det['bbox']['y2'] if isinstance(det, dict) else det.bbox.y2 for det in detections]
            
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

def main(staff_lines_path=STAFF_LINES_PATH, object_detections_path=OBJECT_DETECTIONS_PATH, output_dir=None):
    """
    Main function to process staff line and object detection data
    
    Args:
        staff_lines_path: Path to staff lines JSON file
        object_detections_path: Path to object detections JSON file
        output_dir: Directory to save results (defaults to same directory as input)
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
    enhanced_detections = enhancer.enhance_detections(detections)
    
    # Count original and inferred objects
    original_count = len(detections)
    enhanced_count = len(enhanced_detections)
    inferred_count = sum(1 for det in enhanced_detections if isinstance(det, dict) and det.get("inferred", False))
    
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
        fig = enhancer.visualize_results(detections, enhanced_detections)
        
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
    parser.add_argument("--staff-lines", default=STAFF_LINES_PATH, help="Path to staff lines JSON file")
    parser.add_argument("--detections", default=OBJECT_DETECTIONS_PATH, help="Path to object detections JSON file")
    parser.add_argument("--output-dir", help="Directory to save enhanced detections (defaults to same as input)")
    
    args = parser.parse_args()
    
    main(args.staff_lines, args.detections, args.output_dir)
    
    
# python /homes/es314/omr-objdet-benchmark/scripts/stem_barline_det.py \
#     --staff-lines /homes/es314/omr-objdet-benchmark/scripts/encoding/results/staff_lines/Accidentals-004_staff_lines.json \
#     --detections /homes/es314/omr-objdet-benchmark/scripts/encoding/results/object_detections/Accidentals-004_detections.json \
#     --output-dir /homes/es314/omr-objdet-benchmark/stem-barline-detections