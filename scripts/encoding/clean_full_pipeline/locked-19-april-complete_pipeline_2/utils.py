# /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/utils.py
import json
import numpy as np
import csv

def load_csv_detections(filepath):
    """Load detections from a CSV file."""
    detections = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            for key in row:
                try:
                    row[key] = float(row[key]) if key != 'class_name' else row[key]
                except (ValueError, TypeError):
                    pass
            
            # Create bbox dictionary for consistency
            bbox = {
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'width': float(row['width']),
                'height': float(row['height']),
                'center_x': float(row['center_x']),
                'center_y': float(row['center_y'])
            }
            
            detection = {
                'class_id': int(row['class_id']),
                'class_name': row['class_name'],
                'confidence': float(row['confidence']),
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

def load_json(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_typical_staff_spacing(staff_lines):
    """Calculate typical spacing between staff lines across the page."""
    lines_y = sorted([line['bbox']['center_y'] for line in staff_lines['detections']])
    spacings = [lines_y[i+1] - lines_y[i] for i in range(len(lines_y)-1)]
    
    if spacings:
        return np.median(spacings)
    else:
        return 20  # Default spacing if no lines are detected
    
    
class StaffSystem:
    """Class representing a system of staff lines."""
    def __init__(self, id):
        self.id = id
        self.staves = []  # List of staves, each with 5 line y-positions
        self.elements = []
        self.measures = []
        self.clef = None
        self.key_signature = []
        self.time_signature = None
        self.line_spacing = None
        self.lines = {}  # Dictionary mapping line number to y-position
        
    def split_into_staves(self):
        """Split the staff lines into separate staves (e.g., for piano grand staff)."""
        sorted_lines = sorted(self.lines.items(), key=lambda x: x[1])  # top to bottom
        line_positions = [y for _, y in sorted_lines]

        if len(line_positions) % 5 != 0:
            print(f"⚠️ System {self.id} has {len(line_positions)} lines, not divisible by 5.")
            return

        self.staves = []
        for i in range(0, len(line_positions), 5):
            staff_lines = line_positions[i:i+5]
            self.staves.append(staff_lines)

    def add_line(self, line_num, y_position):
        """Add a staff line to this system."""
        self.lines[line_num] = y_position
        
    def calculate_line_spacing(self):
        """Calculate the average spacing between staff lines."""
        if len(self.lines) < 2:
            self.line_spacing = 20  # Default if not enough lines
            return self.line_spacing
        
        # Sort lines by y-position
        sorted_lines = sorted(self.lines.items(), key=lambda x: x[1])
        distances = []
        
        for i in range(1, len(sorted_lines)):
            distances.append(sorted_lines[i][1] - sorted_lines[i-1][1])
        
        if distances:
            self.line_spacing = sum(distances) / len(distances)
        else:
            self.line_spacing = 20  # Default if no distances calculated
            
        return self.line_spacing
        
    def add_element(self, element):
        """Add a music element to this staff system."""
        self.elements.append(element)
        element.staff_system = self
        
        # If it's a clef, set it as the system's clef
        if element.__class__.__name__ == 'Clef':
            self.clef = element
            
    def y_to_pitch(self, y, clef=None):
        """Convert y-coordinate to pitch based on staff position."""
        if not clef:
            clef = self.clef
            
        if not clef or not self.line_spacing or not self.lines:
            return (None, None)
        
        # Get staff lines sorted by y-position (top to bottom)
        sorted_lines = sorted(self.lines.items(), key=lambda x: x[1])
        line_positions = [pos for _, pos in sorted_lines]
        
        # In treble clef:
        # Line 4 (top): E5
        # Line 3: G4 
        # Line 2 (middle): B4
        # Line 1: D4
        # Line 0 (bottom): F4
        
        # Find which line or space the y-position corresponds to
        for i in range(len(line_positions) - 1):
            # Check if between two lines
            if line_positions[i] <= y <= line_positions[i+1]:
                position = i + (y - line_positions[i]) / (line_positions[i+1] - line_positions[i])
                break
        else:
            # Handle notes above or below the staff
            if y < line_positions[0]:  # Above top line
                position = (y - line_positions[0]) / self.line_spacing
            else:  # Below bottom line
                position = len(line_positions) - 1 + (y - line_positions[-1]) / self.line_spacing
        
        # Map position to pitch name (for treble clef)
        if clef.type == 'G':  # Treble clef
            # Each position corresponds to a pitch, starting from top line
            pitches = ['E5', 'D5', 'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4', 'B3', 'A3']
            
            # Calculate index into pitch array (rounded to nearest half-step)
            index = round(position * 2) 
            
            if 0 <= index < len(pitches):
                pitch_name = pitches[index]
                step = pitch_name[0]
                octave = int(pitch_name[1:])
                return (step, octave)
        
        # Default fallback
        return ('C', 4)


class Measure:
    """Class representing a measure in a staff system."""
    def __init__(self, start_x, end_x, staff_system):
        self.start_x = start_x
        self.end_x = end_x
        self.staff_system = staff_system
        self.elements = []
        
    def add_element(self, element):
        """Add a music element to this measure."""
        self.elements.append(element)
        element.measure = self


def find_closest_staff(y, staves):
    """Return the index and line positions of the staff closest to the y-coordinate."""
    min_dist = float('inf')
    closest_idx = None
    for i, lines in enumerate(staves):
        center_y = sum(lines) / len(lines)
        dist = abs(center_y - y)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx, staves[closest_idx]
