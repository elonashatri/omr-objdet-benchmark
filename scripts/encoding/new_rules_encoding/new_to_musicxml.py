import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, namedtuple
import xml.etree.ElementTree as ET
import math

class MusicElement:
    """Base class for all music elements."""
    def __init__(self, class_id, class_name, confidence, bbox):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.x = bbox['center_x']
        self.y = bbox['center_y']
        self.width = bbox['width']
        self.height = bbox['height']
        self.staff_system = None
        self.measure = None
        self.pitch = None
        self.duration = None
        self.voice = None

class Note(MusicElement):
    """Class representing a note element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.accidental = None
        self.stem_direction = None
        self.beams = []
        self.flag = None
        self.ledger_lines = []
        self.is_chord_member = False
        self.chord = None
        self.duration_type = 'quarter'  # Default
        self.duration = 1.0  # Default quarter note (in quarter notes)
        self.duration_dots = 0
        self.step = None
        self.octave = None
        self.alter = 0  # 0 = natural, 1 = sharp, -1 = flat
        
    def set_pitch(self, step, octave, alter=0):
        """Set the pitch for this note."""
        self.step = step
        self.octave = octave
        self.alter = alter
        self.pitch = f"{step}{octave}"  # Basic string representation
        
    def position_to_xml(self):
        """Generate MusicXML pitch element."""
        pitch = ET.Element('pitch')
        step = ET.SubElement(pitch, 'step')
        step.text = self.step
        
        if self.alter != 0:
            alter = ET.SubElement(pitch, 'alter')
            alter.text = str(self.alter)
            
        octave = ET.SubElement(pitch, 'octave')
        octave.text = str(self.octave)
        
        return pitch

class Rest(MusicElement):
    """Class representing a rest element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.duration = None
        self.duration_type = None
        
        # Set duration based on class name
        if 'restWhole' in class_name:
            self.duration = 4.0
            self.duration_type = 'whole'
        elif 'restHalf' in class_name:
            self.duration = 2.0
            self.duration_type = 'half'
        elif 'restQuarter' in class_name:
            self.duration = 1.0
            self.duration_type = 'quarter'
        elif 'rest8th' in class_name:
            self.duration = 0.5
            self.duration_type = 'eighth'
        elif 'rest16th' in class_name:
            self.duration = 0.25
            self.duration_type = '16th'
        elif 'rest32nd' in class_name:
            self.duration = 0.125
            self.duration_type = '32nd'
        else:
            # Default to quarter rest
            self.duration = 1.0
            self.duration_type = 'quarter'

class Accidental(MusicElement):
    """Class representing an accidental element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.type = None
        self.affected_note = None
        self.is_key_signature = False
        
        # Set the accidental type based on class name
        if 'Sharp' in class_name:
            self.type = 'sharp'
            self.alter = 1
        elif 'Flat' in class_name:
            self.type = 'flat'
            self.alter = -1
        elif 'Natural' in class_name:
            self.type = 'natural'
            self.alter = 0

class Clef(MusicElement):
    """Class representing a clef element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.type = None
        self.line = None
        
        # Set the clef type based on class name
        if 'gClef' in class_name:
            self.type = 'G'
            self.line = 2
        elif 'fClef' in class_name:
            self.type = 'F'
            self.line = 4
        elif 'cClef' in class_name:
            self.type = 'C'
            self.line = 3

class Barline(MusicElement):
    """Class representing a barline element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.bar_style = 'regular'  # Default

class Beam(MusicElement):
    """Class representing a beam element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.connected_notes = []
        self.level = 1  # Default beam level (can be 1, 2, 3, etc. for 8th, 16th, 32nd...)

class Flag(MusicElement):
    """Class representing a flag element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        self.connected_note = None
        self.direction = 'up' if 'Up' in class_name else 'down'
        
        # Set the flag type based on class name
        if 'flag8th' in class_name:
            self.level = 1  # 8th flag
        elif 'flag16th' in class_name:
            self.level = 2  # 16th flag
        elif 'flag32nd' in class_name:
            self.level = 3  # 32nd flag
        else:
            self.level = 1  # Default to 8th flag

class TimeSignatureElement(MusicElement):
    """Class representing a component of a time signature."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        
        # Extract value from class name
        if 'timeSig1' in class_name:
            self.value = 1
        elif 'timeSig2' in class_name:
            self.value = 2
        elif 'timeSig3' in class_name:
            self.value = 3
        elif 'timeSig4' in class_name:
            self.value = 4
        elif 'timeSig5' in class_name:
            self.value = 5
        elif 'timeSig6' in class_name:
            self.value = 6
        elif 'timeSig7' in class_name:
            self.value = 7
        elif 'timeSig8' in class_name:
            self.value = 8
        elif 'timeSig9' in class_name:
            self.value = 9
        elif 'timeSig12' in class_name:
            self.value = 12
        elif 'timeSig16' in class_name:
            self.value = 16
        else:
            self.value = 4  # Default

class StaffSystem:
    """Class representing a system of staff lines."""
    def __init__(self, id):
        self.id = id
        self.lines = {} 
        self.staves = []  # List of staves, each with 5 line y-positions
        self.elements = []
        self.measures = []
        self.clef = None
        self.key_signature = []
        self.time_signature = None
        self.line_spacing = None
        
        
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
        
        self.line_spacing = sum(distances) / len(distances)
        return self.line_spacing
        
    def add_element(self, element):
        """Add a music element to this staff system."""
        self.elements.append(element)
        element.staff_system = self
        
        # If it's a clef, set it as the system's clef
        if isinstance(element, Clef):
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
    # def y_to_pitch(self, y, clef=None):
    #     """
    #     Convert a y-coordinate to a pitch based on staff position.
        
    #     Args:
    #         y: Y-coordinate
    #         clef: Optional clef to use (otherwise uses system's default)
            
    #     Returns:
    #         Tuple of (step, octave)
    #     """
    #     if not clef:
    #         clef = self.clef
            
    #     if not clef or not self.line_spacing or not self.lines:
    #         return (None, None)
        
    #     # Sort staff lines by y-position (top to bottom)
    #     sorted_lines = sorted(self.lines.items(), key=lambda x: x[1])
    #     line_positions = [pos for _, pos in sorted_lines]
        
    #     # Get the middle line position
    #     middle_line_idx = len(line_positions) // 2
    #     middle_line_y = line_positions[middle_line_idx]
        
    #     # Calculate steps from middle line
    #     steps_from_middle = (y - middle_line_y) / (self.line_spacing / 2)
    #     steps_from_middle = round(steps_from_middle)
        
    #     # Convert steps to pitch based on clef
    #     if clef.type == 'G':  # Treble clef
    #         # On a treble clef, the middle line is B4
    #         reference_step = 'B'
    #         reference_octave = 4
    #     elif clef.type == 'F':  # Bass clef
    #         # On a bass clef, the middle line is D3
    #         reference_step = 'D'
    #         reference_octave = 3
    #     elif clef.type == 'C':  # C clef
    #         # Position depends on the line the clef is placed
    #         if clef.line == 3:  # Alto clef
    #             reference_step = 'C'
    #             reference_octave = 4
    #         elif clef.line == 4:  # Tenor clef
    #             reference_step = 'A'
    #             reference_octave = 3
    #         else:
    #             reference_step = 'C'
    #             reference_octave = 4
    #     else:
    #         # Default to treble clef
    #         reference_step = 'B'
    #         reference_octave = 4
        
    #     # Step sequence
    #     step_sequence = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        
    #     # Find position in step sequence
    #     start_idx = step_sequence.index(reference_step)
        
    #     # Calculate new position
    #     steps_in_diatonic_scale = 7  # Number of steps in a diatonic scale
    #     new_idx = (start_idx - steps_from_middle) % steps_in_diatonic_scale
    #     octave_shift = (start_idx - steps_from_middle) // steps_in_diatonic_scale
        
    #     if steps_from_middle > 0 and (start_idx - steps_from_middle) % steps_in_diatonic_scale != 0:
    #         octave_shift -= 1
        
    #     new_step = step_sequence[new_idx]
    #     new_octave = reference_octave + octave_shift
        
    #     return (new_step, new_octave)

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

class KeySignature:
    """Class representing a key signature."""
    def __init__(self, accidentals=None):
        self.accidentals = accidentals if accidentals else []
        self.fifths = self.calculate_fifths()
        
    def calculate_fifths(self):
        """Calculate the number of sharps/flats in this key signature."""
        # Count accidentals
        num_sharps = sum(1 for acc in self.accidentals if acc.type == 'sharp')
        num_flats = sum(1 for acc in self.accidentals if acc.type == 'flat')
        
        # Key signatures are typically all sharps or all flats
        if num_sharps > 0 and num_flats == 0:
            return num_sharps
        elif num_flats > 0 and num_sharps == 0:
            return -num_flats
        else:
            return 0  # C major / A minor

class TimeSignature:
    """Class representing a time signature."""
    def __init__(self, beats=4, beat_type=4):
        self.beats = beats
        self.beat_type = beat_type

class OMRProcessor:
    """Main processor for OMR conversion to MusicXML."""
    def __init__(self, detection_path=None, staff_lines_path=None, 
                 detection_data=None, staff_lines_data=None):
        self.detection_path = detection_path
        self.staff_lines_path = staff_lines_path
        
        # Load data
        self.detections = None
        if detection_path:
            if detection_path.endswith('.csv'):
                self.detections = self.load_csv_detections(detection_path)
            else:
                self.detections = self.load_json_detections(detection_path)
        elif detection_data:
            self.detections = detection_data
            
        self.staff_lines = None
        if staff_lines_path:
            self.staff_lines = self.load_json(staff_lines_path)
        elif staff_lines_data:
            self.staff_lines = staff_lines_data
            
        # Initialize data structures
        self.staff_systems = []
        self.notes = []
        self.rests = []
        self.accidentals = []
        self.clefs = []
        self.barlines = []
        self.beams = []
        self.flags = []
        self.time_signature_elements = []
        self.key_signatures = []
        self.measures = []
        # calculate typical spacing if staff_lines are available
        if self.staff_lines:
            self.calculate_typical_staff_spacing()
    
    
    def calculate_typical_staff_spacing(self):
        """Calculate typical spacing between staff lines across the page."""
        lines_y = sorted([line['bbox']['center_y'] for line in self.staff_lines['detections']])
        spacings = [lines_y[i+1] - lines_y[i] for i in range(len(lines_y)-1)]
        self.typical_staff_spacing = np.median(spacings)
        
    def load_csv_detections(self, filepath):
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
    
    def load_json_detections(self, filepath):
        """Load detections from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'detections' in data:
            return data['detections']
        return data
    
    def load_json(self, filepath):
        """Load JSON data from a file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def process(self):
        """Process the detections and staff lines to generate MusicXML."""
        if not self.detections or not self.staff_lines:
            print("Missing required data. Cannot process.")
            return None
        
        # self.identify_staff_systems()
        self.identify_staff_systems(mode="auto")  # or "piano" if you have a flag
        # self.identify_staff_systems_piano()
        self.process_detected_objects()
        self.assign_to_staff_systems()
        self.analyze_possible_time_signatures()
        self.validate_staff_lines()
        self.calculate_note_pitches()
        self.connect_flags_to_notes()
        self.connect_beams_to_notes()
        self.connect_accidentals_to_notes()
        self.interpret_time_signatures()
        self.identify_key_signatures()
        self.group_notes_into_chords()
        self.infer_barlines()
        self.identify_measures()  # ONLY CALLED ONCE HERE
        self.calculate_note_durations()
        return self.generate_musicxml()
    
    def identify_staff_systems(self, mode="auto"):
        
        if mode == "piano":
            return self.identify_staff_systems_piano()
        else:
            return self.identify_staff_systems_auto()
        
        
    def identify_staff_systems_piano(self):
        """
        Specifically group staff lines into piano systems (pairs of treble and bass staves).
        Assumes staff lines are sorted vertically from top to bottom.
        """
        # Step 1: Sort all staff lines vertically
        all_staff_lines = sorted(
            self.staff_lines['detections'], 
            key=lambda x: x['bbox']['center_y']
        )

        # Step 2: Group lines into individual staves (5 lines per staff)
        staves = []
        current_staff_lines = []

        for line in all_staff_lines:
            if not current_staff_lines:
                current_staff_lines.append(line)
            else:
                prev_line = current_staff_lines[-1]
                vertical_gap = line['bbox']['center_y'] - prev_line['bbox']['center_y']
                
                # Check typical spacing to identify if line belongs to the same staff
                if vertical_gap < self.typical_staff_spacing * 2:
                    current_staff_lines.append(line)
                else:
                    # Start a new staff
                    if len(current_staff_lines) == 5:
                        staves.append(current_staff_lines)
                    else:
                        print(f"Warning: Incorrect number of lines ({len(current_staff_lines)}) in staff.")
                    current_staff_lines = [line]

        # Check last group
        if len(current_staff_lines) == 5:
            staves.append(current_staff_lines)
        else:
            print(f"Warning: Incorrect number of lines ({len(current_staff_lines)}) in last staff.")

        # Step 3: Group staves into piano systems (every 2 staves)
        self.staff_systems.clear()
        for i in range(0, len(staves), 2):
            if i+1 >= len(staves):
                print("Warning: Odd number of staves detected; last staff without pair.")
                break

            system = StaffSystem(i // 2)

            # Combine lines from two staves (upper: treble, lower: bass)
            treble_staff = staves[i]
            bass_staff = staves[i + 1]

            # Assign lines (line numbers top-to-bottom for each staff separately)
            for idx, line in enumerate(sorted(treble_staff, key=lambda l: l['bbox']['center_y'])):
                # Treble staff lines numbered 0-4 (bottom-to-top convention)
                system.add_line(4 - idx, line['bbox']['center_y'])

            for idx, line in enumerate(sorted(bass_staff, key=lambda l: l['bbox']['center_y'])):
                # Bass staff lines numbered 5-9 (bottom-to-top convention for bass staff)
                system.add_line(9 - idx, line['bbox']['center_y'])

            # Calculate line spacing explicitly for this system
            system.calculate_line_spacing()

            # Append system to overall list
            self.staff_systems.append(system)

        print(f"Identified {len(self.staff_systems)} piano staff systems successfully.")

    def identify_staff_systems_auto(self):
        """
        Identify systems automatically by grouping staves based on vertical distance.
        Each system may contain 1 or more staves depending on spacing.
        """
        # Step 1: Sort lines by vertical position
        all_lines = sorted(self.staff_lines['detections'], key=lambda l: l['bbox']['center_y'])
        
        # Step 2: Group lines into staves (sets of 5)
        staves = []
        current_staff = []
        for line in all_lines:
            if not current_staff:
                current_staff.append(line)
            else:
                prev_line = current_staff[-1]
                gap = line['bbox']['center_y'] - prev_line['bbox']['center_y']
                if gap < self.typical_staff_spacing * 2:
                    current_staff.append(line)
                else:
                    if len(current_staff) == 5:
                        staves.append(current_staff)
                    else:
                        print(f"Warning: skipping incomplete staff with {len(current_staff)} lines")
                    current_staff = [line]
        if len(current_staff) == 5:
            staves.append(current_staff)

        # Step 3: Group staves into systems based on vertical spacing
        self.staff_systems = []
        current_system = []
        for i, staff in enumerate(staves):
            if not current_system:
                current_system.append(staff)
            else:
                prev_staff_center = np.mean([l['bbox']['center_y'] for l in current_system[-1]])
                curr_staff_center = np.mean([l['bbox']['center_y'] for l in staff])
                spacing = curr_staff_center - prev_staff_center
                
                if spacing < self.typical_staff_spacing * 6:  # tuning this threshold is key
                    current_system.append(staff)
                else:
                    # Commit current system
                    system = self._build_system_from_staves(current_system, len(self.staff_systems))
                    self.staff_systems.append(system)
                    current_system = [staff]
        
        # Add last system
        if current_system:
            system = self._build_system_from_staves(current_system, len(self.staff_systems))
            self.staff_systems.append(system)

        print(f"[Auto] Identified {len(self.staff_systems)} systems from {len(staves)} staves.")
        
    def split_system_into_staves(system):
        sorted_lines = sorted(system.lines.items(), key=lambda x: x[1])  # top to bottom
        line_positions = [pos for _, pos in sorted_lines]

        if len(line_positions) % 5 != 0:
            print(f"⚠️ System {system.id} has {len(line_positions)} lines, which is not a multiple of 5. Skipping.")
            return

        system.staves = []
        for i in range(0, len(line_positions), 5):
            staff_lines = line_positions[i:i+5]
            system.staves.append(staff_lines)
  
    def _build_system_from_staves(self, staves, system_id):
        system = StaffSystem(system_id)
        line_number = 0
        for staff in staves:
            sorted_lines = sorted(staff, key=lambda l: l['bbox']['center_y'])
            for i, line in enumerate(sorted_lines):
                system.add_line(line_number, line['bbox']['center_y'])
                line_number += 1
        system.calculate_line_spacing()
        return system

    # def identify_staff_systems(self):
    #     """Separate detected staff lines into distinct systems based on vertical spacing."""
    #     all_staff_lines = sorted(self.staff_lines['detections'], key=lambda x: x['bbox']['center_y'])
        
    #     current_system_lines = []
    #     systems = []
        
    #     for i in range(len(all_staff_lines)):
    #         current_line = all_staff_lines[i]
            
    #         if not current_system_lines:
    #             current_system_lines.append(current_line)
    #         else:
    #             prev_line = current_system_lines[-1]
    #             vertical_distance = current_line['bbox']['center_y'] - prev_line['bbox']['center_y']
                
    #             if vertical_distance > self.typical_staff_spacing * 3:
    #                 systems.append(current_system_lines)
    #                 current_system_lines = [current_line]
    #             else:
    #                 current_system_lines.append(current_line)
        
    #     if current_system_lines:
    #         systems.append(current_system_lines)
        
    #     self.staff_systems = []
    #     for i, system_lines in enumerate(systems):
    #         sys = StaffSystem(i)
    #         sorted_sys_lines = sorted(system_lines, key=lambda x: x['bbox']['center_y'])
    #         for j, line in enumerate(sorted_sys_lines):
    #             line_num = len(sorted_sys_lines) - j - 1
    #             sys.add_line(line_num, line['bbox']['center_y'])
            
    #         # Explicitly calculate line spacing here:
    #         sys.calculate_line_spacing()
            
    #         self.staff_systems.append(sys)

        
    def process_staff_systems(self):
        """Process the staff systems from the staff lines data."""
        if 'staff_systems' in self.staff_lines:
            for system_data in self.staff_lines['staff_systems']:
                system = StaffSystem(system_data['id'])
                
                # Find all staff lines for this system
                staff_lines = [line for line in self.staff_lines['detections'] 
                              if line.get('staff_system') == system_data['id']]
                
                # Add staff lines to system
                for line in staff_lines:
                    system.add_line(line['line_number'], line['bbox']['center_y'])
                
                # Calculate staff line spacing
                system.calculate_line_spacing()
                
                self.staff_systems.append(system)
    
    def process_detected_objects(self):
        """Process the detected objects into music elements."""
        if isinstance(self.detections, list):
            detections_list = self.detections
        elif 'detections' in self.detections:
            detections_list = self.detections['detections']
        else:
            detections_list = []
        
        self.notes.clear()
        self.rests.clear()
        self.accidentals.clear()
        self.clefs.clear()
        self.barlines.clear()
        self.beams.clear()
        self.flags.clear()
        self.time_signature_elements.clear()
        
        for det in detections_list:
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get the bounding box
            if 'bbox' in det:
                bbox = det['bbox']
            else:
                # Create bbox from individual coordinates
                bbox = {
                    'x1': det['x1'],
                    'y1': det['y1'],
                    'x2': det['x2'],
                    'y2': det['y2'],
                    'width': det['width'],
                    'height': det['height'],
                    'center_x': det['center_x'],
                    'center_y': det['center_y']
                }
            
            # Process based on class
            if 'notehead' in class_name.lower():
                note = Note(class_id, class_name, confidence, bbox)
                self.notes.append(note)
                
            elif 'rest' in class_name.lower():
                rest = Rest(class_id, class_name, confidence, bbox)
                self.rests.append(rest)
                
            elif 'accidental' in class_name.lower():
                accidental = Accidental(class_id, class_name, confidence, bbox)
                self.accidentals.append(accidental)
                
            elif 'Clef' in class_name or 'clef' in class_name.lower():
                clef = Clef(class_id, class_name, confidence, bbox)
                self.clefs.append(clef)
                
            elif class_name == 'barline':
                barline = Barline(class_id, class_name, confidence, bbox)
                self.barlines.append(barline)
                
            elif 'beam' in class_name.lower():
                beam = Beam(class_id, class_name, confidence, bbox)
                self.beams.append(beam)
                
            elif 'flag' in class_name.lower():
                flag = Flag(class_id, class_name, confidence, bbox)
                self.flags.append(flag)
                
            elif 'timeSig' in class_name:
                time_sig_elem = TimeSignatureElement(class_id, class_name, confidence, bbox)
                self.time_signature_elements.append(time_sig_elem)
        print(f"Notes processed: {len(self.notes)}")
        
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

    
    def validate_staff_lines(self):
        """Validate detected staff lines for consistency."""
        for system in self.staff_systems:
            # Check if we have the expected number of lines (typically 5)
            if len(system.lines) != 5:
                print(f"Warning: Staff system {system.id} has {len(system.lines)} lines, expected 5")
            
            # Check line spacing consistency
            line_positions = sorted(system.lines.values())
            if len(line_positions) >= 2:
                spacings = [line_positions[i+1] - line_positions[i] for i in range(len(line_positions)-1)]
                avg_spacing = sum(spacings) / len(spacings)
                max_spacing = max(spacings)
                min_spacing = min(spacings)
                
                # If max deviation is > 20% of average, warn
                if (max_spacing - min_spacing) / avg_spacing > 0.2:
                    print(f"Warning: Inconsistent staff line spacing in system {system.id}. " 
                          f"Min: {min_spacing:.2f}, Max: {max_spacing:.2f}, Avg: {avg_spacing:.2f}")
    
    def assign_to_staff_systems(self):
        """Assign music elements to staff systems."""
        all_elements = (self.notes + self.rests + self.accidentals + self.clefs + 
                       self.barlines + self.beams + self.flags + self.time_signature_elements)
        
        for element in all_elements:
            # Find the closest staff system
            closest_system = None
            min_distance = float('inf')
            
            for system in self.staff_systems:
                # Get staff line positions
                line_positions = list(system.lines.values())
                if not line_positions:
                    continue
                    
                # Calculate vertical bounds of the staff
                top_line = min(line_positions)
                bottom_line = max(line_positions)
                staff_height = bottom_line - top_line
                
                # For barlines and clefs, check if they span the staff
                if isinstance(element, (Barline, Clef)):
                    element_top = element.bbox['y1']
                    element_bottom = element.bbox['y2']
                    
                    # Check if element overlaps significantly with staff
                    overlap = min(element_bottom, bottom_line) - max(element_top, top_line)
                    if overlap > 0 and overlap > staff_height * 0.5:
                        # Element spans the staff, so assign it
                        distance = 0
                    else:
                        # Calculate distance to nearest staff line
                        if element.y < top_line:
                            distance = top_line - element.y
                        elif element.y > bottom_line:
                            distance = element.y - bottom_line
                        else:
                            distance = 0
                else:
                    # For other elements, check if they're within or near the staff
                    extended_top = top_line - staff_height * 0.5
                    extended_bottom = bottom_line + staff_height * 0.5
                    
                    if extended_top <= element.y <= extended_bottom:
                        # Element is within extended staff range
                        if top_line <= element.y <= bottom_line:
                            distance = 0  # Within actual staff
                        else:
                            # Within extended range but outside actual staff
                            distance = min(abs(element.y - top_line), abs(element.y - bottom_line))
                    else:
                        # Outside extended range
                        distance = min(abs(element.y - extended_top), abs(element.y - extended_bottom))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_system = system
            
            # Assign to the closest staff system
            if closest_system:
                closest_system.add_element(element)
    

    def calculate_note_pitches(self):
        """Calculate pitches for all notes based on staff position."""
        for note in self.notes:
            if not note.staff_system:
                continue
                
            # Use the new direct calculation method instead of y_to_pitch
            step, octave = self.calculate_note_pitch(note)
            
            if step and octave:
                note.set_pitch(step, octave)
                
                # If the note has an accidental, apply the alteration
                if note.accidental:
                    note.alter = note.accidental.alter
                    

    def calculate_note_pitch(self, note):
        """Calculate pitch using correct staff in multi-staff systems."""
        system = note.staff_system
        if not system or not system.staves:
            return ('C', 4)  # fallback

        staff_idx, staff_lines = find_closest_staff(note.y, system.staves)

        # Sort just in case (top to bottom)
        staff_lines = sorted(staff_lines)

        # Calculate line spacing
        spacing = sum(staff_lines[i+1] - staff_lines[i] for i in range(4)) / 4

        # Interpolate space positions
        space_positions = [(staff_lines[i] + staff_lines[i+1]) / 2 for i in range(4)]
        space_positions = [staff_lines[0] - spacing] + space_positions + [staff_lines[-1] + spacing]

        line_pitches = [('E', 4), ('G', 4), ('B', 4), ('D', 5), ('F', 5)]
        space_pitches = [('F', 4), ('A', 4), ('C', 5), ('E', 5), ('G', 5), ('D', 4)]

        # Closest position
        closest_line = min(range(5), key=lambda i: abs(staff_lines[i] - note.y))
        closest_space = min(range(6), key=lambda i: abs(space_positions[i] - note.y))

        if abs(staff_lines[closest_line] - note.y) <= abs(space_positions[closest_space] - note.y):
            return line_pitches[closest_line]
        else:
            return space_pitches[closest_space]

    
    def connect_flags_to_notes(self):
        """Connect flags to the notes they affect."""
        for flag in self.flags:
            if not flag.staff_system:
                continue
                
            closest_note = None
            min_distance = float('inf')
            
            # Find notes in the same staff system
            system_notes = [n for n in self.notes if n.staff_system == flag.staff_system]
            
            for note in system_notes:
                # Calculate horizontal distance
                dx = abs(note.x - flag.x)
                
                # Flag should be close horizontally to the note
                if dx > note.width * 2:  # Arbitrary threshold
                    continue
                    
                # Calculate vertical distance (flag should be above/below note)
                dy = abs(note.y - flag.y)
                
                # Calculate distance metric (weighted)
                distance = dx * 2 + dy
                
                if distance < min_distance:
                    min_distance = distance
                    closest_note = note
            
            # Connect flag to note
            if closest_note:
                flag.connected_note = closest_note
                closest_note.flag = flag
                
                # Set duration based on flag level
                if flag.level == 1:  # 8th flag
                    closest_note.duration = 0.5
                    closest_note.duration_type = 'eighth'
                elif flag.level == 2:  # 16th flag
                    closest_note.duration = 0.25
                    closest_note.duration_type = '16th'
                elif flag.level == 3:  # 32nd flag
                    closest_note.duration = 0.125
                    closest_note.duration_type = '32nd'
    


    def connect_beams_to_notes(self):
        """Connect beams to notes using a robust grouping algorithm."""
        for beam in self.beams:
            if not beam.staff_system:
                continue
                
            # Get all notes in this staff system
            system_notes = [n for n in self.notes if n.staff_system == beam.staff_system]
            
            # Find potential candidates by checking for notes that are "near" this beam
            # horizontally (tolerance: half the beam width)
            beam_center_x = beam.x
            horizontal_tolerance = beam.width * 0.75  # Much more generous
            
            candidate_notes = []
            for note in system_notes:
                # Check horizontal proximity to beam center
                if abs(note.x - beam_center_x) <= horizontal_tolerance:
                    # Add to candidates with distance information
                    horizontal_dist = abs(note.x - beam_center_x)
                    vertical_dist = abs(note.y - beam.y)
                    distance = horizontal_dist + vertical_dist * 0.5  # Weight horizontal more
                    
                    candidate_notes.append((note, distance))
            
            # Take the closest notes (typically 2 for a standard beam)
            # Sort by distance and take the closest
            candidate_notes.sort(key=lambda x: x[1])
            
            # Connect at least 2 notes if we have them (typical beam connects 2+ notes)
            notes_to_connect = [n for n, _ in candidate_notes[:max(2, len(candidate_notes))]]
            
            # Secondary check: ensure the notes are reasonably aligned horizontally
            # (should be arranged left to right without big gaps)
            if notes_to_connect:
                notes_to_connect.sort(key=lambda n: n.x)
                
                # Connect the notes to the beam
                for note in notes_to_connect:
                    beam.connected_notes.append(note)
                    note.beams.append(beam)
                    
                    # Set duration based on beam count
                    if len(note.beams) == 1:
                        note.duration = 0.5  # Eighth note
                        note.duration_type = 'eighth'
                    elif len(note.beams) == 2:
                        note.duration = 0.25  # 16th note
                        note.duration_type = '16th'
                    elif len(note.beams) >= 3:
                        note.duration = 0.125  # 32nd note
                        note.duration_type = '32nd'
                
    def connect_accidentals_to_notes(self):
        """Connect accidentals to the notes they affect."""
        for acc in self.accidentals:
            if not acc.staff_system:
                continue
                
            # Find notes in the same staff system
            system_notes = [n for n in self.notes if n.staff_system == acc.staff_system]
            
            closest_note = None
            min_distance = float('inf')
            
            for note in system_notes:
                # Accidentals typically appear to the left of the note they affect
                if note.x <= acc.x:
                    continue
                    
                # Calculate horizontal and vertical distance
                dx = note.x - acc.x
                dy = abs(note.y - acc.y)
                
                # Weight horizontal distance more than vertical
                distance = dx + 2 * dy
                
                # Vertical tolerance - accidentals should be close to the note vertically
                vertical_tolerance = acc.staff_system.line_spacing * 1.5
                if dy > vertical_tolerance:
                    continue
                    
                if distance < min_distance:
                    min_distance = distance
                    closest_note = note
            
            # Connect the accidental to the note
            if closest_note:
                acc.affected_note = closest_note
                closest_note.accidental = acc
                
                # Set the note's alteration based on the accidental
                if closest_note.step:
                    closest_note.alter = acc.alter
    
    def interpret_time_signatures(self):
        """Interpret time signature elements into complete time signatures."""
        for system in self.staff_systems:
            # Get time signature elements for this system
            time_sig_elements = [e for e in system.elements if isinstance(e, TimeSignatureElement)]
            
            if len(time_sig_elements) >= 2:
                # Sort by vertical position (numerator on top)
                time_sig_elements.sort(key=lambda e: e.y)
                
                # Create time signature with numerator/denominator
                numerator = time_sig_elements[0].value
                denominator = time_sig_elements[1].value
                
                system.time_signature = TimeSignature(beats=numerator, beat_type=denominator)
    
    def identify_key_signatures(self):
        """Identify key signatures based on groups of accidentals at the beginning of staves."""
        for system in self.staff_systems:
            # Find the clef (which marks the start of the staff)
            clefs = [e for e in system.elements if isinstance(e, Clef)]
            if not clefs:
                continue
                
            # Sort clefs by x-position
            clefs.sort(key=lambda c: c.x)
            clef = clefs[0]  # Use leftmost clef
            
            # Find accidentals that could be part of a key signature
            # (appearing right after the clef and before other elements)
            potential_key_accidentals = []
            
            # Find the first non-accidental element after the clef (excluding time signatures)
            first_other_x = float('inf')
            for elem in system.elements:
                if (not isinstance(elem, (Accidental, Clef, TimeSignatureElement)) and
                    elem.x > clef.x):
                    first_other_x = min(first_other_x, elem.x)
            
            # Collect accidentals between clef and first other element
            for acc in [e for e in system.elements if isinstance(e, Accidental)]:
                if clef.x < acc.x < first_other_x:
                    potential_key_accidentals.append(acc)
                    
            # Mark these as key signature accidentals
            for acc in potential_key_accidentals:
                acc.is_key_signature = True
                
            # Create key signature if there are accidentals
            if potential_key_accidentals:
                key_sig = KeySignature(potential_key_accidentals)
                system.key_signature = potential_key_accidentals
                self.key_signatures.append(key_sig)
    
    def group_notes_into_chords(self):
        """Group vertically aligned notes into chords."""
        # Process each staff system separately
        for system in self.staff_systems:
            system_notes = [n for n in system.elements if isinstance(n, Note)]
            
            # Sort notes by x-position
            system_notes.sort(key=lambda n: n.x)
            
            # Group notes by x-position
            x_tolerance = system.line_spacing / 2  # Notes within this distance are aligned
            
            # Group notes by x-position
            x_groups = []
            current_group = []
            
            for note in system_notes:
                if not current_group:
                    current_group.append(note)
                else:
                    ref_x = current_group[0].x
                    if abs(note.x - ref_x) <= x_tolerance:
                        current_group.append(note)
                    else:
                        if len(current_group) > 1:
                            x_groups.append(current_group)
                        current_group = [note]
            
            # Add the last group
            if len(current_group) > 1:
                x_groups.append(current_group)
            
            # For each group of vertically aligned notes, mark them as chord members
            for group in x_groups:
                if len(group) > 1:
                    # Sort by y-position (top to bottom)
                    group.sort(key=lambda n: n.y)
                    
                    for note in group:
                        note.is_chord_member = True
                        note.chord = group
    
    def identify_measures(self):
        """Identify measures using barlines."""
        # Clear existing measures before creating new ones
        self.measures.clear()
        for system in self.staff_systems:
            # Get barlines for this system
            barlines = [b for b in system.elements if isinstance(b, Barline)]
            
            # Sort barlines by x-position
            barlines.sort(key=lambda b: b.x)
            
            if not barlines:
                # If no barlines, create one measure spanning the entire system
                # Estimate system bounds
                elements_x = [e.x for e in system.elements]
                start_x = min(elements_x) if elements_x else 0
                end_x = max(elements_x) if elements_x else 1000
                
                measure = Measure(start_x, end_x, system)
                system.measures.append(measure)
                self.measures.append(measure)
                
                # Add all elements to this measure
                for elem in system.elements:
                    measure.add_element(elem)
            else:
                # Create measures between barlines
                system_elements = system.elements.copy()
                
                # Determine the start of the first measure
                # (use leftmost element or a default value)
                start_x = min([e.x for e in system_elements]) if system_elements else 0
                
                # Create measures
                for i, barline in enumerate(barlines):
                    end_x = barline.x
                    
                    # Create measure
                    measure = Measure(start_x, end_x, system)
                    system.measures.append(measure)
                    self.measures.append(measure)
                    
                    # Add elements that fall within this measure
                    for elem in system_elements:
                        if start_x <= elem.x < end_x:
                            measure.add_element(elem)
                    
                    # Update start_x for next measure
                    start_x = end_x
                
                # If there are elements after the last barline, create a final measure
                if system_elements:
                    max_x = max([e.x for e in system_elements])
                    if max_x > barlines[-1].x:
                        measure = Measure(barlines[-1].x, max_x + 100, system)  # Add padding
                        system.measures.append(measure)
                        self.measures.append(measure)
                        
                        # Add remaining elements
                        for elem in system_elements:
                            if elem.x >= barlines[-1].x:
                                measure.add_element(elem)
    
    
    def calculate_note_durations(self):
        """Calculate durations for notes based on beams and flags."""
        # Note durations are already set when connecting beams and flags,
        # but this method can be used for additional duration calculations
        
        # Set default duration for notes without beams or flags
        for note in self.notes:
            if not note.beams and not note.flag:
                # Default to quarter note
                note.duration = 1.0
                note.duration_type = 'quarter'
                
    
    def infer_barlines(self):
        """
        Analyze cumulative durations and insert barlines according to time signature.
        Should be called after note durations are calculated but before generating MusicXML.
        """
        for system in self.staff_systems:
            # Skip if we already have barlines
            existing_barlines = [e for e in system.elements if isinstance(e, Barline)]
            if existing_barlines:
                print(f"System {system.id} already has {len(existing_barlines)} barlines. Skipping inference.")
                continue
                
            # Get time signature
            time_sig = system.time_signature
            if not time_sig:
                # Default to 4/4 if no time signature found
                print(f"Warning: No time signature found for system {system.id}. Using 4/4 as default.")
                time_sig = TimeSignature(beats=4, beat_type=4)
                system.time_signature = time_sig
            
            # Calculate full measure duration in quarter notes
            measure_duration = time_sig.beats * (4 / time_sig.beat_type)
            
            # Get all notes and rests, sorted by x-position
            elements = [e for e in system.elements 
                    if isinstance(e, (Note, Rest))]
            elements.sort(key=lambda e: e.x)
            
            if not elements:
                print(f"No notes or rests found in system {system.id}. Skipping barline inference.")
                continue
            
            # Track cumulative duration and barline positions
            cumulative_duration = 0
            barline_positions = []
            
            # Get the x range of the staff for positioning barlines
            min_x = min(e.x for e in system.elements)
            max_x = max(e.x for e in system.elements)
            staff_width = max_x - min_x
            
            # Compute position for each element where we reach a full measure
            for i, elem in enumerate(elements):
                # Add current element's duration
                elem_duration = elem.duration if hasattr(elem, 'duration') and elem.duration else 1.0
                cumulative_duration += elem_duration
                
                # Check if we reached a full measure
                if cumulative_duration >= measure_duration:
                    # Determine x position for the barline
                    # If we're exactly at a measure boundary, put it after this element
                    # If we've gone past a measure boundary, estimate where the boundary should be
                    excess = cumulative_duration - measure_duration
                    ratio = 1.0
                    
                    if excess > 0 and i < len(elements) - 1:
                        # Interpolate position based on excess duration
                        next_elem = elements[i + 1]
                        total_elem_duration = elem_duration
                        position_ratio = (elem_duration - excess) / total_elem_duration
                        barline_x = elem.x + (next_elem.x - elem.x) * position_ratio
                    else:
                        # Place after current element
                        barline_x = elem.x + elem.width + 5  # Add a small gap
                    
                    barline_positions.append(barline_x)
                    
                    # Reset for next measure
                    cumulative_duration = excess  # Carry over excess duration
            
            # Create barlines at the calculated positions
            for pos in barline_positions:
                # Calculate height based on staff lines
                if system.lines:
                    top_line = min(system.lines.values())
                    bottom_line = max(system.lines.values())
                    height = bottom_line - top_line + 20  # Add padding
                    y1 = top_line - 10  # Extend above top line
                else:
                    # Default height if no staff lines
                    height = 100
                    y1 = min(e.y for e in system.elements) - 50
                
                # Create barline bbox
                barline_bbox = {
                    'x1': pos - 2,  # Make the barline slightly thick
                    'y1': y1,
                    'x2': pos + 2,
                    'y2': y1 + height,
                    'width': 4,  # Typical barline width
                    'height': height,
                    'center_x': pos,
                    'center_y': y1 + height/2
                }
                
                # Create barline object
                barline = Barline(
                    class_id=-1,  # Use -1 for inferred elements
                    class_name="barline",
                    confidence=1.0,
                    bbox=barline_bbox
                )
                
                # Add to staff system
                system.add_element(barline)
                self.barlines.append(barline)
            
            print(f"Added {len(barline_positions)} inferred barlines to system {system.id}")
        
        # Rebuild measures now that we have added barlines
        # self.identify_measures()

    def analyze_possible_time_signatures(self):
        """
        Attempt to infer the time signature when one is not provided.
        This is a simple analysis based on note groupings and rhythmic patterns.
        """
        for system in self.staff_systems:
            # Skip if time signature already exists
            if system.time_signature:
                continue
                
            # Get notes and rests, sorted by x-position
            elements = [e for e in system.elements 
                    if isinstance(e, (Note, Rest))]
            elements.sort(key=lambda e: e.x)
            
            if not elements:
                continue
                
            # Get existing barlines, if any
            barlines = [e for e in system.elements if isinstance(e, Barline)]
            barlines.sort(key=lambda b: b.x)
            
            # Try to determine measure lengths from existing barlines
            if len(barlines) >= 2:
                # Analyze durations between barlines
                measure_durations = []
                
                for i in range(len(barlines) - 1):
                    start_x = barlines[i].x
                    end_x = barlines[i+1].x
                    
                    # Get elements between these barlines
                    measure_elements = [e for e in elements 
                                    if start_x < e.x < end_x]
                    
                    # Calculate total duration
                    total_duration = sum(e.duration if hasattr(e, 'duration') and e.duration else 1.0 
                                        for e in measure_elements)
                    
                    measure_durations.append(total_duration)
                
                # Use the most common duration as our measure length
                if measure_durations:
                    from collections import Counter
                    duration_counts = Counter(measure_durations)
                    most_common_duration = duration_counts.most_common(1)[0][0]
                    
                    # Determine time signature based on duration
                    if abs(most_common_duration - 4.0) < 0.1:
                        # 4/4 time
                        system.time_signature = TimeSignature(beats=4, beat_type=4)
                    elif abs(most_common_duration - 3.0) < 0.1:
                        # 3/4 time
                        system.time_signature = TimeSignature(beats=3, beat_type=4)
                    elif abs(most_common_duration - 2.0) < 0.1:
                        # 2/4 time
                        system.time_signature = TimeSignature(beats=2, beat_type=4)
                    elif abs(most_common_duration - 6.0/8) < 0.1:
                        # 6/8 time
                        system.time_signature = TimeSignature(beats=6, beat_type=8)
                    else:
                        # Default to 4/4
                        system.time_signature = TimeSignature(beats=4, beat_type=4)
                    
                    print(f"Inferred time signature for system {system.id}: " 
                        f"{system.time_signature.beats}/{system.time_signature.beat_type}")
                    return
            
            # If we couldn't determine from barlines, analyze note groupings
            # (This would be a more complex analysis of rhythmic patterns)
            # For simplicity, we'll default to 4/4 for now
            system.time_signature = TimeSignature(beats=4, beat_type=4)
            print(f"No clear time signature pattern in system {system.id}. Using default 4/4.")
        
    def generate_musicxml(self):
        """Generate MusicXML from the processed musical elements."""
        # Create the root element
        score_partwise = ET.Element('score-partwise', version='4.0')
        
        # Add part-list
        part_list = ET.SubElement(score_partwise, 'part-list')
        
        # Create score-part elements (one per staff system)
        for i, system in enumerate(self.staff_systems):
            score_part = ET.SubElement(part_list, 'score-part', id=f'P{i+1}')
            part_name = ET.SubElement(score_part, 'part-name')
            part_name.text = f'Part {i+1}'
        
        # Create parts
        for i, system in enumerate(self.staff_systems):
            part = ET.SubElement(score_partwise, 'part', id=f'P{i+1}')
            
            # Track current state
            current_clef_type = None
            current_clef_line = None
            current_key_fifths = None
            current_time_sig = None
            
            # Add measures
            for j, measure in enumerate(system.measures):
                measure_elem = ET.SubElement(part, 'measure', number=str(j+1))
                
                # Determine if we need to include attributes
                include_attributes = False
                
                # Calculate key signature fifths value
                key_fifths = 0
                if system.key_signature:
                    num_sharps = sum(1 for acc in system.key_signature if acc.type == 'sharp')
                    num_flats = sum(1 for acc in system.key_signature if acc.type == 'flat')
                    if num_sharps > 0:
                        key_fifths = num_sharps
                    elif num_flats > 0:
                        key_fifths = -num_flats
                
                # Get current clef type
                clef_type = None
                clef_line = None
                if system.clef:
                    clef_type = system.clef.type
                    clef_line = system.clef.line
                
                # Get time signature
                time_sig = system.time_signature
                
                # First measure always gets attributes
                if j == 0:
                    include_attributes = True
                    current_clef_type = clef_type
                    current_clef_line = clef_line
                    current_key_fifths = key_fifths
                    current_time_sig = time_sig
                else:
                    # Check for clef changes
                    if clef_type != current_clef_type or clef_line != current_clef_line:
                        include_attributes = True
                        current_clef_type = clef_type
                        current_clef_line = clef_line
                    
                    # Check for key signature changes
                    if key_fifths != current_key_fifths:
                        include_attributes = True
                        current_key_fifths = key_fifths
                    
                    # Check for time signature changes
                    if time_sig != current_time_sig:
                        include_attributes = True
                        current_time_sig = time_sig
                
                # Add attributes if needed
                if include_attributes:
                    attrs = ET.SubElement(measure_elem, 'attributes')
                    
                    # Add divisions (time base)
                    divisions = ET.SubElement(attrs, 'divisions')
                    divisions.text = '4'  # Quarter note = 4 divisions
                    
                    # Add key signature
                    key = ET.SubElement(attrs, 'key')
                    fifths = ET.SubElement(key, 'fifths')
                    fifths.text = str(key_fifths)
                    
                    # Add time signature if present
                    if time_sig:
                        time = ET.SubElement(attrs, 'time')
                        beats = ET.SubElement(time, 'beats')
                        beats.text = str(time_sig.beats)
                        beat_type = ET.SubElement(time, 'beat-type')
                        beat_type.text = str(time_sig.beat_type)
                    
                    # Add clef if present
                    if clef_type and clef_line:
                        clef_elem = ET.SubElement(attrs, 'clef')
                        sign = ET.SubElement(clef_elem, 'sign')
                        sign.text = clef_type
                        line = ET.SubElement(clef_elem, 'line')
                        line.text = str(clef_line)
                
                # Add notes, rests, and other elements
                # Get all elements in this measure
                all_measure_elements = measure.elements
                
                # Sort elements by x-position
                all_measure_elements.sort(key=lambda e: e.x)
                
                for elem in all_measure_elements:
                    if isinstance(elem, Note):
                        # Skip notes that are part of a chord (except the first one)
                        if elem.is_chord_member and elem.chord[0] != elem:
                            continue
                            
                        # Create note element
                        note_elem = ET.SubElement(measure_elem, 'note')
                        
                        # If this is part of a chord, handle accordingly
                        if elem.is_chord_member and len(elem.chord) > 1:
                            # Add pitch for the first note
                            if elem.step and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))  # Convert to divisions
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                            
                            # Add other chord notes
                            for chord_note in elem.chord[1:]:
                                # Create note element for chord note
                                chord_note_elem = ET.SubElement(measure_elem, 'note')
                                
                                # Add chord tag
                                ET.SubElement(chord_note_elem, 'chord')
                                
                                # Add pitch
                                if chord_note.step and chord_note.octave is not None:
                                    pitch_elem = chord_note.position_to_xml()
                                    chord_note_elem.append(pitch_elem)
                                
                                # Add duration
                                duration = ET.SubElement(chord_note_elem, 'duration')
                                duration.text = str(int(4 * chord_note.duration))
                                
                                # Add type
                                type_elem = ET.SubElement(chord_note_elem, 'type')
                                type_elem.text = chord_note.duration_type
                                
                                # Add accidental if present and not part of key signature
                                if chord_note.accidental and not chord_note.accidental.is_key_signature:
                                    acc_elem = ET.SubElement(chord_note_elem, 'accidental')
                                    acc_elem.text = chord_note.accidental.type
                        else:
                            # Regular note (not part of a chord)
                            # Add pitch
                            if elem.step and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                    
                    elif isinstance(elem, Rest):
                        # Create rest element
                        rest_elem = ET.SubElement(measure_elem, 'note')
                        ET.SubElement(rest_elem, 'rest')
                        
                        # Add duration
                        duration = ET.SubElement(rest_elem, 'duration')
                        duration.text = str(int(4 * elem.duration))
                        
                        # Add type
                        type_elem = ET.SubElement(rest_elem, 'type')
                        type_elem.text = elem.duration_type
        
        # Convert to string
        score_xml = ET.tostring(score_partwise, encoding='utf-8')
        
        # Format with XML declaration and DOCTYPE
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
        doc_type = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
        
        return xml_declaration + doc_type + score_xml.decode('utf-8')
    
    def visualize(self, output_path=None):
        """Visualize the music score with detected elements and relationships."""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Define colors for different element types
        colors = {
            'staff_line': 'black',
            'note': 'green',
            'rest': 'cyan',
            'accidental': 'blue',
            'barline': 'red',
            'clef': 'purple',
            'beam': 'orange',
            'flag': 'magenta',
            'time_sig': 'brown',
            'chord': 'lime',
            'measure': 'gray'
        }
        
        # Draw staff lines
        for system in self.staff_systems:
            for line_num, y in system.lines.items():
                ax.axhline(y=y, color=colors['staff_line'], linestyle='-', alpha=0.5)
                
                # Add line number
                ax.text(system.elements[0].x - 30 if system.elements else 100, 
                       y, f"Line {line_num}", fontsize=8, ha='right', va='center')
        
        # Draw measures
        for measure in self.measures:
            # Get staff system bounds
            system = measure.staff_system
            if system.lines:
                top_y = min(system.lines.values()) - system.line_spacing
                bottom_y = max(system.lines.values()) + system.line_spacing
                
                # Draw measure rectangle
                rect = patches.Rectangle(
                    (measure.start_x, top_y),
                    measure.end_x - measure.start_x, bottom_y - top_y,
                    linewidth=1, edgecolor=colors['measure'], facecolor='none',
                    linestyle='--', alpha=0.3
                )
                ax.add_patch(rect)
                
                # Add measure number
                if measure in system.measures:
                    measure_num = system.measures.index(measure) + 1
                    ax.text(measure.start_x + 10, top_y - 10, f"M{measure_num}",
                          fontsize=8, color=colors['measure'])
        
        # Draw notes
        for note in self.notes:
            rect = patches.Rectangle(
                (note.bbox['x1'], note.bbox['y1']),
                note.width, note.height,
                linewidth=1, edgecolor=colors['note'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add pitch label
            if note.pitch:
                ax.text(note.x, note.y, note.pitch,
                      fontsize=8, ha='center', va='center', color=colors['note'])
            
            # Add duration
            if note.duration_type:
                ax.text(note.x, note.bbox['y2'] + 5, note.duration_type,
                      fontsize=7, ha='center', va='bottom', color=colors['note'])
        
        # Draw rests
        for rest in self.rests:
            rect = patches.Rectangle(
                (rest.bbox['x1'], rest.bbox['y1']),
                rest.width, rest.height,
                linewidth=1, edgecolor=colors['rest'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add rest type
            if rest.duration_type:
                ax.text(rest.x, rest.y, rest.duration_type,
                      fontsize=8, ha='center', va='center', color=colors['rest'])
        
        # Draw accidentals
        for acc in self.accidentals:
            rect = patches.Rectangle(
                (acc.bbox['x1'], acc.bbox['y1']),
                acc.width, acc.height,
                linewidth=1, edgecolor=colors['accidental'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add accidental type
            if acc.type:
                ax.text(acc.x, acc.y - 10, acc.type,
                      fontsize=8, ha='center', color=colors['accidental'])
            
            # Connect to affected note
            if acc.affected_note:
                ax.plot([acc.x, acc.affected_note.x], [acc.y, acc.affected_note.y],
                      'b-', alpha=0.5)
            
            # Mark key signature accidentals differently
            if acc.is_key_signature:
                ax.text(acc.x, acc.bbox['y1'] - 5, "K",
                      fontsize=8, ha='center', color='purple', fontweight='bold')
        
        # Draw clefs
        for clef in self.clefs:
            rect = patches.Rectangle(
                (clef.bbox['x1'], clef.bbox['y1']),
                clef.width, clef.height,
                linewidth=1, edgecolor=colors['clef'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add clef type
            if clef.type:
                ax.text(clef.x, clef.y - 15, f"{clef.type}-clef",
                      fontsize=8, ha='center', color=colors['clef'])
        
        # Draw barlines
        for barline in self.barlines:
            rect = patches.Rectangle(
                (barline.bbox['x1'], barline.bbox['y1']),
                barline.width, barline.height,
                linewidth=1, edgecolor=colors['barline'], facecolor='none'
            )
            ax.add_patch(rect)
        
        # Draw beams
        for beam in self.beams:
            rect = patches.Rectangle(
                (beam.bbox['x1'], beam.bbox['y1']),
                beam.width, beam.height,
                linewidth=1, edgecolor=colors['beam'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Connect beam to notes
            for note in beam.connected_notes:
                ax.plot([note.x, note.x], [note.y, beam.y],
                      color=colors['beam'], linestyle='--', alpha=0.5)
        
        # Draw flags
        for flag in self.flags:
            rect = patches.Rectangle(
                (flag.bbox['x1'], flag.bbox['y1']),
                flag.width, flag.height,
                linewidth=1, edgecolor=colors['flag'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Connect flag to note
            if flag.connected_note:
                ax.plot([flag.x, flag.connected_note.x], [flag.y, flag.connected_note.y],
                      color=colors['flag'], linestyle='--', alpha=0.5)
        
        # Draw time signature elements
        for time_sig in self.time_signature_elements:
            rect = patches.Rectangle(
                (time_sig.bbox['x1'], time_sig.bbox['y1']),
                time_sig.width, time_sig.height,
                linewidth=1, edgecolor=colors['time_sig'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add time sig value
            ax.text(time_sig.x, time_sig.y, str(time_sig.value),
                  fontsize=10, ha='center', va='center', color=colors['time_sig'])
        
        # Highlight chords
        for note in self.notes:
            if note.is_chord_member and note == note.chord[0]:  # Only process first note in chord
                # Draw chord connector
                chord_points = [(n.x, n.y) for n in note.chord]
                xs, ys = zip(*chord_points)
                ax.plot(xs, ys, color=colors['chord'], linestyle='-', alpha=0.7)
        
        # Set axis limits
        all_elements = (self.notes + self.rests + self.accidentals + self.clefs + 
                       self.barlines + self.beams + self.flags + self.time_signature_elements)
        
        if all_elements:
            all_x = [e.x for e in all_elements]
            all_y = [e.y for e in all_elements]
            
            margin = 50
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(max(all_y) + margin, min(all_y) - margin)  # Invert y-axis
        
        # Add title and labels
        ax.set_title('Music Score Analysis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add legend
        legend_elements = [patches.Patch(facecolor='none', edgecolor=color, label=name)
                         for name, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


def process_music_score(detection_file, staff_lines_file, output_xml=None, output_image=None):
    """
    Process a music score and generate MusicXML and visualization.
    
    Args:
        detection_file: Path to detection file (CSV or JSON)
        staff_lines_file: Path to staff lines file (JSON)
        output_xml: Path to save generated MusicXML (optional)
        output_image: Path to save visualization image (optional)
        
    Returns:
        Generated MusicXML string
    """
    # Create processor
    processor = OMRProcessor(detection_file, staff_lines_file)
    
    # Process
    musicxml = processor.process()
    
    # Save MusicXML
    if musicxml and output_xml:
        with open(output_xml, 'w') as f:
            f.write(musicxml)
    
    # Visualize
    processor.visualize(output_image)
    
    return musicxml

# Example usage
if __name__ == "__main__":
    detection_file = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/results/object_detections/beam_groups_8_semiquavers-001_detections.csv"
    staff_lines_file = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/results/staff_lines/beam_groups_8_semiquavers-001_staff_lines.json"
    
    output_xml = "beam_groups_8_semiquavers-001.musicxml"
    output_image = "beam_groups_8_semiquavers-001.png"
    
    
    musicxml = process_music_score(detection_file, staff_lines_file, output_xml, output_image)
    print(f"MusicXML generated and saved to {output_xml}")
    print(f"Visualization saved to {output_image}")