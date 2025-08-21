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
        self.ledger_lines = []
        self.is_chord_member = False
        self.chord = None
        self.duration_type = 'quarter'  # Default
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

class StaffSystem:
    """Class representing a system of staff lines."""
    def __init__(self, id):
        self.id = id
        self.lines = {}  # Map line number to y-position
        self.elements = []
        self.measures = []
        self.clef = None
        self.key_signature = []
        self.time_signature = None
        self.line_spacing = None
        
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
        """
        Convert a y-coordinate to a pitch based on staff position.
        
        Args:
            y: Y-coordinate
            clef: Optional clef to use (otherwise uses system's default)
            
        Returns:
            Tuple of (step, octave)
        """
        if not clef:
            clef = self.clef
            
        if not clef or not self.line_spacing or not self.lines:
            return (None, None)
        
        # Get the y-position of the middle line (line 2)
        middle_line_y = self.lines.get(2, None)
        if middle_line_y is None:
            # Fallback to calculating middle position
            sorted_lines = sorted(self.lines.items(), key=lambda x: x[1])
            if len(sorted_lines) >= 3:
                middle_line_y = sorted_lines[2][1]  # Middle line
            else:
                # Not enough lines
                return (None, None)
        
        # Calculate how many steps above or below the middle line
        # Positive steps = below middle line, negative = above
        steps = (y - middle_line_y) / (self.line_spacing / 2)
        steps = round(steps)  # Round to nearest half step
        
        # Convert steps to pitch based on clef
        if clef.type == 'G':  # Treble clef
            # Middle line (B4) is our reference
            reference_pitch = ('B', 4)
        elif clef.type == 'F':  # Bass clef
            # Middle line (D3) is our reference
            reference_pitch = ('D', 3)
        elif clef.type == 'C':  # Alto/Tenor clef
            # Middle line varies based on clef.line
            if clef.line == 3:  # Alto clef
                reference_pitch = ('C', 4)
            elif clef.line == 4:  # Tenor clef
                reference_pitch = ('A', 3)
            else:
                reference_pitch = ('C', 4)  # Default
        else:
            # Default to treble
            reference_pitch = ('B', 4)
        
        ref_step, ref_octave = reference_pitch
        
        # Steps in the musical scale
        steps_in_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        
        # Find the index of our reference pitch
        ref_index = steps_in_scale.index(ref_step)
        
        # Calculate the new index based on steps
        # Negative steps for upward movement (closer to top of staff)
        new_index = (ref_index + steps) % 7
        new_step = steps_in_scale[int(new_index)]
        
        # Calculate octave change
        octave_change = (ref_index + steps) // 7
        if steps < 0 and (ref_index + steps) % 7 != 0:
            octave_change -= 1
            
        new_octave = ref_octave - octave_change
        
        return (new_step, int(new_octave))

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
        self.accidentals = []
        self.clefs = []
        self.barlines = []
        self.beams = []
        self.key_signatures = []
        self.time_signatures = []
        self.measures = []
        
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
            
        # Process the staff systems
        self.process_staff_systems()
        
        # Process detected objects
        self.process_detected_objects()
        
        # Assign objects to staff systems
        self.assign_to_staff_systems()
        
        # Calculate note pitches
        self.calculate_note_pitches()
        
        # Connect accidentals to notes
        self.connect_accidentals_to_notes()
        
        # Find key signatures
        self.identify_key_signatures()
        
        # Group notes into chords
        self.group_notes_into_chords()
        
        # Identify measures using barlines
        self.identify_measures()
        
        # Calculate note durations
        self.calculate_note_durations()
        
        # Generate MusicXML
        return self.generate_musicxml()
    
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
                
            elif 'accidental' in class_name.lower():
                accidental = Accidental(class_id, class_name, confidence, bbox)
                self.accidentals.append(accidental)
                
            elif 'Clef' in class_name:
                clef = Clef(class_id, class_name, confidence, bbox)
                self.clefs.append(clef)
                
            elif class_name == 'barline':
                barline = Barline(class_id, class_name, confidence, bbox)
                self.barlines.append(barline)
                
            elif 'beam' in class_name.lower():
                beam = Beam(class_id, class_name, confidence, bbox)
                self.beams.append(beam)
    
    def assign_to_staff_systems(self):
        """Assign music elements to staff systems."""
        all_elements = self.notes + self.accidentals + self.clefs + self.barlines + self.beams
        
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
                
                # For barlines and clefs, check if they span the staff
                if isinstance(element, (Barline, Clef)):
                    element_top = element.bbox['y1']
                    element_bottom = element.bbox['y2']
                    
                    # Check if element overlaps significantly with staff
                    overlap = min(element_bottom, bottom_line) - max(element_top, top_line)
                    if overlap > 0 and overlap > (bottom_line - top_line) * 0.5:
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
                    staff_range = bottom_line - top_line
                    extended_top = top_line - staff_range * 0.5  # Extended staff range
                    extended_bottom = bottom_line + staff_range * 0.5
                    
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
                
            # Get the staff system
            system = note.staff_system
            
            # Use the appropriate clef
            clef = system.clef
            if not clef:
                # Default to treble clef if none found
                clef = Clef(0, "gClef", 1.0, {
                    'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
                    'width': 0, 'height': 0,
                    'center_x': 0, 'center_y': 0
                })
                clef.type = 'G'
                clef.line = 2
            
            # Calculate pitch based on y-position
            step, octave = system.y_to_pitch(note.y, clef)
            
            if step and octave:
                note.set_pitch(step, octave)
    
    def connect_accidentals_to_notes(self):
        """Connect accidentals to the notes they affect."""
        # For each accidental, find the closest note to the right
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
            
            # Find the first non-accidental element after the clef
            first_other_x = float('inf')
            for elem in system.elements:
                if not isinstance(elem, Accidental) and elem.x > clef.x:
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
        """Calculate durations for notes based on beams and stems."""
        # For simplicity, we'll use default durations
        # In a more advanced implementation, we would analyze beams, flags, etc.
        for note in self.notes:
            # Default to quarter note
            note.duration = 1.0
            note.duration_type = 'quarter'
            
            # Check if there are beams connecting to this note
            connected_beams = [b for b in self.beams if note in b.connected_notes]
            
            if connected_beams:
                # The number of beams determines the duration
                beam_count = len(connected_beams)
                
                if beam_count == 1:
                    note.duration = 0.5  # Eighth note
                    note.duration_type = 'eighth'
                elif beam_count == 2:
                    note.duration = 0.25  # Sixteenth note
                    note.duration_type = '16th'
                elif beam_count >= 3:
                    note.duration = 0.125  # Thirty-second note
                    note.duration_type = '32nd'
    

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
            current_key_fifths = None
            
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
                
                # First measure always gets attributes
                if j == 0:
                    include_attributes = True
                    current_clef_type = clef_type
                    current_key_fifths = key_fifths
                else:
                    # Check for clef changes
                    if clef_type != current_clef_type:
                        include_attributes = True
                        current_clef_type = clef_type
                    
                    # Check for key signature changes
                    if key_fifths != current_key_fifths:
                        include_attributes = True
                        current_key_fifths = key_fifths
                
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
                    
                    # Add clef if present
                    if clef_type and clef_line:
                        clef_elem = ET.SubElement(attrs, 'clef')
                        sign = ET.SubElement(clef_elem, 'sign')
                        sign.text = clef_type
                        line = ET.SubElement(clef_elem, 'line')
                        line.text = str(clef_line)
                
                # Add notes and other elements
                # Sort elements by x-position
                measure_elements = sorted(measure.elements, key=lambda e: e.x)
                
                for elem in measure_elements:
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
        
        # Convert to string
        return ET.tostring(score_partwise, encoding='utf-8').decode('utf-8')
    
    def visualize(self, output_path=None):
        """Visualize the music score with detected elements and relationships."""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Define colors for different element types
        colors = {
            'staff_line': 'black',
            'note': 'green',
            'accidental': 'blue',
            'barline': 'red',
            'clef': 'purple',
            'beam': 'orange',
            'chord': 'cyan',
            'measure': 'gray'
        }
        
        # Draw staff lines
        for system in self.staff_systems:
            for line_num, y in system.lines.items():
                ax.axhline(y=y, color=colors['staff_line'], linestyle='-', alpha=0.5)
        
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
        
        # Highlight chords
        for note in self.notes:
            if note.is_chord_member and note == note.chord[0]:  # Only process first note in chord
                # Draw chord connector
                chord_points = [(n.x, n.y) for n in note.chord]
                xs, ys = zip(*chord_points)
                ax.plot(xs, ys, color=colors['chord'], linestyle='-', alpha=0.7)
        
        # Set axis limits
        all_elements = self.notes + self.accidentals + self.clefs + self.barlines + self.beams
        if all_elements:
            all_x = [e.x for e in all_elements]
            all_y = [e.y for e in all_elements]
            
            margin = 50
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
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
    staff_lines_file = "//homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/results/staff_lines/beam_groups_8_semiquavers-001_staff_lines.json"
    
    output_xml = "beam_groups_8_semiquavers-001.musicxml"
    output_image = "beam_groups_8_semiquavers-001_detections_score_visualization.png"
    
    musicxml = process_music_score(detection_file, staff_lines_file, output_xml, output_image)
    print(f"MusicXML generated and saved to {output_xml}")
    print(f"Visualization saved to {output_image}")