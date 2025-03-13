import json
import math
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from collections import defaultdict, namedtuple
from enum import Enum


class Pitch:
    """
    Class to handle musical pitch representation and operations.
    """
    STEPS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    
    def __init__(self, step, octave, alter=0):
        self.step = step
        self.octave = octave
        self.alter = alter  # semitone alteration: -1 for flat, 0 for natural, 1 for sharp
    
    @property
    def midi_number(self):
        """Convert pitch to MIDI note number."""
        step_values = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        return 12 * (self.octave + 1) + step_values[self.step] + self.alter
    
    def __str__(self):
        """String representation of the pitch."""
        alter_symbols = {-1: '♭', 0: '', 1: '♯'}
        return f"{self.step}{alter_symbols[self.alter]}{self.octave}"


class ClefType(Enum):
    TREBLE = 'G'
    BASS = 'F'
    ALTO = 'C'
    TENOR = 'C'
    

class OMRProcessor:
    """
    Main processor for Optical Music Recognition that converts
    detected objects into structured MusicXML.
    """
    def __init__(self, detection_path=None, staff_lines_path=None, class_mapping_path=None, 
                 detection_data=None, staff_lines_data=None):
        self.detection_path = detection_path
        self.staff_lines_path = staff_lines_path
        self.class_mapping_path = class_mapping_path
        
        # Load data
        if detection_path:
            self.detections = self.load_json(detection_path)
        else:
            self.detections = detection_data
            
        if staff_lines_path:
            self.staff_lines = self.load_json(staff_lines_path)
        else:
            self.staff_lines = staff_lines_data
        
        # Load class mapping if provided
        self.class_mapping = None
        if class_mapping_path:
            self.class_mapping = self.load_json(class_mapping_path)
        
        # Initialize data structures
        self.staff_systems = []  # Will hold StaffSystem objects
        self.note_elements = []  # Will hold Note objects
        self.accidentals = []    # Will hold Accidental objects
        self.barlines = []       # Will hold Barline objects
        self.clefs = []          # Will hold Clef objects
        self.time_signatures = [] # Will hold TimeSignature objects
        self.beams = []          # Will hold Beam objects
        self.key_signatures = [] # Will hold KeySignature objects
        
        # Constants
        self.STAFF_LINE_DISTANCE = None  # Will be calculated from staff lines
        
    def load_json(self, path):
        """Load JSON data from file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def process(self):
        """Main processing pipeline."""
        if not self.detections or not self.staff_lines:
            print("Missing required data. Cannot process.")
            return None
        
        # Process staff lines
        self.process_staff_systems()
        
        # Calculate staff line distance
        self.calculate_staff_line_distance()
        
        # Process detected objects
        self.process_detected_objects()
        
        # Assign elements to staves
        self.assign_elements_to_staves()
        
        # Identify key signatures
        self.identify_key_signatures()
        
        # Connect accidentals to notes
        self.connect_accidentals_to_notes()
        
        # Calculate note pitches
        self.calculate_note_pitches()
        
        # Identify stems and beams
        self.identify_stems_and_beams()
        
        # Connect beams and stems
        self.connect_beams_and_stems()
        
        # Group notes into chords
        self.group_notes_into_chords()
        
        # Identify measures
        self.identify_measures()
        
        # Estimate note durations
        self.estimate_note_durations()
        
        # Generate MusicXML
        return self.generate_musicxml()
    
    def process_staff_systems(self):
        """Process staff systems from staff lines data."""
        # Extract staff systems from the staff lines data
        if 'staff_systems' in self.staff_lines:
            for system_data in self.staff_lines['staff_systems']:
                staff_system = StaffSystem(system_data['id'])
                
                # Get all staff lines for this system
                staff_lines = [line for line in self.staff_lines['detections'] 
                              if line.get('staff_system') == system_data['id']]
                
                # Group staff lines by line_number
                grouped_lines = defaultdict(list)
                for line in staff_lines:
                    grouped_lines[line['line_number']].append(line)
                
                # Add staff lines to the system
                for line_num, lines in grouped_lines.items():
                    # Use the average y-position if there are multiple lines with the same number
                    if lines:
                        avg_y = sum(line['bbox']['center_y'] for line in lines) / len(lines)
                        staff_system.add_line(line_num, avg_y)
                
                self.staff_systems.append(staff_system)
    
    def calculate_staff_line_distance(self):
        """Calculate the average distance between staff lines."""
        if not self.staff_systems:
            return
        
        distances = []
        for system in self.staff_systems:
            lines = sorted(system.lines.items(), key=lambda x: x[1])  # Sort by y-position
            for i in range(1, len(lines)):
                distances.append(lines[i][1] - lines[i-1][1])
        
        if distances:
            self.STAFF_LINE_DISTANCE = sum(distances) / len(distances)
        else:
            # Default value if we can't calculate
            self.STAFF_LINE_DISTANCE = 20  # Typical value in pixels
    
    def process_detected_objects(self):
        """Process detected objects from YOLOv8 results."""
        # Process objects from detections array
        if 'detections' in self.detections:
            detections_list = self.detections['detections']
        else:
            detections_list = self.detections  # Assume it's already a list
            
        for obj in detections_list:
            class_id = obj['class_id']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Get bbox - handle both formats
            if 'bbox' in obj:
                bbox = obj['bbox']
            else:
                # Create bbox from individual coordinates
                bbox = {
                    'x1': obj['x1'], 
                    'y1': obj['y1'], 
                    'x2': obj['x2'], 
                    'y2': obj['y2'],
                    'width': obj['width'], 
                    'height': obj['height'],
                    'center_x': obj['center_x'], 
                    'center_y': obj['center_y']
                }
            
            # Process based on class
            if 'notehead' in class_name.lower():
                note = Note(
                    x=bbox['center_x'],
                    y=bbox['center_y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence
                )
                self.note_elements.append(note)
            
            elif 'accidental' in class_name.lower():
                accidental_type = None
                if 'Sharp' in class_name:
                    accidental_type = 'sharp'
                elif 'Flat' in class_name:
                    accidental_type = 'flat'
                elif 'Natural' in class_name:
                    accidental_type = 'natural'
                    
                accidental = Accidental(
                    x=bbox['center_x'],
                    y=bbox['center_y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    accidental_type=accidental_type
                )
                self.accidentals.append(accidental)
            
            elif class_name == 'barline':
                barline = Barline(
                    x=bbox['center_x'],
                    y=bbox['center_y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence
                )
                self.barlines.append(barline)
            
            elif 'Clef' in class_name:
                clef_type = None
                if 'gClef' in class_name:
                    clef_type = ClefType.TREBLE
                elif 'fClef' in class_name:
                    clef_type = ClefType.BASS
                elif 'cClef' in class_name:
                    # Determine if alto or tenor based on position
                    clef_type = ClefType.ALTO  # Default to alto
                
                clef = Clef(
                    x=bbox['center_x'],
                    y=bbox['center_y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    clef_type=clef_type
                )
                self.clefs.append(clef)
            
            elif 'beam' in class_name.lower():
                beam = Beam(
                    x=bbox['center_x'],
                    y=bbox['center_y'],
                    width=bbox['width'],
                    height=bbox['height'],
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence
                )
                self.beams.append(beam)
    
    def assign_elements_to_staves(self):
        """Assign musical elements to the appropriate staves."""
        all_elements = (self.note_elements + self.accidentals + 
                       self.barlines + self.clefs + self.beams)
        
        for element in all_elements:
            closest_system = None
            min_distance = float('inf')
            
            for system in self.staff_systems:
                # Get the y-positions of the staff lines
                staff_lines_y = sorted(system.lines.values())
                
                if not staff_lines_y:
                    continue
                
                # Calculate vertical distance to the staff system
                top_line_y = staff_lines_y[0]
                bottom_line_y = staff_lines_y[-1]
                
                # If the element is a barline or clef, it should span the staff
                if isinstance(element, (Barline, Clef)):
                    # Check if the element spans at least part of the staff
                    if (element.bbox['y1'] <= bottom_line_y and 
                        element.bbox['y2'] >= top_line_y):
                        # Element intersects with the staff
                        # Assign based on horizontal position
                        distance = 0
                    else:
                        # Calculate distance to staff
                        if element.y < top_line_y:
                            distance = top_line_y - element.y
                        elif element.y > bottom_line_y:
                            distance = element.y - bottom_line_y
                        else:
                            # Element is within the staff
                            distance = 0
                else:
                    # For other elements, calculate distance to the staff
                    if element.y < top_line_y:
                        distance = top_line_y - element.y
                    elif element.y > bottom_line_y:
                        distance = element.y - bottom_line_y
                    else:
                        # Element is within the staff
                        distance = 0
                
                # Take into account the vertical span of the element
                if distance > 0:
                    # Consider the element's height
                    element_top = element.bbox['y1']
                    element_bottom = element.bbox['y2']
                    
                    # Check if element overlaps with staff
                    if (element_bottom >= top_line_y and element_top <= bottom_line_y):
                        distance = 0
                
                if distance < min_distance:
                    min_distance = distance
                    closest_system = system
            
            if closest_system:
                element.staff_system = closest_system
                closest_system.add_element(element)
    
    def identify_key_signatures(self):
        """Identify key signatures based on accidentals at the beginning of staves."""
        # Organize staves and their elements
        for system in self.staff_systems:
            # Sort elements by x-position
            system.elements.sort(key=lambda e: e.x)
            
            # Find clefs first (they mark the start of the staff)
            clefs = [e for e in system.elements if isinstance(e, Clef)]
            
            if not clefs:
                continue
                
            # Get the x-position right after the clef
            # (key signatures typically follow clefs)
            clef_x = clefs[0].x + clefs[0].width
            
            # Find accidentals that appear right after the clef
            # and before any notes or other elements
            key_accidentals = []
            first_non_accidental_x = float('inf')
            
            for element in system.elements:
                if not isinstance(element, Accidental) and element.x > clef_x:
                    first_non_accidental_x = min(first_non_accidental_x, element.x)
                
                if (isinstance(element, Accidental) and 
                    element.x > clef_x and element.x < first_non_accidental_x):
                    key_accidentals.append(element)
            
            # Sort key accidentals by x-position
            key_accidentals.sort(key=lambda a: a.x)
            
            if key_accidentals:
                # Create a key signature
                key_sig = KeySignature(
                    x=key_accidentals[0].x,
                    y=key_accidentals[0].y,
                    accidentals=key_accidentals,
                    staff_system=system
                )
                self.key_signatures.append(key_sig)
                system.key_signature = key_sig
                
                # Mark these accidentals as part of a key signature
                for acc in key_accidentals:
                    acc.is_key_signature = True
    
    def connect_accidentals_to_notes(self):
        """Connect accidentals to the notes they affect."""
        # For each accidental, find the closest note to the right
        for acc in self.accidentals:
            if acc.is_key_signature:
                continue  # Skip key signature accidentals
                
            closest_note = None
            min_distance = float('inf')
            
            # Only consider notes in the same staff system
            if not acc.staff_system:
                continue
                
            for note in self.note_elements:
                if note.staff_system != acc.staff_system:
                    continue
                    
                # Accidentals typically appear to the left of the note they affect
                if note.x <= acc.x:
                    continue
                    
                # Calculate horizontal and vertical distance
                dx = note.x - acc.x
                dy = abs(note.y - acc.y)
                
                # Weight horizontal distance more than vertical
                distance = dx + 2 * dy
                
                # Additional constraint: accidentals typically affect notes on the same line or space
                # Allow some vertical tolerance
                vertical_tolerance = self.STAFF_LINE_DISTANCE / 2
                
                if dy > vertical_tolerance:
                    continue
                    
                if distance < min_distance:
                    min_distance = distance
                    closest_note = note
            
            # Set the affected note if found
            if closest_note:
                acc.affected_note = closest_note
                closest_note.accidental = acc
    
    def calculate_note_pitches(self):
        """Calculate pitches for notes based on staff position and active clef."""
        for system in self.staff_systems:
            # Find the active clef
            clefs = [e for e in system.elements if isinstance(e, Clef)]
            
            active_clef = None
            if clefs:
                # Use the leftmost clef as the active one
                clefs.sort(key=lambda c: c.x)
                active_clef = clefs[0]
            else:
                # Default to treble clef if none found
                print(f"No clef found for staff system {system.id}, defaulting to treble clef")
                active_clef = Clef(0, 0, 0, 0, -1, "gClef", None, 1.0, ClefType.TREBLE)
            
            # Get notes for this staff system
            notes = [e for e in system.elements if isinstance(e, Note)]
            
            for note in notes:
                self.calculate_note_pitch(note, system, active_clef)
    
    def calculate_note_pitch(self, note, system, active_clef):
        """Calculate pitch for a single note based on staff position and active clef."""
        if not system.lines:
            return
            
        # Get staff lines sorted by y-position (top to bottom)
        staff_lines = sorted([(n, y) for n, y in system.lines.items()], key=lambda x: x[1])
        
        # Find the closest staff line
        closest_line_idx = None
        min_distance = float('inf')
        
        for i, (line_num, line_y) in enumerate(staff_lines):
            distance = abs(note.y - line_y)
            if distance < min_distance:
                min_distance = distance
                closest_line_idx = i
        
        if closest_line_idx is None:
            return
            
        # Determine if the note is on a line or in a space
        closest_line_y = staff_lines[closest_line_idx][1]
        on_line = abs(note.y - closest_line_y) < (self.STAFF_LINE_DISTANCE / 4)
        
        # Calculate how many steps above or below the middle line
        # Middle line is index 2 (the third line from the top)
        steps_from_middle = closest_line_idx - 2
        
        if not on_line:
            # If the note is in a space, adjust steps
            if note.y < closest_line_y:
                # Note is above the closest line
                steps_from_middle -= 0.5
            else:
                # Note is below the closest line
                steps_from_middle += 0.5
        
        # Calculate pitch based on clef
        step, octave = self.get_pitch_from_staff_position(steps_from_middle, active_clef.clef_type)
        
        # Handle ledger lines
        if note.y < staff_lines[0][1] - self.STAFF_LINE_DISTANCE/2:
            # Note is above the staff
            steps_above = round((staff_lines[0][1] - note.y) / self.STAFF_LINE_DISTANCE)
            note.on_ledger_line = steps_above % 2 == 1
            
        elif note.y > staff_lines[-1][1] + self.STAFF_LINE_DISTANCE/2:
            # Note is below the staff
            steps_below = round((note.y - staff_lines[-1][1]) / self.STAFF_LINE_DISTANCE)
            note.on_ledger_line = steps_below % 2 == 1
        
        # Apply accidental if present
        alter = 0
        if hasattr(note, 'accidental') and note.accidental:
            if note.accidental.accidental_type == 'sharp':
                alter = 1
            elif note.accidental.accidental_type == 'flat':
                alter = -1
            # Natural just cancels any previous accidental (alter = 0)
        
        # Create Pitch object
        note.pitch = Pitch(step, octave, alter)
    
    def get_pitch_from_staff_position(self, steps_from_middle, clef_type):
        """Get pitch (step and octave) based on staff position and clef."""
        if clef_type == ClefType.TREBLE:
            # Middle line (steps_from_middle == 0) is B4
            reference_pitch = ('B', 4)
        elif clef_type == ClefType.BASS:
            # Middle line is D3
            reference_pitch = ('D', 3)
        elif clef_type == ClefType.ALTO:
            # Middle line is C4
            reference_pitch = ('C', 4)
        elif clef_type == ClefType.TENOR:
            # Middle line is A3
            reference_pitch = ('A', 3)
        else:
            # Default to treble
            reference_pitch = ('B', 4)
        
        ref_step, ref_octave = reference_pitch
        
        # Convert steps to semitones (approximately)
        # This is a simplification; actual conversion depends on the step
        steps_int = int(steps_from_middle)
        
        # Get index of reference note in the steps array
        ref_idx = Pitch.STEPS.index(ref_step)
        
        # Calculate new index
        new_idx = (ref_idx - steps_int) % 7
        new_step = Pitch.STEPS[new_idx]
        
        # Calculate octave change
        octave_change = (ref_idx - steps_int) // 7
        if ref_idx - steps_int < 0 and (ref_idx - steps_int) % 7 != 0:
            octave_change -= 1
            
        new_octave = ref_octave + octave_change
        
        return new_step, new_octave
    
    def identify_stems_and_beams(self):
        """Identify note stems and connect to beams."""
        # For simplicity, infer stems based on notehead position and beams
        for note in self.note_elements:
            # Check if there's a beam that could connect to this note
            potential_beams = []
            
            for beam in self.beams:
                # Check if the beam is in the same staff system
                if beam.staff_system != note.staff_system:
                    continue
                
                # Check if the beam could reasonably connect to this note
                # (beam spans over the note horizontally)
                if (beam.bbox['x1'] <= note.x <= beam.bbox['x2']):
                    potential_beams.append(beam)
            
            if potential_beams:
                # Sort beams by vertical distance to the note
                potential_beams.sort(key=lambda b: abs(b.y - note.y))
                
                # The closest beam is likely connected to this note
                closest_beam = potential_beams[0]
                
                # Infer the stem direction based on beam position
                stem_up = closest_beam.y < note.y
                
                # Create a stem for this note
                stem = Stem(
                    x=note.x,
                    y=note.y + (-1 if stem_up else 1) * note.height/2,
                    width=1,  # Thin stem
                    height=abs(closest_beam.y - note.y),
                    up=stem_up,
                    note=note
                )
                
                note.stem = stem
                closest_beam.connected_notes.append(note)
            else:
                # No beam, infer stem direction based on position on staff
                if note.staff_system and note.staff_system.lines:
                    # Get the middle of the staff
                    staff_lines_y = list(note.staff_system.lines.values())
                    middle_y = sum(staff_lines_y) / len(staff_lines_y)
                    
                    # Notes above the middle line typically have stems down
                    # Notes below the middle line typically have stems up
                    stem_up = note.y > middle_y
                    
                    # Create a stem with a default length
                    stem_length = self.STAFF_LINE_DISTANCE * 3.5
                    
                    stem = Stem(
                        x=note.x,
                        y=note.y + (-1 if stem_up else 1) * note.height/2,
                        width=1,  # Thin stem
                        height=stem_length,
                        up=stem_up,
                        note=note
                    )
                    
                    note.stem = stem
    
    def connect_beams_and_stems(self):
        """Connect beams to stems based on proximity."""
        for beam in self.beams:
            if not beam.connected_notes:
                # Find notes that could connect to this beam
                for note in self.note_elements:
                    if note.staff_system != beam.staff_system:
                        continue
                    
                    # Check if the note is under the beam horizontally
                    if not (beam.bbox['x1'] <= note.x <= beam.bbox['x2']):
                        continue
                    
                    # Check if the note has a stem
                    if not hasattr(note, 'stem') or not note.stem:
                        continue
                    
                    # Check if the stem could connect to the beam
                    stem = note.stem
                    stem_top = stem.y - stem.height if stem.up else stem.y
                    stem_bottom = stem.y if stem.up else stem.y + stem.height
                    
                    if ((stem.up and beam.y >= stem_top and beam.y <= note.y) or
                        (not stem.up and beam.y <= stem_bottom and beam.y >= note.y)):
                        beam.connected_notes.append(note)
    
    def group_notes_into_chords(self):
        """Group vertically aligned notes into chords."""
        # Sort notes by x-position
        sorted_notes = sorted(self.note_elements, key=lambda n: n.x)
        
        # Group notes by their horizontal position
        x_tolerance = self.STAFF_LINE_DISTANCE / 2  # Notes within this distance are considered aligned
        
        groups = []
        current_group = []
        
        for note in sorted_notes:
            if not current_group:
                current_group.append(note)
            else:
                # Check if this note is aligned with the current group
                ref_x = current_group[0].x
                if abs(note.x - ref_x) <= x_tolerance:
                    current_group.append(note)
                else:
                    # Start a new group
                    if len(current_group) > 1:
                        groups.append(current_group)
                    current_group = [note]
        
        # Add the last group if it has multiple notes
        if len(current_group) > 1:
            groups.append(current_group)
        
        # Create chord objects for each group
        for group in groups:
            # Check if all notes are in the same staff system
            staff_systems = set(note.staff_system for note in group if note.staff_system)
            
            if len(staff_systems) == 1:
                # Create a chord
                system = list(staff_systems)[0]
                
                chord = Chord(
                    notes=group,
                    staff_system=system
                )
                
                # Set the chord reference for each note
                for note in group:
                    note.chord = chord
                
                # Add the chord to the staff system
                system.add_element(chord)
    
    def identify_measures(self):
        """Identify measures based on barlines."""
        # Group elements by staff system
        for system in self.staff_systems:
            # Sort barlines by x-position
            barlines = sorted([e for e in system.elements if isinstance(e, Barline)], 
                            key=lambda x: x.x)
            
            # Create measures between barlines
            measures = []
            prev_x = 0
            
            for barline in barlines:
                # Create a measure from prev_x to barline.x
                measure = Measure(prev_x, barline.x, system)
                
                # Add elements that fall within this measure
                for element in system.elements:
                    if prev_x <= element.x < barline.x:
                        measure.add_element(element)
                
                measures.append(measure)
                prev_x = barline.x
            
            system.measures = measures
    
    def estimate_note_durations(self):
        """Estimate note durations based on beams and note types."""
        for note in self.note_elements:
            # Default to quarter note
            note.duration = 1  # 1 = quarter, 0.5 = eighth, 0.25 = sixteenth, etc.
            note.duration_type = "quarter"
            
            # Check if the note is beamed
            if hasattr(note, 'stem') and note.stem:
                # Count how many beams connect to this note
                beam_count = 0
                
                for beam in self.beams:
                    if note in beam.connected_notes:
                        beam_count += 1
                
                # Adjust duration based on beam count
                if beam_count == 1:
                    note.duration = 0.5  # Eighth note
                    note.duration_type = "eighth"
                elif beam_count == 2:
                    note.duration = 0.25  # Sixteenth note
                    note.duration_type = "16th"
                elif beam_count >= 3:
                    note.duration = 0.125  # Thirty-second note
                    note.duration_type = "32nd"
                
                # If no beams but has a flag
                # (would need flag detection, not implemented here)
                pass
    
    def generate_musicxml(self):
        """Generate MusicXML from processed elements."""
        # Create the root element
        score_partwise = ET.Element('score-partwise', version="4.0")
        
        # Add part-list
        part_list = ET.SubElement(score_partwise, 'part-list')
        
        # Add parts (one per staff system)
        for i, system in enumerate(self.staff_systems):
            # Add to part-list
            score_part = ET.SubElement(part_list, 'score-part', id=f"P{i+1}")
            part_name = ET.SubElement(score_part, 'part-name')
            part_name.text = f"Part {i+1}"
            
            # Create part
            part = ET.SubElement(score_partwise, 'part', id=f"P{i+1}")
            
            # Add measures
            if not system.measures:
                # If no measures identified, create a single measure with all elements
                measure = Measure(0, float('inf'), system)
                measure.elements = system.elements
                system.measures = [measure]
            
            for j, measure in enumerate(system.measures):
                # Create measure
                measure_elem = ET.SubElement(part, 'measure', number=str(j+1))
                
                # Add attributes (first measure or after changes)
                if j == 0 or True:  # Simplified: always include attributes
                    attributes = ET.SubElement(measure_elem, 'attributes')
                    
                    # Add divisions (time base)
                    divisions = ET.SubElement(attributes, 'divisions')
                    divisions.text = "4"  # Quarter note = 4 divisions
                    
                    # Add key signature if present
                    key_sigs = [ks for ks in self.key_signatures if ks.staff_system == system]
                    if key_sigs:
                        key_sig = key_sigs[0]
                        key_elem = ET.SubElement(attributes, 'key')
                        
                        # Count sharps and flats
                        fifths = 0
                        for acc in key_sig.accidentals:
                            if acc.accidental_type == 'sharp':
                                fifths += 1
                            elif acc.accidental_type == 'flat':
                                fifths -= 1
                        
                        fifths_elem = ET.SubElement(key_elem, 'fifths')
                        fifths_elem.text = str(fifths)
                    
                    # Add time signature if present
                    # (Not implemented in this version)
                    
                    # Add clef if present
                    clefs = [e for e in measure.elements if isinstance(e, Clef)]
                    if clefs:
                        clef = clefs[0]  # Use first clef
                        clef_elem = ET.SubElement(attributes, 'clef')
                        sign = ET.SubElement(clef_elem, 'sign')
                        line = ET.SubElement(clef_elem, 'line')
                        
                        if clef.clef_type == ClefType.TREBLE:
                            sign.text = 'G'
                            line.text = '2'
                        elif clef.clef_type == ClefType.BASS:
                            sign.text = 'F'
                            line.text = '4'
                        elif clef.clef_type == ClefType.ALTO:
                            sign.text = 'C'
                            line.text = '3'
                        elif clef.clef_type == ClefType.TENOR:
                            sign.text = 'C'
                            line.text = '4'
                
                # Add notes and other elements
                # Sort elements by x-position
                sorted_elements = sorted(measure.elements, key=lambda x: x.x)
                
                for element in sorted_elements:
                    if isinstance(element, Note):
                        # Skip notes that are part of a chord (except the first one)
                        if hasattr(element, 'chord') and element.chord and element != element.chord.notes[0]:
                            continue
                        
                        # Create note element
                        note_elem = ET.SubElement(measure_elem, 'note')
                        
                        # Add chord tag if this is a chord
                        if hasattr(element, 'chord') and element.chord and len(element.chord.notes) > 1:
                            # This is the first note in a chord
                            # Add subsequent chord notes
                            chord_notes = element.chord.notes
                            
                            # Add the first note's pitch
                            if hasattr(element, 'pitch') and element.pitch:
                                pitch_elem = ET.SubElement(note_elem, 'pitch')
                                step = ET.SubElement(pitch_elem, 'step')
                                step.text = element.pitch.step
                                octave = ET.SubElement(pitch_elem, 'octave')
                                octave.text = str(element.pitch.octave)
                                
                                if element.pitch.alter != 0:
                                    alter = ET.SubElement(pitch_elem, 'alter')
                                    alter.text = str(element.pitch.alter)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * element.duration))  # Convert to divisions
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = element.duration_type
                            
                            # Add stem direction if known
                            if hasattr(element, 'stem') and element.stem:
                                stem_elem = ET.SubElement(note_elem, 'stem')
                                stem_elem.text = 'up' if element.stem.up else 'down'
                            
                            # Add subsequent chord notes
                            for chord_note in chord_notes[1:]:
                                chord_elem = ET.SubElement(measure_elem, 'note')
                                chord_tag = ET.SubElement(chord_elem, 'chord')
                                
                                if hasattr(chord_note, 'pitch') and chord_note.pitch:
                                    pitch_elem = ET.SubElement(chord_elem, 'pitch')
                                    step = ET.SubElement(pitch_elem, 'step')
                                    step.text = chord_note.pitch.step
                                    octave = ET.SubElement(pitch_elem, 'octave')
                                    octave.text = str(chord_note.pitch.octave)
                                    
                                    if chord_note.pitch.alter != 0:
                                        alter = ET.SubElement(pitch_elem, 'alter')
                                        alter.text = str(chord_note.pitch.alter)
                                
                                # Add duration
                                duration = ET.SubElement(chord_elem, 'duration')
                                duration.text = str(int(4 * element.duration))
                                
                                # Add type
                                type_elem = ET.SubElement(chord_elem, 'type')
                                type_elem.text = element.duration_type
                        else:
                            # Regular note (not part of a chord)
                            if hasattr(element, 'pitch') and element.pitch:
                                pitch_elem = ET.SubElement(note_elem, 'pitch')
                                step = ET.SubElement(pitch_elem, 'step')
                                step.text = element.pitch.step
                                octave = ET.SubElement(pitch_elem, 'octave')
                                octave.text = str(element.pitch.octave)
                                
                                if element.pitch.alter != 0:
                                    alter = ET.SubElement(pitch_elem, 'alter')
                                    alter.text = str(element.pitch.alter)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * element.duration))
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = element.duration_type
                            
                            # Add stem direction if known
                            if hasattr(element, 'stem') and element.stem:
                                stem_elem = ET.SubElement(note_elem, 'stem')
                                stem_elem.text = 'up' if element.stem.up else 'down'
        
        # Convert to string
        tree = ET.ElementTree(score_partwise)
        
        # Create a proper MusicXML with DOCTYPE
        xmlstr = ET.tostring(score_partwise, encoding='utf-8')
        
        # Return as pretty-printed XML string with DOCTYPE
        # For proper MusicXML, use lxml or other library to add DOCTYPE
        return xmlstr.decode('utf-8')
    
    def visualize(self, output_path=None):
        """
        Visualize the detected elements and their relationships.
        
        Args:
            output_path: Path to save the visualization image, if None, shows the plot
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(20, 15))
        
        # Create a colormap for different element types
        colors = {
            'staff_line': 'black',
            'note': 'green',
            'accidental': 'blue',
            'barline': 'red',
            'clef': 'purple',
            'beam': 'orange',
            'stem': 'brown',
            'chord': 'olive',
            'measure': 'gray'
        }
        
        # Draw staff lines
        for system in self.staff_systems:
            for line_num, y in system.lines.items():
                ax.axhline(y=y, color=colors['staff_line'], linestyle='-', alpha=0.5)
        
        # Draw notes
        for note in self.note_elements:
            rect = patches.Rectangle(
                (note.bbox['x1'], note.bbox['y1']),
                note.width, note.height,
                linewidth=1, edgecolor=colors['note'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Display pitch
            if hasattr(note, 'pitch') and note.pitch:
                ax.text(note.x, note.y, str(note.pitch),
                       ha='center', va='center', color=colors['note'],
                       fontsize=8)
        
        # Draw stems if present
        for note in self.note_elements:
            if hasattr(note, 'stem') and note.stem:
                stem = note.stem
                if stem.up:
                    # Stem goes up from note
                    ax.plot([stem.x, stem.x], [stem.y, stem.y - stem.height],
                           color=colors['stem'], linewidth=1)
                else:
                    # Stem goes down from note
                    ax.plot([stem.x, stem.x], [stem.y, stem.y + stem.height],
                           color=colors['stem'], linewidth=1)
        
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
                if hasattr(note, 'stem') and note.stem:
                    # Connect beam to stem end
                    stem = note.stem
                    stem_end_y = stem.y - stem.height if stem.up else stem.y + stem.height
                    ax.plot([stem.x, stem.x], [beam.y, stem_end_y],
                          'k--', alpha=0.3)
        
        # Draw accidentals
        for acc in self.accidentals:
            rect = patches.Rectangle(
                (acc.bbox['x1'], acc.bbox['y1']),
                acc.width, acc.height,
                linewidth=1, edgecolor=colors['accidental'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label for the accidental type
            if acc.accidental_type:
                ax.text(acc.x, acc.y - 5, acc.accidental_type,
                      ha='center', va='center', color=colors['accidental'],
                      fontsize=8)
            
            # Connect accidental to affected note
            if acc.affected_note:
                ax.plot([acc.x, acc.affected_note.x], [acc.y, acc.affected_note.y],
                      'b-', alpha=0.5)
        
        # Draw barlines
        for barline in self.barlines:
            rect = patches.Rectangle(
                (barline.bbox['x1'], barline.bbox['y1']),
                barline.width, barline.height,
                linewidth=1, edgecolor=colors['barline'], facecolor='none'
            )
            ax.add_patch(rect)
        
        # Draw clefs
        for clef in self.clefs:
            rect = patches.Rectangle(
                (clef.bbox['x1'], clef.bbox['y1']),
                clef.width, clef.height,
                linewidth=1, edgecolor=colors['clef'], facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label for the clef type
            if clef.clef_type:
                ax.text(clef.x, clef.y - 10, clef.clef_type.value,
                      ha='center', va='center', color=colors['clef'],
                      fontsize=10)
        
        # Draw measures
        for system in self.staff_systems:
            for measure in system.measures:
                # Draw a rectangle around the measure
                if measure.end_x < float('inf'):
                    # Get the min and max y for this staff system
                    if system.lines:
                        min_y = min(system.lines.values()) - self.STAFF_LINE_DISTANCE * 2
                        max_y = max(system.lines.values()) + self.STAFF_LINE_DISTANCE * 2
                        
                        rect = patches.Rectangle(
                            (measure.start_x, min_y),
                            measure.end_x - measure.start_x, max_y - min_y,
                            linewidth=1, edgecolor=colors['measure'], facecolor='none',
                            linestyle='--', alpha=0.3
                        )
                        ax.add_patch(rect)
        
        # Draw chords
        for note in self.note_elements:
            if hasattr(note, 'chord') and note.chord:
                chord = note.chord
                
                # For each chord, connect all notes with a line
                if len(chord.notes) > 1:
                    # Sort notes by vertical position
                    chord_notes = sorted(chord.notes, key=lambda n: n.y)
                    
                    # Connect notes with a line
                    note_positions = [(n.x, n.y) for n in chord_notes]
                    xs, ys = zip(*note_positions)
                    ax.plot(xs, ys, 'g-', alpha=0.3)
        
        # Set limits and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Music Score Elements and Relationships')
        
        # Add a legend
        legend_patches = [patches.Patch(color=color, label=label)
                         for label, color in colors.items()]
        ax.legend(handles=legend_patches, loc='upper right')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


# Helper classes for music elements
class MusicElement:
    """Base class for all music elements."""
    def __init__(self, x, y, width, height, class_id, class_name, bbox, confidence):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.staff_system = None

class Note(MusicElement):
    """Class representing a note element."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pitch = None  # Pitch object
        self.duration = None  # Duration in quarter notes (1 = quarter, 0.5 = eighth, etc.)
        self.duration_type = None  # String like "quarter", "eighth", etc.
        self.stem = None  # Stem object
        self.beam = None  # Connected beam
        self.on_ledger_line = False  # Whether the note is on a ledger line
        self.accidental = None  # Associated accidental
        self.chord = None  # Chord this note belongs to

class Accidental(MusicElement):
    """Class representing an accidental element."""
    def __init__(self, *args, accidental_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accidental_type = accidental_type  # 'sharp', 'flat', 'natural'
        self.affected_note = None
        self.is_key_signature = False  # Whether this is part of a key signature

class Barline(MusicElement):
    """Class representing a barline element."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.barline_type = None  # 'normal', 'double', 'final', etc.

class Clef(MusicElement):
    """Class representing a clef element."""
    def __init__(self, *args, clef_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clef_type = clef_type  # ClefType enum
        self.line = None  # Line where the clef is positioned

class Beam(MusicElement):
    """Class representing a beam element."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connected_notes = []  # Notes connected to this beam

class Stem:
    """Class representing a note stem."""
    def __init__(self, x, y, width, height, up, note):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.up = up  # Whether the stem points up
        self.note = note

class Chord:
    """Class representing a chord (group of vertically aligned notes)."""
    def __init__(self, notes, staff_system):
        self.notes = notes
        self.staff_system = staff_system
        self.x = notes[0].x if notes else 0
        self.y = sum(note.y for note in notes) / len(notes) if notes else 0

class KeySignature:
    """Class representing a key signature."""
    def __init__(self, x, y, accidentals, staff_system):
        self.x = x
        self.y = y
        self.accidentals = accidentals
        self.staff_system = staff_system
        
        # Calculate the number of sharps or flats
        self.num_sharps = sum(1 for acc in accidentals if acc.accidental_type == 'sharp')
        self.num_flats = sum(1 for acc in accidentals if acc.accidental_type == 'flat')

class StaffSystem:
    """Class representing a staff system (group of staff lines)."""
    def __init__(self, system_id):
        self.id = system_id
        self.lines = {}  # Map from line number to y-position
        self.elements = []  # All elements on this staff
        self.measures = []  # Measures identified on this staff
        self.key_signature = None  # Key signature for this staff
    
    def add_line(self, line_num, y_position):
        """Add a staff line to this system."""
        self.lines[line_num] = y_position
    
    def add_element(self, element):
        """Add a music element to this staff system."""
        self.elements.append(element)

class Measure:
    """Class representing a measure in the score."""
    def __init__(self, start_x, end_x, staff_system):
        self.start_x = start_x
        self.end_x = end_x
        self.staff_system = staff_system
        self.elements = []
    
    def add_element(self, element):
        """Add a music element to this measure."""
        self.elements.append(element)


# Usage example
def process_file(detection_path, staff_lines_path, class_mapping_path, output_xml_path, output_image_path):
    """Process a single music score file and generate MusicXML and visualization."""
    # Create processor
    processor = OMRProcessor(detection_path, staff_lines_path, class_mapping_path)
    
    # Process
    musicxml = processor.process()
    
    # Save MusicXML
    if musicxml and output_xml_path:
        with open(output_xml_path, "w") as f:
            f.write(musicxml)
    
    # Visualize
    processor.visualize(output_image_path)
    
    return musicxml

def main():
    # Replace with actual paths
    detection_path = "/homes/es314/omr-objdet-benchmark/results/detections/Accidentals-004_detections.json"
    staff_lines_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/staff_lines/v2-accidentals-004_staff_lines.json"
    class_mapping_path = "/homes/es314/omr-objdet-benchmark/data/class_mapping.json"
    
    # Output paths
    output_xml_path = "output.musicxml"
    output_image_path = "visualization.png"
    
    # Process file
    process_file(detection_path, staff_lines_path, class_mapping_path, output_xml_path, output_image_path)

if __name__ == "__main__":
    main()