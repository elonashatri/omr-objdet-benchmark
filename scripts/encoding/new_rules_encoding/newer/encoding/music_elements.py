import xml.etree.ElementTree as ET

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
import xml.etree.ElementTree as ET

class Note(MusicElement):
    """Class representing a note element with debug logging for pitch assignment."""
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
        self.duration = None
        self.voice = None
        self.accidental = None
        self.stem_direction = None
        self.beams = []
        self.flag = None
        self.ledger_lines = []
        self.is_chord_member = False
        self.chord = None
        self.duration_type = 'quarter'
        self.duration = 1.0
        self.duration_dots = 0
        self.step = None
        self.octave = None
        self.alter = 0

    def set_pitch(self, step, octave, alter=0):
        """Set the pitch for this note and print debug info."""
        self.step = step
        self.octave = octave
        self.alter = alter
        # print(f"DEBUG: Setting pitch for note at y={self.y:.2f} → {self.pitch} (alter={alter})")
        
    @property
    def pitch(self):
        """Dynamically compute the pitch with accidental."""
        if self.step is None or self.octave is None:
            return None
        accidental = '#' if self.alter == 1 else 'b' if self.alter == -1 else ''
        return f"{self.step}{accidental}{self.octave}"

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
    
    def position_to_xml(self):
        """
        Convert the note's position to a MusicXML pitch or unpitched element.
        Handles percussion notes differently.
        """
        if hasattr(self, 'is_percussion') and self.is_percussion:
            # Create unpitched element for percussion notes
            unpitched = ET.Element('unpitched')
            
            # Use the step and octave as display values
            display_step = ET.SubElement(unpitched, 'display-step')
            display_step.text = self.step
            
            display_octave = ET.SubElement(unpitched, 'display-octave')
            display_octave.text = str(self.octave)
            
            return unpitched
        else:
            # Regular pitched note
            pitch = ET.Element('pitch')
            
            step = ET.SubElement(pitch, 'step')
            step.text = self.step
            
            octave = ET.SubElement(pitch, 'octave')
            octave.text = str(self.octave)
            
            # Add alteration if not 0
            if hasattr(self, 'alter') and self.alter != 0:
                alter = ET.SubElement(pitch, 'alter')
                alter.text = str(self.alter)
            
            return pitch

# Just showing this class for now — you'd use DebuggableNote instead of Note in your processor
# and make sure accidental assignment also logs if it's modifying a note's alter.



# class Note(MusicElement):
#     """Class representing a note element."""
#     def __init__(self, class_id, class_name, confidence, bbox):
#         super().__init__(class_id, class_name, confidence, bbox)
#         self.accidental = None
#         self.stem_direction = None
#         self.beams = []
#         self.flag = None
#         self.ledger_lines = []
#         self.is_chord_member = False
#         self.chord = None
#         self.duration_type = 'quarter'  # Default
#         self.duration = 1.0  # Default quarter note (in quarter notes)
#         self.duration_dots = 0
#         self.step = None
#         self.octave = None
#         self.alter = 0  # 0 = natural, 1 = sharp, -1 = flat
        
#     def set_pitch(self, step, octave, alter=0):
#         """Set the pitch for this note."""
#         self.step = step
#         self.octave = octave
#         self.alter = alter
#         self.pitch = f"{step}{octave}"  # Basic string representation
        
#     def position_to_xml(self):
#         """Generate MusicXML pitch element."""
#         pitch = ET.Element('pitch')
#         step = ET.SubElement(pitch, 'step')
#         step.text = self.step
        
#         if self.alter != 0:
#             alter = ET.SubElement(pitch, 'alter')
#             alter.text = str(self.alter)
            
#         octave = ET.SubElement(pitch, 'octave')
#         octave.text = str(self.octave)
        
#         return pitch

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
        print(f"DEBUG: Created accidental {self.type} with alter={self.alter} at y={self.y:.2f}")


class Clef(MusicElement):
    """Class representing a clef element."""
    def __init__(self, class_id, class_name, confidence, bbox):
        super().__init__(class_id, class_name, confidence, bbox)
        
        # Determine clef type from class name
        if 'gClef' in class_name or class_name == 'G':
            self.type = 'G'
            self.line = 2
        elif 'fClef' in class_name or class_name == 'F':
            self.type = 'F'
            self.line = 4
        elif 'cClef' in class_name or class_name == 'C':
            self.type = 'C'
            
            # Determine specific C clef type based on position
            # (would need staff line positions to determine exactly)
            self.line = 3  # Default to alto clef
        elif 'percussion' in class_name.lower():
            self.type = 'percussion'
            self.line = 2
        else:
            # Default to G clef
            self.type = 'G'
            self.line = 2
        
        # Handle transposing clefs
        if '8va' in class_name:
            self.type += '8va'
        elif '8vb' in class_name:
            self.type += '8vb'

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