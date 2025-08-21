# /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/music_elements.py
import xml.etree.ElementTree as ET
class Tie:
    """
    Represents a tie between two notes.
    """
    def __init__(self, x=0, y=0, width=0, height=0, start_note=None, end_note=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.start_note = start_note  # Reference to the starting note
        self.end_note = end_note      # Reference to the ending note
        
        # Update the notes if they're provided
        if start_note:
            start_note.tie_start = True
            start_note.tie = self
        if end_note:
            end_note.tie_end = True
            end_note.tie = self


class Slur:
    """
    Represents a slur connecting multiple notes.
    """
    def __init__(self, x=0, y=0, width=0, height=0, id=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.id = id  # Slur identifier (for multiple concurrent slurs)
        self.notes = []  # List of notes connected by this slur
    
    def add_note(self, note, position="middle"):
        """
        Add a note to this slur.
        
        Args:
            note: The Note object to add
            position: "start", "middle", or "end"
        """
        self.notes.append(note)
        
        # Initialize lists if they don't exist
        if not hasattr(note, 'slur_starts'):
            note.slur_starts = []
        if not hasattr(note, 'slur_ends'):
            note.slur_ends = []
            
        # Mark the note as start or end of slur
        if position == "start":
            note.slur_starts.append(self.id)
        elif position == "end":
            note.slur_ends.append(self.id)
        
        # Store reference to the slur
        if not hasattr(note, 'slurs'):
            note.slurs = []
        note.slurs.append(self)


class Dynamic:
    """
    Represents a dynamic marking (p, f, mf, etc.).
    """
    def __init__(self, x=0, y=0, width=0, height=0, type="f", measure=None, staff=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type  # "p", "pp", "f", "ff", "mf", "sf", etc.
        self.measure = measure  # Reference to the containing measure
        self.staff = staff  # Staff number (for multi-staff systems)
        self.default_x = None  # Optional positioning
        self.default_y = None  # Optional positioning


class GradualDynamic:
    """
    Represents a gradual dynamic marking like crescendo or diminuendo.
    """
    def __init__(self, x=0, y=0, width=0, height=0, type="crescendo", measure=None, staff=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type  # "crescendo", "diminuendo"
        self.measure = measure  # Reference to the containing measure
        self.staff = staff  # Staff number
        self.default_x = None  # Optional positioning
        self.default_y = None  # Optional positioning
        self.start_note = None
        self.end_note = None


class Tuplet:
    """
    Represents a tuplet grouping (triplet, etc.).
    """
    def __init__(self, x=0, y=0, width=0, height=0, actual_notes=3, normal_notes=2):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.actual_notes = actual_notes  # Number of notes in tuplet (e.g., 3 for triplet)
        self.normal_notes = normal_notes  # Number of notes normally (e.g., 2 for triplet)
        self.notes = []  # List of notes in this tuplet
    
    def add_note(self, note, position="middle"):
        """
        Add a note to this tuplet.
        
        Args:
            note: The Note or Rest object to add
            position: "start", "middle", or "end"
        """
        self.notes.append(note)
        
        # Mark note as part of tuplet
        note.is_tuplet = True
        note.tuplet_data = (self.actual_notes, self.normal_notes)
        
        # Mark start/end notes
        if position == "start":
            note.tuplet_start = True
        elif position == "end":
            note.tuplet_end = True
        
        # Store reference to the tuplet
        note.tuplet = self


class TupletBracket:
    """
    Represents a tuplet bracket.
    """
    def __init__(self, x=0, y=0, width=0, height=0, tuplet=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.tuplet = tuplet  # Reference to the tuplet this bracket belongs to


class TupletText:
    """
    Represents tuplet text (3, 5, etc.).
    """
    def __init__(self, x=0, y=0, width=0, height=0, text="3", tuplet=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text  # The tuplet number as text
        self.tuplet = tuplet  # Reference to the tuplet this text belongs to


class Articulation:
    """
    Represents an articulation mark (staccato, accent, etc.).
    """
    def __init__(self, x=0, y=0, width=0, height=0, type="staccato", placement="above", note=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type  # "staccato", "accent", "tenuto", etc.
        self.placement = placement  # "above" or "below"
        
        # Link to note if provided
        if note:
            if not hasattr(note, 'articulations'):
                note.articulations = []
            note.articulations.append(self)
            self.note = note


class Ornament:
    """
    Represents an ornament (trill, mordent, turn, etc.).
    """
    def __init__(self, x=0, y=0, width=0, height=0, type="trill", note=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type  # "trill", "mordent", "turn", etc.
        
        # Link to note if provided
        if note:
            if not hasattr(note, 'ornaments'):
                note.ornaments = []
            note.ornaments.append(self)
            self.note = note


class AugmentationDot:
    """
    Represents a dot that extends the duration of a note or rest.
    """
    def __init__(self, x=0, y=0, width=0, height=0, note_or_rest=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Link to note or rest if provided
        if note_or_rest:
            note_or_rest.augmentation_dots = getattr(note_or_rest, 'augmentation_dots', 0) + 1
            self.note_or_rest = note_or_rest


class Flag:
    """
    Represents a flag on a stem (8th, 16th, 32nd, etc.).
    """
    def __init__(self, x=0, y=0, width=0, height=0, type="8thUp", note=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type  # "8thUp", "8thDown", "16thUp", etc.
        
        # Link to note if provided
        if note:
            note.flag = self
            self.note = note


class Beam:
    """
    Represents a beam connecting multiple notes.
    """
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.notes = []  # List of notes connected by this beam
    
    def add_note(self, note, position="middle"):
        """
        Add a note to this beam.
        
        Args:
            note: The Note object to add
            position: "start", "middle", or "end", or "continue"
        """
        self.notes.append(note)
        
        # Store beam type on the note
        if not hasattr(note, 'beam_types'):
            note.beam_types = []
        
        # Add appropriate beam type
        if position == "start":
            note.beam_types.append(("begin", 1))
        elif position == "end":
            note.beam_types.append(("end", 1))
        elif position == "continue":
            note.beam_types.append(("continue", 1))
        else:  # "middle"
            note.beam_types.append(("continue", 1))
        
        # Store reference to the beam
        if not hasattr(note, 'beams'):
            note.beams = []
        note.beams.append(self)


class TextDirection:
    """
    Represents text directions like "sim.", "sempre", "piu", etc.
    """
    def __init__(self, x=0, y=0, width=0, height=0, text="", measure=None, placement="above"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.measure = measure  # Reference to the containing measure
        self.placement = placement  # "above" or "below"


# Dynamic Mapping Dictionary
# Maps class detector names to MusicXML dynamic element names
DYNAMIC_TYPE_MAP = {
    "dynamicPiano": "p",
    "dynamicForte": "f",
    "dynamicPP": "pp",
    "dynamicFF": "ff",
    "dynamicPPP": "ppp",
    "dynamicFFF": "fff",
    "dynamicFFFF": "ffff",
    "dynamicMF": "mf",
    "dynamicMP": "mp",
    "dynamicSforzato": "sf",
    "dynamicSforzando1": "sfz",
    "dynamicForzando": "fz",
    "dynamicFortePiano": "fp",
    "dynamicSforzatoFF": "sff",
    "Sforzandoff": "sffz"
}

# Articulation Mapping Dictionary
# Maps class detector names to MusicXML articulation element names
ARTICULATION_TYPE_MAP = {
    "articStaccatoAbove": {"type": "staccato", "placement": "above"},
    "articStaccatoBelow": {"type": "staccato", "placement": "below"},
    "articAccentAbove": {"type": "accent", "placement": "above"},
    "articAccentBelow": {"type": "accent", "placement": "below"},
    "articStaccatissimoAbove": {"type": "staccatissimo", "placement": "above"},
    "articStaccatissimoBelow": {"type": "staccatissimo", "placement": "below"},
    "articTenutoAbove": {"type": "tenuto", "placement": "above"},
    "articTenutoBelow": {"type": "tenuto", "placement": "below"},
    "articMarcatoAbove": {"type": "strong-accent", "placement": "above"},
    "articMarcatoBelow": {"type": "strong-accent", "placement": "below"}
}

# Ornament Mapping Dictionary
# Maps class detector names to MusicXML ornament element names
ORNAMENT_TYPE_MAP = {
    "ornamentTrill": "trill-mark",
    "wiggleTrill": "trill-mark",
    "ornamentShortTrill": "trill-mark",
    "ornamentTurn": "turn",
    "ornamentMordent": "mordent",
    "ornamentTremblement": "trill-mark",
    "ornamentPrecompTrillWithMordent": "inverted-mordent",
    "ornamentPrecompAppoggTrill": "turn"
}

# Tuplet Mapping Dictionary
# Maps numeric values to actual/normal note ratios
TUPLET_TYPE_MAP = {
    "T2": (2, 3),  # duplet: 2 in the time of 3
    "T3": (3, 2),  # triplet: 3 in the time of 2
    "T4": (4, 3),  # quadruplet: 4 in the time of 3
    "T5": (5, 4),  # quintuplet: 5 in the time of 4
    "T6": (6, 4),  # sextuplet: 6 in the time of 4
    "T7": (7, 4),  # septuplet: 7 in the time of 4
    "T9": (9, 8),  # nonuplet: 9 in the time of 8
    "T10": (10, 8)  # decuplet: 10 in the time of 8
}

# Text Direction Mapping
TEXT_DIRECTION_MAP = {
    "sub.": "subito",
    "sim.": "simile",
    "sempre": "sempre",
    "piu": "più"
}

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
                # Properties for ties
        self.tie_start = False
        self.tie_end = False
        self.tie = None
        
        # Properties for slurs
        self.slur_starts = []
        self.slur_ends = []
        self.slurs = []
        
        # Properties for articulations
        self.articulations = []
        
        # Properties for ornaments
        self.ornaments = []
        
        # Properties for tuplets
        self.is_tuplet = False
        self.tuplet_data = None  # (actual_notes, normal_notes)
        self.tuplet_start = False
        self.tuplet_end = False
        self.tuplet = None
        
        # Properties for beams
        self.beam_types = []  # List of (type, level) tuples
        self.beams = []
        
        # Properties for augmentation dots
        self.augmentation_dots = 0
        
        # Flag
        self.flag = None
        

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
        # Properties for tuplets
        self.is_tuplet = False
        self.tuplet_data = None  # (actual_notes, normal_notes)
        self.tuplet_start = False
        self.tuplet_end = False
        self.tuplet = None
        
        # Properties for augmentation dots
        self.augmentation_dots = 0
        
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