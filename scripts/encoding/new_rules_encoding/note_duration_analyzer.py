import numpy as np
from collections import defaultdict, namedtuple

class NoteHead:
    """Class representing a detected notehead with its properties."""
    def __init__(self, detection):
        self.detection = detection
        self.bbox = detection['bbox']
        self.x = self.bbox['center_x']
        self.y = self.bbox['center_y']
        self.width = self.bbox['width']
        self.height = self.bbox['height']
        self.class_name = detection['class_name']
        
        # Attributes to be set during processing
        self.staff_system = None
        self.staff_position = None  # Position on the staff (line number or between lines)
        self.pitch = None
        self.stem = None
        self.beams = []
        self.flags = []
        self.accidental = None
        self.duration = 1.0  # Default to quarter note (will be updated during processing)
        self.duration_type = 'quarter'  # String representation of duration
        self.is_in_chord = False
        self.chord = None

class Beam:
    """Class representing a beam connecting multiple noteheads."""
    def __init__(self, detection):
        self.detection = detection
        self.bbox = detection['bbox']
        self.x = self.bbox['center_x']
        self.y = self.bbox['center_y']
        self.width = self.bbox['width']
        self.height = self.bbox['height']
        
        # Attributes to be set during processing
        self.connected_notes = []
        self.level = None  # Level of the beam (primary, secondary, etc.)

class Stem:
    """Class representing a note stem."""
    def __init__(self, detection=None, x=None, y=None, height=None, is_up=None):
        if detection:
            self.detection = detection
            self.bbox = detection['bbox']
            self.x = self.bbox['center_x']
            self.y = self.bbox['center_y']
            self.width = self.bbox['width']
            self.height = self.bbox['height']
        else:
            # Create a stem from explicit parameters (for inferred stems)
            self.detection = None
            self.x = x
            self.y = y
            self.width = 1.0  # Default stem width
            self.height = height if height is not None else 3.5 * 20  # Default height based on staff line distance
        
        self.is_up = is_up  # Direction of the stem
        self.note = None  # Associated notehead

class Flag:
    """Class representing a note flag."""
    def __init__(self, detection):
        self.detection = detection
        self.bbox = detection['bbox']
        self.x = self.bbox['center_x']
        self.y = self.bbox['center_y']
        self.width = self.bbox['width']
        self.height = self.bbox['height']
        
        # Attributes to be set during processing
        self.associated_note = None
        self.count = 1  # Number of flags (affects duration)

class DurationAnalyzer:
    """
    Analyzes note durations based on noteheads, stems, beams, and flags.
    """
    def __init__(self, staff_line_distance=20):
        self.staff_line_distance = staff_line_distance
        self.noteheads = []
        self.beams = []
        self.stems = []
        self.flags = []
    
    def add_elements_from_detections(self, detections):
        """
        Add music elements from detection data.
        
        Args:
            detections: List of detection dictionaries
        """
        for det in detections:
            class_name = det['class_name']
            
            if 'notehead' in class_name.lower():
                self.noteheads.append(NoteHead(det))
            elif 'beam' in class_name.lower():
                self.beams.append(Beam(det))
            elif 'stem' in class_name.lower():
                self.stems.append(Stem(det))
            elif 'flag' in class_name.lower():
                self.flags.append(Flag(det))
    
    def infer_stems(self):
        """
        Infer stems for noteheads that don't have explicit stem detections.
        This creates synthetic stem objects based on notehead positions and beams.
        """
        # First, link explicit stems to noteheads
        for stem in self.stems:
            # Find the closest notehead to this stem
            closest_note = None
            min_distance = float('inf')
            
            for note in self.noteheads:
                # Calculate horizontal distance (stems should be very close horizontally)
                dx = abs(note.x - stem.x)
                
                # Only consider noteheads that are close enough horizontally
                if dx > self.staff_line_distance / 2:
                    continue
                
                # Stems are typically attached to the right or left side of the notehead
                # and extend either up or down
                note_top = note.y - note.height / 2
                note_bottom = note.y + note.height / 2
                
                # Check if stem passes through or touches the notehead
                if (stem.bbox['y1'] <= note_bottom and stem.bbox['y2'] >= note_top):
                    # Stem intersects with notehead
                    distance = dx
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_note = note
            
            if closest_note:
                # Link the stem to the notehead
                stem.note = closest_note
                closest_note.stem = stem
                
                # Determine stem direction
                stem.is_up = (stem.bbox['y1'] < closest_note.y)
        
        # For noteheads without stems, infer stems based on beams or standard rules
        for note in self.noteheads:
            if note.stem is not None:
                continue  # Already has a stem
            
            # Check if there are beams that could connect to this note
            potential_beams = []
            for beam in self.beams:
                # Check if the beam is horizontally aligned with the note
                if (beam.bbox['x1'] <= note.x <= beam.bbox['x2']):
                    potential_beams.append(beam)
            
            if potential_beams:
                # Sort beams by vertical distance to the note
                potential_beams.sort(key=lambda b: abs(b.y - note.y))
                
                # The closest beam is likely connected to this note
                closest_beam = potential_beams[0]
                
                # Infer stem direction based on beam position
                is_up = closest_beam.y < note.y
                
                # Create a stem for this note
                stem = Stem(
                    x=note.x,
                    y=note.y,
                    height=abs(closest_beam.y - note.y),
                    is_up=is_up
                )
                
                # Link stem and note
                stem.note = note
                note.stem = stem
                self.stems.append(stem)
                
                # Link note to beam
                closest_beam.connected_notes.append(note)
                note.beams.append(closest_beam)
            else:
                # No beam, infer stem direction based on position on staff
                # Notes above the middle line typically have stems down
                # Notes below the middle line typically have stems up
                if note.staff_position is not None:
                    # Middle line is usually position 2 (0-indexed)
                    is_up = note.staff_position >= 2
                else:
                    # Default if staff position is unknown
                    is_up = True
                
                # Create a stem with standard length
                stem_length = self.staff_line_distance * 3.5
                
                stem = Stem(
                    x=note.x,
                    y=note.y,
                    height=stem_length,
                    is_up=is_up
                )
                
                # Link stem and note
                stem.note = note
                note.stem = stem
                self.stems.append(stem)
    
    def connect_beams_to_stems(self):
        """Connect beams to note stems based on proximity."""
        for beam in self.beams:
            # Skip beams that already have connected notes
            if beam.connected_notes:
                continue
            
            # Find stems that could connect to this beam
            for note in self.noteheads:
                if not note.stem:
                    continue
                
                stem = note.stem
                
                # Check if the note is under the beam horizontally
                if not (beam.bbox['x1'] <= note.x <= beam.bbox['x2']):
                    continue
                
                # Check if the stem could connect to the beam vertically
                stem_top = note.y - stem.height if stem.is_up else note.y
                stem_bottom = note.y if stem.is_up else note.y + stem.height
                
                if ((stem.is_up and beam.y >= stem_top and beam.y <= note.y) or
                    (not stem.is_up and beam.y <= stem_bottom and beam.y >= note.y)):
                    beam.connected_notes.append(note)
                    note.beams.append(beam)
    
    def connect_flags_to_notes(self):
        """Connect flags to notes based on proximity."""
        for flag in self.flags:
            # Find the closest note with a stem
            closest_note = None
            min_distance = float('inf')
            
            for note in self.noteheads:
                if not note.stem:
                    continue
                
                # Flags are typically attached to the end of stems
                stem = note.stem
                
                # Calculate stem endpoint
                stem_end_x = stem.x
                stem_end_y = note.y - stem.height if stem.is_up else note.y + stem.height
                
                # Calculate distance to flag
                distance = np.sqrt((stem_end_x - flag.x)**2 + (stem_end_y - flag.y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_note = note
            
            # Link flag to note if found and close enough
            if closest_note and min_distance < self.staff_line_distance * 1.5:
                flag.associated_note = closest_note
                closest_note.flags.append(flag)
    
    def analyze_durations(self):
        """
        Analyze and set durations for all noteheads based on stems, beams, and flags.
        """
        for note in self.noteheads:
            # Default duration is quarter note
            duration = 1.0
            duration_type = 'quarter'
            
            # Check for beams and flags
            if note.beams or note.flags:
                if note.beams:
                    # Count unique beam levels
                    # This is a simplification - in reality, we would need to 
                    # determine beam levels based on vertical positions
                    beam_levels = len(note.beams)
                    
                    # Adjust duration based on beam count
                    if beam_levels == 1:
                        duration = 0.5  # Eighth note
                        duration_type = 'eighth'
                    elif beam_levels == 2:
                        duration = 0.25  # Sixteenth note
                        duration_type = '16th'
                    elif beam_levels >= 3:
                        duration = 0.125  # Thirty-second note
                        duration_type = '32nd'
                
                if note.flags:
                    # Count flags
                    flag_count = sum(flag.count for flag in note.flags)
                    
                    # Adjust duration based on flag count
                    if flag_count == 1:
                        duration = 0.5  # Eighth note
                        duration_type = 'eighth'
                    elif flag_count == 2:
                        duration = 0.25  # Sixteenth note
                        duration_type = '16th'
                    elif flag_count >= 3:
                        duration = 0.125  # Thirty-second note
                        duration_type = '32nd'
            
            # Set the duration and type
            note.duration = duration
            note.duration_type = duration_type
    
    def analyze(self):
        """
        Run the complete duration analysis pipeline.
        """
        # Step 1: Infer stems for noteheads if needed
        self.infer_stems()
        
        # Step 2: Connect beams to stems
        self.connect_beams_to_stems()
        
        # Step 3: Connect flags to notes
        self.connect_flags_to_notes()
        
        # Step 4: Analyze durations
        self.analyze_durations()
        
        return {
            'noteheads': self.noteheads,
            'stems': self.stems,
            'beams': self.beams,
            'flags': self.flags
        }
    
    def visualize(self, output_path=None):
        """
        Visualize the duration analysis results.
        
        Args:
            output_path: Path to save the visualization image
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw noteheads
        for note in self.noteheads:
            rect = patches.Rectangle(
                (note.bbox['x1'], note.bbox['y1']),
                note.width, note.height,
                linewidth=1, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add duration label
            ax.text(note.x, note.y, note.duration_type,
                   ha='center', va='center', fontsize=8, color='green')
        
        # Draw stems
        for stem in self.stems:
            if stem.detection:
                # Explicit stem detection
                rect = patches.Rectangle(
                    (stem.bbox['x1'], stem.bbox['y1']),
                    stem.width, stem.height,
                    linewidth=1, edgecolor='blue', facecolor='none'
                )
                ax.add_patch(rect)
            else:
                # Inferred stem
                if stem.is_up:
                    stem_start = (stem.x, stem.note.y)
                    stem_end = (stem.x, stem.note.y - stem.height)
                else:
                    stem_start = (stem.x, stem.note.y)
                    stem_end = (stem.x, stem.note.y + stem.height)
                
                ax.plot([stem_start[0], stem_end[0]], [stem_start[1], stem_end[1]],
                       'b-', linewidth=1)
        
        # Draw beams
        for beam in self.beams:
            rect = patches.Rectangle(
                (beam.bbox['x1'], beam.bbox['y1']),
                beam.width, beam.height,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Draw connections to notes
            for note in beam.connected_notes:
                ax.plot([note.x, note.x], [note.y, beam.y], 'r--', alpha=0.5)
        
        # Draw flags
        for flag in self.flags:
            rect = patches.Rectangle(
                (flag.bbox['x1'], flag.bbox['y1']),
                flag.width, flag.height,
                linewidth=1, edgecolor='purple', facecolor='none'
            )
            ax.add_patch(rect)
        
        # Set axis limits
        all_x = [note.bbox['x1'] for note in self.noteheads] + [note.bbox['x2'] for note in self.noteheads]
        all_y = [note.bbox['y1'] for note in self.noteheads] + [note.bbox['y2'] for note in self.noteheads]
        
        if self.beams:
            all_x.extend([beam.bbox['x1'] for beam in self.beams] + [beam.bbox['x2'] for beam in self.beams])
            all_y.extend([beam.bbox['y1'] for beam in self.beams] + [beam.bbox['y2'] for beam in self.beams])
        
        margin = 50
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Note Duration Analysis')
        
        # Add legend
        legend_elements = [
            patches.Patch(edgecolor='green', facecolor='none', label='Notehead'),
            patches.Patch(edgecolor='blue', facecolor='none', label='Stem'),
            patches.Patch(edgecolor='red', facecolor='none', label='Beam'),
            patches.Patch(edgecolor='purple', facecolor='none', label='Flag')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save or show the visualization
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

# Example usage:
def analyze_durations(detections, staff_line_distance=20, output_path=None):
    """
    Analyze note durations from detection data.
    
    Args:
        detections: List of detection dictionaries
        staff_line_distance: Distance between staff lines in pixels
        output_path: Path to save visualization
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = DurationAnalyzer(staff_line_distance)
    analyzer.add_elements_from_detections(detections)
    results = analyzer.analyze()
    
    if output_path:
        analyzer.visualize(output_path)
    
    return results