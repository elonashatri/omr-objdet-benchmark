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
        """Convert y-coordinate to pitch based on staff position with better handling of notes outside the staff."""
        if not clef:
            clef = self.clef
            
        if not clef or not self.line_spacing or not self.lines:
            return (None, None)
        
        # Get staff lines sorted by y-position (top to bottom)
        sorted_lines = sorted(self.lines.items(), key=lambda x: x[1])
        line_positions = [pos for _, pos in sorted_lines]
        
        # In treble clef, standard line pitches (bottom to top)
        line_pitches = ['E4', 'G4', 'B4', 'D5', 'F5']
        
        # Determine if note is within or outside the staff
        top_line = line_positions[0]
        bottom_line = line_positions[-1]
        
        # Calculate step size based on staff line spacing
        step_size = self.line_spacing / 2  # Half step is space between line and adjacent space
        
        # Calculate how many steps above or below the staff
        if y < top_line:
            # Note is above the staff
            steps_above = round((top_line - y) / step_size)
            
            # Extend pattern upward from top line (F5)
            # Each step up gives us: F5, G5, A5, B5, C6, D6, E6, F6, etc.
            extended_pitches = ['F5', 'G5', 'A5', 'B5', 'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6', 'C7']
            
            if steps_above < len(extended_pitches):
                pitch_name = extended_pitches[steps_above]
            else:
                # If extremely high, use a reasonable upper limit
                pitch_name = 'C8'
                
        elif y > bottom_line:
            # Note is below the staff
            steps_below = round((y - bottom_line) / step_size)
            
            # Extend pattern downward from bottom line (E4)
            # Each step down gives us: E4, D4, C4, B3, A3, G3, F3, etc.
            extended_pitches = ['E4', 'D4', 'C4', 'B3', 'A3', 'G3', 'F3', 'E3', 'D3', 'C3', 'B2']
            
            if steps_below < len(extended_pitches):
                pitch_name = extended_pitches[steps_below]
            else:
                # If extremely low, use a reasonable lower limit
                pitch_name = 'C2'
        else:
            # Note is within the staff - use standard position calculation
            for i in range(len(line_positions) - 1):
                if line_positions[i] <= y <= line_positions[i+1]:
                    # Interpolate position between lines
                    pos_ratio = (y - line_positions[i]) / (line_positions[i+1] - line_positions[i])
                    position = i + pos_ratio
                    
                    # Determine if on a line or in a space
                    rounded_pos = round(position * 2) / 2  # Round to nearest half-step
                    
                    # Map to pitch
                    if clef.type == 'G':  # Treble clef
                        # Each line position (0, 1, 2, 3, 4) maps to a specific pitch
                        # Each space position (0.5, 1.5, 2.5, 3.5) also maps to a specific pitch
                        all_pitches = ['E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5']
                        
                        # Convert position to index in all_pitches
                        index = int(rounded_pos * 2)
                        if 0 <= index < len(all_pitches):
                            pitch_name = all_pitches[index]
                        else:
                            return ('C', 4)  # Default fallback
                    else:
                        return ('C', 4)  # Only treble clef implemented
                    
                    break
            else:
                # This shouldn't happen if the note is within the staff
                return ('C', 4)
        
        # Extract step and octave from pitch name
        step = pitch_name[0]
        octave = int(pitch_name[1:])
        
        return (step, octave)
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