# /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/processor.py
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import csv
import generate_musicxml

from music_elements import (
    Note, Rest, Accidental, Clef, Barline, Beam, Flag, 
    TimeSignatureElement, KeySignature, TimeSignature,
    Tie, Slur, Dynamic, GradualDynamic, Tuplet, TupletBracket, TupletText,
    Articulation, Ornament, AugmentationDot, TextDirection,
    # Add these mapping dictionaries
    DYNAMIC_TYPE_MAP, ARTICULATION_TYPE_MAP, ORNAMENT_TYPE_MAP, TUPLET_TYPE_MAP, TEXT_DIRECTION_MAP
)
from staff_system import StaffSystem, Measure, find_closest_staff

# Import utils functions - if there's an issue with the import, we have local fallbacks
try:
    from utils import (
        load_csv_detections, load_json_detections, load_json, 
        calculate_typical_staff_spacing
    )
except ImportError:
    # Define fallback functions if imports fail
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
                self.detections = load_csv_detections(detection_path)
            else:
                self.detections = load_json_detections(detection_path)
        elif detection_data:
            self.detections = detection_data
            
        self.staff_lines = None
        if staff_lines_path:
            self.staff_lines = load_json(staff_lines_path)
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
            self.typical_staff_spacing = calculate_typical_staff_spacing(self.staff_lines)
    


    def process(self):
        from debug_chord import debug_chord_connections, debug_leftover_chord_attributes
        """Process the detections and staff lines to generate MusicXML."""
        if not self.detections or not self.staff_lines:
            print("Missing required data. Cannot process.")
            return None
        
        # STEP 1: Identify staff systems
        self.identify_staff_systems_auto()
        
        # STEP 2: Process detected objects and assign to staff systems
        self.process_detected_objects()
        self.assign_to_staff_systems()
        self.validate_staff_lines()

        # Process systemic barlines (both detected and inferred)
        self.process_systemic_barlines()
        self.identify_systemic_barlines()
        
    
        # STEP 3: Process clefs and time signatures (needed for barline detection)
        self.interpret_time_signatures()
        if not any(system.time_signature for system in self.staff_systems):
            self.analyze_possible_time_signatures()
        
        # STEP 4: Process initial note properties (needed for barline detection)
        self.calculate_note_pitches()
        self.connect_flags_to_notes()
        self.connect_beams_to_notes()
        self.calculate_note_durations()  # Initial duration calculation

        # STEP 5: Enhanced barline detection 
        print("\n=== RUNNING ENHANCED BARLINE DETECTION ===")
        self.master_barline_inference()
        
        # STEP 6: Identify measures with enhanced barlines
        self.identify_measures()
        
        # STEP 7: Connect musical elements (MOVED AFTER measure identification)
        self.connect_music_elements()
        
        # STEP 8: Process note relationships
        self.group_notes_into_chords()  # Do chord detection before accidentals
        debug_chord_connections(self)
        debug_leftover_chord_attributes(self)
        
        # STEP 9: Process key signatures and accidentals
        self.identify_key_signatures()
        self.connect_accidentals_to_notes()  # Respects measure boundaries
        
        # STEP 10: Final pass for durations and validation
        self.calculate_note_durations()  # Final duration calculation
        
        # Final verification
        print("\n=== CHECKING FINAL NOTE PITCHES AFTER ACCIDENTALS ===")
        for i, note in enumerate(self.notes):
            print(f"Note {i}: x={note.x:.1f}, y={note.y:.1f}, step={note.step}, octave={note.octave}, alter={note.alter}, pitch={note.pitch}")
        
        return self.generate_musicxml()
    
    def identify_staff_systems(self, mode="auto"):
        """Choose the right staff system identification method."""
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
        
        # Initial detection of systemic barlines
        systemic_barlines = []
        for det in self.detections:
            if isinstance(det, dict) and det.get('class_name') == 'systemicBarline':
                systemic_barlines.append(det)
            elif isinstance(det, list) and 'detections' in det:
                for sub_det in det['detections']:
                    if sub_det.get('class_name') == 'systemicBarline':
                        systemic_barlines.append(sub_det)
        
        print(f"Found {len(systemic_barlines)} systemic barlines for initial grouping")
        
        # If we detected systemic barlines, use them to inform staff grouping
        if systemic_barlines:
            # For vocal scores with systemic barlines, use a different grouping approach
            # First staff might be solo, others paired
            
            # Analyze the layout - count number of staff lines above first systemic barline
            if len(staves) >= 2:  # Need at least 2 staves
                # For vocal scores, group staves in pairs except for the first if it's solo
                # Determine if first staff is solo
                first_solo = True
                
                # Build systems accordingly
                if first_solo and len(staves) >= 3:
                    # First system has one staff, rest have pairs
                    # First system (solo)
                    system = self._build_system_from_staves([staves[0]], 0)
                    self.staff_systems.append(system)
                    
                    # Remaining systems in pairs
                    for i in range(1, len(staves), 2):
                        if i+1 < len(staves):
                            system = self._build_system_from_staves([staves[i], staves[i+1]], len(self.staff_systems))
                            self.staff_systems.append(system)
                        else:
                            # Handle odd number of staves
                            system = self._build_system_from_staves([staves[i]], len(self.staff_systems))
                            self.staff_systems.append(system)
                else:
                    # Default: group in pairs
                    for i in range(0, len(staves), 2):
                        if i+1 < len(staves):
                            system = self._build_system_from_staves([staves[i], staves[i+1]], len(self.staff_systems))
                            self.staff_systems.append(system)
                        else:
                            # Handle odd number of staves
                            system = self._build_system_from_staves([staves[i]], len(self.staff_systems))
                            self.staff_systems.append(system)
            else:
                # Fallback to standard approach
                for i, staff in enumerate(staves):
                    system = self._build_system_from_staves([staff], i)
                    self.staff_systems.append(system)
        else:
            # Standard approach when no systemic barlines are detected
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

        # Ensure staves are explicitly split (needed for pitch estimation)
        for system in self.staff_systems:
            system.split_into_staves()
            
            # Mark multi-staff systems
            if len(system.staves) > 1:
                system.is_multi_staff = True
                print(f"System {system.id} marked as multi-staff with {len(system.staves)} staves")
            else:
                system.is_multi_staff = False

        print(f"[Auto] Identified {len(self.staff_systems)} systems from {len(staves)} staves.")
        for i, system in enumerate(self.staff_systems):
            print(f"  System {i}: {len(system.staves)} staves")
        
    def _build_system_from_staves(self, staves, system_id):
        """Build a StaffSystem from a list of staves."""
        system = StaffSystem(system_id)
        line_number = 0
        for staff in staves:
            sorted_lines = sorted(staff, key=lambda l: l['bbox']['center_y'])
            for i, line in enumerate(sorted_lines):
                system.add_line(line_number, line['bbox']['center_y'])
                line_number += 1
        system.calculate_line_spacing()
        return system
    
    def validate_staff_lines(self):
        """Validate detected staff lines for consistency."""
        for system in self.staff_systems:
            # Check if we have the expected number of lines (typically 5)
            if len(system.lines) != 5 and len(system.lines) != 10:  # piano systems have 10
                print(f"Warning: Staff system {system.id} has {len(system.lines)} lines, expected 5 or 10")
            
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
    

    def connect_music_elements(self):
        """Connect musical elements based on spatial relationships."""
        print("\n=== CONNECTING MUSICAL ELEMENTS ===")
        
        # Connect ties to the closest notes
        for tie in self.ties:
            # Find notes that are horizontally aligned with this tie
            horizontal_margin = tie.width * 1.2  # Allow some margin
            vertical_margin = tie.height * 1.0
            
            # Find notes to the left and right of this tie
            left_notes = [n for n in self.notes if
                        n.x < tie.x and 
                        abs(n.x - tie.x) < horizontal_margin and
                        abs(n.y - tie.y) < vertical_margin]
            
            right_notes = [n for n in self.notes if
                        n.x > tie.x and 
                        abs(n.x - tie.x) < horizontal_margin and
                        abs(n.y - tie.y) < vertical_margin]
            
            # Sort by horizontal distance
            left_notes.sort(key=lambda n: abs(n.x - tie.x))
            right_notes.sort(key=lambda n: abs(n.x - tie.x))
            
            # If we found notes on both sides, connect them with this tie
            if left_notes and right_notes:
                # Check if notes are from the same staff system
                same_system_pairs = []
                for left_note in left_notes:
                    left_system = left_note.staff_system
                    left_staff_id = self._get_staff_id(left_note)
                    
                    for right_note in right_notes:
                        right_system = right_note.staff_system
                        right_staff_id = self._get_staff_id(right_note)
                        
                        # Only connect notes from the same system AND same staff
                        if left_system == right_system and left_staff_id == right_staff_id:
                            same_system_pairs.append((left_note, right_note))
                
                # If we have valid pairs, use the closest one
                if same_system_pairs:
                    # Sort by sum of distances
                    same_system_pairs.sort(key=lambda pair: 
                        abs(pair[0].x - tie.x) + abs(pair[1].x - tie.x))
                    
                    start_note, end_note = same_system_pairs[0]
                    
                    # Update tie with the notes
                    tie.start_note = start_note
                    tie.end_note = end_note
                    
                    # Update notes with tie information
                    start_note.tie_start = True
                    start_note.tie = tie
                    end_note.tie_end = True
                    end_note.tie = tie
                    
                    print(f"Connected tie between notes at ({start_note.x}, {start_note.y}) and ({end_note.x}, {end_note.y}) on same staff")
                else:
                    print(f"Skipped tie at ({tie.x}, {tie.y}) - notes are on different staves")
    
        
        # Connect slurs to notes
        for slur in self.slurs:
            # Similar approach as for ties, but slurs can span multiple notes
            horizontal_span = slur.width * 1.5  # Slurs are usually wider
            vertical_margin = slur.height * 1.2
            
            # Group notes by staff and system
            staff_groups = {}
            
            # Find notes within horizontal span
            for note in self.notes:
                if abs(note.x - slur.x) < horizontal_span and abs(note.y - slur.y) < vertical_margin:
                    system = note.staff_system
                    if system:
                        staff_id = self._get_staff_id(note)
                        key = (system.id, staff_id)
                        if key not in staff_groups:
                            staff_groups[key] = []
                        staff_groups[key].append(note)
            
            # Process each staff group separately
            for (system_id, staff_id), staff_notes in staff_groups.items():
                # Sort notes by x-position
                staff_notes.sort(key=lambda n: n.x)
                
                # If we have notes for this staff, add them to the slur
                if staff_notes:
                    # Add notes to slur
                    for i, note in enumerate(staff_notes):
                        position = "start" if i == 0 else "end" if i == len(staff_notes) - 1 else "middle"
                        slur.add_note(note, position)
                    
                    print(f"Connected slur to {len(staff_notes)} notes on staff {staff_id} in system {system_id}")
        

        
        # Connect dynamics to measures
        # In connect_music_elements()
        # print("\nDEBUG: Dynamic measure connection")
        for dynamic in self.dynamics:
            print(f"Connecting dynamic {dynamic.type} at ({dynamic.x:.1f}, {dynamic.y:.1f})")
            
            # First, find the correct staff system
            closest_system = None
            min_system_distance = float('inf')
            
            for system in self.staff_systems:
                # Get vertical bounds of this system
                if hasattr(system, 'lines') and system.lines:
                    system_top = min(system.lines.values())
                    system_bottom = max(system.lines.values())
                    
                    # Calculate vertical distance
                    if dynamic.y < system_top:
                        distance = system_top - dynamic.y
                    elif dynamic.y > system_bottom:
                        distance = dynamic.y - system_bottom
                    else:
                        # Dynamic is within system bounds
                        distance = 0
                    
                    if distance < min_system_distance:
                        min_system_distance = distance
                        closest_system = system
            
            if not closest_system:
                print(f"  WARNING: Could not find a system for dynamic {dynamic.type}")
                continue
                
            print(f"  Found closest system {closest_system.id}")
            
            # Now find the right measure within that system
            found_measure = False
            if hasattr(closest_system, 'measures') and closest_system.measures:
                for measure in closest_system.measures:
                    if not hasattr(measure, 'start_x') or not hasattr(measure, 'end_x'):
                        continue
                        
                    if measure.start_x <= dynamic.x < measure.end_x:
                        dynamic.measure = measure
                        dynamic.staff = 1  # Default to first staff
                        found_measure = True
                        print(f"  Connected dynamic {dynamic.type} to measure at x={measure.start_x:.1f}")
                        break
                
                if not found_measure and closest_system.measures:
                    # Fallback: find closest measure by x-position
                    closest_measure = None
                    min_distance = float('inf')
                    
                    for measure in closest_system.measures:
                        if not hasattr(measure, 'start_x') or not hasattr(measure, 'end_x'):
                            continue
                            
                        measure_mid_x = (measure.start_x + measure.end_x) / 2
                        distance = abs(dynamic.x - measure_mid_x)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_measure = measure
                    
                    if closest_measure:
                        dynamic.measure = closest_measure
                        dynamic.staff = 1
                        print(f"  Fallback: Connected dynamic {dynamic.type} to closest measure at x={closest_measure.start_x:.1f}")
                        found_measure = True
            
            if not found_measure:
                print(f"  WARNING: Could not find a measure for dynamic {dynamic.type} in system {closest_system.id}")
        
        # Add this after the dynamics connection code in connect_music_elements()
        # Connect gradual dynamics to measures IN THE CORRECT SYSTEM
        # print("\nDEBUG: Gradual dynamic measure connection")
        for gradual in self.gradual_dynamics:
            # print(f"Connecting gradual dynamic {gradual.type} at ({gradual.x:.1f}, {gradual.y:.1f})")
            
            # First, find the correct staff system
            closest_system = None
            min_system_distance = float('inf')
            
            for system in self.staff_systems:
                # Get vertical bounds of this system
                if hasattr(system, 'lines') and system.lines:
                    system_top = min(system.lines.values())
                    system_bottom = max(system.lines.values())
                    
                    # Calculate vertical distance
                    if gradual.y < system_top:
                        distance = system_top - gradual.y
                    elif gradual.y > system_bottom:
                        distance = gradual.y - system_bottom
                    else:
                        # Gradual dynamic is within system bounds
                        distance = 0
                    
                    if distance < min_system_distance:
                        min_system_distance = distance
                        closest_system = system
            
            if not closest_system:
                print(f"  WARNING: Could not find a system for gradual dynamic {gradual.type}")
                continue
                
            print(f"  Found closest system {closest_system.id}")
            
            # Now find the right measure within that system
            found_measure = False
            if hasattr(closest_system, 'measures') and closest_system.measures:
                for measure in closest_system.measures:
                    if not hasattr(measure, 'start_x') or not hasattr(measure, 'end_x'):
                        continue
                        
                    if measure.start_x <= gradual.x < measure.end_x:
                        gradual.measure = measure
                        gradual.staff = 1  # Default to first staff
                        found_measure = True
                        print(f"  Connected gradual dynamic {gradual.type} to measure at x={measure.start_x:.1f}")
                        break
                
                if not found_measure and closest_system.measures:
                    # Fallback: find closest measure by x-position
                    closest_measure = None
                    min_distance = float('inf')
                    
                    for measure in closest_system.measures:
                        if not hasattr(measure, 'start_x') or not hasattr(measure, 'end_x'):
                            continue
                            
                        measure_mid_x = (measure.start_x + measure.end_x) / 2
                        distance = abs(gradual.x - measure_mid_x)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_measure = measure
                    
                    if closest_measure:
                        gradual.measure = closest_measure
                        gradual.staff = 1
                        print(f"  Fallback: Connected gradual dynamic {gradual.type} to closest measure at x={closest_measure.start_x:.1f}")
                        found_measure = True
            
            if not found_measure:
                print(f"  WARNING: Could not find a measure for gradual dynamic {gradual.type} in system {closest_system.id}")

        # Also add fallback for gradual dynamics at the end of the method, similar to dynamics
        # Connect articulations to notes
        for articulation in self.articulations:
            # Find the closest note
            closest_note = None
            min_distance = float('inf')
            
            for note in self.notes:
                # Calculate Euclidean distance
                distance = ((note.x - articulation.x) ** 2 + (note.y - articulation.y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_note = note
            
            # If the closest note is within a reasonable distance, connect them
            if closest_note and min_distance < articulation.width * 2:
                articulation.note = closest_note
                
                # Ensure the note has an articulations list
                if not hasattr(closest_note, 'articulations'):
                    closest_note.articulations = []
                    
                closest_note.articulations.append(articulation)
                print(f"Connected articulation to note at ({closest_note.x}, {closest_note.y})")
        
        # Connect ornaments to notes (similar to articulations)
        for ornament in self.ornaments:
            closest_note = None
            min_distance = float('inf')
            
            for note in self.notes:
                distance = ((note.x - ornament.x) ** 2 + (note.y - ornament.y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_note = note
            
            if closest_note and min_distance < ornament.width * 2:
                ornament.note = closest_note
                
                if not hasattr(closest_note, 'ornaments'):
                    closest_note.ornaments = []
                    
                closest_note.ornaments.append(ornament)
                print(f"Connected ornament to note at ({closest_note.x}, {closest_note.y})")
                
        # Add to the end of connect_music_elements()
        # Fallback for unconnected dynamics - connect to nearest measure
        unconnected_dynamics = [d for d in self.dynamics if not hasattr(d, 'measure') or d.measure is None]
        if unconnected_dynamics:
            print(f"\nAttempting fallback connection for {len(unconnected_dynamics)} unconnected dynamics")
            
            # For each unconnected dynamic, find the closest measure
            for dynamic in unconnected_dynamics:
                closest_measure = None
                min_distance = float('inf')
                
                for system in self.staff_systems:
                    for measure in system.measures:
                        # Calculate distance to measure center
                        if hasattr(measure, 'start_x') and hasattr(measure, 'end_x'):
                            measure_center_x = (measure.start_x + measure.end_x) / 2
                            distance = abs(dynamic.x - measure_center_x)
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_measure = measure
                
                if closest_measure:
                    dynamic.measure = closest_measure
                    dynamic.staff = 1  # Default to first staff
                    print(f"Fallback: Connected dynamic {dynamic.type} at ({dynamic.x:.1f}, {dynamic.y:.1f})")
                    print(f"          to nearest measure at x={closest_measure.start_x:.1f}-{closest_measure.end_x:.1f}")
                    
                    
                    

    
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
        
        # Add new lists for expression elements
        self.dynamics = []
        self.gradual_dynamics = []
        self.ties = []
        self.slurs = []
        self.articulations = []
        self.ornaments = []
        self.tuplets = []
        self.text_directions = []
        
        
        
        for det in detections_list:
            class_id = det['class_id']
            class_name = det.get('class_name', '')
            confidence = det['confidence']
            
            # Get the bounding box
            if 'bbox' in det:
                bbox = det['bbox']
            else:
                # Create bbox from individual coordinates
                bbox = {
                    'x1': det.get('x1', 0),
                    'y1': det.get('y1', 0),
                    'x2': det.get('x2', 0),
                    'y2': det.get('y2', 0),
                    'width': det.get('width', 0),
                    'height': det.get('height', 0),
                    'center_x': det.get('center_x', 0),
                    'center_y': det.get('center_y', 0)
                }
            
            # Process based on class
            if 'notehead' in class_name.lower():
                note = Note(class_id, class_name, confidence, bbox)
                self.notes.append(note)
                
            elif 'rest' in class_name.lower():
                rest = Rest(class_id, class_name, confidence, bbox)
                
                # Extract rest type from class_name for accurate duration
                rest_type_map = {
                    'whole': {'duration': 4.0, 'type': 'whole'},
                    'half': {'duration': 2.0, 'type': 'half'},
                    'quarter': {'duration': 1.0, 'type': 'quarter'},
                    'eighth': {'duration': 0.5, 'type': 'eighth'},
                    '16th': {'duration': 0.25, 'type': '16th'},
                    '32nd': {'duration': 0.125, 'type': '32nd'},
                    '64th': {'duration': 0.0625, 'type': '64th'}
                }
                
                # Try to extract rest type from name
                for rest_type, values in rest_type_map.items():
                    if rest_type in class_name.lower():
                        rest.duration = values['duration']
                        rest.duration_type = values['type']
                        rest.rest_type_confidence = confidence  # Store confidence in the type
                        break
                
                # Default if no specific type found
                if not hasattr(rest, 'duration'):
                    rest.duration = 1.0  # Default to quarter rest
                    rest.duration_type = 'quarter'
                    rest.rest_type_confidence = 0.5  # Lower confidence in default
                    
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
            
            elif class_name == 'systemicBarline':
                # Create a special barline that spans multiple staves
                barline = Barline(class_id, class_name, confidence, bbox)
                barline.is_systemic = True  # Mark as spanning multiple staves
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
                
            # Process dynamic marks
            elif class_name.lower() in ['dynamic', 'dynamicforte', 'dynamicpiano', 'dynamicmf', 'dynamicmp', 'dynamicff', 'dynamicpp'] or class_name in DYNAMIC_TYPE_MAP:
                # Get the dynamic type from the mapping or use a default
                dynamic_type = DYNAMIC_TYPE_MAP.get(class_name, "mf")  # Default to mf if not found
                
                # Create dynamic object
                dynamic = Dynamic(
                    x=bbox['center_x'], 
                    y=bbox['center_y'], 
                    width=bbox['width'], 
                    height=bbox['height'],
                    type=dynamic_type
                )
                self.dynamics.append(dynamic)
                print(f"Processed dynamic: {class_name} -> {dynamic_type}")
                
            # Process gradual dynamics (crescendo/diminuendo)
            elif 'gradualdynamic' in class_name.lower() or 'cresc' in class_name.lower() or 'dim' in class_name.lower():
                # Determine if it's crescendo or diminuendo
                if 'cresc' in class_name.lower():
                    grad_type = "crescendo"
                elif any(x in class_name.lower() for x in ['dim', 'decresc', 'decres']):
                    grad_type = "diminuendo"
                else:
                    grad_type = "crescendo"  # Default
                    
                # Create gradual dynamic object
                gradual_dynamic = GradualDynamic(
                    x=bbox['center_x'], 
                    y=bbox['center_y'], 
                    width=bbox['width'], 
                    height=bbox['height'],
                    type=grad_type
                )
                self.gradual_dynamics.append(gradual_dynamic)
                print(f"Processed gradual dynamic: {class_name} -> {grad_type}")
                
            # Process ties
            elif 'tie' in class_name.lower():
                tie = Tie(
                    x=bbox['center_x'], 
                    y=bbox['center_y'], 
                    width=bbox['width'], 
                    height=bbox['height']
                )
                self.ties.append(tie)
                print(f"Processed tie at ({bbox['center_x']}, {bbox['center_y']})")
                
            # Process slurs
            elif 'slur' in class_name.lower():
                slur = Slur(
                    x=bbox['center_x'], 
                    y=bbox['center_y'], 
                    width=bbox['width'], 
                    height=bbox['height']
                )
                self.slurs.append(slur)
                print(f"Processed slur at ({bbox['center_x']}, {bbox['center_y']})")
                
            # Process articulations
            elif any(artic in class_name.lower() for artic in ['staccato', 'accent', 'tenuto', 'marcato']):
                # Look up in the articulation map
                artic_info = None
                for key, value in ARTICULATION_TYPE_MAP.items():
                    if key.lower() in class_name.lower():
                        artic_info = value
                        break
                        
                if artic_info:
                    articulation = Articulation(
                        x=bbox['center_x'], 
                        y=bbox['center_y'], 
                        width=bbox['width'], 
                        height=bbox['height'],
                        type=artic_info['type'],
                        placement=artic_info['placement']
                    )
                    self.articulations.append(articulation)
                    print(f"Processed articulation: {class_name} -> {artic_info['type']}")
                
            # Process ornaments
            elif any(orn in class_name.lower() for orn in ['trill', 'mordent', 'turn']):
                # Look up in the ornament map
                orn_type = None
                for key, value in ORNAMENT_TYPE_MAP.items():
                    if key.lower() in class_name.lower():
                        orn_type = value
                        break
                        
                if orn_type:
                    ornament = Ornament(
                        x=bbox['center_x'], 
                        y=bbox['center_y'], 
                        width=bbox['width'], 
                        height=bbox['height'],
                        type=orn_type
                    )
                    self.ornaments.append(ornament)
                    print(f"Processed ornament: {class_name} -> {orn_type}")
                    
            # Process tuplets
            elif 'tuplet' in class_name.lower() or (class_name.startswith('T') and class_name[1:].isdigit()):
                # Try to determine tuplet type
                if class_name in TUPLET_TYPE_MAP:
                    actual_notes, normal_notes = TUPLET_TYPE_MAP[class_name]
                else:
                    actual_notes, normal_notes = 3, 2  # Default to triplet
                    
                tuplet = Tuplet(
                    x=bbox['center_x'], 
                    y=bbox['center_y'], 
                    width=bbox['width'], 
                    height=bbox['height'],
                    actual_notes=actual_notes,
                    normal_notes=normal_notes
                )
                self.tuplets.append(tuplet)
                print(f"Processed tuplet: {class_name} -> {actual_notes}:{normal_notes}")
            
            # Process any other class we don't specifically handle
            # else:
            #     print(f"Unhandled element type: {class_name}")
        
        print(f"Notes processed: {len(self.notes)}")
        print(f"Dynamics processed: {len(self.dynamics)}")
        print(f"Gradual dynamics processed: {len(self.gradual_dynamics)}")
        print(f"Ties processed: {len(self.ties)}")
        print(f"Slurs processed: {len(self.slurs)}")
    


    def assign_to_staff_systems(self):
        """Assign music elements to staff systems."""
        # First, identify potential systemic barlines
        for barline in self.barlines:
            if 'systemicBarline' in getattr(barline, 'class_name', ''):
                barline.is_systemic = True
            elif hasattr(barline, 'bbox') and barline.bbox['height'] > self.typical_staff_spacing * 6:
                barline.is_systemic = True
                barline.class_name = 'systemicBarline'
        
        # Now assign elements to staff systems
        all_elements = (self.notes + self.rests + self.accidentals + self.clefs + 
                    self.beams + self.flags + self.time_signature_elements)
        
        # First, assign non-barline elements
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
                
                # Check vertical alignment
                if element.y < top_line:
                    distance = top_line - element.y
                elif element.y > bottom_line:
                    distance = element.y - bottom_line
                else:
                    distance = 0  # Element is within staff bounds
                    
                if distance < min_distance:
                    min_distance = distance
                    closest_system = system
            
            # Assign to the closest system
            if closest_system:
                closest_system.add_element(element)
        
        # Then, handle barlines separately
        for barline in self.barlines:
            if getattr(barline, 'is_systemic', False):
                # Find all systems that this systemic barline spans
                affected_systems = []
                barline_top = barline.bbox['y1']
                barline_bottom = barline.bbox['y2']
                
                for system in self.staff_systems:
                    # Get vertical bounds of this system
                    if system.lines:
                        system_top = min(system.lines.values())
                        system_bottom = max(system.lines.values())
                        
                        # Check if barline spans this system
                        if (barline_top <= system_bottom and barline_bottom >= system_top):
                            affected_systems.append(system)
                            system.is_multi_staff = True
                
                # If we found systems, assign this barline to all of them
                if affected_systems:
                    print(f"Systemic barline at x={barline.x:.1f} spans {len(affected_systems)} systems")
                    for system in affected_systems:
                        system.add_element(barline)
                else:
                    # Otherwise, find closest system as fallback
                    closest_system = None
                    min_distance = float('inf')
                    
                    for system in self.staff_systems:
                        line_positions = list(system.lines.values())
                        if not line_positions:
                            continue
                            
                        top_line = min(line_positions)
                        bottom_line = max(line_positions)
                        center_y = (top_line + bottom_line) / 2
                        
                        distance = abs(barline.y - center_y)
                        if distance < min_distance:
                            min_distance = distance
                            closest_system = system
                    
                    if closest_system:
                        closest_system.add_element(barline)
            else:
                # Regular barline - find closest system
                closest_system = None
                min_distance = float('inf')
                
                for system in self.staff_systems:
                    line_positions = list(system.lines.values())
                    if not line_positions:
                        continue
                        
                    top_line = min(line_positions)
                    bottom_line = max(line_positions)
                    center_y = (top_line + bottom_line) / 2
                    
                    distance = abs(barline.y - center_y)
                    if distance < min_distance:
                        min_distance = distance
                        closest_system = system
                
                if closest_system:
                    closest_system.add_element(barline)
    def calculate_note_pitches(self):
        """Calculate pitches for all notes based on staff position."""
        for note in self.notes:
            if not note.staff_system:
                continue

            system = note.staff_system
            # Determine clef type to use
            clef_type = "G"  # Default to treble clef
            if system.clef:
                clef_type = system.clef.type  # Get clef type from staff system
            # Make sure staves are split
            if not system.staves:
                system.split_into_staves()

            if system.staves:
                staff_idx, staff_lines = find_closest_staff(note.y, system.staves)
                step, octave = self.calculate_note_pitch_from_staff(note, staff_idx, staff_lines, clef_type)
            else:
                # Fallback (if staves not split correctly)
                step, octave = system.y_to_pitch(note.y, system.clef)

            if step and octave:
                note.set_pitch(step, octave, note.alter)
                
                # After all pitches are calculated, add debug to verify
        print("\n=== CHECKING FINAL NOTE PITCHES AFTER CALCULATION ===")
        for i, note in enumerate(self.notes):
            print(f"Note {i}: x={note.x}, y={note.y}, step={note.step}, octave={note.octave}, pitch={note.pitch}")


    def calculate_note_pitch_from_staff(self, note, staff_idx, staff_lines, clef_type="G"):
        """
        Calculate note pitch based on position relative to staff lines with comprehensive
        clef handling including transposing clefs and C clefs.
        
        Args:
            note: Note object with x, y coordinates
            staff_idx: Index of the staff this note belongs to
            staff_lines: List of y-positions of the staff lines (top to bottom)
            clef_type: Type of clef (G, F, C, G8va, G8vb, F8vb, percussion, etc.)
            
        Returns:
            Tuple of (step, octave) representing the pitch
        """
        # Sort staff lines from top to bottom
        staff_lines = sorted(staff_lines)
        
        # Calculate spacing between staff lines
        spacings = [staff_lines[i+1] - staff_lines[i] for i in range(len(staff_lines) - 1)]
        spacing = sum(spacings) / len(spacings) if spacings else 10
        
        # Define pitch mappings for different clef types
        # Each entry is a tuple: (line_pitches, space_pitches)
        # Line pitches are given from top to bottom (lines 1-5)
        # Space pitches are given from top to bottom (spaces 1-4)
        clef_mappings = {
            # Standard G clef (treble)
            "G": ([('F', 5), ('D', 5), ('B', 4), ('G', 4), ('E', 4)],  # lines
                [('E', 5), ('C', 5), ('A', 4), ('F', 4)]),           # spaces
            
            # G clef aliases
            "gClef": ([('F', 5), ('D', 5), ('B', 4), ('G', 4), ('E', 4)], 
                    [('E', 5), ('C', 5), ('A', 4), ('F', 4)]),
            
            # G clef 8va (sounds one octave higher)
            "G8va": ([('F', 6), ('D', 6), ('B', 5), ('G', 5), ('E', 5)],
                    [('E', 6), ('C', 6), ('A', 5), ('F', 5)]),
            
            "gClef8va": ([('F', 6), ('D', 6), ('B', 5), ('G', 5), ('E', 5)],
                        [('E', 6), ('C', 6), ('A', 5), ('F', 5)]),
            
            # G clef 8vb (sounds one octave lower)
            "G8vb": ([('F', 4), ('D', 4), ('B', 3), ('G', 3), ('E', 3)],
                    [('E', 4), ('C', 4), ('A', 3), ('F', 3)]),
            
            "gClef8vb": ([('F', 4), ('D', 4), ('B', 3), ('G', 3), ('E', 3)],
                        [('E', 4), ('C', 4), ('A', 3), ('F', 3)]),
            
            # Standard F clef (bass)
            "F": ([('A', 3), ('F', 3), ('D', 3), ('B', 2), ('G', 2)],
                [('G', 3), ('E', 3), ('C', 3), ('A', 2)]),
            
            # F clef aliases
            "fClef": ([('A', 3), ('F', 3), ('D', 3), ('B', 2), ('G', 2)],
                    [('G', 3), ('E', 3), ('C', 3), ('A', 2)]),
            
            # F clef 8vb (sounds one octave lower)
            "F8vb": ([('A', 2), ('F', 2), ('D', 2), ('B', 1), ('G', 1)],
                    [('G', 2), ('E', 2), ('C', 2), ('A', 1)]),
            
            "fClef8vb": ([('A', 2), ('F', 2), ('D', 2), ('B', 1), ('G', 1)],
                        [('G', 2), ('E', 2), ('C', 2), ('A', 1)]),
            
            # C clef - Alto (C on middle line)
            "C": ([('E', 4), ('C', 4), ('A', 3), ('F', 3), ('D', 3)],
                [('D', 4), ('B', 3), ('G', 3), ('E', 3)]),
            
            "cClef": ([('E', 4), ('C', 4), ('A', 3), ('F', 3), ('D', 3)],
                    [('D', 4), ('B', 3), ('G', 3), ('E', 3)]),
            
            # C clef - Tenor (C on 4th line)
            "cClefTenor": ([('G', 4), ('E', 4), ('C', 4), ('A', 3), ('F', 3)],
                        [('F', 4), ('D', 4), ('B', 3), ('G', 3)]),
            
            # C clef - Soprano (C on bottom line)
            "cClefSoprano": ([('D', 5), ('B', 4), ('G', 4), ('E', 4), ('C', 4)],
                            [('C', 5), ('A', 4), ('F', 4), ('D', 4)]),
            
            # Percussion clef (unpitched)
            "percussion": ([('B', 4), ('A', 4), ('F', 4), ('D', 4), ('F', 3)],  # arbitrary mapping
                        [('C', 5), ('G', 4), ('E', 4), ('C', 4)]),
            
            "unpitchedPercussionClef1": ([('B', 4), ('A', 4), ('F', 4), ('D', 4), ('F', 3)],
                                        [('C', 5), ('G', 4), ('E', 4), ('C', 4)])
        }
        
        # Select appropriate pitch mapping based on clef type
        if clef_type in clef_mappings:
            line_pitches, space_pitches = clef_mappings[clef_type]
        else:
            # Default to G clef if unsupported clef
            print(f"WARNING: Unknown clef type '{clef_type}', defaulting to G clef.")
            line_pitches, space_pitches = clef_mappings["G"]
        
        # Handle special case for percussion clef
        is_percussion = clef_type in ["percussion", "unpitchedPercussionClef1"]
        
        # Find if the note is on a line
        for i, line_y in enumerate(staff_lines):
            if abs(note.y - line_y) < spacing * 0.3:  # Note is on a line
                if 0 <= i < len(line_pitches):
                    if is_percussion:
                        # For percussion, return a conventional pitch but set a percussion flag
                        step, octave = line_pitches[i]
                        note.is_percussion = True
                        return step, octave
                    else:
                        return line_pitches[i]
        
        # Check if the note is in a space
        for i in range(len(staff_lines) - 1):
            space_y = (staff_lines[i] + staff_lines[i+1]) / 2
            if abs(note.y - space_y) < spacing * 0.3:  # Note is in a space
                if i < len(space_pitches):
                    if is_percussion:
                        # For percussion, return a conventional pitch but set a percussion flag
                        step, octave = space_pitches[i]
                        note.is_percussion = True
                        return step, octave
                    else:
                        return space_pitches[i]
        
        # Handle ledger lines above staff
        if note.y < staff_lines[0] - spacing * 0.5:
            # Note is above the staff
            steps_above = round((staff_lines[0] - note.y) / spacing)
            
            # Start from the top line pitch
            top_step, top_octave = line_pitches[0]
            step_sequence = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            step_idx = step_sequence.index(top_step)
            
            # Calculate pitch stepping up from the top line
            curr_octave = top_octave
            for _ in range(steps_above):
                step_idx = (step_idx + 1) % 7
                if step_idx == 0:  # Reached C, increment octave
                    curr_octave += 1
            
            if is_percussion:
                note.is_percussion = True
            
            return step_sequence[step_idx], curr_octave
        
        # Handle ledger lines below staff
        elif note.y > staff_lines[-1] + spacing * 0.5:
            # Note is below the staff
            steps_below = round((note.y - staff_lines[-1]) / spacing)
            
            # Start from the bottom line pitch
            bottom_step, bottom_octave = line_pitches[-1]
            step_sequence = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            step_idx = step_sequence.index(bottom_step)
            
            # Calculate pitch stepping down from the bottom line
            curr_octave = bottom_octave
            for _ in range(steps_below):
                step_idx = (step_idx - 1) % 7
                if step_idx == 6:  # Reached B, decrement octave
                    curr_octave -= 1
            
            if is_percussion:
                note.is_percussion = True
            
            return step_sequence[step_idx], curr_octave
        
        # Default fallback
        if is_percussion:
            note.is_percussion = True
        
        return ('C', 4)

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
        """
        Connect beams to notes using stem-detector style robust logic that
        focuses on vertical alignment without distance limitations.
        """
        # Clear any existing beams from all notes first
        for note in self.notes:
            if hasattr(note, 'beams'):
                note.beams = []
                
        for beam in self.beams:
            if not beam.staff_system:
                continue
                
            # Get all notes in this staff system and same staff
            beam_staff_id = self._get_staff_id(beam) if hasattr(self, '_get_staff_id') else 0
            system_notes = [n for n in self.notes 
                            if n.staff_system == beam.staff_system 
                            and self._get_staff_id(n) == beam_staff_id]
            
            # Skip if no notes in this staff
            if not system_notes:
                continue
            
            # Calculate typical notehead dimensions
            avg_note_height = sum(note.height for note in system_notes) / len(system_notes) if system_notes else 10
            avg_note_width = sum(note.width for note in system_notes) / len(system_notes) if system_notes else 10
            
            # Extract beam dimensions
            beam_x1, beam_y1 = beam.bbox['x1'], beam.bbox['y1']
            beam_x2, beam_y2 = beam.bbox['x2'], beam.bbox['y2']
            
            # Calculate beam slope
            beam_slope = (beam_y2 - beam_y1) / (beam_x2 - beam_x1 + 1e-6)  # Avoid div by zero
            
            # Determine beam direction based on slope
            beam_direction = 'up' if beam_slope < 0 else 'down'
            
            # Add a margin to beam span
            margin = avg_note_width
            extended_x1 = beam_x1 - margin
            extended_x2 = beam_x2 + margin
            
            # Find all notes that are horizontally aligned with the beam (including margin)
            candidate_notes = []
            for note in system_notes:
                # Check horizontal alignment with beam span
                note_x = note.x
                if extended_x1 <= note_x <= extended_x2:
                    # Calculate y-position where the beam would be at this x-position
                    # Using the beam slope formula: y = y1 + slope * (x - x1)
                    projected_beam_y = beam_y1 + beam_slope * (note_x - beam_x1)
                    
                    # Calculate vertical distance from note to beam
                    vertical_dist = note.y - projected_beam_y
                    
                    # Store the note with its projected beam position and distance
                    candidate_notes.append({
                        'note': note,
                        'projected_beam_y': projected_beam_y,
                        'vertical_dist': vertical_dist,
                        'horizontal_dist': min(abs(note_x - beam_x1), abs(note_x - beam_x2))
                    })
            
            # Group notes by x-position to handle chords
            x_groups = {}
            for candidate in candidate_notes:
                note = candidate['note']
                # Group notes that are horizontally close
                x_key = round(note.x / 5) * 5  # Group within 5 pixels
                if x_key not in x_groups:
                    x_groups[x_key] = []
                x_groups[x_key].append(candidate)
            
            # Select one note from each x-position based on stem direction
            selected_notes = []
            for x_key, candidates in sorted(x_groups.items()):
                # Skip if no candidates at this x-position
                if not candidates:
                    continue
                    
                # For stem-up beams, select notes below the beam
                # For stem-down beams, select notes above the beam
                filtered_candidates = []
                for candidate in candidates:
                    vertical_dist = candidate['vertical_dist']
                    
                    # Notes for up-stem beams should be below the beam (positive vertical_dist)
                    # Notes for down-stem beams should be above the beam (negative vertical_dist)
                    if (beam_direction == 'up' and vertical_dist > 0) or \
                    (beam_direction == 'down' and vertical_dist < 0):
                        filtered_candidates.append(candidate)
                
                # If no notes match the direction criteria, take the closest one in any direction
                if not filtered_candidates and candidates:
                    filtered_candidates = candidates
                
                # Select the note with the smallest absolute vertical distance
                if filtered_candidates:
                    best_candidate = min(filtered_candidates, 
                                        key=lambda c: abs(c['vertical_dist']))
                    selected_notes.append(best_candidate)
            
            # Connect beam to selected notes if we have at least 2
            connected_notes = []
            if len(selected_notes) >= 2:
                for candidate in selected_notes:
                    note = candidate['note']
                    
                    # Add note to beam's connected notes
                    connected_notes.append(note)
                    
                    # Add beam to note's beams list
                    if not hasattr(note, 'beams'):
                        note.beams = []
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
            
            # Store the connected notes
            beam.connected_notes = connected_notes
                    
    def connect_accidentals_to_notes(self):
        """
        Connect accidentals to notes with a balanced approach that respects measure boundaries.
        """
        # First identify key signature accidentals
        for system in self.staff_systems:
            clefs = [e for e in system.elements if isinstance(e, Clef)]
            if clefs:
                leftmost_clef = min(clefs, key=lambda c: c.x)
                
                # Find the first non-accidental, non-clef element
                first_other_element = None
                for elem in sorted(system.elements, key=lambda e: e.x):
                    if not isinstance(elem, (Accidental, Clef)) and elem.x > leftmost_clef.x:
                        first_other_element = elem
                        break
                        
                if first_other_element:
                    # Accidentals between clef and first other element are key signature
                    key_sig_x_limit = first_other_element.x
                    for acc in [a for a in system.elements if isinstance(a, Accidental)]:
                        if leftmost_clef.x < acc.x < key_sig_x_limit:
                            acc.is_key_signature = True
                            print(f"Identified key signature accidental: {acc.type} at ({acc.x:.1f}, {acc.y:.1f})")
        
        # PHASE 1: Initial flexible connections
        for system in self.staff_systems:
            print(f"\n=== INITIAL ACCIDENTAL CONNECTION FOR SYSTEM {system.id} ===")
            staff_spacing = system.line_spacing if hasattr(system, 'line_spacing') else 10
            
            # Get all non-key signature accidentals and notes in the system
            system_accidentals = [a for a in system.elements if isinstance(a, Accidental) 
                                and not getattr(a, 'is_key_signature', False)]
            system_notes = [n for n in system.elements if isinstance(n, Note)]
            
            # Skip if no accidentals or notes
            if not system_accidentals or not system_notes:
                continue
                
            # Track assigned accidentals and notes
            assigned_accidentals = set()
            notes_with_accidentals = set()
            
            # Find chord accidental groups (vertical stacks)
            x_tolerance = staff_spacing * 0.7  # More generous for initial phase
            acc_x_groups = {}
            
            for acc in system_accidentals:
                x_key = round(acc.x / x_tolerance) * x_tolerance
                if x_key not in acc_x_groups:
                    acc_x_groups[x_key] = []
                acc_x_groups[x_key].append(acc)
            
            # Process chord accidental groups
            for x_key, acc_group in acc_x_groups.items():
                if len(acc_group) >= 2:  # Potential chord accidentals
                    print(f"Found potential chord accidental group with {len(acc_group)} accidentals")
                    
                    # Sort accidentals vertically
                    acc_group.sort(key=lambda a: a.y)
                    
                    # Look for potential chord notes
                    note_x_groups = {}
                    for note in system_notes:
                        if note in notes_with_accidentals:
                            continue
                            
                        # Group notes by x-position
                        x_key = round(note.x / x_tolerance) * x_tolerance
                        if x_key not in note_x_groups:
                            note_x_groups[x_key] = []
                        note_x_groups[x_key].append(note)
                    
                    # Check each note group
                    for note_x, note_group in sorted(note_x_groups.items()):
                        # Only consider notes to the right of accidentals
                        if min(n.x for n in note_group) <= max(a.x for a in acc_group):
                            continue
                            
                        # Check horizontal distance - be generous
                        horizontal_distance = min(n.x for n in note_group) - max(a.x for a in acc_group)
                        if horizontal_distance > staff_spacing * 5:  # Very generous
                            continue
                        
                        # Look for chord patterns
                        if len(note_group) >= 2:
                            note_group.sort(key=lambda n: n.y)
                            
                            # Check if group sizes are compatible
                            if abs(len(note_group) - len(acc_group)) <= 1:
                                # Calculate vertical spreads
                                acc_spread = acc_group[-1].y - acc_group[0].y
                                note_spread = note_group[-1].y - note_group[0].y
                                
                                # If spreads are similar, this is likely a chord
                                if abs(acc_spread - note_spread) < staff_spacing * min(len(acc_group), len(note_group)) * 1.5:
                                    print(f"  Found matching chord pattern: {len(acc_group)} accidentals → {len(note_group)} notes")
                                    
                                    # Match notes to accidentals by vertical position
                                    num_to_match = min(len(acc_group), len(note_group))
                                    for i in range(num_to_match):
                                        acc = acc_group[i]
                                        # Find best matching note by vertical position
                                        unmatched_notes = [n for n in note_group if n not in notes_with_accidentals]
                                        if not unmatched_notes:
                                            break
                                            
                                        # Get reference point for this accidental
                                        if acc.type == 'flat':
                                            ref_y = acc.y + (acc.height * 0.3)
                                        else:
                                            ref_y = acc.y
                                            
                                        best_note = min(unmatched_notes, key=lambda n: abs(n.y - ref_y))
                                        
                                        # Connect if vertical distance is reasonable
                                        vertical_dist = abs(best_note.y - ref_y)
                                        if vertical_dist < staff_spacing * 1.2:  # Generous
                                            acc.affected_note = best_note
                                            best_note.accidental = acc
                                            best_note.alter = acc.alter
                                            
                                            assigned_accidentals.add(acc)
                                            notes_with_accidentals.add(best_note)
                                            
                                            print(f"    Chord connection: {acc.type} at ({acc.x:.1f}, {acc.y:.1f}) → {best_note.pitch} at ({best_note.x:.1f}, {best_note.y:.1f})")
                                    
                                    # If we matched any notes, consider this group done
                                    if len(assigned_accidentals.intersection(acc_group)) > 0:
                                        break
            
            # Process remaining individual accidentals
            remaining_accidentals = [a for a in system_accidentals if a not in assigned_accidentals]
            for acc in sorted(remaining_accidentals, key=lambda a: a.x):
                candidate_notes = []
                
                for note in [n for n in system_notes if n not in notes_with_accidentals]:
                    # Note must be to the right of accidental
                    if note.x <= acc.x:
                        continue
                    
                    # Check vertical alignment
                    if acc.type == 'flat':
                        ref_y = acc.y + (acc.height * 0.3)
                    else:
                        ref_y = acc.y
                        
                    vertical_distance = abs(ref_y - note.y)
                    horizontal_distance = note.x - acc.x
                    
                    # Be generous with distances
                    if vertical_distance < staff_spacing * 1.2 and horizontal_distance < staff_spacing * 4:
                        score = vertical_distance * 3 + horizontal_distance * 0.5
                        candidate_notes.append((note, score))
                
                # Connect to best matching note
                if candidate_notes:
                    candidate_notes.sort(key=lambda x: x[1])
                    best_note = candidate_notes[0][0]
                    
                    acc.affected_note = best_note
                    best_note.accidental = acc
                    best_note.alter = acc.alter
                    
                    notes_with_accidentals.add(best_note)
                    print(f"  Individual connection: {acc.type} at ({acc.x:.1f}, {acc.y:.1f}) → {best_note.pitch} at ({best_note.x:.1f}, {best_note.y:.1f})")
                else:
                    print(f"  WARNING: Could not find a note for accidental {acc.type} at ({acc.x:.1f}, {acc.y:.1f})")
        
        # PHASE 2: Validate and enforce measure boundaries
        print("\n=== VALIDATING MEASURE BOUNDARIES ===")
        for system in self.staff_systems:
            for measure_idx, measure in enumerate(system.measures):
                measure_notes = [n for n in measure.elements if isinstance(n, Note)]
                
                # Check for notes with accidentals from outside this measure
                for note in measure_notes:
                    if hasattr(note, 'accidental') and note.accidental:
                        acc = note.accidental
                        
                        # Check if accidental is in a different measure
                        acc_measure = None
                        for m_idx, m in enumerate(system.measures):
                            if acc in m.elements:
                                acc_measure = m
                                acc_measure_idx = m_idx
                                break
                        
                        if acc_measure != measure:
                            print(f"  Breaking invalid cross-measure connection:")
                            print(f"    {acc.type} at ({acc.x:.1f}, {acc.y:.1f}) in measure {acc_measure_idx+1}")
                            print(f"    connected to note {note.pitch} at ({note.x:.1f}, {note.y:.1f}) in measure {measure_idx+1}")
                            
                            # Break the connection
                            note.accidental = None
                            note.alter = 0
                            acc.affected_note = None
            
            # Process measure boundaries
            self._process_measure_accidentals(system)
            
            # Handle naturals
            self._process_naturals(system)

        
    def _process_measure_accidentals(self, system):
        """Process accidentals within measure boundaries with detailed debugging."""
        # print(f"\n=== PROCESSING ACCIDENTALS BY MEASURE FOR SYSTEM {system.id} ===")
        
        # Get measures for this system
        if not hasattr(system, 'measures') or not system.measures:
            print("  No measures defined for this system.")
            return
        
        # Process each measure separately
        for measure_idx, measure in enumerate(system.measures):
            # print(f"\nMeasure {measure_idx+1} (x={measure.start_x:.1f} to x={measure.end_x:.1f}):")
            
            # Get notes and accidentals in this measure
            measure_notes = [e for e in measure.elements if isinstance(e, Note)]
            
            # Print all notes and their current state
            print(f"  Contains {len(measure_notes)} notes:")
            for note in measure_notes:
                acc_info = ""
                if hasattr(note, 'accidental') and note.accidental:
                    acc_info = f" (has {note.accidental.type} accidental)"
                print(f"    Note at ({note.x:.1f}, {note.y:.1f}): {note.pitch}{acc_info}")
            
            # Track altered pitches in this measure
            altered_notes = {}  # Dict to track which pitches have been altered
            
            # Sort notes by x-position 
            measure_notes.sort(key=lambda n: n.x)
            
            # Process notes in order
            for note in measure_notes:
                if hasattr(note, 'accidental') and note.accidental:
                    # This note has an accidental directly
                    altered_notes[note.step] = note.alter
                    print(f"    → Setting {note.step} to alter={note.alter} for remainder of measure {measure_idx+1}")
                elif note.step in altered_notes:
                    # Apply alteration from earlier in measure
                    old_alter = note.alter
                    note.alter = altered_notes[note.step]
                    print(f"    → Applying previous alteration: changing {note.step}{old_alter if old_alter != 0 else ''}{note.octave} to {note.step}{note.alter if note.alter != 0 else ''}{note.octave}")
            
            # Print final note states for this measure
            print(f"  Final note states in measure {measure_idx+1}:")
            for note in measure_notes:
                print(f"    {note.pitch} at ({note.x:.1f}, {note.y:.1f})")

                    

    def validate_accidental_placement(self):
        """
        Validate that accidentals are properly placed according to music engraving rules.
        This is useful to detect potential issues in the OMR recognition process.
        """
        placement_issues = []
        
        for system in self.staff_systems:
            # Get system's staff line spacing
            staff_spacing = system.line_spacing if hasattr(system, 'line_spacing') else 10
            
            # Get all non-key-signature accidentals in this system
            accidentals = [a for a in system.elements if isinstance(a, Accidental) and not a.is_key_signature]
            
            for acc in accidentals:
                # Skip accidentals without affected notes
                if not acc.affected_note:
                    placement_issues.append(f"Orphaned accidental at ({acc.x:.1f}, {acc.y:.1f})")
                    continue
                
                note = acc.affected_note
                
                # Check vertical alignment with accidental-specific rules
                if acc.type == 'flat':
                    # For flats, use the curved portion as reference
                    reference_y = acc.y + (acc.height * 0.3)
                    vertical_offset = abs(reference_y - note.y)
                else:
                    # For sharps and naturals, use center
                    vertical_offset = abs(acc.y - note.y)
                
                if vertical_offset > staff_spacing * 0.5:
                    placement_issues.append(
                        f"{acc.type} at ({acc.x:.1f}, {acc.y:.1f}) is not vertically aligned with its note at {note.pitch}")
                
                # Check horizontal spacing
                horizontal_gap = note.x - acc.x
                min_gap = staff_spacing * 0.2  # Minimum acceptable gap
                max_gap = staff_spacing * 0.6  # Maximum acceptable gap
                
                if horizontal_gap < min_gap:
                    placement_issues.append(
                        f"{acc.type} at ({acc.x:.1f}, {acc.y:.1f}) is too close to its note {note.pitch}")
                elif horizontal_gap > max_gap:
                    placement_issues.append(
                        f"{acc.type} at ({acc.x:.1f}, {acc.y:.1f}) is too far from its note {note.pitch}")
        
        # Print summary of issues
        if placement_issues:
            print("\n=== ACCIDENTAL PLACEMENT ISSUES ===")
            for issue in placement_issues:
                print(f"WARNING: {issue}")
            print(f"Found {len(placement_issues)} placement issues")
        else:
            print("All accidentals are properly placed according to engraving rules.")
        
        return placement_issues
            
    def _process_naturals(self, system):
        """Process natural signs to cancel key signature alterations."""
        # Extract key signature information
        key_sig_notes = []
        if hasattr(system, 'key_signature') and system.key_signature:
            for acc in system.key_signature:
                # Find which note step this key signature accidental affects
                # This is simplified - would need to be based on actual staff position
                key_sig_notes.append((acc.y, acc.alter))
        
        # Process all naturals
        for acc in [a for a in system.elements if isinstance(a, Accidental) and a.type == 'natural']:
            if acc.affected_note:
                # Natural cancels any alteration from key signature
                acc.affected_note.alter = 0

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
        """
        Group notes into chords with specialized handling for vertically stacked notes
        with tall stems connecting them. Notes are grouped by staff to prevent
        cross-staff chord detection.
        """
        # FIRST: Clear any existing chord relationships to prevent leftover attributes
        for note in self.notes:
            note.is_chord_member = False
            if hasattr(note, 'chord'):
                note.chord = []
        # Process each staff system separately
        for system in self.staff_systems:
            # Group notes by staff within this system
            staff_notes = {}
            for note in [n for n in system.elements if isinstance(n, Note)]:
                staff_id = self._get_staff_id(note)
                if staff_id not in staff_notes:
                    staff_notes[staff_id] = []
                staff_notes[staff_id].append(note)
            
            # Process each staff's notes separately
            for staff_id, staff_notes_list in staff_notes.items():
                # Skip if no notes in this staff
                if not staff_notes_list:
                    continue
                    
                # Calculate average note dimensions and staff spacing
                avg_width = sum(n.width for n in staff_notes_list) / len(staff_notes_list) if staff_notes_list else 20
                avg_height = sum(n.height for n in staff_notes_list) / len(staff_notes_list) if staff_notes_list else 20
                staff_spacing = system.line_spacing if hasattr(system, 'line_spacing') and system.line_spacing else 10
                
                # STEP 1: Group notes based on horizontal proximity
                horizontal_tolerance = avg_width * 1.2  # Less aggressive tolerance
                
                # First, sort notes by x-position
                sorted_notes = sorted(staff_notes_list, key=lambda n: n.x)
                
                # Identify potential chord groups
                potential_chords = []
                
                # For each note, check if others are within horizontal tolerance (ONLY WITHIN SAME STAFF)
                for i, note in enumerate(sorted_notes):
                    chord_group = [note]
                    
                    # Only check notes within the same staff
                    for j, other_note in enumerate(sorted_notes):
                        if i == j:
                            continue
                        
                        # Verify horizontal alignment is close
                        if abs(note.x - other_note.x) <= horizontal_tolerance:
                            # Include notes regardless of vertical separation initially
                            chord_group.append(other_note)
                    
                    # Only consider as potential chord if multiple notes
                    if len(chord_group) > 1:
                        potential_chords.append(chord_group)
                
                # STEP 2: Merge overlapping chord groups and remove duplicates
                merged_chords = []
                while potential_chords:
                    current = potential_chords.pop(0)
                    current_set = set(current)
                    
                    # Check for overlaps with other groups
                    i = 0
                    while i < len(potential_chords):
                        other = potential_chords[i]
                        other_set = set(other)
                        
                        # If there's significant overlap, merge groups
                        if len(current_set.intersection(other_set)) > 0:
                            current_set = current_set.union(other_set)
                            potential_chords.pop(i)
                        else:
                            i += 1
                    
                    merged_chords.append(list(current_set))
                
                # STEP 3: Special handling for vertically stacked adjacent notes (D4-E4 case)
                for chord_group in merged_chords:
                    # Sort vertically (top to bottom)
                    chord_group.sort(key=lambda n: n.y)
                    
                    # Check if this might be a vertical stack with close notes
                    if len(chord_group) >= 2:
                        # Calculate all vertical gaps between adjacent notes
                        gaps = [chord_group[i+1].y - chord_group[i].y for i in range(len(chord_group)-1)]
                        
                        # Check for any suspiciously small gaps that might indicate adjacent notes
                        # But ensure we don't discard valid chords
                        
                        # The critical insight: Actual chord notes should be at least 1 staff space apart
                        # BUT we need to be careful not to break actual chord detection
                        valid_spacing = True
                        for gap in gaps:
                            # If gap is smaller than half a staff space, it's suspiciously small
                            # This likely means notes are on adjacent lines/spaces - not a chord
                            if 0 < gap < staff_spacing / 2:
                                # BUT - check if the boundingboxes actually overlap vertically
                                # If they do, it's more likely to be a true chord
                                for i in range(len(chord_group)-1):
                                    note1 = chord_group[i]
                                    note2 = chord_group[i+1]
                                    
                                    # Calculate vertical overlap
                                    v_overlap = min(note1.bbox['y2'], note2.bbox['y2']) - max(note1.bbox['y1'], note2.bbox['y1'])
                                    
                                    # If bounding boxes have significant vertical overlap, 
                                    # it's likely these are real chord notes
                                    if v_overlap > note1.height * 0.3 or v_overlap > note2.height * 0.3:
                                        # Keep as chord
                                        pass
                                    else:
                                        # Not a chord - notes are just on adjacent lines/spaces
                                        valid_spacing = False
                                        break
                                
                                if not valid_spacing:
                                    break
                        
                        # If spacing isn't valid, split this group
                        if not valid_spacing:
                            # Don't assign as chord members
                            continue
                    
                    # This is a valid chord
                    for note in chord_group:
                        note.is_chord_member = True
                        note.chord = chord_group
                
                # STEP 4: Perform vertical alignment check for tall stems
                # This is crucial for catching notes connected by a tall stem
                for i, note1 in enumerate(staff_notes_list):
                    # Skip if already in a chord
                    if note1.is_chord_member:
                        continue
                    
                    staff_id1 = self._get_staff_id(note1)
                    
                    # Find vertically aligned notes (potentially connected by stems)
                    stem_group = [note1]
                    for j, note2 in enumerate(staff_notes_list):
                        if i == j or note2.is_chord_member:
                            continue
                        
                        staff_id2 = self._get_staff_id(note2)
                        
                        # Ensure the notes are from the same staff
                        if staff_id1 != staff_id2:
                            continue
                        
                        # Check for vertical alignment (tight horizontal tolerance)
                        if abs(note1.x - note2.x) <= avg_width * 0.7:
                            # Check for reasonable vertical separation
                            if abs(note1.y - note2.y) >= staff_spacing:
                                stem_group.append(note2)
                    
                    # If we found multiple notes vertically aligned, it's likely a stem connection
                    if len(stem_group) > 1:
                        # Sort by y-position
                        stem_group.sort(key=lambda n: n.y)
                        
                        # Verify that they're spaced appropriately
                        valid_stem_group = True
                        for k in range(1, len(stem_group)):
                            # Ensure notes aren't too close
                            if stem_group[k].y - stem_group[k-1].y < staff_spacing * 0.7:
                                valid_stem_group = False
                                break
                        
                        if valid_stem_group:
                            # Mark as a chord
                            for note in stem_group:
                                note.is_chord_member = True
                                note.chord = stem_group
                                
                # STEP 5: Special handling for D4-E4 specific case (adjacent lines with large bounding box)
                # Look for pairs of notes on adjacent lines that have large bounding boxes
                # indicating they may be connected by a stem
                for i, note1 in enumerate(staff_notes_list):
                    # Skip if already in a chord
                    if note1.is_chord_member:
                        continue
                    
                    staff_id1 = self._get_staff_id(note1)
                    
                    for j, note2 in enumerate(staff_notes_list):
                        if i == j or note2.is_chord_member:
                            continue
                        
                        staff_id2 = self._get_staff_id(note2)
                        
                        # Ensure the notes are from the same staff
                        if staff_id1 != staff_id2:
                            continue
                        
                        # Check for close horizontal alignment
                        if abs(note1.x - note2.x) <= avg_width * 0.7:
                            # Check for adjacent staff positions
                            # These would typically be 1 staff space apart
                            dy = abs(note1.y - note2.y)
                            
                            # Check if notes are exactly 1 line/space apart
                            if abs(dy - staff_spacing) < staff_spacing * 0.3:
                                # Check if both have unusually tall bounding boxes
                                # Normal note height is approximately 1/2 to 1 staff space
                                if note1.height > staff_spacing * 1.5 or note2.height > staff_spacing * 1.5:
                                    # This is likely our D4-E4 case - they're connected by a stem
                                    stem_group = [note1, note2]
                                    # Sort by y-position
                                    stem_group.sort(key=lambda n: n.y)
                                    
                                    # Mark as chord
                                    for note in stem_group:
                                        note.is_chord_member = True
                                        note.chord = stem_group
                
                # STEP 6: Share musical properties within chords
                for note in staff_notes_list:
                    if note.is_chord_member and len(note.chord) > 1:
                        # Share duration and other musical properties
                        primary_note = note.chord[0]
                        
                        for chord_note in note.chord:
                            if hasattr(primary_note, 'duration') and primary_note.duration:
                                chord_note.duration = primary_note.duration
                                chord_note.duration_type = primary_note.duration_type
                                
                            if hasattr(primary_note, 'beams') and primary_note.beams:
                                chord_note.beams = primary_note.beams
                                
                            if hasattr(primary_note, 'flag') and primary_note.flag:
                                chord_note.flag = primary_note.flag

    def _get_staff_id(self, note):
        """
        Determine which staff a note belongs to based on vertical position.
        Returns the staff ID (0-based index).
        """
        if not note.staff_system or not hasattr(note.staff_system, 'staves'):
            return 0  # Default staff ID
        
        system = note.staff_system
        staves = system.staves
        
        if not staves:
            return 0
        
        # For each staff, check if the note falls within its vertical range
        for i, staff_lines in enumerate(staves):
            if not staff_lines:
                continue
                
            # Get the vertical bounds of this staff
            top_line = min(staff_lines)
            bottom_line = max(staff_lines)
            
            # Add padding for ledger lines (adjust this if needed)
            staff_height = bottom_line - top_line
            padding = staff_height * 0.75
            
            # Check if note falls within this staff's range
            if top_line - padding <= note.y <= bottom_line + padding:
                return i
        
        # If not found, find the closest staff
        closest_staff = 0
        min_distance = float('inf')
        
        for i, staff_lines in enumerate(staves):
            if not staff_lines:
                continue
                
            # Calculate distance to staff center
            staff_center = sum(staff_lines) / len(staff_lines)
            distance = abs(note.y - staff_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_staff = i
        
        return closest_staff

    def _check_for_stem_sided_notes(self, notes_group, system):
        """
        Check for notes that appear on opposite sides of a stem, but only within the same staff.
        
        Args:
            notes_group: List of Note objects to check
            system: The StaffSystem these notes belong to
            
        Returns:
            List of lists, each containing notes likely connected by a stem
        """
        # Group notes by staff first
        staff_groups = {}
        for note in notes_group:
            staff_id = self._get_staff_id(note)
            if staff_id not in staff_groups:
                staff_groups[staff_id] = []
            staff_groups[staff_id].append(note)
        
        # Process each staff separately
        overlapping_groups = []
        
        for staff_id, staff_notes in staff_groups.items():
            # Sort by x-position, then y-position
            sorted_notes = sorted(staff_notes, key=lambda n: (n.x, n.y))
            if not sorted_notes:
                continue
                
            current_group = [sorted_notes[0]]
            
            for i in range(1, len(sorted_notes)):
                current = sorted_notes[i]
                previous = sorted_notes[i-1]
                # Check for x-overlap
                if (current.x <= previous.x + previous.width and 
                    current.x + current.width >= previous.x):
                    # If vertical distance is appropriate for stemmed notes
                    if abs(current.y - previous.y) > system.line_spacing * 2:
                        current_group.append(current)
                    else:
                        # Start new group if not vertically distant enough
                        overlapping_groups.append(current_group)
                        current_group = [current]
                else:
                    overlapping_groups.append(current_group)
                    current_group = [current]
            
            if current_group:
                overlapping_groups.append(current_group)
        
        return overlapping_groups

    def _detect_potential_stem_groups(self, notes, system):
        """
        Detect groups of notes that appear to be connected by a stem,
        but only within the same staff.
        
        Args:
            notes: List of Note objects to analyze
            system: The StaffSystem these notes belong to
            
        Returns:
            List of lists, each containing notes likely connected by a stem
        """
        # Group notes by staff first
        staff_groups = {}
        for note in notes:
            staff_id = self._get_staff_id(note)
            if staff_id not in staff_groups:
                staff_groups[staff_id] = []
            staff_groups[staff_id].append(note)
        
        # Process each staff separately
        potential_stem_groups = []
        
        for staff_id, staff_notes in staff_groups.items():
            for note in staff_notes:
                # Look for notes vertically aligned with this one WITHIN THE SAME STAFF
                aligned_notes = [n for n in staff_notes if 
                            abs(n.x - note.x) < note.width/2 and
                            n != note]
                if aligned_notes:
                    # Sort by y-position
                    stem_group = [note] + aligned_notes
                    stem_group.sort(key=lambda n: n.y)
                    potential_stem_groups.append(stem_group)
        
        # Remove duplicates and subsets
        unique_groups = []
        for group in potential_stem_groups:
            if not any(set(group).issubset(set(g)) for g in unique_groups if group != g):
                unique_groups.append(group)
        
        return unique_groups


    def identify_systemic_barlines(self):
        """
        Identify barlines that span multiple staves and reassign them as systemic barlines.
        """
        print("\n=== IDENTIFYING SYSTEMIC BARLINES ===")
        
        # First, identify all possible systemic barlines by checking their height
        for barline in self.barlines:
            # Get the vertical extent of the barline
            height = barline.bbox['height']
            
            # If the barline is significantly taller than a typical staff
            if height > self.typical_staff_spacing * 6:
                barline.is_systemic = True
                barline.class_name = 'systemicBarline'
                print(f"Identified tall barline at x={barline.x:.1f} as systemic (height={height:.1f})")
        
        # Now, assign systemic barlines to all staves they span
        for barline in self.barlines:
            if getattr(barline, 'is_systemic', False) or barline.class_name == 'systemicBarline':
                barline_top = barline.bbox['y1']
                barline_bottom = barline.bbox['y2']
                
                # Find all systems this barline affects
                affected_systems = []
                for system in self.staff_systems:
                    if system.lines:
                        system_top = min(system.lines.values())
                        system_bottom = max(system.lines.values())
                        
                        # Check if barline spans a significant portion of this system
                        if (barline_top <= system_top + 20 and 
                            barline_bottom >= system_bottom - 20):
                            affected_systems.append(system)
                
                # Mark affected systems as multi-staff
                for system in affected_systems:
                    if len(affected_systems) > 1 or len(system.staves) > 1:
                        system.is_multi_staff = True
                        print(f"Marked system {system.id} as multi-staff due to systemic barline")
                
                # If this barline spans multiple systems, duplicate it for each system
                if len(affected_systems) > 1:
                    print(f"Systemic barline at x={barline.x:.1f} spans {len(affected_systems)} systems")
                    
                    # Ensure this barline is in all affected systems' elements
                    for system in affected_systems:
                        if barline not in system.elements:
                            system.add_element(barline)
                        
    def process_systemic_barlines(self):
        """Process detected systemic barlines and mark corresponding systems as multi-staff."""
        print("\n=== PROCESSING DETECTED SYSTEMIC BARLINES ===")
        
        # Find all detected systemic barlines
        systemic_barlines = []
        for system in self.staff_systems:
            for element in system.elements:
                if getattr(element, 'class_name', '') == 'systemicBarline':
                    element.is_systemic = True
                    systemic_barlines.append(element)
                    # Mark this system as multi-staff
                    system.is_multi_staff = True
                    print(f"Found systemic barline in system {system.id}")
        
        print(f"Processed {len(systemic_barlines)} detected systemic barlines")

    def identify_measures(self):
        """Identify measures using barlines with detailed debugging."""
        # Clear existing measures
        self.measures.clear()
        for system in self.staff_systems:
            system.measures.clear()
            print(f"\n=== IDENTIFYING MEASURES FOR SYSTEM {system.id} ===")
            
            # Get barlines for this system
            barlines = [b for b in system.elements if isinstance(b, Barline)]
            # print(f"Found {len(barlines)} barlines at positions: {[round(b.x, 1) for b in sorted(barlines, key=lambda b: b.x)]}")
            
            # Sort barlines by x-position
            barlines.sort(key=lambda b: b.x)
            
            # Create measures
            if not barlines:
                # If no barlines, create one measure for the entire system
                elements_x = [e.x for e in system.elements]
                start_x = min(elements_x) if elements_x else 0
                end_x = max(elements_x) if elements_x else 1000
                
                measure = Measure(start_x, end_x, system)
                system.measures.append(measure)
                self.measures.append(measure)
                
                print(f"Created single measure from {start_x:.1f} to {end_x:.1f}")
                
                # Add ALL elements to this measure
                for elem in system.elements:
                    measure.add_element(elem)
            else:
                # Create measures between barlines
                system_elements = system.elements.copy()
                
                # Determine the start of the first measure
                start_x = min([e.x for e in system_elements]) if system_elements else 0
                
                # Create measures and assign elements
                for i, barline in enumerate(barlines):
                    end_x = barline.x
                    
                    # Create measure
                    measure = Measure(start_x, end_x, system)
                    system.measures.append(measure)
                    self.measures.append(measure)
                    
                    print(f"Created measure {i+1} from x={start_x:.1f} to x={end_x:.1f}")
                    
                    # Add ALL elements that fall within this measure's x-range
                    assigned_notes = []
                    assigned_accidentals = []
                    for elem in system_elements:
                        if start_x <= elem.x < end_x:
                            measure.add_element(elem)
                            if isinstance(elem, Note):
                                assigned_notes.append(elem)
                            elif isinstance(elem, Accidental) and not getattr(elem, 'is_key_signature', False):
                                assigned_accidentals.append(elem)
                    
                    print(f"  Assigned {len(assigned_notes)} notes to measure {i+1}")
                    # if assigned_notes:
                        # Show note details (x, y, pitch)
                        # for note in assigned_notes:
                        #     print(f"    Note at ({note.x:.1f}, {note.y:.1f}): {note.pitch}")
                    
                    print(f"  Assigned {len(assigned_accidentals)} accidentals to measure {i+1}")
                    if assigned_accidentals:
                        # Show accidental details
                        for acc in assigned_accidentals:
                            note_info = f" → {acc.affected_note.pitch}" if hasattr(acc, 'affected_note') and acc.affected_note else "unassigned"
                            print(f"    {acc.type} at ({acc.x:.1f}, {acc.y:.1f}){note_info}")
                    
                    # Update start_x for next measure
                    start_x = end_x
                
                # Create final measure after the last barline
                max_x = max([e.x for e in system_elements]) if system_elements else (barlines[-1].x + 100)
                if max_x > barlines[-1].x:
                    measure = Measure(barlines[-1].x, max_x + 100, system)
                    system.measures.append(measure)
                    self.measures.append(measure)
                    
                    print(f"Created final measure from x={barlines[-1].x:.1f} to x={max_x+100:.1f}")
                    
                    # Add remaining elements
                    assigned_notes = []
                    assigned_accidentals = []
                    for elem in system_elements:
                        if elem.x >= barlines[-1].x:
                            measure.add_element(elem)
                            if isinstance(elem, Note):
                                assigned_notes.append(elem)
                            elif isinstance(elem, Accidental) and not getattr(elem, 'is_key_signature', False):
                                assigned_accidentals.append(elem)
                    
                    print(f"  Assigned {len(assigned_notes)} notes to final measure")
                    # if assigned_notes:
                    #     for note in assigned_notes:
                    #         print(f"    Note at ({note.x:.1f}, {note.y:.1f}): {note.pitch}")
                    
                    print(f"  Assigned {len(assigned_accidentals)} accidentals to final measure")
                    if assigned_accidentals:
                        for acc in assigned_accidentals:
                            note_info = f" → {acc.affected_note.pitch}" if hasattr(acc, 'affected_note') and acc.affected_note else "unassigned"
                            print(f"    {acc.type} at ({acc.x:.1f}, {acc.y:.1f}){note_info}")
                            
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
        """
        Generate MusicXML by delegating to the specialized module.
        This is a wrapper around the standalone generate_musicxml module.
        """
        return generate_musicxml.generate_musicxml(self)
    
    
    
    # Additional helper method for step-by-step analysis
    def analyze_barline_detection_steps(self, system_index=0):
        """
        Perform step-by-step analysis to see which methods contribute to barline detection
        for a specific staff system.
        """
        if system_index >= len(self.staff_systems):
            print(f"Invalid system index: {system_index}. Only {len(self.staff_systems)} systems available.")
            return
        
        system = self.staff_systems[system_index]
        print(f"\n=== STEP-BY-STEP BARLINE ANALYSIS FOR SYSTEM {system_index+1} ===")
        
        # Get existing barlines
        existing_barlines = [e for e in system.elements if isinstance(e, Barline)]
        existing_barlines.sort(key=lambda b: b.x)
        existing_x = [b.x for b in existing_barlines]
        
        print(f"Existing barlines: {len(existing_barlines)} at positions: {[round(x) for x in existing_x]}")
        
        # Run each analysis method individually
        
        # 1. Duration-based analysis
        duration_based, duration_conf = self._duration_based_barlines(system)
        print(f"\n1. Duration-based analysis found {len(duration_based)} candidates:")
        for i, (pos, conf) in enumerate(zip(duration_based, duration_conf)):
            print(f"   {i+1}. Position: {round(pos)}, Confidence: {conf:.2f}")
        
        # 2. Beam group analysis
        beam_boundaries = self._beam_group_boundaries(system)
        print(f"\n2. Beam group analysis found {len(beam_boundaries)} candidates:")
        for i, pos in enumerate(beam_boundaries):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # 3. Rhythmic pattern analysis
        time_sig = self._analyze_rhythmic_patterns(system)
        if time_sig:
            print(f"\n3. Rhythmic pattern analysis inferred time signature: {time_sig.beats}/{time_sig.beat_type}")
        else:
            print("\n3. Rhythmic pattern analysis found no clear time signature.")
        
        # 4. Phrase structure analysis
        phrase_predictions = self._analyze_phrase_structure(system, existing_barlines)
        print(f"\n4. Phrase structure analysis found {len(phrase_predictions)} candidates:")
        for i, pos in enumerate(phrase_predictions):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # 5. Harmonic analysis
        harmonic_points = self._analyze_harmonic_progression(system)
        print(f"\n5. Harmonic analysis found {len(harmonic_points)} candidates:")
        for i, pos in enumerate(harmonic_points):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # 6. Melodic contour analysis
        melodic_points = self.analyze_melodic_contour(system)
        print(f"\n6. Melodic contour analysis found {len(melodic_points)} candidates:")
        for i, pos in enumerate(melodic_points):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # 7. Metric pattern analysis
        metric_points = self.analyze_metric_patterns(system)
        print(f"\n7. Metric pattern analysis found {len(metric_points)} candidates:")
        for i, pos in enumerate(metric_points):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # 8. Note grouping analysis
        grouping_points = self.analyze_note_groupings(system)
        print(f"\n8. Note grouping analysis found {len(grouping_points)} candidates:")
        for i, pos in enumerate(grouping_points):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # 9. Rest pattern analysis
        rest_points = self.analyze_rest_patterns(system)
        print(f"\n9. Rest pattern analysis found {len(rest_points)} candidates:")
        for i, pos in enumerate(rest_points):
            print(f"   {i+1}. Position: {round(pos)}")
        
        # Collect all candidates and weights
        all_candidates = []
        all_candidates.extend(duration_based)
        all_candidates.extend(beam_boundaries)
        all_candidates.extend(phrase_predictions)
        all_candidates.extend(harmonic_points)
        all_candidates.extend(melodic_points)
        all_candidates.extend(metric_points)
        all_candidates.extend(grouping_points)
        all_candidates.extend(rest_points)
        
        # Get parameters
        params = self.tune_barline_detection_parameters(system)
        
        # Generate weights
        all_weights = []
        all_weights.extend([c * params['duration_weight'] for c in duration_conf])
        all_weights.extend([params['beam_boundary_weight']] * len(beam_boundaries))
        all_weights.extend([params['phrase_structure_weight']] * len(phrase_predictions))
        all_weights.extend([params['harmonic_weight']] * len(harmonic_points))
        all_weights.extend([params['melodic_contour_weight']] * len(melodic_points))
        all_weights.extend([params['metric_pattern_weight']] * len(metric_points))
        all_weights.extend([params['note_grouping_weight']] * len(grouping_points))
        all_weights.extend([params['rest_pattern_weight']] * len(rest_points))
        
        # Final clustering
        final_positions = []
        if all_candidates:
            final_positions = self._weighted_cluster_barlines(all_candidates, all_weights)
        
        print(f"\nFinal consensus: {len(final_positions)} barline positions:")
        for i, pos in enumerate(final_positions):
            print(f"   {i+1}. Position: {round(pos)}")
        
        return final_positions

    # Method to visualize barline detection results
    def visualize_barline_detection(self, system_index=0, output_path='barline_visualization.png'):
        """
        Generate a visualization of barline detection results.
        This requires matplotlib.
        
        Args:
            system_index: Index of the staff system to visualize
            output_path: Path to save the output image
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
        except ImportError:
            print("Matplotlib is required for visualization. Please install with 'pip install matplotlib'")
            return
        
        if system_index >= len(self.staff_systems):
            print(f"Invalid system index: {system_index}. Only {len(self.staff_systems)} systems available.")
            return
        
        system = self.staff_systems[system_index]
        
        # Get all relevant elements
        notes = [e for e in system.elements if isinstance(e, Note)]
        rests = [e for e in system.elements if isinstance(e, Rest)]
        barlines = [e for e in system.elements if isinstance(e, Barline)]
        clefs = [e for e in system.elements if isinstance(e, Clef)]
        
        # Sort barlines by x-position
        barlines.sort(key=lambda b: b.x)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Set system boundaries
        min_x = min([e.x for e in system.elements]) - 50 if system.elements else 0
        max_x = max([e.x for e in system.elements]) + 50 if system.elements else 1000
        
        min_y = min([system.lines[line] for line in system.lines]) - 50 if system.lines else 0
        max_y = max([system.lines[line] for line in system.lines]) + 50 if system.lines else 500
        
        # Draw staff lines
        for y in system.lines.values():
            ax.axhline(y=y, color='black', linestyle='-', alpha=0.3)
        
        # Draw elements
        for note in notes:
            ax.add_patch(patches.Rectangle((note.x, note.y - note.height/2), 
                                        note.width, note.height, 
                                        linewidth=1, edgecolor='b', facecolor='none'))
        
        for rest in rests:
            ax.add_patch(patches.Rectangle((rest.x, rest.y - rest.height/2), 
                                        rest.width, rest.height, 
                                        linewidth=1, edgecolor='g', facecolor='none'))
        
        # Draw original barlines
        for barline in barlines:
            if hasattr(barline, 'is_inferred') and barline.is_inferred:
                # Inferred barlines
                ax.axvline(x=barline.x, color='r', linestyle='-', alpha=0.8)
            else:
                # Original barlines
                ax.axvline(x=barline.x, color='k', linestyle='-', alpha=0.8)
        
        # Analyze barline candidates
        final_positions = self.analyze_barline_detection_steps(system_index)
        
        # Draw new candidate barlines
        for pos in final_positions:
            ax.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
        
        # Set plot limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)  # Inverted y-axis to match image coordinates
        
        # Add labels
        ax.set_title(f'Barline Detection Analysis - System {system_index + 1}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Add legend
        ax.plot([], [], 'k-', label='Original Barlines')
        ax.plot([], [], 'r--', label='Detected Barlines')
        ax.plot([], [], 'b-', label='Notes')
        ax.plot([], [], 'g-', label='Rests')
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
        

    def comprehensive_barline_inference(self):
        """
        Master method that combines multiple approaches for robust barline inference.
        This should be called after note and beam processing but before MusicXML generation.
        """
        print("\n=== PERFORMING COMPREHENSIVE BARLINE INFERENCE ===")

        for system_idx, system in enumerate(self.staff_systems):
            print(f"\nAnalyzing System {system_idx+1}")
            
            # Get existing barlines
            existing_barlines = [e for e in system.elements if isinstance(e, Barline)]
            existing_barlines.sort(key=lambda b: b.x)
            
            # If we already have sufficient barlines, skip inference
            if len(existing_barlines) > 3:
                print(f"  System {system_idx+1} already has {len(existing_barlines)} barlines. No inference needed.")
                continue
                
            # Collect barline candidates from multiple methods
            candidates = []
            confidences = []  # Store confidence scores separately
            
            # 1. Time signature-based analysis
            duration_based, duration_conf = self._duration_based_barlines(system)
            candidates.extend(duration_based)
            confidences.extend([c * 0.8 for c in duration_conf])  # Base confidence
            print(f"  Duration-based analysis found {len(duration_based)} candidate barlines.")
            
            # 2. Analyze beam groups to avoid breaking them across barlines
            beam_boundaries = self._beam_group_boundaries(system)
            candidates.extend(beam_boundaries)
            confidences.extend([0.7] * len(beam_boundaries))
            print(f"  Beam group analysis found {len(beam_boundaries)} candidate barlines.")
            
            # 3. Analyze rhythmic patterns
            time_sig = self._analyze_rhythmic_patterns(system)
            if time_sig and not system.time_signature:
                system.time_signature = time_sig
                print(f"  Inferred time signature: {time_sig.beats}/{time_sig.beat_type}")
            
            # 4. Analyze phrase structure
            phrase_predictions = self._analyze_phrase_structure(system, existing_barlines)
            candidates.extend(phrase_predictions)
            confidences.extend([0.6] * len(phrase_predictions))
            print(f"  Phrase structure analysis found {len(phrase_predictions)} candidate barlines.")
            
            # 5. Analyze harmonic progression
            harmonic_points = self._analyze_harmonic_progression(system)
            candidates.extend(harmonic_points)
            confidences.extend([0.65] * len(harmonic_points))
            print(f"  Harmonic analysis found {len(harmonic_points)} candidate barlines.")
            
            # Combine and filter candidates
            if candidates:
                # Cluster nearby candidates (they're likely detecting the same barline)
                final_positions = self._cluster_barline_candidates(candidates, confidences)
                print(f"  After clustering, found {len(final_positions)} barline positions.")
                
                # Create barlines at the consensus positions (non-overlapping with existing)
                new_barlines = self._create_barlines_at_positions(system, final_positions, existing_barlines)
                print(f"  Created {len(new_barlines)} new barlines.")
                
                # After adding new barlines, we need to update measures
                if new_barlines:
                    self.identify_measures()
            else:
                print("  No barline candidates found.")


    def _duration_based_barlines(self, system):
        """
        Enhanced time signature-based analysis for barline position inference,
        giving priority to well-detected rests.
        """
        # ADD THIS CODE AT THE BEGINNING - skip if we already have enough barlines
        existing_barlines = [b for b in system.elements if isinstance(b, Barline)]
        # Only proceed with inference if we have very few barlines
        if len(existing_barlines) > 2:  # Adjust this threshold as needed
            print(f"  System already has {len(existing_barlines)} barlines. Skipping duration-based inference.")
            return [], []  # Return empty lists to skip inference
        
        # Get time signature (or default to 4/4)
        time_sig = system.time_signature
        if not time_sig:
            time_sig = self._analyze_rhythmic_patterns(system)
            if not time_sig:
                time_sig = TimeSignature(beats=4, beat_type=4)
                print("  No time signature found. Using 4/4 as default.")
            system.time_signature = time_sig
        
        # Calculate expected measure duration
        measure_duration = time_sig.beats * (4 / time_sig.beat_type)
        
        # Get notes and rests sorted by x-position
        elements = [e for e in system.elements if isinstance(e, (Note, Rest))]
        elements.sort(key=lambda e: e.x)
        
        if not elements:
            return [], []
        
        # Special handling for whole rests - they're always one full measure
        for i, elem in enumerate(elements):
            if isinstance(elem, Rest) and hasattr(elem, 'duration_type') and elem.duration_type == 'whole':
                # Force whole rests to have exact measure duration
                elem.duration = measure_duration
                elem.is_full_measure_rest = True
                print(f"  Found whole rest at x={elem.x:.1f} - setting to full measure duration {measure_duration}")
        
        # Calculate progressive note durations with enhanced rest handling
        cumulative_duration = 0
        barline_positions = []
        barline_confidences = []  # Store confidence scores
        
        # Track potential barline positions after whole or significant rests
        rest_barline_positions = []
        
        # Get beam groups to avoid breaking them across barlines
        beam_groups = self._identify_beam_groups(system)
            
        # Get system dimensions for normalization
        min_x = min(e.x for e in system.elements) if system.elements else 0
        max_x = max(e.x for e in system.elements) if system.elements else 1000
        system_width = max_x - min_x
        
        # Track position of the last strong beat for better spacing
        last_strong_beat_x = min_x
        
        # Progressive analysis
        for i, elem in enumerate(elements):
            # Get element duration
            elem_duration = elem.duration if hasattr(elem, 'duration') and elem.duration else 1.0
            
            # Before adding duration, check if this would cross a measure boundary
            would_cross_boundary = (cumulative_duration + elem_duration >= measure_duration)
            
            # Check if element is in a beam group
            elem_in_beam_group = any(elem in group for group in beam_groups)
            
            # Calculate barline position based on durations
            if would_cross_boundary and not elem_in_beam_group:
                # Calculate excess beyond measure boundary
                excess = (cumulative_duration + elem_duration) - measure_duration
                
                if excess > 0 and elem_duration > 0:
                    # Interpolate position based on excess duration
                    position_ratio = (elem_duration - excess) / elem_duration
                    
                    # Calculate barline x-position
                    if i < len(elements) - 1:
                        next_elem = elements[i + 1]
                        barline_x = elem.x + (next_elem.x - elem.x) * position_ratio
                    else:
                        # Last element - place after it
                        barline_x = elem.x + elem.width * 2
                else:
                    # Place after current element
                    barline_x = elem.x + elem.width * 1.5
                
                # Calculate confidence based on multiple factors
                
                # 1. Duration accuracy - how close are we to the exact measure duration
                duration_confidence = 1.0 - min(0.5, abs(excess) / measure_duration)
                
                # 2. Spacing regularity - is the barline spacing consistent
                if barline_positions:
                    last_barline = barline_positions[-1]
                    expected_spacing = system_width / (measure_duration * 4)  # Rough estimate
                    actual_spacing = barline_x - last_barline
                    spacing_confidence = 1.0 - min(0.5, abs(actual_spacing - expected_spacing) / expected_spacing)
                else:
                    spacing_confidence = 0.8  # First barline gets decent confidence
                
                # 3. Strong beat alignment
                beat_position = (cumulative_duration % measure_duration) / (measure_duration / time_sig.beats)
                strong_beat_confidence = 1.0 - min(0.3, abs(beat_position - round(beat_position)))
                
                # Combine confidence factors (weighted average)
                confidence = (0.5 * duration_confidence + 
                            0.3 * spacing_confidence + 
                            0.2 * strong_beat_confidence)
                
                barline_positions.append(barline_x)
                barline_confidences.append(confidence)
                
                # Reset for next measure
                cumulative_duration = excess if excess > 0 else 0
                last_strong_beat_x = barline_x
            else:
                # Update cumulative duration
                cumulative_duration += elem_duration
        
        
        # Add special weighting for barlines inferred from rest positions
        for pos in rest_barline_positions:
            if pos not in barline_positions:  # Avoid duplicates
                barline_positions.append(pos)
                # Give high confidence to rest-based barlines
                barline_confidences.append(0.9)  
        
        return barline_positions, barline_confidences

    def _analyze_rest_based_barlines(self, system):
        """
        Analyze rest patterns to identify high-confidence barline positions,
        prioritizing whole rests and recognizing common patterns.
        """
        # Get time signature
        time_sig = system.time_signature
        if not time_sig:
            time_sig = TimeSignature(beats=4, beat_type=4)
        
        # Calculate measure duration
        measure_duration = time_sig.beats * (4 / time_sig.beat_type)
        
        # Get all rests and notes, sorted by x-position
        rests = [e for e in system.elements if isinstance(e, Rest)]
        notes = [e for e in system.elements if isinstance(e, Note)]
        all_elements = rests + notes
        all_elements.sort(key=lambda e: e.x)
        
        if not rests:
            return []
        
        barline_positions = []
        barline_confidences = []
        
        # First pass: whole rest analysis (highest priority)
        for rest in rests:
            if hasattr(rest, 'duration_type') and rest.duration_type == 'whole':
                # Whole rests are complete measures - they should have barlines on both sides
                # Find closest elements before and after this rest
                elements_before = [e for e in all_elements if e.x < rest.x]
                elements_after = [e for e in all_elements if e.x > rest.x]
                
                if elements_before:
                    prev_elem = max(elements_before, key=lambda e: e.x)
                    # Position barline after previous element
                    barline_x = prev_elem.x + prev_elem.width + 5
                    barline_positions.append(barline_x)
                    barline_confidences.append(0.95)  # Very high confidence
                
                if elements_after:
                    next_elem = min(elements_after, key=lambda e: e.x)
                    # Position barline before next element
                    barline_x = rest.x + rest.width + 5
                    barline_positions.append(barline_x)
                    barline_confidences.append(0.95)  # Very high confidence
        
        # Second pass: analyze cumulative durations using accurate rest types
        if all_elements:
            cumulative_duration = 0
            
            for i, elem in enumerate(all_elements):
                # Get element duration with high confidence in rest durations
                if isinstance(elem, Rest) and hasattr(elem, 'duration'):
                    elem_duration = elem.duration
                elif hasattr(elem, 'duration') and elem.duration:
                    elem_duration = elem.duration
                else:
                    elem_duration = 1.0  # Default
                
                # Check if adding this would cross a measure boundary
                if cumulative_duration + elem_duration >= measure_duration:
                    # Calculate excess beyond measure boundary
                    excess = (cumulative_duration + elem_duration) - measure_duration
                    
                    if excess > 0 and elem_duration > 0:
                        # Interpolate position based on excess duration
                        position_ratio = (elem_duration - excess) / elem_duration
                        
                        # Calculate barline x-position
                        if i < len(all_elements) - 1:
                            next_elem = all_elements[i + 1]
                            barline_x = elem.x + (next_elem.x - elem.x) * position_ratio
                        else:
                            # Last element - place after it
                            barline_x = elem.x + elem.width * 2
                    else:
                        # Place after current element
                        barline_x = elem.x + elem.width * 1.5
                    
                    # Calculate confidence - higher for divisions after rests
                    confidence = 0.9 if isinstance(elem, Rest) else 0.8
                    
                    barline_positions.append(barline_x)
                    barline_confidences.append(confidence)
                    
                    # Reset for next measure
                    cumulative_duration = excess if excess > 0 else 0
                else:
                    # Update cumulative duration
                    cumulative_duration += elem_duration
        
        return barline_positions, barline_confidences

    def _identify_beam_groups(self, system):
        """
        Identify groups of notes connected by beams.
        Notes in beam groups must remain in the same measure.
        """
        beam_groups = []
        
        # Get all notes with beams
        notes_with_beams = [n for n in system.elements 
                        if isinstance(n, Note) and hasattr(n, 'beams') and n.beams]
        
        # Get all beams in the system
        beams = [b for b in system.elements if isinstance(b, Beam)]
        
        # Group notes by connected beams
        processed_notes = set()
        
        # First approach: use the beam's connected_notes property if available
        for beam in beams:
            if hasattr(beam, 'connected_notes') and beam.connected_notes:
                if any(note not in processed_notes for note in beam.connected_notes):
                    # Get unprocessed notes from this beam
                    group = [note for note in beam.connected_notes 
                            if note not in processed_notes]
                    
                    # Add the group if it's not empty
                    if group:
                        beam_groups.append(group)
                        processed_notes.update(group)
        
        # Second approach: for notes not covered by the first approach
        for note in notes_with_beams:
            if note in processed_notes:
                continue
            
            # Start a new group with this note
            group = [note]
            processed_notes.add(note)
            
            # Find all notes sharing beams with this note
            shared_beams = note.beams.copy() if hasattr(note, 'beams') else []
            
            for other_note in notes_with_beams:
                if other_note in processed_notes or other_note == note:
                    continue
                
                # Check if they share any beams
                other_beams = other_note.beams if hasattr(other_note, 'beams') else []
                if any(b1 is b2 for b1 in shared_beams for b2 in other_beams):
                    group.append(other_note)
                    processed_notes.add(other_note)
                    # Add this note's beams to our search
                    shared_beams.extend([b for b in other_beams if b not in shared_beams])
            
            # Only add groups with multiple notes
            if len(group) > 1:
                beam_groups.append(sorted(group, key=lambda n: n.x))
        
        # Sort each group by x-position
        for group in beam_groups:
            group.sort(key=lambda n: n.x)
        
        return beam_groups

    def _beam_group_boundaries(self, system):
        """
        Identify potential barline positions at beam group boundaries.
        """
        beam_groups = self._identify_beam_groups(system)
        
        # Sort groups by their leftmost note
        beam_groups.sort(key=lambda g: g[0].x if g else float('inf'))
        
        barline_positions = []
        
        # Find boundaries between beam groups where barlines could be placed
        for i in range(len(beam_groups) - 1):
            current_group = beam_groups[i]
            next_group = beam_groups[i+1]
            
            if not current_group or not next_group:
                continue
            
            # Get rightmost note of current group and leftmost of next group
            right_note = max(current_group, key=lambda n: n.x)
            left_note = min(next_group, key=lambda n: n.x)
            
            # Calculate a good position between groups
            gap = left_note.x - (right_note.x + right_note.width)
            
            if gap > 0:
                barline_x = right_note.x + right_note.width + (gap * 0.5)
                barline_positions.append(barline_x)
        
        return barline_positions

    def _analyze_rhythmic_patterns(self, system):
        """
        Analyze note patterns to infer time signature.
        Returns a TimeSignature object or None if no pattern is detected.
        """
        elements = [e for e in system.elements if isinstance(e, (Note, Rest))]
        elements.sort(key=lambda e: e.x)
        
        if not elements:
            return None
        
        # Extract duration patterns
        durations = []
        for e in elements:
            # Try to get the most accurate duration
            if hasattr(e, 'duration') and e.duration:
                durations.append(e.duration)
            elif hasattr(e, 'duration_type'):
                # Map duration_type to actual duration
                duration_map = {
                    'whole': 4.0, 
                    'half': 2.0, 
                    'quarter': 1.0, 
                    'eighth': 0.5, 
                    '16th': 0.25, 
                    '32nd': 0.125
                }
                durations.append(duration_map.get(e.duration_type, 1.0))
            else:
                # Default to quarter note
                durations.append(1.0)
        
        # Look for groups of consistent duration totals
        group_sizes = [2, 3, 4, 6]  # Common time signatures: 2/4, 3/4, 4/4, 6/8
        best_match = None
        least_variance = float('inf')
        
        for size in group_sizes:
            # Group durations
            groups = [sum(durations[i:i+size]) for i in range(0, len(durations), size) 
                    if i+size <= len(durations)]
            
            if len(groups) < 2:  # Need at least two groups to compare
                continue
            
            # Calculate variance between groups
            mean_duration = sum(groups) / len(groups)
            variance = sum((g - mean_duration)**2 for g in groups) / len(groups)
            
            # Normalize by group size
            normalized_variance = variance / size
            
            if normalized_variance < least_variance:
                least_variance = normalized_variance
                best_match = size
        
        # Check if the variance is low enough to indicate a pattern
        if least_variance > 0.3:  # Threshold can be tuned
            return None
        
        # Determine time signature from best matching group size
        if best_match == 2:
            # Check if it's 2/4 or 2/2
            if sum(1 for d in durations if d >= 2.0) > len(durations) / 3:
                return TimeSignature(beats=2, beat_type=2)
            else:
                return TimeSignature(beats=2, beat_type=4)
        
        elif best_match == 3:
            # Check if it's 3/4 or 3/8
            if sum(1 for d in durations if d <= 0.5) > len(durations) / 2:
                return TimeSignature(beats=3, beat_type=8)
            else:
                return TimeSignature(beats=3, beat_type=4)
        
        elif best_match == 4:
            # Almost always 4/4
            return TimeSignature(beats=4, beat_type=4)
        
        elif best_match == 6:
            # Check if it's 6/8 or 6/4 by looking at typical note durations
            eighth_notes = sum(1 for d in durations if abs(d - 0.5) < 0.1)
            quarter_notes = sum(1 for d in durations if abs(d - 1.0) < 0.1)
            
            if eighth_notes > quarter_notes:
                return TimeSignature(beats=6, beat_type=8)
            else:
                return TimeSignature(beats=6, beat_type=4)
        
        # Default
        return TimeSignature(beats=4, beat_type=4)

    def _analyze_phrase_structure(self, system, existing_barlines):
        """
        Analyze note groupings for common phrase lengths.
        Uses existing barlines to predict where others should be.
        """
        if len(existing_barlines) < 2:
            return []
        
        # Calculate typical measure width from existing barlines
        measure_widths = [existing_barlines[i+1].x - existing_barlines[i].x 
                        for i in range(len(existing_barlines)-1)]
        
        if not measure_widths:
            return []
        
        avg_measure_width = sum(measure_widths) / len(measure_widths)
        
        # Predict locations of missing barlines based on typical phrase structure
        # Common structures are 2, 4, 8 measure phrases
        
        # Go forward from last barline
        last_barline_x = existing_barlines[-1].x
        forward_predictions = [last_barline_x + avg_measure_width * i 
                            for i in range(1, 5)]  # Predict next 4 measures
        
        # Go backward from first barline
        first_barline_x = existing_barlines[0].x
        backward_predictions = [first_barline_x - avg_measure_width * i 
                            for i in range(1, 3)]  # Predict previous 2 measures
        
        # Check for missing intermediate barlines
        intermediate_predictions = []
        for i in range(len(existing_barlines) - 1):
            start_x = existing_barlines[i].x
            end_x = existing_barlines[i+1].x
            width = end_x - start_x
            
            # If the gap is significantly larger than the average,
            # it might be missing one or more barlines
            if width > avg_measure_width * 1.7:
                num_missing = round(width / avg_measure_width) - 1
                for j in range(1, num_missing + 1):
                    intermediate_x = start_x + (width * j / (num_missing + 1))
                    intermediate_predictions.append(intermediate_x)
        
        # Combine all predictions
        all_predictions = forward_predictions + backward_predictions + intermediate_predictions
        
        # Filter out predictions that would be too close to existing barlines
        min_distance = avg_measure_width * 0.7
        filtered_predictions = []
        
        for pred_x in all_predictions:
            # Check if this predicted position is too close to an existing barline
            if all(abs(pred_x - barline.x) > min_distance for barline in existing_barlines):
                filtered_predictions.append(pred_x)
        
        return filtered_predictions

    def _analyze_harmonic_progression(self, system):
        """
        Analyze chord patterns to identify potential barline positions.
        """
        # Get all notes and sort by x-position
        notes = [n for n in system.elements if isinstance(n, Note)]
        notes.sort(key=lambda n: n.x)
        
        if not notes:
            return []
        
        # Identify vertical note stacks (potential chords)
        chord_positions = []
        processed_notes = set()
        
        # Get the typical staff spacing for this system
        staff_spacing = 10  # Default
        if hasattr(system, 'line_spacing'):
            staff_spacing = system.line_spacing
        
        horizontal_tolerance = staff_spacing * 0.5
        
        for note in notes:
            if note in processed_notes:
                continue
            
            # Find notes that are vertically aligned with this one
            aligned_notes = [n for n in notes 
                            if abs(n.x - note.x) < horizontal_tolerance and n != note]
            
            if aligned_notes:
                # We have a potential chord
                chord_notes = [note] + aligned_notes
                processed_notes.update(chord_notes)
                
                # Calculate the x-position of this chord
                chord_x = sum(n.x for n in chord_notes) / len(chord_notes)
                chord_positions.append((chord_x, len(chord_notes)))
        
        # If we have at least 3 chords, analyze their spacings
        if len(chord_positions) < 3:
            return []
        
        # Sort chords by x-position
        chord_positions.sort(key=lambda c: c[0])
        
        # Analyze chord spacing patterns
        barline_candidates = []
        for i in range(len(chord_positions) - 1):
            current_chord_x, current_size = chord_positions[i]
            next_chord_x, next_size = chord_positions[i+1]
            
            # Calculate spacing
            spacing = next_chord_x - current_chord_x
            
            # Check for large gaps that might indicate measure boundaries
            # or for chord progression patterns that typically end measures
            
            # Larger chords (3+ notes) often mark strong beats which could be measure starts
            if current_size >= 3 and i > 0:
                # Potential barline before this chord
                barline_candidates.append(current_chord_x - (spacing * 0.2))
            
            # Large gaps often indicate measure boundaries
            if i < len(chord_positions) - 2:
                next_next_chord_x = chord_positions[i+2][0]
                next_spacing = next_next_chord_x - next_chord_x
                
                # If this gap is significantly larger than the next one, it might be a measure boundary
                if spacing > next_spacing * 1.5:
                    barline_candidates.append(current_chord_x + (spacing * 0.8))
        
        return barline_candidates

    def _cluster_barline_candidates(self, barline_positions, confidences):
        """
        Cluster nearby barline candidates to find consensus positions.
        
        Args:
            barline_positions: List of x-coordinates for potential barlines
            confidences: Corresponding confidence scores for each position
            
        Returns:
            List of consensus barline positions
        """
        if not barline_positions:
            return []
        
        # Sort positions and their confidences together
        position_data = sorted(zip(barline_positions, confidences), key=lambda x: x[0])
        sorted_positions = [p[0] for p in position_data]
        sorted_confidences = [p[1] for p in position_data]
        
        # Define clustering threshold (how close positions need to be to be considered the same barline)
        # Adapt this based on the typical spacing between elements
        threshold = 20  # Pixels, can be adjusted
        
        # Perform clustering
        clusters = []
        current_cluster = [(sorted_positions[0], sorted_confidences[0])]
        
        for i in range(1, len(sorted_positions)):
            if sorted_positions[i] - sorted_positions[i-1] < threshold:
                # Add to current cluster
                current_cluster.append((sorted_positions[i], sorted_confidences[i]))
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [(sorted_positions[i], sorted_confidences[i])]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        # Calculate consensus position for each cluster using weighted average
        consensus_positions = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            positions, confs = zip(*cluster)
            total_conf = sum(confs)
            
            if total_conf > 0:
                # Weighted average by confidence
                weighted_pos = sum(p * c for p, c in zip(positions, confs)) / total_conf
                consensus_positions.append(weighted_pos)
            else:
                # Simple average if all confidences are zero
                consensus_positions.append(sum(positions) / len(positions))
        
        return consensus_positions


    def _create_barlines_at_positions(self, system, barline_positions, existing_barlines=None):
        """
        Create barline objects at the specified positions, with improved validation
        to ensure barlines only appear in musically appropriate locations.
        
        Returns: List of newly created barlines
        """
        if not barline_positions:
            return []
        
        # Convert existing_barlines to x-positions for comparison
        if existing_barlines:
            existing_positions = [b.x for b in existing_barlines]
        else:
            existing_positions = []
        
        # First, validate that we're not adding barlines too close to existing ones
        min_distance = 40  # Significantly increase minimum distance (was 15)
        
        filtered_positions = []
        for pos in barline_positions:
            # Additional checks to prevent problematic barlines:
            
            # 1. Don't add barlines that are too close together
            if any(abs(pos - existing_pos) < 50 for existing_pos in existing_positions):
                print(f"  Rejecting barline at x={pos:.1f}: too close to existing barline")
                continue
            
            # 2. Don't add barlines in the middle of beamed note groups
            beam_spans = []
            for beam in [e for e in system.elements if isinstance(e, Beam)]:
                if hasattr(beam, 'connected_notes') and len(beam.connected_notes) >= 2:
                    notes = beam.connected_notes
                    beam_start = min(n.x for n in notes)
                    beam_end = max(n.x + n.width for n in notes)
                    beam_spans.append((beam_start, beam_end))
            
            if any(start < pos < end for start, end in beam_spans):
                print(f"  Rejecting barline at x={pos:.1f}: would break beamed note group")
                continue
                
            # 3. Don't add barlines in spaces with very few elements
            # Find nearest elements to this position
            elements_within_range = [e for e in system.elements 
                                if isinstance(e, (Note, Rest)) and 
                                abs(e.x - pos) < 100]  # Within 100 pixels
            
            if len(elements_within_range) < 2:
                print(f"  Rejecting barline at x={pos:.1f}: insufficient musical content nearby")
                continue
            
        # Get all musical elements (notes and rests)
        musical_elements = [e for e in system.elements if isinstance(e, (Note, Rest))]
        musical_elements.sort(key=lambda e: e.x)
        
        if not musical_elements:
            return []  # No musical elements to use as reference
        
        # Second, validate by checking for content between barlines
        content_validated_positions = []
        
        # Combine existing and candidate positions for validation
        all_positions = sorted(existing_positions + filtered_positions)
        
        for i, pos in enumerate(filtered_positions):
            # Find position in the combined list
            idx = all_positions.index(pos)
            
            # Find previous and next barline positions
            prev_pos = float('-inf')
            next_pos = float('inf')
            
            if idx > 0:
                prev_pos = all_positions[idx-1]
            if idx < len(all_positions) - 1:
                next_pos = all_positions[idx+1]
            
            # Count musical elements between this barline and adjacent ones
            elements_before = sum(1 for e in musical_elements if prev_pos < e.x < pos)
            elements_after = sum(1 for e in musical_elements if pos < e.x < next_pos)
            
            # Only add if we have musical content on both sides (or at boundary)
            valid_position = True
            
            # Check if this creates an empty measure
            if prev_pos > float('-inf') and elements_before == 0:
                valid_position = False
                print(f"  Rejecting barline at x={pos:.1f}: no content after previous barline")
            
            if next_pos < float('inf') and elements_after == 0:
                valid_position = False
                print(f"  Rejecting barline at x={pos:.1f}: no content before next barline")
            
            # If valid, add to final list
            if valid_position:
                content_validated_positions.append(pos)
        
        # Calculate staff line positions for barline height
        if system.lines:
            top_line = min(system.lines.values())
            bottom_line = max(system.lines.values())
            height = bottom_line - top_line + 20  # Add padding
            y1 = top_line - 10  # Extend above top line
        else:
            # Default height if no staff lines
            height = 100
            y1 = min(e.y for e in system.elements) - 50 if system.elements else 100
        
        # Create barlines
        new_barlines = []
        for pos in content_validated_positions:
            # Create barline bbox
            barline_bbox = {
                'x1': pos - 2,  # Make the barline slightly thick
                'y1': y1,
                'x2': pos + 2,
                'y2': y1 + height,
                'width': 4,
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
            
            # Mark this barline as inferred (for visualization)
            barline.is_inferred = True
            
            # Add to staff system
            system.add_element(barline)
            self.barlines.append(barline)
            new_barlines.append(barline)
        
        return new_barlines

    def detect_vertical_lines_in_image(self, image_path=None, system=None):
        """
        Optional method to detect vertical lines (barlines) in the source image.
        This would require OpenCV or another image processing library.
        
        IMPORTANT: This is a stub implementation - you would need to implement 
        proper image processing using OpenCV or another library.
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Load the image
        # 2. Perform edge detection
        # 3. Use Hough transform to detect vertical lines
        # 4. Filter for lines that span the staff height
        # 5. Return x-positions of detected lines
        
        if not image_path or not system:
            return []
        
        try:
            # Pseudo-code for image-based barline detection
            """
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Get staff dimensions
            if system.lines:
                top_line = min(system.lines.values())
                bottom_line = max(system.lines.values())
                staff_height = bottom_line - top_line
            else:
                return []
            
            # Edge detection
            edges = cv2.Canny(img, 50, 150)
            
            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=staff_height*0.8, maxLineGap=20)
            
            barline_candidates = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is vertical
                    if abs(x2 - x1) < 10:  # Nearly vertical
                        line_height = abs(y2 - y1)
                        if line_height > staff_height * 0.8:
                            # This is likely a barline
                            barline_x = (x1 + x2) / 2
                            barline_candidates.append(barline_x)
            
            return barline_candidates
            """
            
            # For now, return an empty list
            return []
        
        except Exception as e:
            print(f"Error in image processing: {str(e)}")
            return []            
 
    # Additional music theory-based methods for barline detection


    def analyze_melodic_contour(self, system):
        """
        Analyze melodic contour to identify potential phrase endings.
        Phrases often end with descending contours or cadential patterns.
        """
        notes = [n for n in system.elements if isinstance(n, Note)]
        
        # Skip if too few notes
        if len(notes) < 8:
            return []
        
        # Sort notes by x-position
        notes.sort(key=lambda n: n.x)
        
        # Track melodic direction changes
        direction_changes = []
        
        # Convert note pitches to numeric values for contour analysis
        pitch_values = []
        
        for note in notes:
            if hasattr(note, 'step') and hasattr(note, 'octave') and note.octave is not None:
                # Calculate base value from octave (C4 = 60, C5 = 72, etc.)
                base = note.octave * 12
                
                # Add step offset (C=0, D=2, E=4, F=5, G=7, A=9, B=11)
                step_offsets = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                step_value = step_offsets.get(note.step, 0)
                
                # Add accidental adjustment
                alter = note.alter if hasattr(note, 'alter') else 0
                
                # Final value
                value = base + step_value + alter
                pitch_values.append((note, value))
        
        # Analyze pitch contours for phrase endings
        if len(pitch_values) < 3:
            return []
        
        potential_endings = []
        
        # Look for descending contours (common at phrase endings)
        for i in range(2, len(pitch_values) - 1):
            p1 = pitch_values[i-2][1]
            p2 = pitch_values[i-1][1]
            p3 = pitch_values[i][1]
            
            # Check for descending pattern
            if p1 > p2 > p3:
                # This might be a phrase ending
                note = pitch_values[i][0]
                potential_endings.append(note.x + note.width)
        
        # Look for common cadential patterns
        for i in range(len(pitch_values) - 3):
            # Get 4 consecutive pitch values
            values = [pitch_values[i+j][1] % 12 for j in range(4)]  # Use pitch class (mod 12)
            
            # Check for cadential patterns:
            # Perfect cadence (V-I): 7,2,11,0 or 7,2,7,0 (G,D,B,C or G,D,G,C)
            # Plagal cadence (IV-I): 5,9,0,5 or 5,9,0,9 (F,A,C,F or F,A,C,A)
            
            cadence_patterns = [
                # Perfect cadence patterns
                [7, 2, 11, 0],  # G,D,B,C
                [7, 2, 7, 0],   # G,D,G,C
                [7, 11, 2, 0],  # G,B,D,C
                
                # Plagal cadence patterns
                [5, 9, 0, 5],   # F,A,C,F
                [5, 9, 0, 9],   # F,A,C,A
                [5, 0, 9, 0],   # F,C,A,C
            ]
            
            if any(all(abs(values[j] - pattern[j]) <= 1 for j in range(4)) 
                for pattern in cadence_patterns):
                # Likely cadential pattern - place barline after it
                note = pitch_values[i+3][0]
                potential_endings.append(note.x + note.width * 1.5)
        
        return potential_endings
    def analyze_metric_patterns(self, system):
        """
        Analyze metric patterns to identify strong and weak beats.
        Barlines typically precede strong beats.
        """
        # Get time signature
        time_sig = system.time_signature
        if not time_sig:
            time_sig = self._analyze_rhythmic_patterns(system)
            if not time_sig:
                time_sig = TimeSignature(beats=4, beat_type=4)
        
        # Get notes
        elements = [e for e in system.elements if isinstance(e, (Note, Rest))]
        elements.sort(key=lambda e: e.x)
        
        if not elements:
            return []
        
        # Calculate measure duration
        measure_duration = time_sig.beats * (4 / time_sig.beat_type)
        
        # Determine beat structure based on time signature
        if time_sig.beats in [2, 4, 6]:
            # Simple duple/quadruple meter
            # Pattern of strong-weak (2/4) or strong-weak-weak-weak (4/4)
            pattern = [1.0] + [0.5] * (time_sig.beats - 1)
        elif time_sig.beats == 3:
            # Simple triple meter (3/4)
            pattern = [1.0, 0.3, 0.5]
        elif time_sig.beats in [9, 12]:
            # Compound meter
            pattern = [1.0] + [0.3, 0.5] * ((time_sig.beats // 3) - 1) + [0.3, 0.5]
        else:
            # Default pattern
            pattern = [1.0] + [0.5] * (time_sig.beats - 1)
        
        # Normalize pattern
        pattern_sum = sum(pattern)
        pattern = [p / pattern_sum for p in pattern]
        
        # Track cumulative duration and calculate barline positions
        cumulative_duration = 0
        barline_positions = []
        metric_weights = []  # Store metric weight of each element
        
        for elem in elements:
            # Get duration
            elem_duration = elem.duration if hasattr(elem, 'duration') and elem.duration else 1.0
            
            # Calculate position in the measure
            position_in_measure = cumulative_duration % measure_duration
            
            # Calculate metric weight based on beat position
            beat_position = position_in_measure / (measure_duration / time_sig.beats)
            beat_index = int(beat_position)
            beat_weight = pattern[beat_index % len(pattern)]
            
            # Adjust weight based on how close we are to the exact beat
            offset = beat_position - beat_index
            if offset > 0.1 and offset < 0.9:
                # Reduce weight for off-beat positions
                beat_weight *= (1 - min(offset, 1-offset))
            
            metric_weights.append((elem, beat_weight))
            
            # Update cumulative duration
            cumulative_duration += elem_duration
        
        # Analyze for strong beats that likely start measures
        potential_barlines = []
        
        # Find local maxima in metric weights
        for i in range(1, len(metric_weights) - 1):
            prev_elem, prev_weight = metric_weights[i-1]
            curr_elem, curr_weight = metric_weights[i]
            next_elem, next_weight = metric_weights[i+1]
            
            # Local maximum indicates a strong beat
            if curr_weight > prev_weight and curr_weight >= next_weight:
                if curr_weight > 0.7:  # High enough to be a strong beat
                    # Potential barline before this element
                    barline_x = curr_elem.x - 10  # Place slightly before the element
                    potential_barlines.append(barline_x)
        
        return potential_barlines

    def analyze_note_groupings(self, system):
        """
        Analyze note groupings that suggest measure boundaries.
        """
        elements = [e for e in system.elements if isinstance(e, (Note, Rest))]
        elements.sort(key=lambda e: e.x)
        
        if not elements:
            return []
        
        # Calculate spacing between elements
        spacings = []
        for i in range(1, len(elements)):
            prev = elements[i-1]
            curr = elements[i]
            gap = curr.x - (prev.x + prev.width)
            spacings.append((gap, i))
        
        if not spacings:
            return []
        
        # Calculate median spacing
        median_spacing = sorted(s[0] for s in spacings)[len(spacings)//2]
        
        # Look for unusually large gaps that might indicate measure boundaries
        threshold = max(median_spacing * 2, 20)  # At least twice the median spacing
        
        large_gaps = [s for s in spacings if s[0] > threshold]
        
        barline_positions = []
        for gap, index in large_gaps:
            prev_elem = elements[index-1]
            curr_elem = elements[index]
            
            # Position barline in the gap
            barline_x = prev_elem.x + prev_elem.width + (gap * 0.5)
            barline_positions.append(barline_x)
        
        return barline_positions

    def analyze_rest_patterns(self, system):
        """
        Analyze rest patterns to identify potential measure boundaries.
        Full-measure rests are strong indicators of measure boundaries.
        """
        rests = [r for r in system.elements if isinstance(r, Rest)]
        
        if not rests:
            return []
        
        # Get time signature
        time_sig = system.time_signature
        if not time_sig:
            time_sig = TimeSignature(beats=4, beat_type=4)
        
        # Calculate measure duration
        measure_duration = time_sig.beats * (4 / time_sig.beat_type)
        
        barline_positions = []
        
        # Check for whole or multi-beat rests
        for rest in rests:
            # Get duration
            if hasattr(rest, 'duration_type'):
                if rest.duration_type == 'whole':
                    # Whole rest - definitely a measure boundary
                    barline_positions.append(rest.x - 5)  # Before the rest
                    barline_positions.append(rest.x + rest.width + 5)  # After the rest
                elif rest.duration_type == 'half':
                    # Half rest might indicate measure boundary if in 2/4 time
                    if time_sig.beats == 2 and time_sig.beat_type == 4:
                        barline_positions.append(rest.x - 5)
                        barline_positions.append(rest.x + rest.width + 5)
            elif hasattr(rest, 'duration') and rest.duration:
                # Check if this rest is a significant portion of a measure
                if rest.duration >= measure_duration * 0.75:
                    barline_positions.append(rest.x - 5)
                    barline_positions.append(rest.x + rest.width + 5)
        
        return barline_positions

    def tune_barline_detection_parameters(self, system):
        """
        Dynamically tune detection parameters based on system properties.
        """
        params = {
            'duration_weight': 0.8,
            'beam_boundary_weight': 0.7,
            'phrase_structure_weight': 0.6,
            'harmonic_weight': 0.65,
            'melodic_contour_weight': 0.55,
            'metric_pattern_weight': 0.75,
            'note_grouping_weight': 0.6,
            'rest_pattern_weight': 0.7
        }
        
        # Adjust weights based on music style (inferred from system properties)
        
        # 1. Check for complexity
        note_count = len([e for e in system.elements if isinstance(e, Note)])
        
        if note_count > 100:  # Complex piece with many notes
            # For complex music, rely more on duration and metric patterns
            params['duration_weight'] = 0.9
            params['metric_pattern_weight'] = 0.85
            params['beam_boundary_weight'] = 0.8
        elif note_count < 30:  # Simple piece
            # For simpler pieces, visual spacing is more reliable
            params['note_grouping_weight'] = 0.75
            params['rest_pattern_weight'] = 0.8
            
        # 2. Check for beam density
        beam_count = len([e for e in system.elements if isinstance(e, Beam)])
        if beam_count > 20:  # Lots of beams (likely faster music)
            # For heavily beamed music, beam boundaries are important
            params['beam_boundary_weight'] = 0.85
            params['duration_weight'] = 0.75
        
        # 3. Check for time signature
        if system.time_signature:
            # Compound meters need more attention to metric patterns
            if system.time_signature.beats in [6, 9, 12]:
                params['metric_pattern_weight'] = 0.9
                params['duration_weight'] = 0.85
            # Complex meters need more attention to groupings
            elif system.time_signature.beats in [5, 7]:
                params['metric_pattern_weight'] = 0.9
                params['note_grouping_weight'] = 0.75
        
        return params


    def master_barline_inference(self):
        """
        Master barline detection method that integrates all approaches with weighted combination.
        Works with both completely missing and partially detected barlines.
        """
        print("\n=== PERFORMING MASTER BARLINE INFERENCE ===")
        
        # Track statistics for detection success
        total_barlines_added = 0
        
        for system_idx, system in enumerate(self.staff_systems):
            print(f"\nAnalyzing System {system_idx+1}")
            
            # Get existing barlines
            existing_barlines = [e for e in system.elements if isinstance(e, Barline)]
            existing_barlines.sort(key=lambda b: b.x)
            
            # Check for missing barlines by analyzing gaps between existing barlines
            missing_barlines = self._detect_missing_barlines(system, existing_barlines)
            
            if not missing_barlines:
                print(f"  No significant gaps detected between existing barlines in system {system_idx+1}.")
                print(f"  System has {len(existing_barlines)} barlines that appear properly spaced.")
                continue
            
            print(f"  System {system_idx+1} has {len(existing_barlines)} barlines but appears to be missing approximately {missing_barlines} more.")
            
            # Tune parameters for this system
            params = self.tune_barline_detection_parameters(system)
            
            # Collect barline candidates from all methods
            candidates = []
            weights = []
            
            # 1. Rest-based analysis (HIGHEST PRIORITY)
            rest_positions, rest_conf = self._analyze_rest_based_barlines(system)
            candidates.extend(rest_positions)
            weights.extend([c * 1.2 for c in rest_conf])  # Higher weight for rest-based positions
            print(f"  Rest-based analysis found {len(rest_positions)} candidate barlines.")
            
            # 2. Duration-based analysis
            duration_based, duration_conf = self._duration_based_barlines(system)
            candidates.extend(duration_based)
            weights.extend([c * params['duration_weight'] for c in duration_conf])
            print(f"  Duration-based analysis found {len(duration_based)} candidate barlines.")
        
            
            # 2. Beam group boundaries
            beam_boundaries = self._beam_group_boundaries(system)
            candidates.extend(beam_boundaries)
            weights.extend([params['beam_boundary_weight']] * len(beam_boundaries))
            print(f"  Beam group analysis found {len(beam_boundaries)} candidate barlines.")
            
            # 3. Analyze rhythmic patterns (infer time signature if needed)
            time_sig = self._analyze_rhythmic_patterns(system)
            if time_sig and not system.time_signature:
                system.time_signature = time_sig
                print(f"  Inferred time signature: {time_sig.beats}/{time_sig.beat_type}")
            
            # 4. Phrase structure analysis - this works well with existing barlines!
            phrase_predictions = self._analyze_phrase_structure(system, existing_barlines)
            candidates.extend(phrase_predictions)
            weights.extend([params['phrase_structure_weight']] * len(phrase_predictions))
            print(f"  Phrase structure analysis found {len(phrase_predictions)} candidate barlines.")
            
            # 5. Analyze spacing between existing barlines - critical for partial detection!
            gap_barlines = self._analyze_barline_gaps(system, existing_barlines)
            candidates.extend(gap_barlines)
            weights.extend([0.85] * len(gap_barlines))  # High confidence for gap analysis
            print(f"  Gap analysis found {len(gap_barlines)} candidate barlines.")
            
            # 6. Other analysis methods
            harmonic_points = self._analyze_harmonic_progression(system)
            candidates.extend(harmonic_points)
            weights.extend([params['harmonic_weight']] * len(harmonic_points))
            
            melodic_points = self.analyze_melodic_contour(system)
            candidates.extend(melodic_points)
            weights.extend([params['melodic_contour_weight']] * len(melodic_points))
            
            metric_points = self.analyze_metric_patterns(system)
            candidates.extend(metric_points)
            weights.extend([params['metric_pattern_weight']] * len(metric_points))
            
            grouping_points = self.analyze_note_groupings(system)
            candidates.extend(grouping_points)
            weights.extend([params['note_grouping_weight']] * len(grouping_points))
            
            rest_points = self.analyze_rest_patterns(system)
            candidates.extend(rest_points)
            weights.extend([params['rest_pattern_weight']] * len(rest_points))
            
            # Combine and filter candidates
            if candidates:
                # Cluster nearby candidates using weighted clustering
                final_positions = self._weighted_cluster_barlines(candidates, weights)
                print(f"  After clustering, found {len(final_positions)} consensus barline positions.")
                
                # Create barlines at the consensus positions
                new_barlines = self._create_barlines_at_positions(system, final_positions, existing_barlines)
                print(f"  Created {len(new_barlines)} new barlines.")
                total_barlines_added += len(new_barlines)
                
                # After adding new barlines, update measures
                if new_barlines:
                    self.identify_measures()
                    
                    # Validate the resulting measures
                    self._validate_measure_consistency(system)
            else:
                print("  No barline candidates found.")
        
        print(f"\nBarline Inference Complete: Added {total_barlines_added} barlines across all systems.")
        return total_barlines_added > 0



    def _detect_missing_barlines(self, system, existing_barlines):
        """
        Analyze if a system is likely missing barlines by examining:
        1. Gaps between existing barlines
        2. Musical content distribution
        3. Time signature vs. expected barline count
        
        Returns: Estimated number of missing barlines
        """
        # Get musical elements (notes and rests)
        musical_elements = [e for e in system.elements if isinstance(e, (Note, Rest))]
        musical_elements.sort(key=lambda e: e.x)
        
        if not musical_elements:
            return 0  # No content to analyze
        
        if not existing_barlines:
            # No barlines at all - definitely missing some if we have content
            return len(musical_elements) // 4  # Reasonable estimate based on content
        
        # Check for excessively large gaps between barlines
        barline_xs = [b.x for b in existing_barlines]
        barline_xs.sort()
        
        # Get min and max x from musical elements (not system boundaries)
        min_x = min(e.x for e in musical_elements)
        max_x = max(e.x for e in musical_elements)
        
        # Add system boundaries for gap analysis (but only if they contain elements)
        gaps = []
        if min_x < barline_xs[0]:
            gaps.append(barline_xs[0] - min_x)  # Gap before first barline
            
        # Interior gaps
        interior_gaps = [barline_xs[i+1] - barline_xs[i] for i in range(len(barline_xs) - 1)]
        gaps.extend(interior_gaps)
        
        if max_x > barline_xs[-1]:
            gaps.append(max_x - barline_xs[-1])  # Gap after last barline
        
        # Calculate typical gap (use median to avoid outliers)
        if len(gaps) > 2:
            typical_gap = sorted(gaps)[len(gaps) // 2]
        else:
            # Safer calculation with smaller number of gaps
            non_zero_gaps = [g for g in gaps if g > 0]
            if non_zero_gaps:
                typical_gap = sum(non_zero_gaps) / len(non_zero_gaps)
            else:
                # Default if all gaps are zero - avoid division by zero
                typical_gap = 100  # A reasonable default value
        
        # Ensure typical_gap is never zero to avoid division by zero
        typical_gap = max(typical_gap, 1.0)
        
        # Count notes in each gap to ensure we're not just looking at empty spaces
        gap_content = []
        
        # First gap (before first barline)
        if min_x < barline_xs[0]:
            elements_before = sum(1 for e in musical_elements if e.x < barline_xs[0])
            gap_content.append((barline_xs[0] - min_x, elements_before))
        
        # Interior gaps
        for i in range(len(barline_xs) - 1):
            elements_in_gap = sum(1 for e in musical_elements 
                                if barline_xs[i] < e.x < barline_xs[i+1])
            gap_content.append((barline_xs[i+1] - barline_xs[i], elements_in_gap))
        
        # Last gap (after last barline)
        if max_x > barline_xs[-1]:
            elements_after = sum(1 for e in musical_elements if e.x > barline_xs[-1])
            gap_content.append((max_x - barline_xs[-1], elements_after))
        
        # Identify gaps that are significantly larger than typical AND have enough content
        large_gaps = []
        for i, (gap_size, element_count) in enumerate(gap_content):
            # Only consider gaps with reasonable content
            if element_count < 3:
                continue
                
            if gap_size > typical_gap * 1.6:  # Gap is 60% larger than typical
                # This gap might contain missing barlines
                missing_count = round(gap_size / typical_gap) - 1
                
                # Limit based on content (need enough notes for multiple measures)
                missing_count = min(missing_count, element_count // 3)
                
                if missing_count > 0:
                    if i == 0 and min_x < barline_xs[0]:
                        desc = f"before first barline ({missing_count} barlines)"
                    elif i == len(gap_content) - 1 and max_x > barline_xs[-1]:
                        desc = f"after last barline ({missing_count} barlines)"
                    else:
                        idx = i - (1 if min_x < barline_xs[0] else 0)
                        desc = f"between barlines {idx} and {idx+1} ({missing_count} barlines)"
                        
                    large_gaps.append((i, gap_size, missing_count, element_count, desc))
        
        # Estimate total missing based on large gaps
        total_missing = sum(g[2] for g in large_gaps)
        
        # Debug output
        if large_gaps:
            print(f"  Detected {len(large_gaps)} unusually large gaps in barlines:")
            for i, gap, count, elements, desc in large_gaps:
                print(f"    Gap {i}: {gap:.1f} pixels with {elements} elements {desc}")
        
        # Secondary check: if we have a time signature, compare expected vs. actual barline count
        if system.time_signature:
            # Estimate expected measures based on note/rest count and time signature
            time_sig = system.time_signature
            beats_per_measure = time_sig.beats
            
            # Calculate total content duration
            total_duration = 0
            for elem in musical_elements:
                elem_duration = elem.duration if hasattr(elem, 'duration') and elem.duration else 1.0
                total_duration += elem_duration
            
            # If we have enough duration info, use it to estimate measures
            if total_duration > 0:
                measure_duration = time_sig.beats * (4 / time_sig.beat_type)
                expected_measures = max(1, round(total_duration / measure_duration))
                expected_barlines = expected_measures + 1
                
                barline_diff = expected_barlines - len(existing_barlines)
                if barline_diff > 0:
                    print(f"  Based on content duration and time signature, expecting ~{expected_measures} measures "
                        f"({expected_barlines} barlines), missing ~{barline_diff}")
                    
                    # Reconcile with gap-based estimate
                    total_missing = max(total_missing, barline_diff)
        
        return total_missing

    def _analyze_barline_gaps(self, system, existing_barlines):
        """
        Analyze gaps between existing barlines to predict where missing barlines should be.
        """
        if len(existing_barlines) < 2:
            return []
        
        # Get sorted barline positions
        barline_xs = [b.x for b in existing_barlines]
        barline_xs.sort()
        
        # Calculate gaps between consecutive barlines
        gaps = [barline_xs[i+1] - barline_xs[i] for i in range(len(barline_xs) - 1)]
        
        # Get system boundaries
        min_x = min(e.x for e in system.elements) if system.elements else 0
        max_x = max(e.x for e in system.elements) if system.elements else 1000
        
        # Add boundary gaps if there's content outside the barlines
        boundary_gaps = []
        if min_x < barline_xs[0]:
            boundary_gaps.append((None, barline_xs[0] - min_x, min_x, barline_xs[0], "start"))
            
        if max_x > barline_xs[-1]:
            boundary_gaps.append((len(barline_xs)-1, max_x - barline_xs[-1], barline_xs[-1], max_x, "end"))
        
        # Calculate typical gap (median is more robust to outliers)
        if len(gaps) >= 3:
            typical_gap = sorted(gaps)[len(gaps)//2]
        else:
            # With few gaps, use mean of non-zero gaps to avoid division by zero
            non_zero_gaps = [g for g in gaps if g > 0]
            typical_gap = sum(non_zero_gaps) / len(non_zero_gaps) if non_zero_gaps else 100  # Default if all gaps are zero
        
        # Ensure typical_gap is never zero to avoid division by zero
        typical_gap = max(typical_gap, 1.0)
        
        # Find large gaps that might contain missing barlines
        candidate_positions = []
        
        # Process interior gaps
        for i, gap in enumerate(gaps):
            # Count musical elements in this gap
            elements_in_gap = sum(1 for e in system.elements 
                                if isinstance(e, (Note, Rest)) and
                                barline_xs[i] < e.x < barline_xs[i+1])
            
            # Skip gaps with no content
            if elements_in_gap == 0:
                continue
                
            # Only consider gaps with musical content
            if gap > typical_gap * 1.5 and elements_in_gap >= 3:
                # Determine how many barlines should be in this gap
                missing_count = round(gap / typical_gap) - 1
                missing_count = min(missing_count, elements_in_gap // 3)  # Ensure enough elements per measure
                
                if missing_count < 1:
                    continue
                    
                # Group notes into clusters to find natural divisions
                element_positions = [e.x for e in system.elements 
                                    if isinstance(e, (Note, Rest)) and
                                    barline_xs[i] < e.x < barline_xs[i+1]]
                
                # Calculate gaps between adjacent notes
                if len(element_positions) >= 2:
                    note_gaps = [(element_positions[j+1] - element_positions[j]) 
                            for j in range(len(element_positions)-1)]
                    
                    # Find large gaps between notes (potential measure boundaries)
                    note_gap_threshold = max(20, sum(note_gaps) / len(note_gaps) * 1.5)
                    
                    large_note_gaps = [(j, gap) for j, gap in enumerate(note_gaps) 
                                    if gap > note_gap_threshold]
                    
                    # Sort by gap size (largest first)
                    large_note_gaps.sort(key=lambda x: x[1], reverse=True)
                    
                    # Use largest note gaps for barline positions
                    for j in range(min(missing_count, len(large_note_gaps))):
                        gap_idx, _ = large_note_gaps[j]
                        # Place barline in the middle of the gap
                        gap_middle = (element_positions[gap_idx] + element_positions[gap_idx+1]) / 2
                        candidate_positions.append(gap_middle)
                else:
                    # Fallback: divide evenly if clustering fails
                    start_x = barline_xs[i]
                    end_x = barline_xs[i+1]
                    sub_gap = gap / (missing_count + 1)
                    
                    for j in range(1, missing_count + 1):
                        pos = start_x + sub_gap * j
                        candidate_positions.append(pos)
        
        # Process boundary gaps similarly
        for idx, gap_size, start_pos, end_pos, gap_type in boundary_gaps:
            # Count elements in boundary gap
            elements_in_gap = sum(1 for e in system.elements 
                                if isinstance(e, (Note, Rest)) and
                                start_pos < e.x < end_pos)
            
            if elements_in_gap < 3:
                continue
                
            if gap_size > typical_gap * 1.5:
                missing_count = round(gap_size / typical_gap) - 1
                missing_count = min(missing_count, elements_in_gap // 3)
                
                if missing_count < 1:
                    continue
                    
                # Try to find natural breaks
                element_positions = [e.x for e in system.elements 
                                if isinstance(e, (Note, Rest)) and
                                start_pos < e.x < end_pos]
                
                if len(element_positions) >= 2:
                    # Similar clustering approach as above
                    # [same code as for interior gaps]
                    note_gaps = [(element_positions[j+1] - element_positions[j]) 
                            for j in range(len(element_positions)-1)]
                    
                    note_gap_threshold = max(20, sum(note_gaps) / len(note_gaps) * 1.5)
                    
                    large_note_gaps = [(j, gap) for j, gap in enumerate(note_gaps) 
                                    if gap > note_gap_threshold]
                    
                    large_note_gaps.sort(key=lambda x: x[1], reverse=True)
                    
                    for j in range(min(missing_count, len(large_note_gaps))):
                        gap_idx, _ = large_note_gaps[j]
                        gap_middle = (element_positions[gap_idx] + element_positions[gap_idx+1]) / 2
                        candidate_positions.append(gap_middle)
                else:
                    # Fallback: divide evenly
                    sub_gap = gap_size / (missing_count + 1)
                    for j in range(1, missing_count + 1):
                        pos = start_pos + sub_gap * j
                        candidate_positions.append(pos)
        
        return candidate_positions

    def _weighted_cluster_barlines(self, positions, weights):
        """
        Enhanced clustering algorithm that uses weights to determine
        consensus barline positions.
        """
        if not positions:
            return []
        
        # Combine positions and weights and sort by position
        data = sorted(zip(positions, weights), key=lambda x: x[0])
        
        # Determine clustering threshold adaptively
        pos_diffs = [data[i+1][0] - data[i][0] for i in range(len(data)-1)]
        if pos_diffs:
            # Use smaller of median spacing and 20px
            threshold = min(sorted(pos_diffs)[len(pos_diffs)//2], 20)
        else:
            threshold = 20
        
        # Perform clustering
        clusters = []
        current_cluster = [data[0]]
        
        for i in range(1, len(data)):
            if data[i][0] - data[i-1][0] < threshold:
                # Add to current cluster
                current_cluster.append(data[i])
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [data[i]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        # Calculate weighted consensus position for each cluster
        consensus_positions = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            positions, weights = zip(*cluster)
            total_weight = sum(weights)
            
            if total_weight > 0:
                # Weighted average by confidence/weight
                weighted_pos = sum(p * w for p, w in zip(positions, weights)) / total_weight
                
                # Calculate cluster strength (total weight)
                cluster_strength = total_weight
                
                # Only keep strong clusters (those with multiple supporting methods)
                if cluster_strength > 0.8 or len(cluster) >= 2:
                    consensus_positions.append(weighted_pos)
            else:
                # Fallback to simple average if all weights are zero
                consensus_positions.append(sum(positions) / len(positions))
        
        return consensus_positions

    def _validate_measure_consistency(self, system):
        """
        Validate and adjust measures for consistency based on music theory.
        """
        # Get time signature for this system
        time_sig = system.time_signature
        if not time_sig:
            time_sig = TimeSignature(beats=4, beat_type=4)
        
        # Expected measure duration in quarter notes
        expected_duration = time_sig.beats * (4 / time_sig.beat_type)
        
        # Check each measure's total duration
        for i, measure in enumerate(system.measures):
            elements = [e for e in measure.elements if isinstance(e, (Note, Rest))]
            
            # Calculate total duration
            total_duration = 0
            for elem in elements:
                elem_duration = elem.duration if hasattr(elem, 'duration') and elem.duration else 1.0
                total_duration += elem_duration
            
            # Check if measure is significantly shorter or longer than expected
            ratio = total_duration / expected_duration
            
            if ratio < 0.6 or ratio > 1.4:
                print(f"  Warning: Measure {i+1} has abnormal duration: {total_duration:.2f} beats "
                    f"(expected {expected_duration:.2f})")
                
                # Could implement automatic correction here
                # But for now, just flag the issue           
        
        
        
        
# Example implementation to test the new barline detection functionality


def test_enhanced_barline_detection():
    """
    Test function to demonstrate the enhanced barline detection.
    """
    # This would be the code you'd run to test the new barline detection
    
    # 1. Create processor with sample data
    processor = OMRProcessor(detection_path='example_detections.json',
                             staff_lines_path='example_staff_lines.json')
    
    # 2. Process score without enhanced barline detection for comparison
    processor.identify_staff_systems_auto()
    processor.process_detected_objects()
    processor.assign_to_staff_systems()
    processor.identify_measures()
    
    # Count original barlines
    original_barlines = []
    for system in processor.staff_systems:
        system_barlines = [e for e in system.elements if isinstance(e, Barline)]
        original_barlines.extend(system_barlines)
    
    print(f"Original barline count: {len(original_barlines)}")
    
    # 3. Apply enhanced barline detection
    processor.master_barline_inference()
    
    # Regenerate measures
    processor.identify_measures()
    
    # Count enhanced barlines
    enhanced_barlines = []
    for system in processor.staff_systems:
        system_barlines = [e for e in system.elements if isinstance(e, Barline)]
        enhanced_barlines.extend(system_barlines)
    
    print(f"Enhanced barline count: {len(enhanced_barlines)}")
    print(f"Added {len(enhanced_barlines) - len(original_barlines)} new barlines")
    
    # 4. Generate and compare MusicXML output
    original_xml = processor.generate_musicxml()
    
    # Save results
    with open('original_output.musicxml', 'w') as f:
        f.write(original_xml)
    
    with open('enhanced_output.musicxml', 'w') as f:
        f.write(processor.generate_musicxml())
    
    print("Test complete. Comparison files generated.")

