import os
import json
import numpy as np
from collections import defaultdict

class PitchIdentifier:
    """
    Class for identifying pitches of noteheads in a music score
    based on their position relative to staff lines and taking
    into account key signatures and accidentals.
    """
    
    def __init__(self, debug=True):
        # Set debug flag
        self.debug = debug
        
        # Define the base note names by staff position
        # For treble clef (G clef)
        self.treble_notes = {
            -5: "C4", -4: "D4", -3: "E4", -2: "F4", -1: "G4", 
            0: "A4", 1: "B4", 2: "C5", 3: "D5", 4: "E5", 
            5: "F5", 6: "G5", 7: "A5", 8: "B5", 9: "C6", 
            10: "D6", 11: "E6", 12: "F6"
        }
        
        # For bass clef (F clef)
        self.bass_notes = {
            -5: "E2", -4: "F2", -3: "G2", -2: "A2", -1: "B2", 
            0: "C3", 1: "D3", 2: "E3", 3: "F3", 4: "G3", 
            5: "A3", 6: "B3", 7: "C4", 8: "D4", 9: "E4", 
            10: "F4", 11: "G4", 12: "A4"
        }
        
        # Define accidental symbols
        self.accidental_classes = {
            "accidentalSharp": 1,     # Sharp raises by one semitone
            "accidentalFlat": -1,     # Flat lowers by one semitone
            "accidentalNatural": 0,   # Natural cancels previous accidentals
            "accidentalDoubleSharp": 2,  # Double sharp raises by two semitones
            "accidentalDoubleFlat": -2   # Double flat lowers by two semitones
        }
        
        # Define key signature accidentals
        # Order of sharps: F C G D A E B
        self.sharp_key_signatures = {
            1: ["F"],       # G major / E minor (1 sharp: F#)
            2: ["F", "C"],  # D major / B minor (2 sharps: F#, C#)
            3: ["F", "C", "G"],  # A major / F# minor (3 sharps)
            4: ["F", "C", "G", "D"],  # E major / C# minor (4 sharps)
            5: ["F", "C", "G", "D", "A"],  # B major / G# minor (5 sharps)
            6: ["F", "C", "G", "D", "A", "E"],  # F# major / D# minor (6 sharps)
            7: ["F", "C", "G", "D", "A", "E", "B"]  # C# major / A# minor (7 sharps)
        }
        
        # Order of flats: B E A D G C F
        self.flat_key_signatures = {
            1: ["B"],       # F major / D minor (1 flat: Bb)
            2: ["B", "E"],  # Bb major / G minor (2 flats: Bb, Eb)
            3: ["B", "E", "A"],  # Eb major / C minor (3 flats)
            4: ["B", "E", "A", "D"],  # Ab major / F minor (4 flats)
            5: ["B", "E", "A", "D", "G"],  # Db major / Bb minor (5 flats)
            6: ["B", "E", "A", "D", "G", "C"],  # Gb major / Eb minor (6 flats)
            7: ["B", "E", "A", "D", "G", "C", "F"]  # Cb major / Ab minor (7 flats)
        }
        
        # Note to semitone mapping (for accidental calculations)
        self.note_to_semitone = {
            "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11
        }
        
        # Semitone to note mapping (for converting back after applying accidentals)
        self.semitone_to_note = {
            0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
            6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"
        }
        
        # Alternative enharmonic spellings
        self.enharmonic_spellings = {
            "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb",
            "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"
        }
    
    def log(self, message):
        """Print debug message if debug mode is enabled"""
        if self.debug:
            print(f"[PitchIdentifier] {message}")
    
    def identify_clef(self, system_symbols):
        """
        Identify the clef in a staff system
        
        Args:
            system_symbols: List of symbols in the staff system
            
        Returns:
            String indicating clef type ("treble", "bass", or "unknown")
        """
        for symbol in system_symbols:
            class_name = symbol.get("class_name", "")
            if class_name == "gClef":
                return "treble"
            elif class_name == "fClef":
                return "bass"
        
        # Default to treble if no clef is found
        # self.log("No clef found, defaulting to treble")
        return "treble"
        
    def identify_key_signature(self, system_symbols):
        """
        Identify the key signature in a staff system
        
        Args:
            system_symbols: List of symbols in the staff system
            
        Returns:
            Dictionary with key signature information
        """
        # Group accidentals that appear at the beginning of the staff
        sharps = []
        flats = []
        
        # Get staff position for each accidental
        for symbol in system_symbols:
            class_name = symbol.get("class_name", "").lower()
            if "accidental" not in class_name:
                continue
                
            # Get position
            x_pos = symbol["bbox"]["center_x"]
            y_pos = symbol["bbox"]["center_y"]
            
            # Only consider accidentals at the beginning of the staff (first 20% of width)
            # and not associated with notes (key signature accidentals)
            is_isolated = True
            for other in system_symbols:
                if "noteheadBlack" in other.get("class_name", "").lower():
                    # Check if accidental is close to this notehead horizontally
                    if (abs(other["bbox"]["center_x"] - x_pos) < 30 and 
                        abs(other["bbox"]["center_y"] - y_pos) < 20):
                        is_isolated = False
                        break
            
            if not is_isolated:
                continue
                
            # Count sharps and flats
            if "sharp" in class_name and "natural" not in class_name and "double" not in class_name:
                sharps.append(y_pos)
            elif "flat" in class_name and "natural" not in class_name and "double" not in class_name:
                flats.append(y_pos)
        
        # Determine key signature
        key_sig = {"type": "none", "count": 0, "notes": []}
        
        if len(sharps) > 0 and len(flats) == 0:
            key_sig["type"] = "sharp"
            key_sig["count"] = len(sharps)
            if key_sig["count"] in self.sharp_key_signatures:
                key_sig["notes"] = self.sharp_key_signatures[key_sig["count"]]
        elif len(flats) > 0 and len(sharps) == 0:
            key_sig["type"] = "flat"
            key_sig["count"] = len(flats)
            if key_sig["count"] in self.flat_key_signatures:
                key_sig["notes"] = self.flat_key_signatures[key_sig["count"]]
        
        return key_sig
    
    def get_staff_position(self, notehead, staff_lines):
        """
        Calculate the position of a notehead relative to the staff lines
        
        Args:
            notehead: Notehead symbol data
            staff_lines: List of staff line data for the system
            
        Returns:
            Integer representing staff position (0 = middle line, positive = above, negative = below)
        """
        if not staff_lines:
            # self.log(f"WARNING: No staff lines found for notehead at {notehead['bbox']['center_x']}, {notehead['bbox']['center_y']}")
            return 0  # Default position if no staff lines
        
        # Sort staff lines by vertical position
        sorted_lines = sorted(staff_lines, key=lambda x: x["bbox"]["center_y"])
        
        # Get y positions of all staff lines
        line_positions = [line["bbox"]["center_y"] for line in sorted_lines]
        
        # Get notehead position
        notehead_y = notehead["bbox"]["center_y"]
        
        # Calculate staff line spacing (average)
        if len(line_positions) < 2:
            # Default if not enough staff lines
            staff_spacing = 10
            # self.log(f"Not enough staff lines to calculate spacing, using default: {staff_spacing}")
        else:
            # Calculate the average spacing between adjacent staff lines
            spacings = [line_positions[i+1] - line_positions[i] 
                    for i in range(len(line_positions)-1)]
            staff_spacing = sum(spacings) / len(spacings)
            # self.log(f"Calculated staff spacing: {staff_spacing}")
        
        # Find the closest staff line
        line_distances = [abs(notehead_y - line_y) for line_y in line_positions]
        closest_line_index = line_distances.index(min(line_distances))
        closest_line_y = line_positions[closest_line_index]
        
        # Calculate the distance from the notehead to the closest staff line
        distance_to_line = abs(notehead_y - closest_line_y)
        
        # Determine if the notehead is above or below the staff line
        is_above = notehead_y < closest_line_y
        
        # Calculate the vertical offset in terms of staff line units
        # If the distance is less than half a staff line spacing, consider it on the line
        if distance_to_line < staff_spacing / 2:
            offset = 0
        else:
            # Calculate how many half-spaces away from the line the notehead is
            offset = round(distance_to_line / (staff_spacing / 2))
            
            # Adjust the sign based on whether it's above or below the line
            offset = offset if is_above else -offset
        
        # Calculate the final staff position
        # Middle line is typically index 2 in a 5-line staff
        middle_line_index = 2
        position = (closest_line_index - middle_line_index) * 2 + offset
        
        # self.log(f"Detailed position calculation:")
        # self.log(f"  Notehead Y: {notehead_y}")
        # self.log(f"  Closest line Y: {closest_line_y}")
        # self.log(f"  Staff spacing: {staff_spacing}")
        # self.log(f"  Distance to line: {distance_to_line}")
        # self.log(f"  Is above: {is_above}")
        # self.log(f"  Offset: {offset}")
        # self.log(f"  Final staff position: {position}")
        
        return position
    
    def apply_key_signature(self, note_name, key_sig):
        """
        Apply key signature accidentals to a note name
        
        Args:
            note_name: Base note name (e.g., "C4")
            key_sig: Key signature information
            
        Returns:
            Updated note name with appropriate accidentals
        """
        if key_sig["type"] == "none" or not key_sig["notes"]:
            return note_name
        
        # Extract the note letter from the note name
        note_letter = note_name[0]
        
        # Check if this note is affected by the key signature
        if note_letter in key_sig["notes"]:
            # Get the base note and octave
            base_note = note_name[0]
            octave = note_name[1:] if len(note_name) > 1 else ""
            
            # Apply the appropriate accidental
            if key_sig["type"] == "sharp":
                return f"{base_note}#{octave}"
            elif key_sig["type"] == "flat":
                return f"{base_note}b{octave}"
        
        return note_name
    
    def apply_local_accidental(self, note_name, accidental_type):
        """
        Apply a local accidental to a note name
        
        Args:
            note_name: Note name (e.g., "C4" or "C#4")
            accidental_type: Type of accidental to apply
            
        Returns:
            Updated note name with appropriate accidental
        """
        # Extract the base note and octave
        if '#' in note_name or 'b' in note_name:
            # Already has an accidental
            base_note = note_name[0]
            octave = note_name[2:] if len(note_name) > 2 else ""
        else:
            base_note = note_name[0]
            octave = note_name[1:] if len(note_name) > 1 else ""
        
        # Apply the appropriate accidental
        if accidental_type == "accidentalSharp":
            return f"{base_note}#{octave}"
        elif accidental_type == "accidentalFlat":
            return f"{base_note}b{octave}"
        elif accidental_type == "accidentalNatural":
            return f"{base_note}{octave}"
        elif accidental_type == "accidentalDoubleSharp":
            return f"{base_note}x{octave}"  # Use 'x' for double sharp
        elif accidental_type == "accidentalDoubleFlat":
            return f"{base_note}bb{octave}"  # Use 'bb' for double flat
        
        return note_name
    
    def find_local_accidentals(self, notehead, system_symbols, measure_bounds=None):
        """
        Find local accidentals that affect a given notehead
        
        Args:
            notehead: Notehead symbol data
            system_symbols: List of symbols in the staff system
            measure_bounds: Optional bounds of the current measure
            
        Returns:
            Accidental symbol if found, None otherwise
        """
        notehead_x = notehead["bbox"]["center_x"]
        notehead_y = notehead["bbox"]["center_y"]
        
        # Look for linked accidentals first (from the linked_symbols field)
        if "linked_symbols" in notehead:
            for link in notehead["linked_symbols"]:
                if link["type"] == "has_accidental":
                    # Find the referenced accidental
                    for symbol in system_symbols:
                        # Check if this is the linked accidental by id
                        symbol_id = symbol.get("id", system_symbols.index(symbol))
                        if symbol_id == link["id"]:
                            if "accidental" in symbol.get("class_name", "").lower():
                                # self.log(f"Found linked accidental: {symbol['class_name']}")
                                return symbol["class_name"]
        
        # If no linked accidental, search for accidentals near this notehead
        for symbol in system_symbols:
            class_name = symbol.get("class_name", "").lower()
            if "accidental" not in class_name:
                continue
                
            symbol_x = symbol["bbox"]["center_x"]
            symbol_y = symbol["bbox"]["center_y"]
            
            # Check if accidental is to the left of the notehead
            is_to_left = symbol_x < notehead_x
            
            # Check if accidental is at the same vertical position (same note)
            is_same_height = abs(symbol_y - notehead_y) < 15  # Adjust threshold as needed
            
            # Check if accidental is close enough horizontally
            is_close = notehead_x - symbol_x < 50  # Adjust threshold as needed
            
            if is_to_left and is_same_height and is_close:
                # self.log(f"Found nearby accidental: {symbol['class_name']}")
                return symbol["class_name"]
        
        return None
    
    # def identify_notehead_pitches(self, linked_data):
    def identify_notehead_pitches(self, linked_data):
        """
        Identify pitches for all noteheads in the linked data
        
        Args:
            linked_data: Dictionary containing detected staff lines and musical symbols
            
        Returns:
            Updated linked data with pitch information for noteheads
        """
        self.log("Starting pitch identification...")
        
        # Create a deep copy of the input data to avoid modifying it
        result = linked_data.copy()
        if "detections" in result:
            result["detections"] = [detection.copy() for detection in result["detections"]]
        
        # Count symbols by type for debugging
        symbol_counts = defaultdict(int)
        for symbol in result.get("detections", []):
            class_name = symbol.get("class_name", "unknown")
            symbol_counts[class_name] += 1
        
        self.log(f"Symbol counts: {dict(symbol_counts)}")
        
        # Extract staff lines
        staff_lines = [s for s in result.get("detections", []) 
                    if s.get("class_name", "") == "staff_line"]
        self.log(f"Found {len(staff_lines)} staff lines")
        
        # Group staff lines by staff system
        staff_systems_lines = defaultdict(list)
        for line in staff_lines:
            # Use staff_system field if available
            staff_id = line.get("staff_system")
            if staff_id is not None:
                staff_systems_lines[staff_id].append(line)
        
        self.log(f"Grouped staff lines into {len(staff_systems_lines)} staff systems")
        
        # Group all symbols by staff system
        staff_systems_symbols = defaultdict(list)
        for symbol in result.get("detections", []):
            # Try both staff_system and staff_assignment
            staff_id = symbol.get("staff_system", symbol.get("staff_assignment"))
            if staff_id is not None:
                staff_systems_symbols[staff_id].append(symbol)
        
        self.log(f"Grouped all symbols into {len(staff_systems_symbols)} staff systems")
        
        # Track active accidentals in each measure per staff line and note letter
        # Format: {staff_id: {measure_id: {note_letter: accidental_type}}}
        active_accidentals = defaultdict(lambda: defaultdict(dict))
        
        # Process each staff system
        for staff_id, system_symbols in staff_systems_symbols.items():
            self.log(f"Processing staff system {staff_id} with {len(system_symbols)} symbols")
            
            # Get staff lines for this system
            system_staff_lines = staff_systems_lines.get(staff_id, [])
            
            # Identify clef for this system
            clef_type = self.identify_clef(system_symbols)
            # self.log(f"Identified clef: {clef_type}")
            
            # Identify key signature for this system
            key_sig = self.identify_key_signature(system_symbols)
            # self.log(f"Identified key signature: {key_sig}")
            
            # Find all noteheads in this system
            noteheads = [s for s in system_symbols if s.get("class_name", "") == "noteheadBlack"]
            # self.log(f"Found {len(noteheads)} noteheads in system {staff_id}")
            
            # Sort noteheads by x-coordinate (left to right)
            noteheads.sort(key=lambda x: x["bbox"]["center_x"])
            
            # Process each notehead
            for notehead in noteheads:
                # self.log(f"Processing notehead at {notehead['bbox']['center_x']}, {notehead['bbox']['center_y']}")
                
                # Get the measure ID if available (for tracking accidentals)
                measure_id = notehead.get("measure_id", 0)
                
                # Get staff position
                position = self.get_staff_position(notehead, system_staff_lines)
                
                # Get base note name based on clef and position
                if clef_type == "treble":
                    base_note = self.treble_notes.get(position, "?")
                else:  # bass clef
                    base_note = self.bass_notes.get(position, "?")
                
                # self.log(f"Base note: {base_note} (position: {position})")
                
                # Extract the note letter (without octave)
                note_letter = base_note[0] if base_note and len(base_note) > 0 else "?"
                
                # Apply key signature
                note_name = self.apply_key_signature(base_note, key_sig)
                # self.log(f"After key signature: {note_name}")
                
                # Check for local accidentals that affect this notehead
                accidental_type = self.find_local_accidentals(notehead, system_symbols)
                
                # If a local accidental is found, apply it and update active accidentals
                if accidental_type:
                    # self.log(f"Found local accidental: {accidental_type}")
                    note_name = self.apply_local_accidental(note_name, accidental_type)
                    active_accidentals[staff_id][measure_id][note_letter] = accidental_type
                # Otherwise, check if this note is affected by an active accidental in this measure
                elif note_letter in active_accidentals[staff_id][measure_id]:
                    prev_accidental = active_accidentals[staff_id][measure_id][note_letter]
                    # self.log(f"Applying active accidental from measure: {prev_accidental}")
                    note_name = self.apply_local_accidental(note_name, prev_accidental)
                
                # self.log(f"Final pitch: {note_name}")
                
                # Add pitch information to the notehead
                notehead["pitch"] = {
                    "note_name": note_name,
                    "staff_position": position,
                    "clef": clef_type,
                    "key_signature": key_sig,
                    "local_accidental": accidental_type if accidental_type else None
                }
            
            # Count noteheads with pitch information after processing
            noteheads = [s for s in result.get("detections", []) if s.get("class_name", "") == "noteheadBlack"]
            noteheads_with_pitch = [n for n in noteheads if "pitch" in n]
            # self.log(f"After processing: {len(noteheads_with_pitch)}/{len(noteheads)} noteheads have pitch information")
        
        return result
    

    def process_file(self, input_file, output_file=None):
        """
        Process a file containing linked staff lines and symbols
        """
        # self.log(f"Processing file: {input_file}")
        
        # Load input data
        with open(input_file, 'r') as f:
            linked_data = json.load(f)
        
        # Check if we have a "symbols" field instead of "detections"
        if "symbols" in linked_data and not "detections" in linked_data:
            # self.log("Found 'symbols' key instead of 'detections', adapting structure")
            linked_data["detections"] = linked_data["symbols"]
        
        # If we still don't have detections, create an empty list
        if "detections" not in linked_data:
            linked_data["detections"] = []
        
        # Count symbols
        detections = linked_data.get("detections", [])
        noteheads = [d for d in detections if "notehead" in d.get("class_name", "").lower()]
        staff_lines = [d for d in detections if "staff" in d.get("class_name", "").lower()]
        
        # self.log(f"Loaded data with {len(detections)} detections ({len(noteheads)} noteheads, {len(staff_lines)} staff lines)")
        
        # Identify pitches
        result = self.identify_notehead_pitches(linked_data)
        
        # Generate output path if not provided
        if output_file is None:
            input_base = os.path.splitext(input_file)[0]
            output_file = f"{input_base}_pitched.json"
        
        # Save result
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # self.log(f"Saved pitch information to {output_file}")
        return output_file


def main():
    """Main function to run the pitch identification pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Identify pitches for noteheads in music scores")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file with linked data")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Run pitch identification
    identifier = PitchIdentifier(debug=args.debug)
    identifier.process_file(args.input, args.output)


if __name__ == "__main__":
    main()