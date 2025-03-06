import json
import os
import sys

def identify_pitch(notehead, staff_lines, clef_type="treble"):
    """Simple pitch identification function"""
    # Get staff line positions
    line_positions = sorted([line["bbox"]["center_y"] for line in staff_lines])
    
    # Get notehead position
    notehead_y = notehead["bbox"]["center_y"]
    
    # Calculate staff spacing
    if len(line_positions) >= 2:
        spacings = [line_positions[i+1] - line_positions[i] for i in range(len(line_positions)-1)]
        staff_spacing = sum(spacings) / len(spacings)
    else:
        staff_spacing = 25  # Default
    
    # Get middle line position
    if len(line_positions) >= 5:
        middle_line = line_positions[2]
    else:
        middle_line = line_positions[len(line_positions) // 2]
    
    # Calculate position relative to middle line
    position = round((middle_line - notehead_y) / (staff_spacing / 2))
    
    # Map to note names
    treble_notes = {
        -5: "C4", -4: "D4", -3: "E4", -2: "F4", -1: "G4", 
        0: "A4", 1: "B4", 2: "C5", 3: "D5", 4: "E5", 
        5: "F5", 6: "G5", 7: "A5", 8: "B5", 9: "C6"
    }
    
    bass_notes = {
        -5: "E2", -4: "F2", -3: "G2", -2: "A2", -1: "B2", 
        0: "C3", 1: "D3", 2: "E3", 3: "F3", 4: "G3", 
        5: "A3", 6: "B3", 7: "C4", 8: "D4", 9: "E4"
    }
    
    if clef_type == "treble":
        note_name = treble_notes.get(position, f"Unknown({position})")
    else:
        note_name = bass_notes.get(position, f"Unknown({position})")
    
    return {
        "note_name": note_name,
        "staff_position": position,
        "clef": clef_type
    }

def process_file(input_file, output_file):
    """Process merged detections file and add pitch information"""
    print(f"Processing: {input_file}")
    
    # Load input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get all detections
    detections = data.get("detections", [])
    
    # Group by staff assignment
    staff_systems = {}
    staff_lines = {}
    
    for detection in detections:
        staff_id = detection.get("staff_system", detection.get("staff_assignment"))
        if staff_id is not None:
            if staff_id not in staff_systems:
                staff_systems[staff_id] = []
            staff_systems[staff_id].append(detection)
            
            # Track staff lines
            if "staff_line" in detection.get("class_name", "").lower():
                if staff_id not in staff_lines:
                    staff_lines[staff_id] = []
                staff_lines[staff_id].append(detection)
    
    print(f"Found {len(staff_systems)} staff systems")
    
    # Process each staff system
    for staff_id, system_detections in staff_systems.items():
        # Find clef
        clef_type = "treble"  # Default
        for detection in system_detections:
            if "gclef" in detection.get("class_name", "").lower():
                clef_type = "treble"
                break
            elif "fclef" in detection.get("class_name", "").lower():
                clef_type = "bass"
                break
        
        # Process noteheads
        for detection in system_detections:
            if "notehead" in detection.get("class_name", "").lower():
                # Add pitch information
                detection["pitch"] = identify_pitch(
                    detection, 
                    staff_lines.get(staff_id, []),
                    clef_type
                )
    
    # Save result
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved result to: {output_file}")
    
    # Count noteheads with pitch
    noteheads = [d for d in detections if "notehead" in d.get("class_name", "").lower()]
    noteheads_with_pitch = [n for n in noteheads if "pitch" in n]
    print(f"Added pitch information to {len(noteheads_with_pitch)}/{len(noteheads)} noteheads")

# Main function
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_pitch.py input_file output_file")
        sys.exit(1)
    
    process_file(sys.argv[1], sys.argv[2])