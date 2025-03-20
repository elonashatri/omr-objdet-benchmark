"""
beam_analyzer.py - Analyzes beam groups and rhythmic durations for notes
"""
import json
import math

def find_beam_groups(data):
    """Find and organize all beam groups in the data"""
    if isinstance(data, str):
        data = json.loads(data)
        
    symbols = data.get('symbols', data.get('detections', []))
    
    # Extract beams and their linked noteheads
    beams = [s for s in symbols if s.get('class_name') == 'beam']
    beam_groups = {}
    
    for i, beam in enumerate(beams):
        beam_id = f"beam_{i}"
        connected_notes = []
        
        if 'linked_symbols' in beam:
            for link in beam['linked_symbols']:
                if link['type'] == 'connects_notehead':
                    note_id = link['id']
                    connected_notes.append(note_id)
        
        if connected_notes:
            beam_groups[beam_id] = {
                'note_ids': connected_notes,
                'notes': []
            }
    
    # Add the actual note objects to each beam group and sort by x position
    for beam_id, group in beam_groups.items():
        for note_id in group['note_ids']:
            if 0 <= note_id < len(symbols):
                note = symbols[note_id]
                if note['class_name'] == 'noteheadBlack':
                    group['notes'].append({
                        'id': note_id,
                        'note': note,
                        'x_pos': note['bbox']['center_x']
                    })
        
        # Sort notes in the beam group by x position
        group['notes'].sort(key=lambda n: n['x_pos'])
    
    return beam_groups

def identify_note_durations(data, default_beam_duration='16th'):
    """Identify durations for all notes"""
    if isinstance(data, str):
        data = json.loads(data)
        
    symbols = data.get('symbols', data.get('detections', []))
    
    # Find noteheads and flags
    noteheads = [s for s in symbols if s.get('class_name') == 'noteheadBlack']
    flags = [s for s in symbols if 'flag' in s.get('class_name', '').lower()]
    
    # Get beam groups
    beam_groups = find_beam_groups(data)
    
    # Create mapping from note ID to beam group
    note_to_beam = {}
    for beam_id, group in beam_groups.items():
        for note_info in group['notes']:
            note_to_beam[note_info['id']] = beam_id
    
    # Map flags to nearby notes
    flagged_notes = {}
    for flag in flags:
        flag_x = flag['bbox']['center_x']
        flag_y = flag['bbox']['center_y']
        flag_class = flag['class_name']
        
        # Determine the duration based on flag type
        if 'flag32nd' in flag_class:
            duration = '32nd'
            divisions = 2
        elif 'flag16th' in flag_class:
            duration = '16th'
            divisions = 4
        elif 'flag8th' in flag_class:
            duration = 'eighth'
            divisions = 8
        else:
            continue  # Skip unrecognized flags
        
        # Find closest note
        closest_note = None
        min_distance = float('inf')
        
        for i, note in enumerate(noteheads):
            note_x = note['bbox']['center_x']
            note_y = note['bbox']['center_y']
            distance = math.sqrt((note_x - flag_x)**2 + (note_y - flag_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_note = (i, note)
        
        if closest_note and min_distance < 50:  # Threshold for flag-to-note association
            note_id, note = closest_note
            flagged_notes[note_id] = {
                'duration': duration,
                'divisions': divisions
            }
    
    # Process all noteheads
    note_durations = {}
    for i, note in enumerate(noteheads):
        # Check if note has an explicit flag
        if i in flagged_notes:
            duration = flagged_notes[i]['duration']
            divisions = flagged_notes[i]['divisions']
        # Check if note is part of a beam group
        elif i in note_to_beam:
            # Beamed notes default to 16th notes unless specified otherwise
            if default_beam_duration == '32nd':
                duration = '32nd'
                divisions = 2
            else:
                duration = '16th'
                divisions = 4
                
            # Also note the beam position
            beam_id = note_to_beam[i]
            beam_group = beam_groups[beam_id]
            note_index = next(j for j, n in enumerate(beam_group['notes']) if n['id'] == i)
            
            if note_index == 0:
                beam_pos = 'begin'
            elif note_index == len(beam_group['notes']) - 1:
                beam_pos = 'end'
            else:
                beam_pos = 'continue'
                
            note_durations[i] = {
                'duration': duration,
                'divisions': divisions,
                'beam_id': beam_id,
                'beam_pos': beam_pos
            }
            continue
        else:
            # Default for isolated notes (assuming quarter notes)
            duration = 'quarter'
            divisions = 16
        
        note_durations[i] = {
            'duration': duration,
            'divisions': divisions
        }
    
    return {
        'note_durations': note_durations,
        'beam_groups': beam_groups
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        duration_info = identify_note_durations(data)
        print(json.dumps(duration_info, indent=2))
    else:
        print("Usage: python beam_analyzer.py input.json")