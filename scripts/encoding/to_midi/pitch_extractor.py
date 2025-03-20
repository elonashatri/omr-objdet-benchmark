"""
pitch_extractor.py - Extracts pitch information from OMR JSON data
"""
import json

def extract_key_signature(data):
    """Extract key signature information from the first note with pitch data"""
    symbols = data.get('detections', data.get('detections', []))
    
    # Find the first noteheadBlack with pitch information
    for symbol in symbols:
        if symbol.get('class_name') == 'noteheadBlack' and 'pitch' in symbol:
            key_sig = symbol['pitch'].get('key_signature', {})
            if key_sig:
                return key_sig
    
    # Default to C major if no key signature found
    return {"type": "natural", "count": 0, "notes": []}

def get_fifths_from_key_signature(key_sig):
    """Convert key signature to fifths value for MusicXML"""
    if not key_sig:
        return 0
        
    key_type = key_sig.get('type', 'natural')
    count = key_sig.get('count', 0)
    
    if key_type == 'sharp':
        return count
    elif key_type == 'flat':
        return -count
    else:
        return 0

def extract_accidentals(data):
    """Extract mapping of accidentals to the notes they modify"""
    symbols = data.get('detections', data.get('detections', []))
    accidentals = {}
    
    for i, symbol in enumerate(symbols):
        if 'accidental' in symbol.get('class_name', '').lower():
            if 'linked_symbols' in symbol:
                for link in symbol['linked_symbols']:
                    if link['type'] == 'modifies_note':
                        note_id = link['id']
                        accidental_type = symbol['class_name']
                        accidentals[note_id] = accidental_type
    
    return accidentals

def get_complete_pitch_info(note, symbols, accidentals_map, note_id):
    """Get complete pitch information for a note, including accidentals"""
    pitch_info = note.get('pitch', {})
    
    # Default values if pitch info isn't available
    if not pitch_info:
        return {
            'step': 'C',
            'octave': '4',
            'alter': 0,
            'accidental': None
        }
    
    # Extract note name from pitch info
    note_name = pitch_info.get('note_name')
    staff_position = pitch_info.get('staff_position')
    
    print(f"Debug - Note {note_id}: name={note_name}, staff_pos={staff_position}")
    
    # If note_name isn't available, try to use staff_position
    if not note_name and staff_position is not None:
        # Map staff positions to note names (assuming treble clef)
        # staff_pos 0 = A4, -1 = G4, -2 = F4, etc.
        position_to_note = {
            0: 'A4',  # Middle line
            -1: 'G4',
            -2: 'F4',
            -3: 'E4',
            -4: 'D4',
            -5: 'C4',
            -6: 'B3',
            -7: 'A3',
            1: 'B4',
            2: 'C5',
            3: 'D5',
            4: 'E5'
        }
        note_name = position_to_note.get(staff_position, 'C4')
        print(f"Debug - Derived note name from staff position: {note_name}")
    
    # If we still don't have a note name, use default
    if not note_name:
        note_name = 'C4'
    
    # Initialize values
    basic_note = 'C'
    octave = '4'
    alter = 0
    accidental = None
    
    # Parse the note name more carefully
    if note_name:
        # Extract the basic note (first character)
        basic_note = note_name[0].upper()
        
        # Look for accidentals in the note name
        if '#' in note_name or 'sharp' in note_name.lower():
            alter = 1
            accidental = 'sharp'
        elif 'b' in note_name or 'flat' in note_name.lower():
            alter = -1
            accidental = 'flat'
        
        # Extract octave (the last digit in the string)
        digits = [c for c in note_name if c.isdigit()]
        if digits:
            octave = digits[-1]
    
    # Check for local accidental in pitch info (overrides note name)
    local_acc = pitch_info.get('local_accidental')
    if local_acc:
        if 'Sharp' in local_acc:
            alter = 1
            accidental = 'sharp'
        elif 'Flat' in local_acc:
            alter = -1
            accidental = 'flat'
        elif 'Natural' in local_acc:
            alter = 0
            accidental = 'natural'
    
    # Check if the note has a linked accidental (highest priority)
    if note_id in accidentals_map:
        acc_type = accidentals_map[note_id]
        if 'Sharp' in acc_type:
            alter = 1
            accidental = 'sharp'
        elif 'Flat' in acc_type:
            alter = -1
            accidental = 'flat'
        elif 'Natural' in acc_type:
            alter = 0
            accidental = 'natural'
    
    # Also check key signature - if the note is F and key sig has F# 
    # then apply sharp unless overridden by a local accidental or linked accidental
    key_sig = pitch_info.get('key_signature', {})
    key_notes = key_sig.get('notes', [])
    
    if not local_acc and not (note_id in accidentals_map):
        if basic_note in key_notes:
            if key_sig.get('type') == 'sharp':
                alter = 1
                accidental = 'sharp'
            elif key_sig.get('type') == 'flat':
                alter = -1
                accidental = 'flat'
    
    return {
        'step': basic_note,
        'octave': octave,
        'alter': alter,
        'accidental': accidental
    }

def process_json_for_pitch(json_data):
    """Process JSON data to extract all pitch information"""
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
        
    symbols = data.get('detections', data.get('detections', []))
    
    # Debug: Print number of noteheads found
    noteheads = [s for s in symbols if s.get('class_name') == 'noteheadBlack']
    print(f"Debug - Found {len(noteheads)} noteheads")
    
    # Debug: Dump the first notehead with pitch info (if any)
    for note in noteheads:
        if 'pitch' in note:
            print(f"Debug - Sample notehead pitch: {json.dumps(note['pitch'], indent=2)}")
            break
    
    # Extract key signature information
    key_signature = extract_key_signature(data)
    fifths = get_fifths_from_key_signature(key_signature)
    
    # Extract accidentals mapping
    accidentals_map = extract_accidentals(data)
    
    # Process each notehead
    note_pitch_info = {}
    for i, symbol in enumerate(symbols):
        if symbol.get('class_name') == 'noteheadBlack':
            note_id = i
            # Debug: Print raw pitch data
            if 'pitch' in symbol:
                print(f"Debug - Note {note_id} raw pitch: {symbol['pitch']}")
            pitch_info = get_complete_pitch_info(symbol, symbols, accidentals_map, note_id)
            note_pitch_info[note_id] = pitch_info
    
    return {
        'key_signature': {
            'fifths': fifths,
            'mode': 'major' if fifths >= 0 else 'minor',
            'raw': key_signature
        },
        'notes': note_pitch_info,
        'accidentals': accidentals_map
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        result = process_json_for_pitch(data)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python pitch_extractor.py input.json")