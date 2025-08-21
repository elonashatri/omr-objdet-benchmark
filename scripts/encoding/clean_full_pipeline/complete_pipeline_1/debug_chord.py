def debug_chord_connections(processor):
    """
    Debug function to print details about chord connections in the score.
    Shows which notes are connected and which staff they belong to.
    
    Args:
        processor: The OMRProcessor instance with processed notes
    """
    print("\n=== DEBUGGING CHORD CONNECTIONS ===")
    chord_groups = {}  # Group chords by a unique ID
    chord_id = 0
    
    # Find all chord notes
    chord_notes = [n for n in processor.notes if hasattr(n, 'is_chord_member') and n.is_chord_member]
    
    # Group chord notes by their chord reference
    for note in chord_notes:
        if not hasattr(note, 'chord') or not note.chord:
            continue
            
        chord_found = False
        # Check if this chord is already in our groups
        for group_id, group_notes in chord_groups.items():
            if any(n in group_notes for n in note.chord):
                # This chord belongs to an existing group
                for n in note.chord:
                    if n not in group_notes:
                        group_notes.append(n)
                chord_found = True
                break
        
        # If not found, create a new group
        if not chord_found:
            chord_groups[chord_id] = note.chord
            chord_id += 1
    
    # Print details for each chord group
    print(f"Found {len(chord_groups)} chord groups")
    
    for group_id, notes in chord_groups.items():
        print(f"\nChord Group {group_id}:")
        print(f"  Contains {len(notes)} notes")
        
        # Sort notes by vertical position
        notes.sort(key=lambda n: n.y)
        
        # Print details for each note in the chord
        for i, note in enumerate(notes):
            # Get staff and system info
            staff_id = processor._get_staff_id(note) if hasattr(processor, '_get_staff_id') else "unknown"
            system_id = note.staff_system.id if hasattr(note, 'staff_system') else "unknown"
            
            # Get pitch info
            pitch = f"{note.step}{note.octave}" if hasattr(note, 'step') and hasattr(note, 'octave') else "unknown"
            
            # Print note details
            print(f"  Note {i+1}: x={note.x:.1f}, y={note.y:.1f}, pitch={pitch}, staff={staff_id}, system={system_id}")
            
        # Check if this chord spans multiple staves or systems
        staff_ids = set(processor._get_staff_id(n) for n in notes if hasattr(processor, '_get_staff_id'))
        system_ids = set(n.staff_system.id for n in notes if hasattr(n, 'staff_system'))
        
        if len(staff_ids) > 1:
            print(f"  WARNING: This chord spans multiple staves: {staff_ids}")
        if len(system_ids) > 1:
            print(f"  WARNING: This chord spans multiple systems: {system_ids}")
            
    # Also check the visualization code that draws connections
    print("\n=== CHECKING VISUALIZATION CODE ===")
    try:
        import inspect
        from encoding_visualization import visualize_score, visualize_overlay
        
        # Check if visualization functions exist
        if 'visualize_score' in locals():
            vis_code = inspect.getsource(visualize_score)
            if "chord" in vis_code.lower():
                print("visualize_score() contains chord-related code that may be drawing connections")
                
                # Look for specific patterns in the visualization code
                lines = vis_code.split('\n')
                for i, line in enumerate(lines):
                    if "chord" in line.lower() and "draw" in line.lower():
                        print(f"Line {i+1}: {line.strip()}")
        
        if 'visualize_overlay' in locals():
            vis_code = inspect.getsource(visualize_overlay)
            if "chord" in vis_code.lower():
                print("visualize_overlay() contains chord-related code that may be drawing connections")
                
                # Look for specific patterns in the visualization code
                lines = vis_code.split('\n')
                for i, line in enumerate(lines):
                    if "chord" in line.lower() and "draw" in line.lower():
                        print(f"Line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Could not inspect visualization code: {e}")
    
    return chord_groups


def debug_leftover_chord_attributes(processor):
    """
    Debug function to find notes that have leftover chord attributes
    even when they shouldn't be part of any chord.
    
    Args:
        processor: The OMRProcessor instance with processed notes
    """
    print("\n=== DEBUGGING LEFTOVER CHORD ATTRIBUTES ===")
    
    # Track notes with chord attributes
    notes_with_chord_attr = []
    
    # Check all notes for chord attributes
    for i, note in enumerate(processor.notes):
        has_is_chord_member = hasattr(note, 'is_chord_member') and note.is_chord_member
        has_chord = hasattr(note, 'chord') and note.chord
        
        if has_is_chord_member or has_chord:
            staff_id = processor._get_staff_id(note) if hasattr(processor, '_get_staff_id') else "unknown"
            system_id = note.staff_system.id if hasattr(note, 'staff_system') else "unknown"
            
            # Get pitch info
            pitch = f"{note.step}{note.octave}" if hasattr(note, 'step') and hasattr(note, 'octave') else "unknown"
            
            # Add to list
            notes_with_chord_attr.append({
                'index': i,
                'pitch': pitch,
                'staff': staff_id,
                'system': system_id,
                'x': note.x,
                'y': note.y,
                'is_chord_member': has_is_chord_member,
                'has_chord': has_chord,
                'chord_length': len(note.chord) if has_chord else 0
            })
    
    # Report findings
    if notes_with_chord_attr:
        print(f"Found {len(notes_with_chord_attr)} notes with leftover chord attributes:")
        for info in notes_with_chord_attr:
            print(f"  Note {info['index']}: pitch={info['pitch']}, x={info['x']:.1f}, y={info['y']:.1f}")
            print(f"    staff={info['staff']}, system={info['system']}")
            print(f"    is_chord_member={info['is_chord_member']}, chord_length={info['chord_length']}")
            
            # If it has a chord, print details of the chord members
            if info['has_chord']:
                note = processor.notes[info['index']]
                print(f"    Chord members:")
                for j, chord_note in enumerate(note.chord):
                    chord_staff = processor._get_staff_id(chord_note) if hasattr(processor, '_get_staff_id') else "unknown"
                    chord_pitch = f"{chord_note.step}{chord_note.octave}" if hasattr(chord_note, 'step') and hasattr(chord_note, 'octave') else "unknown"
                    print(f"      Member {j}: pitch={chord_pitch}, staff={chord_staff}, x={chord_note.x:.1f}, y={chord_note.y:.1f}")
    else:
        print("No notes with leftover chord attributes found.")
        
    return notes_with_chord_attr