#!/usr/bin/env python3
"""
MIDI Generation Module for OMR Pipeline

This module converts processed musical elements into a MIDI file.
It works with the OMRProcessor class to translate detected notes, 
rests, and other musical elements into a playable MIDI output.

Usage:
    Typically called from the complete_pipeline.py as the final step
    after MusicXML generation.
"""

import os
import sys
from collections import defaultdict
import math
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

def generate_midi(processor, output_path):
    """
    Generate a MIDI file from the processed music elements.
    
    Args:
        processor: The OMRProcessor containing musical elements
        output_path: Path to save the MIDI file
        
    Returns:
        Path to the generated MIDI file
    """
    print(f"\n=== GENERATING MIDI OUTPUT ===")
    
    # Create a new MIDI file with ticks_per_beat=480 (standard value)
    mid = MidiFile(ticks_per_beat=480)
    
    # Process each staff system as a separate track
    for system_idx, system in enumerate(processor.staff_systems):
        print(f"Processing System {system_idx+1} for MIDI")
        
        # Get time signature if available
        time_sig = system.time_signature
        if not time_sig:
            time_sig_numerator = 4
            time_sig_denominator = 4
        else:
            time_sig_numerator = time_sig.beats
            time_sig_denominator = time_sig.beat_type
        
        # Get key signature (expressed as number of sharps/flats)
        key_sig = 0  # Default to C major / A minor (0 sharps/flats)
        if hasattr(system, 'staff_key_signatures') and system.staff_key_signatures:
            # Use first staff's key signature for simplicity
            first_key = next(iter(system.staff_key_signatures.values()))
            sharps = first_key.get('sharps', 0)
            flats = first_key.get('flats', 0)
            key_sig = sharps if sharps > 0 else -flats
        
        # Process each staff in the system as a separate MIDI track
        num_staves = len(system.staves) if hasattr(system, 'staves') else 1
        
        for staff_idx in range(num_staves):
            # Create a new track for this staff
            track = MidiTrack()
            mid.tracks.append(track)
            
            # Name the track based on system and staff
            track_name = f"System {system_idx+1}, Staff {staff_idx+1}"
            track.append(MetaMessage('track_name', name=track_name, time=0))
            
            # Set instrument based on clef type (approximation)
            instrument = 0  # Default: Acoustic Grand Piano
            # Get clef type for this staff
            clef_type = None
            clefs = [e for e in system.elements if hasattr(e, 'class_name') and 'clef' in e.class_name.lower()]
            if clefs:
                for clef in clefs:
                    if hasattr(processor, '_get_staff_id') and processor._get_staff_id(clef) == staff_idx:
                        clef_type = clef.type if hasattr(clef, 'type') else None
                        break
            
            # Choose instrument based on clef
            if clef_type:
                if 'G' in clef_type or 'treble' in clef_type.lower():
                    instrument = 0  # Acoustic Grand Piano (treble)
                elif 'F' in clef_type or 'bass' in clef_type.lower():
                    instrument = 0  # Also Piano for bass (same instrument)
                elif 'C' in clef_type or 'alto' in clef_type.lower():
                    instrument = 41  # Viola
                else:
                    instrument = 0  # Default to piano
            
            # Add program change to set instrument
            track.append(Message('program_change', program=instrument, time=0))
            
            # Add time signature
            track.append(MetaMessage('time_signature', 
                                   numerator=time_sig_numerator,
                                   denominator=time_sig_denominator,
                                   clocks_per_click=24,
                                   notated_32nd_notes_per_beat=8,
                                   time=0))
            
            # Add key signature
            # MIDI format: positive = sharps, negative = flats, 0 = C major / A minor
            track.append(MetaMessage('key_signature', key=key_sig, time=0))
            
            # Set tempo (default to 120 BPM)
            tempo = bpm2tempo(120)
            track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
            
            # Get all notes that belong to this staff
            staff_elements = []
            for measure in system.measures:
                for elem in measure.elements:
                    # Check if element belongs to this staff
                    if hasattr(processor, '_get_staff_id') and hasattr(elem, 'x'):
                        elem_staff_id = processor._get_staff_id(elem)
                        if elem_staff_id == staff_idx:
                            staff_elements.append(elem)
            
            # Only include notes (skip other elements like barlines)
            notes = [e for e in staff_elements if hasattr(e, 'class_name') and 'notehead' in e.class_name.lower()]
            
            # Sort notes by onset time (if available) or x-position
            if all(hasattr(n, 'onset') for n in notes):
                notes.sort(key=lambda n: n.onset)
            else:
                notes.sort(key=lambda n: n.x)
            
            # Process each note
            note_on_events = []
            for note in notes:
                # Skip notes that are chord members but not the first note in the chord
                if (hasattr(note, 'is_chord_member') and note.is_chord_member and 
                    hasattr(note, 'chord') and note.chord and note.chord[0] != note):
                    continue
                
                # Get MIDI note number
                midi_note = note_to_midi_number(note)
                if midi_note is None:
                    continue  # Skip notes that can't be converted
                
                # Calculate note duration in ticks
                duration_quarter_notes = note.duration if hasattr(note, 'duration') and note.duration else 1.0
                duration_ticks = int(duration_quarter_notes * mid.ticks_per_beat)
                
                # Calculate delta time (time since previous event)
                if hasattr(note, 'onset') and note.onset is not None:
                    onset = note.onset
                else:
                    # If no onset time, estimate from position in list
                    onset = notes.index(note) * 0.25  # Simple approximation
                
                # Determine delta time from previous event
                if note_on_events:
                    delta_onset = onset - note_on_events[-1][0]
                    delta_ticks = int(delta_onset * mid.ticks_per_beat)
                else:
                    delta_ticks = 0  # First event starts at time 0
                
                # Store onset and note data for this event
                note_on_events.append((onset, delta_ticks, midi_note, duration_ticks))
                
                # Handle chord notes
                if hasattr(note, 'is_chord_member') and note.is_chord_member and hasattr(note, 'chord'):
                    for chord_note in note.chord:
                        if chord_note == note:
                            continue  # Skip the main note (already processed)
                        
                        # Get MIDI note number for chord note
                        chord_midi_note = note_to_midi_number(chord_note)
                        if chord_midi_note is not None:
                            # Chord notes have same timing as main note
                            note_on_events.append((onset, 0, chord_midi_note, duration_ticks))
            
            # Sort by onset time and add to track
            note_on_events.sort(key=lambda x: x[0])
            
            # Process each note on/off event
            active_notes = {}  # Track active notes and when they should end
            current_time = 0
            
            # Organize notes by time
            time_map = defaultdict(list)
            for onset, delta_ticks, midi_note, duration in note_on_events:
                time_map[onset].append((midi_note, duration))
            
            # Sort time points
            time_points = sorted(time_map.keys())
            
            # Process each time point
            last_time = 0
            for t in time_points:
                # Calculate delta time since last event
                delta = t - last_time
                if delta > 0:
                    delta_ticks = int(delta * mid.ticks_per_beat)
                else:
                    delta_ticks = 0
                
                # Process note-off events that should occur before this time point
                notes_to_remove = []
                for note_num, end_time in active_notes.items():
                    if end_time <= t:
                        # Note should end before or at this time point
                        off_delta = end_time - last_time
                        if off_delta > 0:
                            off_delta_ticks = int(off_delta * mid.ticks_per_beat)
                            # Add note off event
                            track.append(Message('note_off', note=note_num, velocity=64, time=off_delta_ticks))
                            # Update last time
                            last_time = end_time
                            delta = t - last_time  # Recalculate delta
                        else:
                            # Note ends at same time as previous event
                            track.append(Message('note_off', note=note_num, velocity=64, time=0))
                        
                        notes_to_remove.append(note_num)
                
                # Remove processed notes
                for note_num in notes_to_remove:
                    del active_notes[note_num]
                
                # Calculate final delta for current time point
                if delta > 0:
                    delta_ticks = int(delta * mid.ticks_per_beat)
                else:
                    delta_ticks = 0
                
                # Add note-on events for this time point
                first_note = True
                for midi_note, duration in time_map[t]:
                    # First note gets the delta time, others get 0
                    if first_note:
                        track.append(Message('note_on', note=midi_note, velocity=64, time=delta_ticks))
                        first_note = False
                    else:
                        track.append(Message('note_on', note=midi_note, velocity=64, time=0))
                    
                    # Schedule note-off event
                    end_time = t + (duration / mid.ticks_per_beat)
                    active_notes[midi_note] = end_time
                
                # Update last time
                last_time = t
            
            # Process any remaining active notes
            for note_num, end_time in sorted(active_notes.items(), key=lambda x: x[1]):
                delta = end_time - last_time
                delta_ticks = int(delta * mid.ticks_per_beat) if delta > 0 else 0
                track.append(Message('note_off', note=note_num, velocity=64, time=delta_ticks))
                last_time = end_time
                
            # End of track
            track.append(MetaMessage('end_of_track', time=0))
        
    # Save the MIDI file
    mid.save(output_path)
    print(f"MIDI file saved to: {output_path}")
    return output_path

def note_to_midi_number(note):
    """
    Convert a note to a MIDI note number.
    
    Args:
        note: A Note object with step, octave, and alter properties
        
    Returns:
        MIDI note number (0-127) or None if conversion isn't possible
    """
    # Check if note has required attributes
    if not hasattr(note, 'step') or not hasattr(note, 'octave') or note.octave is None:
        return None
    
    # Base values for each note in an octave
    base_values = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    # Get base value for this note's step
    if note.step not in base_values:
        return None
    
    base = base_values[note.step]
    
    # Apply alteration (sharps/flats)
    if hasattr(note, 'alter'):
        base += note.alter
    
    # Compute MIDI note number (C4 = 60)
    midi_note = 12 * (note.octave + 1) + base
    
    # Ensure note is within MIDI range (0-127)
    if 0 <= midi_note <= 127:
        return midi_note
    
    return None

def add_to_pipeline(processor, output_dir, music_xml_path=None):
    """
    Add MIDI generation to the OMR pipeline.
    
    Args:
        processor: The OMRProcessor object with musical elements
        output_dir: Directory to save the MIDI file
        music_xml_path: Path to MusicXML file (for naming consistency)
        
    Returns:
        Path to the generated MIDI file or None if generation fails
    """
    try:
        # Create MIDI output path based on MusicXML path or default name
        if music_xml_path:
            midi_path = os.path.splitext(music_xml_path)[0] + '.mid'
        else:
            # Create a default path if no MusicXML path provided
            img_stem = ""
            if hasattr(processor, 'detection_path') and processor.detection_path:
                img_stem = os.path.splitext(os.path.basename(processor.detection_path))[0]
                img_stem = img_stem.replace('_detections', '').replace('_combined', '')
            else:
                img_stem = "output"
                
            midi_path = os.path.join(output_dir, f"{img_stem}.mid")
        
        # Generate MIDI file
        return generate_midi(processor, midi_path)
    
    except Exception as e:
        print(f"Error generating MIDI: {e}")
        import traceback
        traceback.print_exc()
        return None