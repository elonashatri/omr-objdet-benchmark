#!/usr/bin/env python3
"""
omr_json_to_musicxml.py - Convert OMR JSON output to MusicXML
                         with proper pitch detection and handling of accidentals

This script coordinates the entire conversion process:
1. Extract pitch information from the JSON
2. Analyze beam groups and note durations
3. Generate MusicXML output

Usage:
  python omr_json_to_musicxml.py --input input.json --output output.musicxml --time 4/4 --verbose

Arguments:
  --input, -i       Input JSON file from OMR system
  --output, -o      Output MusicXML file
  --time, -t        Time signature (default: 4/4)
  --verbose, -v     Show detailed processing information
  --beamall, -b     Treat all beamed notes as 32nd notes
"""

import argparse
import json
import os
import tempfile
from pitch_extractor import process_json_for_pitch
from beam_analyzer import identify_note_durations
from musicxml_generator import process_data_for_musicxml

def main():
    parser = argparse.ArgumentParser(description='Convert OMR JSON to MusicXML with accurate pitch detection')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output MusicXML file (default: input_file_base.musicxml)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose information')
    parser.add_argument('-t', '--time', default='4/4', help='Time signature (default: 4/4)')
    parser.add_argument('-b', '--beamall', action='store_true', help='Treat all notes in beam groups as 32nd notes')
    args = parser.parse_args()

    # Determine output filename if not specified
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}.musicxml"
    
    if args.verbose:
        print(f"Processing {args.input} -> {args.output}")
        print(f"Time signature: {args.time}")
        if args.beamall:
            print("All beamed notes will be treated as 32nd notes")
    
    # Load the JSON data
    with open(args.input, 'r') as file:
        data = json.load(file)
    
    # Step 1: Extract pitch information
    if args.verbose:
        print("Extracting pitch information...")
    pitch_info = process_json_for_pitch(data)
    
    # Step 2: Analyze beam groups and note durations
    if args.verbose:
        print("Analyzing beam groups and note durations...")
    default_beam_duration = '32nd' if args.beamall else '16th'
    rhythm_info = identify_note_durations(data, default_beam_duration)
    
    # Step 3: Generate MusicXML
    if args.verbose:
        print("Generating MusicXML...")
    musicxml = process_data_for_musicxml(data, pitch_info, rhythm_info, args.time)
    
    # Write the output
    # with open(args.output, 'w') as file:
    #     file.write(musicxml)
        
    try:
        # Existing code to process and write the file
        with open(args.output, 'w') as file:
            file.write(musicxml)
        print(f"Successfully wrote to {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    if args.verbose:
        print(f"Successfully created MusicXML file: {args.output}")
    else:
        print(f"Created: {args.output}")

if __name__ == "__main__":
    main()