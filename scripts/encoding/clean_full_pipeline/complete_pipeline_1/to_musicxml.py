#/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/to_musicxml.py
"""
Main entry point for OMR (Optical Music Recognition) processing.
"""
import os
import argparse
from scripts.encoding.new_rules_encoding.newer.processor import OMRProcessor
from scripts.encoding.clean_full_pipeline.complete_pipeline_1.link_visualization import visualize_score, visualize_overlay  # ✅ Import overlay visualizer

def process_music_score(detection_file, staff_lines_file, output_xml=None, output_image=None,
                        staff_mode="piano", original_image=None):
    """
    Process a music score and generate MusicXML and visualizations.
    
    Args:
        detection_file: Path to detection file (CSV or JSON)
        staff_lines_file: Path to staff lines file (JSON)
        output_xml: Path to save generated MusicXML (optional)
        output_image: Path to save visualization image (optional)
        staff_mode: Mode for staff system identification ('auto' or 'piano')
        original_image: Path to the original image for overlay visualization (optional)
        
    Returns:
        Generated MusicXML string
    """
    # Create processor
    processor = OMRProcessor(detection_file, staff_lines_file)
    
    # Process
    musicxml = processor.process()
    
    # Save MusicXML
    if musicxml and output_xml:
        with open(output_xml, 'w') as f:
            f.write(musicxml)
    
    # Visualization: symbolic layout
    if output_image:
        visualize_score(processor, output_image)
    
    # Visualization: overlay on original image
    if original_image:
        overlay_path = output_image.replace('.png', '_overlay.png') if output_image else 'overlay.png'
        visualize_overlay(processor, original_image, overlay_path)
    
    return musicxml

def main():
    """Command line interface for OMR processing."""
    parser = argparse.ArgumentParser(description='Process music score images to MusicXML.')
    parser.add_argument('--detection', '-d', required=True, 
                        help='Path to detection file (CSV or JSON)')
    parser.add_argument('--staff-lines', '-s', required=True,
                        help='Path to staff lines file (JSON)')
    parser.add_argument('--output-xml', '-o', 
                        help='Path to save output MusicXML file')
    parser.add_argument('--output-image', '-i',
                        help='Path to save symbolic layout visualization')
    parser.add_argument('--original-image', '-r',
                        help='Path to the original score image for overlay visualization')  # ✅ New arg
    parser.add_argument('--staff-mode', '-m', choices=['auto', 'piano'], default='piano',
                        help='Mode for staff system identification')
    
    args = parser.parse_args()
    
    # Determine output XML path if not specified
    if not args.output_xml:
        basename = os.path.splitext(os.path.basename(args.detection))[0]
        args.output_xml = f"{basename}.musicxml"
    
    # Determine output image path if not specified
    if not args.output_image:
        basename = os.path.splitext(os.path.basename(args.detection))[0]
        args.output_image = f"{basename}.png"
    
    # Process score
    musicxml = process_music_score(
        args.detection, 
        args.staff_lines, 
        args.output_xml, 
        args.output_image,
        args.staff_mode,
        args.original_image  # ✅ Pass new argument
    )
    
    print(f"MusicXML generated and saved to {args.output_xml}")
    print(f"Symbolic visualization saved to {args.output_image}")
    if args.original_image:
        overlay_path = args.output_image.replace('.png', '_overlay.png')
        print(f"Overlay visualization saved to {overlay_path}")

if __name__ == "__main__":
    main()
