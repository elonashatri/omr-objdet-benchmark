import os
import uuid
import json
import base64
import subprocess
import base64
import traceback
import yaml
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = '/tmp/omr_uploads'
RESULTS_FOLDER = '/tmp/omr_results'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create directories with proper permissions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Available models with their paths
AVAILABLE_MODELS = {
    # 'train7': '/homes/es314/omr-objdet-benchmark/runs/detect/train7/weights/best.pt',
    'train8': '/homes/es314/omr-objdet-benchmark/runs/detect/train8/weights/best.pt',
    'train11': '/homes/es314/omr-objdet-benchmark/runs/detect/train11/weights/best.pt',
    # 'train13': '/homes/es314/omr-objdet-benchmark/runs/detect/train13/weights/best.pt',
    'staffline_extreme': '/homes/es314/omr-objdet-benchmark/runs/staffline_extreme/weights/best.pt'
}

# Paths to scripts and configurations
PIPELINE_SCRIPT = "/homes/es314/omr-objdet-benchmark/scripts/encoding/complete_pipeline.py"
CLASS_MAPPING_FILE = "/homes/es314/omr-objdet-benchmark/data/class_mapping.json"



def find_pitched_json(base_name):
    """
    Searches for the '_pitched.json' file in multiple possible locations
    Returns the full file path if found, otherwise returns None.
    
    Args:
        base_name (str): Base name of the processed file (UUID + original filename)
        
    Returns:
        str or None: Path to the pitched JSON file if found, None otherwise
    """
    # List of possible locations for pitched JSON files
    possible_locations = [
        # Standard location in pitched_data subdirectory
        os.path.join(RESULTS_FOLDER, base_name, "pitched_data"),
        # Alternative location directly in pitched subdirectory
        os.path.join(RESULTS_FOLDER, base_name, "pitched"),
        # Fallback to main directory
        os.path.join(RESULTS_FOLDER, base_name)
    ]
    
    print(f"🔍 Looking for pitched JSON for base_name: {base_name}")
    
    # Try each possible location
    for location in possible_locations:
        print(f"  Checking directory: {location}")
        
        if not os.path.exists(location):
            print(f"  ⚠️ Directory does not exist: {location}")
            continue
            
        # Find all JSON files in this directory
        json_files = [f for f in os.listdir(location) if f.endswith(".json")]
        
        if not json_files:
            print(f"  ⚠️ No JSON files found in: {location}")
            continue
            
        # First try to find files with _pitched.json pattern
        pitched_files = [f for f in json_files if f.endswith("_pitched.json")]
        
        if pitched_files:
            # Sort by last modified time to get the latest
            pitched_files.sort(key=lambda f: os.path.getmtime(os.path.join(location, f)), reverse=True)
            selected_file = os.path.join(location, pitched_files[0])
            print(f"  ✅ Found pitched JSON file: {selected_file}")
            return selected_file
        
        # If no specific pitched files, try any JSON as a fallback
        print(f"  ⚠️ No *_pitched.json files, trying any JSON as fallback")
        json_files.sort(key=lambda f: os.path.getmtime(os.path.join(location, f)), reverse=True)
        selected_file = os.path.join(location, json_files[0])
        print(f"  ✅ Using fallback JSON file: {selected_file}")
        return selected_file
    
    # If we got here, no JSON files were found
    print(f"❌ No JSON files found in any expected locations for {base_name}")
    return None


@app.route('/available_models', methods=['GET'])
def get_available_models():
    """
    Endpoint to retrieve list of available models
    """
    models = [
        # 'train7',
        'train8', 
        'train11', 
        # 'train13', 
        'staffline_extreme'
    ]
    return jsonify(models)


@app.route('/model_info/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """
    Endpoint to retrieve information about a specific model from its args.yaml file
    """
    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": f"Model {model_name} not found"}), 404
    
    model_path = AVAILABLE_MODELS[model_name]
    model_dir = os.path.dirname(model_path)
    yaml_path = os.path.join(model_dir, "args.yaml")
    
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                model_info = yaml.safe_load(f)
            return jsonify(model_info)
        else:
            # If args.yaml doesn't exist, try other common names
            alt_paths = [
                os.path.join(model_dir, "opt.yaml"),
                os.path.join(model_dir, "hyp.yaml"),
                os.path.join(model_dir, "config.yaml")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        model_info = yaml.safe_load(f)
                    return jsonify(model_info)
                    
            # If no YAML file is found, return basic info
            return jsonify({
                "name": model_name,
                "path": model_path,
                "note": "No detailed configuration found"
            })
    except Exception as e:
        print(f"Error loading model info for {model_name}: {str(e)}")
        return jsonify({
            "error": f"Error loading model info: {str(e)}",
            "name": model_name,
            "path": model_path
        }), 500


@app.route('/run_omr_pipeline', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    if 'model' not in request.form:
        return jsonify({"error": "No model selected"}), 400
    
    file = request.files['file']
    selected_model = request.form['model']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if selected_model not in AVAILABLE_MODELS:
        return jsonify({
            "error": "Invalid model selected", 
            "available_models": list(AVAILABLE_MODELS.keys())
        }), 400
    
    # Verify that class mapping file exists
    if not os.path.exists(CLASS_MAPPING_FILE):
        return jsonify({"error": f"Class mapping file not found: {CLASS_MAPPING_FILE}"}), 500
    
    # Generate unique ID and filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # Save the uploaded file
        file.save(filepath)
        
        # Check if the file was saved correctly
        if not os.path.exists(filepath):
            return jsonify({"error": "Failed to save uploaded file"}), 500
            
        # Create a clean output directory structure
        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(RESULTS_FOLDER, base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the path for the selected model
        model_path = AVAILABLE_MODELS[selected_model]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 500
            
        # Check if pipeline script exists
        if not os.path.exists(PIPELINE_SCRIPT):
            return jsonify({"error": f"Pipeline script not found: {PIPELINE_SCRIPT}"}), 500
        
        # Build command with properly formatted arguments
        command = [
            '/homes/es314/.conda/envs/omr_benchmark/bin/python',  # Use the full path to Python
            PIPELINE_SCRIPT, 
            '--image', filepath,
            '--model', model_path,
            '--class_mapping', CLASS_MAPPING_FILE,
            '--output_dir', output_dir,
            # '--debug'  # Always run in debug mode for better error reporting
        ]
        
        # Add optional parameters if provided
        if 'beam_all' in request.form and request.form['beam_all'] == 'true':
            command.append('--beam-all')
            
        if 'time_signature' in request.form:
            command.extend(['--time-signature', request.form['time_signature']])
        
        # Print command for debugging
        print(f"Running command: {' '.join(command)}")
        
        # Run the pipeline
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check for errors
        if result.returncode != 0:
            print(f"Pipeline error: {result.stderr}")
            return jsonify({
                "error": "Pipeline execution failed",
                "stderr": result.stderr,
                "command": " ".join(command)
            }), 500
        
        # Process and return results
        processed_results = process_pipeline_results(output_dir, filepath)
        processed_results['model'] = selected_model
        
        return jsonify(processed_results)
    
    except Exception as e:
        print(f"Exception: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")
            
            
def process_pipeline_results(results_dir, original_image_path):
    """
    Consistent process for finding and returning visualizations.
    Ensures predictable behavior by using explicit paths and priorities.
    
    Args:
        results_dir (str): Directory containing pipeline results.
        original_image_path (str): Path to the original image.
    
    Returns:
        dict: Processed results with base64 encoded images and detection data.
    """
    print(f"Processing results from: {results_dir}")
    
    # Get basename for constructing paths
    basename = os.path.basename(results_dir)
    
    # Define EXACT paths for each visualization with no ambiguity
    exact_paths = {
        "original": original_image_path,
        # Standard linked visualization paths
        "linked": os.path.join(results_dir, f"{basename}_detections_linked_visualization.png"),
        "linked_alt": os.path.join(results_dir, "linked", f"{basename}_detections_linked_visualization.png"),
        # New path pattern for merged linked visualization
        "linked_merged": os.path.join(results_dir, "linked_data", f"{basename}_merged_linked_visualization.png"),
        
        # Standard pitched visualization paths
        "pitched": os.path.join(results_dir, f"{basename}_pitched_visualization.png"),
        "pitched_alt": os.path.join(results_dir, "visualizations", f"{basename}_pitched_visualization.png"),
        
        # Pitch data paths
        "pitch_data": os.path.join(results_dir, "pitched", f"{basename}_pitched.json"),
        "pitch_data_alt": os.path.join(results_dir, "pitched_data", f"{basename}_pitched.json"),
    }
    
    # Check which files actually exist
    file_exists = {key: os.path.exists(path) for key, path in exact_paths.items()}
    
    # Print which files were found
    print("File availability:")
    for key, exists in file_exists.items():
        print(f"- {key}: {'FOUND' if exists else 'NOT FOUND'} at {exact_paths[key]}")
    
    # Determine which paths to use with explicit fallback strategy
    linked_path = None
    if file_exists["linked_merged"]:
        linked_path = exact_paths["linked_merged"]
    elif file_exists["linked"]:
        linked_path = exact_paths["linked"]
    elif file_exists["linked_alt"]:
        linked_path = exact_paths["linked_alt"]
    
    pitched_path = exact_paths["pitched"] if file_exists["pitched"] else (
                  exact_paths["pitched_alt"] if file_exists["pitched_alt"] else None)
    
    pitch_data_path = None
    if file_exists["pitch_data"]:
        pitch_data_path = exact_paths["pitch_data"]
    elif file_exists["pitch_data_alt"]:
        pitch_data_path = exact_paths["pitch_data_alt"]
    
    # Final selection report
    print("Selected files:")
    print(f"- Original: {exact_paths['original'] if file_exists['original'] else 'NOT FOUND'}")
    print(f"- Linked: {linked_path if linked_path else 'NOT FOUND'}")
    print(f"- Pitched: {pitched_path if pitched_path else 'NOT FOUND'}")
    print(f"- Pitch data: {pitch_data_path if pitch_data_path else 'NOT FOUND'}")
    
    # Function to safely encode images
    def encode_image(path, name):
        if not path or not os.path.exists(path):
            print(f"{name} not available at path: {path}")
            return None
        
        try:
            with open(path, 'rb') as f:
                data = f.read()
                if not data:
                    print(f"{name} file is empty")
                    return None
                
                encoded = base64.b64encode(data).decode('utf-8')
                print(f"Successfully encoded {name}: {len(data)} bytes → {len(encoded)} chars")
                return encoded
        except Exception as e:
            print(f"Error encoding {name}: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    # Encode images
    original_base64 = encode_image(exact_paths["original"], "Original image")
    linked_base64 = encode_image(linked_path, "Linked visualization") 
    pitched_base64 = encode_image(pitched_path, "Pitched visualization")
    
    # Load pitch data
    pitch_data = {}
    if pitch_data_path and os.path.exists(pitch_data_path):
        try:
            with open(pitch_data_path, 'r') as f:
                pitch_data = json.load(f)
            print(f"Successfully loaded pitch data from {pitch_data_path}")
        except Exception as e:
            print(f"Error loading pitch data: {e}")
    
    # Create response with consistent naming
    response = {
        "original_image": f"data:image/png;base64,{original_base64}" if original_base64 else None,
        "linked_image": f"data:image/png;base64,{linked_base64}" if linked_base64 else None,
        "pitched_image": f"data:image/png;base64,{pitched_base64}" if pitched_base64 else None,
        "pitch_data": pitch_data,
        # Always include base_name for client reference
        "base_name": basename,
        # Add metadata to help identify what's in the response
        "metadata": {
            "has_original": original_base64 is not None,
            "has_linked": linked_base64 is not None,
            "has_pitched": pitched_base64 is not None,
            "has_pitch_data": bool(pitch_data),
            "linked_path_used": linked_path if linked_path else "none",
            "pitched_path_used": pitched_path if pitched_path else "none"
        }
    }
    
    # Add processed_image for backward compatibility, ALWAYS use linked visualization when available
    if linked_base64:
        response["processed_image"] = f"data:image/png;base64,{linked_base64}"
    elif pitched_base64:  # Fall back to pitched only if linked is not available
        response["processed_image"] = f"data:image/png;base64,{pitched_base64}"
    
    # Print what's in the final response
    print("Response contains:")
    print(f"- original_image: {'YES' if response['original_image'] else 'NO'}")
    print(f"- linked_image: {'YES' if response['linked_image'] else 'NO'}")
    print(f"- pitched_image: {'YES' if response['pitched_image'] else 'NO'}")
    print(f"- processed_image: {'YES' if 'processed_image' in response else 'NO'}")
    print(f"- base_name: {basename}")
    
    return response

            
            # workin function
# def process_pipeline_results(results_dir, original_image_path):
#     """
#     Consistent process for finding and returning visualizations.
#     Ensures predictable behavior by using explicit paths and priorities.
    
#     Args:
#         results_dir (str): Directory containing pipeline results.
#         original_image_path (str): Path to the original image.
    
#     Returns:
#         dict: Processed results with base64 encoded images and detection data.
#     """
#     print(f"Processing results from: {results_dir}")
    
#     # Get basename for constructing paths
#     basename = os.path.basename(results_dir)
    
#     # Define EXACT paths for each visualization with no ambiguity
#     exact_paths = {
#         "original": original_image_path,
#         "linked": os.path.join(results_dir, f"{basename}_detections_linked_visualization.png"),
#         "linked_alt": os.path.join(results_dir, "linked", f"{basename}_detections_linked_visualization.png"),
#         "pitched": os.path.join(results_dir, f"{basename}_pitched_visualization.png"),
#         "pitched_alt": os.path.join(results_dir, "visualizations", f"{basename}_pitched_visualization.png"),
#         "pitch_data": os.path.join(results_dir, "pitched", f"{basename}_pitched.json"),
#     }
    
#     # Check which files actually exist
#     file_exists = {key: os.path.exists(path) for key, path in exact_paths.items()}
    
#     # Print which files were found
#     print("File availability:")
#     for key, exists in file_exists.items():
#         print(f"- {key}: {'FOUND' if exists else 'NOT FOUND'} at {exact_paths[key]}")
    
#     # Determine which paths to use with explicit fallback strategy
#     linked_path = exact_paths["linked"] if file_exists["linked"] else (
#                  exact_paths["linked_alt"] if file_exists["linked_alt"] else None)
    
#     pitched_path = exact_paths["pitched"] if file_exists["pitched"] else (
#                   exact_paths["pitched_alt"] if file_exists["pitched_alt"] else None)
    
#     pitch_data_path = exact_paths["pitch_data"] if file_exists["pitch_data"] else None
    
#     # Final selection report
#     print("Selected files:")
#     print(f"- Original: {exact_paths['original'] if file_exists['original'] else 'NOT FOUND'}")
#     print(f"- Linked: {linked_path if linked_path else 'NOT FOUND'}")
#     print(f"- Pitched: {pitched_path if pitched_path else 'NOT FOUND'}")
#     print(f"- Pitch data: {pitch_data_path if pitch_data_path else 'NOT FOUND'}")
    
#     # Function to safely encode images
#     def encode_image(path, name):
#         if not path or not os.path.exists(path):
#             print(f"{name} not available at path: {path}")
#             return None
        
#         try:
#             with open(path, 'rb') as f:
#                 data = f.read()
#                 if not data:
#                     print(f"{name} file is empty")
#                     return None
                
#                 encoded = base64.b64encode(data).decode('utf-8')
#                 print(f"Successfully encoded {name}: {len(data)} bytes → {len(encoded)} chars")
#                 return encoded
#         except Exception as e:
#             print(f"Error encoding {name}: {e}")
#             import traceback
#             print(traceback.format_exc())
#             return None
    
#     # Encode images
#     original_base64 = encode_image(exact_paths["original"], "Original image")
#     linked_base64 = encode_image(linked_path, "Linked visualization") 
#     pitched_base64 = encode_image(pitched_path, "Pitched visualization")
    
#     # Load pitch data
#     pitch_data = {}
#     if pitch_data_path and os.path.exists(pitch_data_path):
#         try:
#             with open(pitch_data_path, 'r') as f:
#                 pitch_data = json.load(f)
#             print(f"Successfully loaded pitch data")
#         except Exception as e:
#             print(f"Error loading pitch data: {e}")
    
#     # Create response with consistent naming
#     response = {
#         "original_image": f"data:image/png;base64,{original_base64}" if original_base64 else None,
#         "linked_image": f"data:image/png;base64,{linked_base64}" if linked_base64 else None,
#         "pitched_image": f"data:image/png;base64,{pitched_base64}" if pitched_base64 else None,
#         "pitch_data": pitch_data,
#         # Add the base_name explicitly to the response so client can use it for conversion
#         "base_name": basename,
#         # Add metadata to help identify what's in the response
#         "metadata": {
#             "has_original": original_base64 is not None,
#             "has_linked": linked_base64 is not None,
#             "has_pitched": pitched_base64 is not None,
#             "has_pitch_data": bool(pitch_data),
#             "linked_path_used": linked_path if linked_path else "none",
#             "pitched_path_used": pitched_path if pitched_path else "none",
#             "results_dir": results_dir  # Include the results directory for debugging
#         }
#     }
    
#     # Add processed_image for backward compatibility, ALWAYS use linked visualization when available
#     if linked_base64:
#         response["processed_image"] = f"data:image/png;base64,{linked_base64}"
#     elif pitched_base64:  # Fall back to pitched only if linked is not available
#         response["processed_image"] = f"data:image/png;base64,{pitched_base64}"
    
#     # Print what's in the final response
#     print("Response contains:")
#     print(f"- original_image: {'YES' if response['original_image'] else 'NO'}")
#     print(f"- linked_image: {'YES' if response['linked_image'] else 'NO'}")
#     print(f"- pitched_image: {'YES' if response['pitched_image'] else 'NO'}")
#     print(f"- processed_image: {'YES' if 'processed_image' in response else 'NO'}")
#     print(f"- base_name: {basename}")
    
#     return response
# def process_pipeline_results(results_dir, original_image_path):
#     """
#     Consistent process for finding and returning visualizations.
#     Ensures predictable behavior by using explicit paths and priorities.
    
#     Args:
#         results_dir (str): Directory containing pipeline results.
#         original_image_path (str): Path to the original image.
    
#     Returns:
#         dict: Processed results with base64 encoded images and detection data.
#     """
#     print(f"Processing results from: {results_dir}")
    
#     # Get basename for constructing paths
#     basename = os.path.basename(results_dir)
    
#     # Define EXACT paths for each visualization with no ambiguity
#     exact_paths = {
#         "original": original_image_path,
#         "linked": os.path.join(results_dir, f"{basename}_detections_linked_visualization.png"),
#         "linked_alt": os.path.join(results_dir, "linked", f"{basename}_detections_linked_visualization.png"),
#         "pitched": os.path.join(results_dir, f"{basename}_pitched_visualization.png"),
#         "pitched_alt": os.path.join(results_dir, "visualizations", f"{basename}_pitched_visualization.png"),
#         "pitch_data": os.path.join(results_dir, "pitched", f"{basename}_pitched.json"),
#     }
    
#     # Check which files actually exist
#     file_exists = {key: os.path.exists(path) for key, path in exact_paths.items()}
    
#     # Print which files were found
#     print("File availability:")
#     for key, exists in file_exists.items():
#         print(f"- {key}: {'FOUND' if exists else 'NOT FOUND'} at {exact_paths[key]}")
    
#     # Determine which paths to use with explicit fallback strategy
#     linked_path = exact_paths["linked"] if file_exists["linked"] else (
#                  exact_paths["linked_alt"] if file_exists["linked_alt"] else None)
    
#     pitched_path = exact_paths["pitched"] if file_exists["pitched"] else (
#                   exact_paths["pitched_alt"] if file_exists["pitched_alt"] else None)
    
#     pitch_data_path = exact_paths["pitch_data"] if file_exists["pitch_data"] else None
    
#     # Final selection report
#     print("Selected files:")
#     print(f"- Original: {exact_paths['original'] if file_exists['original'] else 'NOT FOUND'}")
#     print(f"- Linked: {linked_path if linked_path else 'NOT FOUND'}")
#     print(f"- Pitched: {pitched_path if pitched_path else 'NOT FOUND'}")
#     print(f"- Pitch data: {pitch_data_path if pitch_data_path else 'NOT FOUND'}")
    
#     # Function to safely encode images
#     def encode_image(path, name):
#         if not path or not os.path.exists(path):
#             print(f"{name} not available at path: {path}")
#             return None
        
#         try:
#             with open(path, 'rb') as f:
#                 data = f.read()
#                 if not data:
#                     print(f"{name} file is empty")
#                     return None
                
#                 encoded = base64.b64encode(data).decode('utf-8')
#                 print(f"Successfully encoded {name}: {len(data)} bytes → {len(encoded)} chars")
#                 return encoded
#         except Exception as e:
#             print(f"Error encoding {name}: {e}")
#             import traceback
#             print(traceback.format_exc())
#             return None
    
#     # Encode images
#     original_base64 = encode_image(exact_paths["original"], "Original image")
#     linked_base64 = encode_image(linked_path, "Linked visualization") 
#     pitched_base64 = encode_image(pitched_path, "Pitched visualization")
    
#     # Load pitch data
#     pitch_data = {}
#     if pitch_data_path and os.path.exists(pitch_data_path):
#         try:
#             with open(pitch_data_path, 'r') as f:
#                 pitch_data = json.load(f)
#             print(f"Successfully loaded pitch data")
#         except Exception as e:
#             print(f"Error loading pitch data: {e}")
    
#     # Create response with consistent naming
#     response = {
#         "original_image": f"data:image/png;base64,{original_base64}" if original_base64 else None,
#         "linked_image": f"data:image/png;base64,{linked_base64}" if linked_base64 else None,
#         "pitched_image": f"data:image/png;base64,{pitched_base64}" if pitched_base64 else None,
#         "pitch_data": pitch_data,
#         # Add metadata to help identify what's in the response
#         "metadata": {
#             "has_original": original_base64 is not None,
#             "has_linked": linked_base64 is not None,
#             "has_pitched": pitched_base64 is not None,
#             "has_pitch_data": bool(pitch_data),
#             "linked_path_used": linked_path if linked_path else "none",
#             "pitched_path_used": pitched_path if pitched_path else "none"
#         }
#     }
    
#     # Add processed_image for backward compatibility, ALWAYS use linked visualization when available
#     if linked_base64:
#         response["processed_image"] = f"data:image/png;base64,{linked_base64}"
#     elif pitched_base64:  # Fall back to pitched only if linked is not available
#         response["processed_image"] = f"data:image/png;base64,{pitched_base64}"
    
#     # Print what's in the final response
#     print("Response contains:")
#     print(f"- original_image: {'YES' if response['original_image'] else 'NO'}")
#     print(f"- linked_image: {'YES' if response['linked_image'] else 'NO'}")
#     print(f"- pitched_image: {'YES' if response['pitched_image'] else 'NO'}")
#     print(f"- processed_image: {'YES' if 'processed_image' in response else 'NO'}")
    
#     return response


# @app.route('/convert_to_musicxml', methods=['POST'])
# def convert_to_musicxml():
#     try:
#         # Get JSON data from request
#         if 'json_data' not in request.json:
#             return jsonify({"error": "Missing JSON data"}), 400
            
#         json_data = request.json['json_data']
        
#         # Create a temporary file for the JSON input
#         json_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_input.json")
#         musicxml_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_output.musicxml")
        
#         # Write the JSON to a file
#         with open(json_path, 'w') as f:
#             json.dump(json_data, f)
        
#         # Get time signature if provided
#         time_signature = request.json.get('time_signature', '4/4')
        
#         # Run the JSON to MusicXML converter
#         converter_script = "/homes/es314/omr-objdet-benchmark/scripts/encoding/to_midi/json_to_musicxml.py"
#         command = [
#             '/homes/es314/.conda/envs/omr_benchmark/bin/python',
#             converter_script,
#             '-i', json_path,  # Changed back to -i
#             '-o', musicxml_path,  # Changed back to -o
#             '-t', time_signature  # Changed back to -t
#         ]
        
#         # Add --beamall flag if requested (keep this parameter as is)
#         if request.json.get('beam_all', False):
#             command.append('--beamall')
            
#         # Add -v flag for debugging (change back to single hyphen)
#         if request.json.get('verbose', False):
#             command.append('-v')
        
#         print(f"Running MusicXML conversion: {' '.join(command)}")
        
#         # Run the converter
#         result = subprocess.run(command, capture_output=True, text=True)
        
#         # Check for errors
#         if result.returncode != 0:
#             print(f"MusicXML conversion error: {result.stderr}")
#             return jsonify({
#                 "error": "MusicXML conversion failed",
#                 "stderr": result.stderr,
#                 "command": " ".join(command)
#             }), 500
        
#         # Read the generated MusicXML file
#         with open(musicxml_path, 'r') as f:
#             musicxml_content = f.read()
        
#         # Clean up temporary files
#         os.remove(json_path)
#         os.remove(musicxml_path)
        
#         # Return the MusicXML as a response
#         return jsonify({
#             "musicxml": musicxml_content,
#             "message": "Conversion successful"
#         })
    
#     except Exception as e:
#         print(f"Exception in MusicXML conversion: {str(e)}")
#         print(traceback.format_exc())
#         return jsonify({
#             "error": "An unexpected error occurred",
#             "details": str(e),
#             "traceback": traceback.format_exc()
#         }), 500
       
@app.route('/api/list_musicxml_files', methods=['GET'])
def list_musicxml_files():
    files_dir = "/homes/es314/musicxml_files"
    files = []
    if os.path.exists(files_dir):
        for file in os.listdir(files_dir):
            if file.endswith(".musicxml"):
                file_path = os.path.join(files_dir, file)
                files.append({
                    "name": file,
                    "path": file_path,
                    "date": os.path.getmtime(file_path),
                    "size": os.path.getsize(file_path)
                })
    
    # Sort by date, newest first
    files.sort(key=lambda x: x["date"], reverse=True)
    return jsonify({"files": files})

@app.route('/api/download_musicxml/<base_name>', methods=['GET'])
def download_musicxml(base_name):
    file_path = f"/homes/es314/musicxml_files/{base_name}_output.musicxml"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=f"{base_name}.musicxml")
    else:
        return jsonify({"error": "File not found"}), 404
 
@app.route('/convert_to_musicxml', methods=['POST'])
def convert_to_musicxml():
    try:
        print(f"📥 Received MusicXML conversion request: {request.json}")

        # Validate JSON structure
        if not request.is_json:
            return jsonify({"error": "Invalid request format, expected JSON"}), 400

        # Support both old and new API formats
        if 'json_data' in request.json and request.json.get('json_data'):
            # Old API: Direct JSON data provided
            print("⚠️ Using legacy API format with direct json_data")
            
            # Create a temporary file for the JSON input
            json_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_input.json")
            
            # Write the JSON to a file
            with open(json_path, 'w') as f:
                json.dump(request.json['json_data'], f)
                
        elif 'base_name' in request.json:
            # New API: Using base_name to locate the JSON file
            base_name = request.json['base_name']
            print(f"🔍 Using base_name to locate pitched JSON: {base_name}")
            
            # Find the correct _pitched.json file
            json_path = find_pitched_json(base_name)
            
            if not json_path:
                error_msg = f"No valid JSON file found for base_name: {base_name}"
                print(f"❌ {error_msg}")
                return jsonify({
                    "error": error_msg,
                    "help": "Make sure you're passing the correct base_name from the previous pipeline run"
                }), 404
        else:
            error_msg = "Request must include either 'json_data' or 'base_name'"
            print(f"❌ {error_msg}")
            return jsonify({
                "error": error_msg,
                "help": "Use the base_name returned from /run_omr_pipeline endpoint"
            }), 400

        print(f"📄 Using JSON file: {json_path}")

        # Generate a unique output MusicXML file path
        musicxml_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_output.musicxml")

        # Get time signature from request or default to '4/4'
        time_signature = request.json.get('time_signature', '4/4')

        # Command for MusicXML conversion
        converter_script = "/homes/es314/omr-objdet-benchmark/scripts/encoding/to_midi/json_to_musicxml.py"
        command = [
            '/homes/es314/.conda/envs/omr_benchmark/bin/python',
            converter_script,
            '-i', json_path,
            '-o', musicxml_path,
            '-t', time_signature
        ]

        # Add --beamall flag if requested
        if request.json.get('beam_all', False):
            command.append('--beamall')

        # Add -v flag for debugging
        if request.json.get('verbose', False):
            command.append('-v')

        print(f"🚀 Running MusicXML conversion: {' '.join(command)}")

        # Execute the conversion
        result = subprocess.run(command, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"❌ MusicXML conversion error: {result.stderr}")
            return jsonify({
                "error": "MusicXML conversion failed",
                "stderr": result.stderr,
                "command": " ".join(command)
            }), 500

        # Check if output file exists and is not empty
        if not os.path.exists(musicxml_path) or os.path.getsize(musicxml_path) == 0:
            print(f"❌ MusicXML file was not created or is empty: {musicxml_path}")
            return jsonify({
                "error": "MusicXML file was not created or is empty",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        # Read the generated MusicXML file
        with open(musicxml_path, 'r') as f:
            musicxml_content = f.read()

        # Cleanup temporary files
        try:
            # Only cleanup the temporary JSON if we created it ourselves
            if 'json_data' in request.json:
                os.remove(json_path)
            os.remove(musicxml_path)
        except Exception as cleanup_error:
            print(f"⚠️ Warning during cleanup: {cleanup_error}")

        return jsonify({
            "musicxml": musicxml_content,
            "message": "Conversion successful"
        })

    except Exception as e:
        print(f"❌ Exception in MusicXML conversion: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

def main():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()       
        