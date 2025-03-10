import os
import uuid
import json
import base64
import subprocess
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
    'train7': '/homes/es314/runs/detect/train7/weights/best.pt',
    'train8': '/homes/es314/runs/detect/train8/weights/best.pt',
    'train11': '/homes/es314/runs/detect/train11/weights/best.pt',
    'train13': '/homes/es314/runs/detect/train13/weights/best.pt',
    'staffline_extreme': '/homes/es314/runs/staffline_extreme/weights/best.pt'
}

# Paths to scripts and configurations
PIPELINE_SCRIPT = "/homes/es314/omr-objdet-benchmark/scripts/encoding/complete_pipeline.py"
CLASS_MAPPING_FILE = "/homes/es314/omr-objdet-benchmark/data/class_mapping.json"


@app.route('/available_models', methods=['GET'])
def get_available_models():
    """
    Endpoint to retrieve list of available models
    """
    models = [
        'train7',
        'train8', 
        'train11', 
        'train13', 
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
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # Save the uploaded file
        file.save(filepath)
        
        # Check if the file was saved correctly
        if not os.path.exists(filepath):
            return jsonify({"error": "Failed to save uploaded file"}), 500
            
        # Construct the command to run the pipeline script
        output_dir = os.path.join(RESULTS_FOLDER, filename)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the path for the selected model
        model_path = AVAILABLE_MODELS[selected_model]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model file not found: {model_path}"}), 500
            
        # Check if pipeline script exists
        if not os.path.exists(PIPELINE_SCRIPT):
            return jsonify({"error": f"Pipeline script not found: {PIPELINE_SCRIPT}"}), 500
        
        command = [
            '/homes/es314/.conda/envs/omr_benchmark/bin/python',  # Use the full path to Python
            PIPELINE_SCRIPT, 
            '--image', filepath,
            '--model', model_path,
            '--class_mapping', CLASS_MAPPING_FILE,
            '--output_dir', output_dir
        ]
        
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
    Process pipeline results to create a response suitable for frontend
    
    Args:
        results_dir (str): Directory containing pipeline results
        original_image_path (str): Path to original image
    
    Returns:
        dict: Processed results with base64 encoded images and detection data
    """
    # The complete_pipeline.py creates subdirectories
    visualizations_dir = os.path.join(results_dir, "visualizations")
    pitched_dir = os.path.join(results_dir, "pitched_data")
    
    # Print debug information
    print(f"Looking for results in: {results_dir}")
    print(f"Original image path: {original_image_path}")
    
    if os.path.exists(visualizations_dir):
        print(f"Visualization directory exists: {visualizations_dir}")
        viz_files = [f for f in os.listdir(visualizations_dir) if f.endswith('_pitched_visualization.png')]
        print(f"Found visualization files: {viz_files}")
    else:
        print(f"Visualization directory does not exist")
        viz_files = []
    
    if os.path.exists(pitched_dir):
        print(f"Pitched directory exists: {pitched_dir}")
        pitch_files = [f for f in os.listdir(pitched_dir) if f.endswith('_pitched.json')]
        print(f"Found pitch files: {pitch_files}")
    else:
        print(f"Pitched directory does not exist")
        pitch_files = []
    
    # Default to first file if multiple exist
    viz_path = os.path.join(visualizations_dir, viz_files[0]) if viz_files else original_image_path
    pitch_path = os.path.join(pitched_dir, pitch_files[0]) if pitch_files else None
    
    # Read and encode images
    print(f"Using visualization path: {viz_path}")
    try:
        with open(viz_path, 'rb') as img_file:
            viz_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading visualization image: {str(e)}")
        viz_base64 = ""
    
    try:
        with open(original_image_path, 'rb') as img_file:
            original_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading original image: {str(e)}")
        original_base64 = ""
    
    # Read pitch data
    detections = []
    if pitch_path and os.path.exists(pitch_path):
        print(f"Reading pitch data from: {pitch_path}")
        try:
            with open(pitch_path, 'r') as f:
                pitch_data = json.load(f)
                detections = pitch_data.get('detections', [])
                print(f"Found {len(detections)} detections")
        except Exception as e:
            print(f"Error reading pitch data: {str(e)}")
    else:
        print(f"No pitch data file found")
    
    # If no visualization was found, fallback to using a message
    has_visualization = viz_path != original_image_path and viz_base64
    
    return {
        "original_image": f"data:image/png;base64,{original_base64}" if original_base64 else None,
        "processed_image": f"data:image/png;base64,{viz_base64}" if viz_base64 else None,
        "detections": detections,
        "has_visualization": has_visualization,
        "message": "Pipeline completed but no visualization was generated" if not has_visualization else ""
    }

def main():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()