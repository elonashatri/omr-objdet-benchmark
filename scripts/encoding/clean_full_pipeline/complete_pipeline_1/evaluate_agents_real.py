# evaluate_agents_real.py
import os
import glob
import json
import tempfile
import shutil
from complete_pipeline import detect_structure_elements, detect_music_symbols
from music_agents import MusicScoreAgent

def test_agent_with_real_data(test_images_dir, num_images=7):
    """Test agent with real detection data"""
    agent = MusicScoreAgent()
    
    test_images = glob.glob(f"{test_images_dir}/*.png")[:num_images]
    decisions = []
    
    print("Testing agent with real detection data...")
    
    # Model paths
    structure_model = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/may_2023_ex001.onnx"
    structure_mapping = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/mapping.txt"
    symbol_model = "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt"
    symbol_mapping = "/homes/es314/omr-objdet-benchmark/data/class_mapping.json"
    
    for image_path in test_images:
        print(f"\nProcessing {os.path.basename(image_path)}...")
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        struct_dir = os.path.join(temp_dir, "structure")
        symbol_dir = os.path.join(temp_dir, "symbol")
        os.makedirs(struct_dir, exist_ok=True)
        os.makedirs(symbol_dir, exist_ok=True)
        
        try:
            # Run actual detections
            print("  Running structure detection...")
            struct_path = detect_structure_elements(
                image_path, structure_model, structure_mapping, struct_dir
            )
            
            print("  Running symbol detection...")
            symbol_path = detect_music_symbols(
                image_path, symbol_model, symbol_mapping, symbol_dir
            )
            
            # Test agent with real data
            print("  Analyzing with agent...")
            recommendations = agent.analyze_and_recommend(struct_path, symbol_path, image_path)
            
            decision = {
                'image': os.path.basename(image_path),
                'density': recommendations['density'],
                'total_detections': recommendations['total_detections'],
                'structure_count': recommendations['structure_count'],
                'symbol_count': recommendations['symbol_count'],
                'rest_count': recommendations['rest_count'],
                'conf_threshold': recommendations['high_conf_threshold'],
                'overlap_threshold': recommendations['overlap_threshold'],
                'notes': recommendations['processing_notes']
            }
            
            decisions.append(decision)
            
            print(f"  Results: {decision['total_detections']} total detections, "
                  f"density={decision['density']:.2f}, conf={decision['conf_threshold']}")
            if decision['notes']:
                print(f"  Agent notes: {decision['notes']}")
                
        except Exception as e:
            print(f"  Error: {e}")
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return decisions

def analyze_agent_adaptivity(decisions):
    """Analyze if agent is making different decisions"""
    if len(decisions) < 2:
        print("Need at least 2 test images to analyze adaptivity")
        return
        
    print("\n=== Agent Adaptivity Analysis ===")
    
    # Check if parameters vary
    conf_thresholds = [d['conf_threshold'] for d in decisions]
    overlap_thresholds = [d['overlap_threshold'] for d in decisions]
    
    unique_conf = set(conf_thresholds)
    unique_overlap = set(overlap_thresholds)
    
    print(f"Confidence thresholds used: {unique_conf}")
    print(f"Overlap thresholds used: {unique_overlap}")
    
    if len(unique_conf) > 1 or len(unique_overlap) > 1:
        print("✓ Agent is adaptive - makes different decisions for different images")
    else:
        print("⚠ Agent not adaptive - same decisions for all images")
        
    # Show decision reasoning
    print("\nDecision reasoning:")
    for d in decisions:
        print(f"{d['image']}: {d['notes']}")

if __name__ == "__main__":
    test_dir = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/diverse-examples"
    
    print("Testing agent with real detection data...")
    decisions = test_agent_with_real_data(test_dir, 2)
    
    # Analyze results
    analyze_agent_adaptivity(decisions)
    
    # Save results
    with open('/tmp/real_agent_test_results.json', 'w') as f:
        json.dump(decisions, f, indent=2)
    
    print(f"\nDetailed results saved to /tmp/real_agent_test_results.json")