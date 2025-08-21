# evaluate_agents.py
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from complete_pipeline import process_image
from music_agents import MusicScoreAgent

def count_detections(merged_path):
    """Count total detections in merged file"""
    try:
        with open(merged_path, 'r') as f:
            data = json.load(f)
        return len(data.get('detections', []))
    except:
        return 0

def measure_detection_quality(merged_detections_path):
    """Measure quality metrics from detection file"""
    try:
        with open(merged_detections_path, 'r') as f:
            data = json.load(f)
        
        detections = data.get('detections', [])
        
        if not detections:
            return {'total_detections': 0, 'avg_confidence': 0, 'low_confidence_count': 0}
        
        confidences = [d.get('confidence', 0) for d in detections]
        
        return {
            'total_detections': len(detections),
            'avg_confidence': np.mean(confidences),
            'low_confidence_count': len([c for c in confidences if c < 0.3]),
            'high_confidence_count': len([c for c in confidences if c > 0.7])
        }
    except Exception as e:
        print(f"Error measuring quality: {e}")
        return {'total_detections': 0, 'avg_confidence': 0, 'low_confidence_count': 0}

def evaluate_agent_impact(test_images_dir, num_images=3):
    """Compare baseline vs agent results"""
    results = []
    
    test_images = glob.glob(f"{test_images_dir}/*.png")[:num_images]
    print(f"Testing {len(test_images)} images...")
    
    for image_path in test_images:
        print(f"\nTesting {os.path.basename(image_path)}")
        
        try:
            # Run with agent (your current implementation)
            agent_result = process_image(
                image_path,
                "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/may_2023_ex001.onnx",
                "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex001/mapping.txt",
                "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt",
                "/homes/es314/omr-objdet-benchmark/data/class_mapping.json",
                "/tmp/agent_test"
            )
            
            agent_quality = measure_detection_quality(agent_result['merged'])
            
            comparison = {
                'image': os.path.basename(image_path),
                'agent_detections': agent_quality['total_detections'],
                'agent_avg_confidence': agent_quality['avg_confidence'],
                'agent_low_conf_count': agent_quality['low_confidence_count']
            }
            
            print(f"  Agent: {agent_quality['total_detections']} detections, "
                  f"avg conf: {agent_quality['avg_confidence']:.3f}")
            
            results.append(comparison)
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
    
    return results

def test_agent_decisions(test_images_dir, num_images=5):
    """Test if agent makes different decisions on different images"""
    agent = MusicScoreAgent()
    
    test_images = glob.glob(f"{test_images_dir}/*.png")[:num_images]
    decisions = []
    
    print("Testing agent decision making...")
    
    for image_path in test_images:
        # Create dummy detection files for testing
        dummy_structure = {"detections": [{"class_name": "barline", "confidence": 0.8}] * 10}
        dummy_symbol = {"detections": [{"class_name": "noteBlack", "confidence": 0.7}] * 20}
        
        # Save dummy files
        struct_path = f"/tmp/dummy_struct_{os.path.basename(image_path)}.json"
        symbol_path = f"/tmp/dummy_symbol_{os.path.basename(image_path)}.json"
        
        with open(struct_path, 'w') as f:
            json.dump(dummy_structure, f)
        with open(symbol_path, 'w') as f:
            json.dump(dummy_symbol, f)
        
        # Test agent
        recommendations = agent.analyze_and_recommend(struct_path, symbol_path, image_path)
        
        decision = {
            'image': os.path.basename(image_path),
            'density': recommendations['density'],
            'conf_threshold': recommendations['high_conf_threshold'],
            'overlap_threshold': recommendations['overlap_threshold'],
            'notes': recommendations['processing_notes']
        }
        
        decisions.append(decision)
        
        print(f"{os.path.basename(image_path)}: density={decision['density']:.2f}, "
              f"conf={decision['conf_threshold']}, overlap={decision['overlap_threshold']}")
        
        # Clean up
        os.remove(struct_path)
        os.remove(symbol_path)
    
    return decisions

def quick_test():
    """Quick test to see if agent is working"""
    print("=== Quick Agent Test ===")
    
    # Test with a few images from your examples
    test_dir = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/diverse-examples"
    
    if os.path.exists(test_dir):
        decisions = test_agent_decisions(test_dir, 2)
        
        # Check if agent makes different decisions
        unique_decisions = set((d['conf_threshold'], d['overlap_threshold']) for d in decisions)
        
        if len(unique_decisions) > 1:
            print("✓ Agent makes different decisions on different images")
        else:
            print("⚠ Agent makes same decision on all images")
            
        return decisions
    else:
        print(f"Test directory not found: {test_dir}")
        return []

if __name__ == "__main__":
    # Run quick test
    results = quick_test()
    
    # Save results
    with open('/tmp/agent_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to /tmp/agent_test_results.json")