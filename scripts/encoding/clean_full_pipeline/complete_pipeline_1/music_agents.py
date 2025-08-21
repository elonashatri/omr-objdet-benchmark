# music_agents.py - Updated version
import cv2
import numpy as np
import json
import os
import subprocess

class MusicScoreAgent:
    def __init__(self):
        self.model_expertise = {
            "restwhole": "structure",
            "resthalf": "symbol", 
            "restquarter": "symbol",
            "barline": "structure",
            "stem": "structure",
            "beam": "structure"
        }
    
    def analyze_and_recommend(self, structure_detections_path, symbol_detections_path, image_path):
        """Analyze REAL detection data and return processing recommendations"""
        
        # Load actual detection data
        structure_data = self._load_json(structure_detections_path)
        symbol_data = self._load_json(symbol_detections_path)
        
        # Get real detection counts
        structure_count = len(structure_data.get('detections', []))
        symbol_count = len(symbol_data.get('detections', []))
        total_detections = structure_count + symbol_count
        
        # Calculate density based on actual image size
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        density = total_detections / (width * height / 10000)
        
        # Analyze symbol types
        all_detections = structure_data.get('detections', []) + symbol_data.get('detections', [])
        rest_count = len([d for d in all_detections if 'rest' in d.get('class_name', '').lower()])
        beam_count = len([d for d in all_detections if 'beam' in d.get('class_name', '').lower()])
        
        # Better decision logic with more realistic thresholds
        recommendations = {
            'density': density,
            'total_detections': total_detections,
            'structure_count': structure_count,
            'symbol_count': symbol_count,
            'rest_count': rest_count,
            'beam_count': beam_count,
            'high_conf_threshold': 0.5,  # default
            'overlap_threshold': 0.5,    # default
            'processing_notes': []
        }
        
        # More nuanced agent decisions
        if density > 1.5:  # Very dense score (raised threshold)
            recommendations['high_conf_threshold'] = 0.6
            recommendations['overlap_threshold'] = 0.3
            recommendations['processing_notes'].append(f"Very dense score (density={density:.1f}) - using conservative thresholds")
            
        elif density > 1.0:  # Moderately dense
            recommendations['high_conf_threshold'] = 0.55
            recommendations['overlap_threshold'] = 0.4
            recommendations['processing_notes'].append(f"Dense score (density={density:.1f}) - using moderate thresholds")
            
        # Adjust for rest-heavy scores
        rest_ratio = rest_count / max(total_detections, 1)
        if rest_ratio > 0.15:  # More than 15% rests
            recommendations['overlap_threshold'] = 0.7
            recommendations['processing_notes'].append(f"Rest-heavy score ({rest_count} rests, {rest_ratio:.1%}) - using specialized rest handling")
            
        # Beam complexity - use higher threshold
        if beam_count > 100:  # Very complex rhythms
            recommendations['high_conf_threshold'] = 0.4
            recommendations['processing_notes'].append(f"Extremely complex rhythmic score ({beam_count} beams) - using aggressive beam-aware processing")
        elif beam_count > 50:  # Moderately complex
            recommendations['high_conf_threshold'] = 0.45
            recommendations['processing_notes'].append(f"Complex rhythmic score ({beam_count} beams) - using beam-aware processing")
            
        # Simple scores get different treatment
        if beam_count < 20 and density < 0.5:
            recommendations['high_conf_threshold'] = 0.3
            recommendations['processing_notes'].append(f"Simple score ({beam_count} beams, density={density:.1f}) - using standard processing")
            
        return recommendations
    
    def _load_json(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {"detections": []}

    def run_quick_detection(self, image_path):
        """Run a quick detection to get real data for analysis"""
        # This would run your actual detection pipeline
        # For now, return None to indicate we need real detection files
        return None