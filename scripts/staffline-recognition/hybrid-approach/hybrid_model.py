from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import torch

class HybridStafflineDetector:
    def __init__(self, model_path, staffline_class_id, conf=0.25):
        """
        Hybrid model combining YOLO with classical line detection for stafflines
        
        Args:
            model_path: Path to the trained YOLO model
            staffline_class_id: Class ID for stafflines (0-indexed)
            conf: Confidence threshold for YOLO detection
        """
        self.yolo_model = YOLO(model_path)
        self.staffline_class_id = staffline_class_id
        self.conf = conf
        
    def detect(self, image_path, save_path=None, visualize=False):
        """
        Run detection with hybrid approach
        
        Args:
            image_path: Path to the input image
            save_path: Path to save visualization (optional)
            visualize: Whether to create visualization
            
        Returns:
            results: YOLO results with enhanced staffline detection
        """
        # 1. Run standard YOLO detection
        results = self.yolo_model.predict(image_path, conf=self.conf)[0]
        
        # 2. Extract staffline predictions (if any)
        staffline_boxes = []
        for i, box in enumerate(results.boxes):
            if int(box.cls.item()) == self.staffline_class_id:
                xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                staffline_boxes.append({
                    'xyxy': xyxy,
                    'conf': float(box.conf.item())
                })
        
        # 3. Run specialized staffline detection
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            return results
            
        enhanced_boxes = self._detect_stafflines(img, staffline_boxes)
        
        # 4. Merge results
        # We'll create a new set of boxes to add to the YOLO results
        new_boxes = []
        for box in enhanced_boxes:
            # Only add if confidence is above threshold
            if box['conf'] >= self.conf:
                new_boxes.append({
                    'xyxy': box['xyxy'],
                    'conf': box['conf'],
                    'cls': self.staffline_class_id
                })
        
        # 5. Visualize if requested
        if visualize or save_path:
            self._visualize_results(img, results, enhanced_boxes, save_path)
            
        # 6. Add new boxes to results (this is a simplified approach)
        # In a real implementation, you'd modify the YOLO Results object
        print(f"Added {len(new_boxes)} stafflines via specialized detection")
        
        return results, new_boxes
    
    def _detect_stafflines(self, img, existing_boxes):
        """
        Specialized staffline detection using classical CV techniques
        """
        enhanced_boxes = existing_boxes.copy()
        
        # Convert to grayscale for line detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle different lighting conditions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Remove small noise
        kernel = np.ones((1, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detect lines using probabilistic Hough transform
        # Parameters tuned for stafflines
        lines = cv2.HoughLinesP(
            binary, 
            rho=1, 
            theta=np.pi/180,
            threshold=100,  # Minimum points to form a line
            minLineLength=img.shape[1]//4,  # At least 1/4 of image width
            maxLineGap=20  # Allow small gaps
        )
        
        if lines is None:
            return enhanced_boxes
        
        # Process detected lines
        h, w = img.shape[:2]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip if not horizontal-ish (stafflines are mostly horizontal)
            if abs(y2 - y1) > 5:  # Allow slight slope
                continue
                
            # Skip if too short
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < w * 0.2:  # At least 20% of image width
                continue
                
            # Compute line properties
            y_center = (y1 + y2) / 2
            height = max(3, abs(y2 - y1))  # Ensure minimum height
            
            # Check if this line overlaps with existing stafflines
            is_new_line = True
            for box in enhanced_boxes:
                y_min, y_max = box['xyxy'][1], box['xyxy'][3]
                if y_min <= y_center <= y_max or \
                   abs(y_center - (y_min + y_max)/2) < height:
                    is_new_line = False
                    break
            
            if is_new_line:
                # Convert to YOLO box format (xyxy)
                xyxy = np.array([x1, y_center - height/2, x2, y_center + height/2])
                
                # Add to enhanced boxes
                enhanced_boxes.append({
                    'xyxy': xyxy,
                    'conf': 0.7  # Assign reasonable confidence
                })
        
        return enhanced_boxes
    
    def _visualize_results(self, img, yolo_results, enhanced_boxes, save_path=None):
        """Visualize detection results"""
        # Create copy of image for visualization
        vis_img = img.copy()
        
        # Draw YOLO boxes (non-staffline)
        for i, box in enumerate(yolo_results.boxes):
            if int(box.cls.item()) != self.staffline_class_id:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(vis_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        
        # Draw enhanced staffline boxes
        for box in enhanced_boxes:
            xyxy = box['xyxy'].astype(int)
            # Use red color for stafflines
            cv2.rectangle(vis_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 1)
            
            # Add confidence text
            conf = box['conf']
            cv2.putText(vis_img, f"{conf:.2f}", (xyxy[0], xyxy[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, vis_img)
            print(f"Visualization saved to {save_path}")
        
        # Display if no save path
        if save_path is None:
            cv2.imshow("Hybrid Staffline Detection", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Hybrid Staffline Detection")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="results/hybrid_detection", help="Output directory")
    parser.add_argument("--staffline_class_id", type=int, required=True, help="Class ID for stafflines (0-indexed)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = HybridStafflineDetector(args.model, args.staffline_class_id, args.conf)
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        output_path = os.path.join(args.output, f"{input_path.stem}_hybrid.jpg")
        results, new_boxes = detector.detect(str(input_path), output_path, args.visualize)
        print(f"Processed {input_path.name}: Found {len(new_boxes)} stafflines")
    else:
        # Directory
        os.makedirs(args.output, exist_ok=True)
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        for img_path in image_paths:
            output_path = os.path.join(args.output, f"{img_path.stem}_hybrid.jpg")
            results, new_boxes = detector.detect(str(img_path), output_path, args.visualize)
            print(f"Processed {img_path.name}: Found {len(new_boxes)} stafflines")

if __name__ == "__main__":
    main()