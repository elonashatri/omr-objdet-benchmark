# Save as staffline_postprocess.py
import numpy as np
import cv2

def enhance_staffline_predictions(image_path, predictions, staffline_class_id, min_confidence=0.25):
    """Enhance staffline predictions using image processing techniques"""
    # Read the original image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract staffline predictions
    staffline_boxes = []
    for pred in predictions:
        if pred.cls == staffline_class_id and pred.conf >= min_confidence:
            staffline_boxes.append(pred.xyxy[0].cpu().numpy())  # Convert to numpy
    
    # If we have few stafflines detected, try to find more
    if len(staffline_boxes) < 5:  # Typical staff has 5 lines
        # Use Hough transform to detect horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=gray.shape[1]//3, maxLineGap=20)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if horizontal (small y difference)
                if abs(y2 - y1) < 5:
                    # Convert to YOLO format
                    x_center = (x1 + x2) / 2 / gray.shape[1]
                    y_center = (y1 + y2) / 2 / gray.shape[0]
                    width = abs(x2 - x1) / gray.shape[1]
                    height = max(abs(y2 - y1), 3) / gray.shape[0]  # Minimum height
                    
                    # Check if this line overlaps with existing predictions
                    new_line = True
                    for box in staffline_boxes:
                        box_y_center = (box[1] + box[3]) / 2 / gray.shape[0]
                        if abs(box_y_center - y_center) < 0.01:  # 1% of image height
                            new_line = False
                            break
                    
                    if new_line:
                        # Add as a new prediction with confidence 0.3
                        predictions.append([staffline_class_id, x_center, y_center, 
                                          width, height, 0.3])
    
    return predictions