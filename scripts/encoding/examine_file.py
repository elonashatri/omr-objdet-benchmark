import json
import sys

filename = sys.argv[1]
print(f"Examining file: {filename}")

try:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    detections = data.get("detections", [])
    print(f"File contains {len(detections)} detections")
    
    noteheads = [d for d in detections if "notehead" in d.get("class_name", "").lower()]
    staff_lines = [d for d in detections if "staff" in d.get("class_name", "").lower()]
    
    print(f"Found {len(noteheads)} noteheads and {len(staff_lines)} staff lines")
    
    if noteheads:
        print("\nSample notehead:")
        print(json.dumps(noteheads[0], indent=2))
    
    if staff_lines:
        print("\nSample staff line:")
        print(json.dumps(staff_lines[0], indent=2))
        
except Exception as e:
    print(f"Error examining file: {e}")