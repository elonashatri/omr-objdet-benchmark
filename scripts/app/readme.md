# Optical Music Recognition (OMR) Pipeline

## Project Structure
```
omr-pipeline/
│
├── backend/
│   ├── app.py              # Flask backend server
│   └── your_omr_pipeline.py # Your existing OMR pipeline script
│
├── frontend/
│   ├── src/
│   │   └── App.js          # React frontend application
│   ├── package.json
│   └── README.md
│
└── deployment/
    └── deploy.sh           # Deployment script
```

## Prerequisites
- Python 3.8+
- Node.js 14+
- CUDA 12.1 (for GPU support)
- pip, npm
- NGINX (for production deployment)

## Backend Setup

### 1. Create Virtual Environment
```bash
python3 -m venv omr_env
source omr_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Edit `backend/app.py` and update:
- `MODEL_PATH`: Path to your YOLOv8 model
- `CLASS_MAPPING_FILE`: Path to class mapping JSON
- Adjust `UPLOAD_FOLDER` and `RESULTS_FOLDER`

### 4. Run Backend
```bash
python backend/app.py
```

## Frontend Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Run Development Server
```bash
npm start
```

## Production Deployment

### 1. Build Frontend
```bash
cd frontend
npm run build
```

### 2. Use Deployment Script
```bash
sudo ./deployment/deploy.sh
```

## API Endpoints

### POST `/run_omr_pipeline`
- **Description**: Run full OMR pipeline on uploaded image
- **Request**: Multipart form-data with `file` parameter
- **Response**:
  ```json
  {
    "original_image": "base64 encoded image",
    "processed_image": "base64 encoded visualization",
    "detections": [
      {
        "class": "Notehead",
        "pitch": "C4",
        "confidence": 0.95,
        "bbox": [x, y, width, height]
      }
    ]
  }
  ```

## Troubleshooting
- Ensure all paths are correct
- Check CUDA and GPU compatibility
- Verify model and class mapping files exist
- Check permission issues with upload/result folders

## Environment
- Python: 3.8+
- PyTorch: 2.5.1
- CUDA: 12.1
- YOLOv8
- React: 18+