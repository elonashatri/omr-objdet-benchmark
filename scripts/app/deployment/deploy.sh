#!/bin/bash

# Project Setup and Deployment Script for OMR Pipeline

# Ensure script is run with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo" 
   exit 1
fi

# Variables
PROJECT_DIR="/opt/omr-pipeline"
VENV_NAME="omr_env"
FRONTEND_DIR="$PROJECT_DIR/frontend"
BACKEND_DIR="$PROJECT_DIR/backend"

# Create project directory
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Setup Python Virtual Environment
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# Install backend dependencies
pip install flask flask-cors opencv-python-headless numpy werkzeug ultralytics torch torchvision

# Deactivate virtual environment
deactivate

# Create systemd service for backend
cat > /etc/systemd/system/omr-backend.service << EOL
[Unit]
Description=OMR Pipeline Backend Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BACKEND_DIR
ExecStart=$PROJECT_DIR/$VENV_NAME/bin/python $BACKEND_DIR/app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOL

# Setup NGINX configuration
cat > /etc/nginx/sites-available/omr-pipeline << EOL
server {
    listen 80;
    server_name omr.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;  # Frontend React server
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    location /api/ {
        proxy_pass http://localhost:5000;  # Backend Flask server
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOL

# Enable and start services
systemctl enable omr-backend
systemctl start omr-backend

# Nginx configuration
ln -s /etc/nginx/sites-available/omr-pipeline /etc/nginx/sites-enabled/
nginx -t  # Test configuration
systemctl restart nginx

echo "OMR Pipeline deployment complete!"