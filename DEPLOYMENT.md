# 🚀 Deployment Guide

## Quick Start

### 1️⃣ Option A: Simple Launch (Recommended)

**Windows:**
```batch
# Double-click run.bat
# OR in command prompt:
run.bat
```

**Linux/Mac/Windows:**
```bash
python run.py
```

### 2️⃣ Option B: Direct Backend Launch

```bash
cd backend
python app.py
```

### 3️⃣ Option C: Production Mode

```bash
# Set environment
export FLASK_ENV=production
export FLASK_DEBUG=False

# Run with Gunicorn (Linux/Mac)
cd backend
pip install gunicorn 
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Run with Waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

---

## 📁 Project Structure (After Organization)

```
genetic-mutation-prioritization/
│
├── 🖥️ Backend (API Server)
│   ├── backend/
│   │   ├── app.py              # Main Flask application
│   │   ├── config.py           # Configuration management
│   │   └── __init__.py         # Backend package init
│
├── 🎨 Frontend (Web Interface)  
│   ├── frontend/
│   │   ├── templates/
│   │   │   └── index.html      # Main web interface
│   │   └── static/
│   │       ├── css/style.css   # Styling
│   │       └── js/app.js       # JavaScript logic
│
├── 🧠 Core ML Pipeline
│   ├── src/                    # Source code
│   │   ├── models/             # ML model architectures
│   │   ├── preprocessing/      # Data processing
│   │   ├── evaluation/         # Metrics and plotting
│   │   ├── uncertainty/        # Confidence estimation
│   │   ├── ensemble/           # Multi-model approaches  
│   │   ├── aggregation/        # Gene-level analysis
│   │   └── utils/              # Utilities
│
├── 📓 Research Pipeline
│   ├── notebooks/              # Jupyter notebooks (00-06)
│   └── configs/                # YAML configuration
│
├── 📊 Data & Results
│   ├── data/                   # Raw, interim, processed data
│   └── reports/                # Model outputs and plots
│
├── 🚀 Launch Scripts
│   ├── run.py                  # Python launcher
│   ├── run.bat                 # Windows batch script
│   ├── setup.py               # Package installation
│   └── requirements.txt       # Dependencies
│
└── 📚 Documentation
    ├── README.md               # Main documentation
    ├── WEB_APP_GUIDE.md        # API documentation  
    ├── QUICK_START.md          # 30-second setup
    ├── DEPLOYMENT.md           # This file
    └── .env.example            # Environment template
```

---

## 🔧 Environment Setup

### Required Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Server Settings
HOST=127.0.0.1
PORT=5000

# Model Configuration  
DEFAULT_MODEL=mlp
USE_GPU=true

# File Upload
MAX_FILE_SIZE=16777216  # 16MB
UPLOAD_FOLDER=data/uploads
```

### Development Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install in development mode (optional)
pip install -e .

# 5. Run the application
python run.py
```

---

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)

**Create `docker-compose.yml`:**
```yaml
version: '3.8'

services:
  genetic-ai:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
      - HOST=0.0.0.0
      - PORT=5000
    restart: unless-stopped
```

**Deploy:**
```bash
docker-compose up -d
```

### Option 2: Standard Docker

**Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads reports/results/checkpoints

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV HOST=0.0.0.0
ENV PORT=5000

# Run the application
CMD ["python", "run.py"]
```

**Build and Run:**
```bash
# Build image
docker build -t genetic-mutation-ai:latest .

# Run container
docker run -d \
  --name genetic-ai \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  genetic-mutation-ai:latest
```

---

## ☁️ Cloud Deployment

### Heroku

**1. Create `Procfile`:**
```
web: cd backend && gunicorn app:app --bind 0.0.0.0:$PORT
```

**2. Deploy:**
```bash
# Install Heroku CLI
heroku login
heroku create your-app-name

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-secret-key

# Deploy
git push heroku main
```

### AWS EC2

**1. Launch EC2 instance (Ubuntu 20.04)**

**2. Setup script:**
```bash
#!/bin/bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv nginx

# Clone repository
git clone https://github.com/your-username/genetic-mutation-ai.git
cd genetic-mutation-ai

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Gunicorn
pip install gunicorn

# Create systemd service
sudo tee /etc/systemd/system/genetic-ai.service > /dev/null <<EOF
[Unit]
Description=Genetic Mutation AI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/genetic-mutation-ai/backend
Environment=PATH=/home/ubuntu/genetic-mutation-ai/venv/bin
ExecStart=/home/ubuntu/genetic-mutation-ai/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl start genetic-ai
sudo systemctl enable genetic-ai

# Configure Nginx
sudo tee /etc/nginx/sites-available/genetic-ai > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    client_max_body_size 20M;
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/genetic-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

echo "✅ Deployment complete! Visit http://your-domain.com"
```

### Google Cloud Run

**1. Create `cloudbuild.yaml`:**
```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/genetic-ai', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/genetic-ai']
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'run'
  - 'deploy'
  - 'genetic-ai'
  - '--image'
  - 'gcr.io/$PROJECT_ID/genetic-ai'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
  - '--allow-unauthenticated'
```

**2. Deploy:**
```bash
gcloud builds submit --config cloudbuild.yaml
```

---

## 🔐 Production Security

### HTTPS Setup (Let's Encrypt)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Security Headers

**Update Nginx configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration (auto-generated by certbot)
    
    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'" always;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:5000;
        # ... proxy headers
    }

    client_max_body_size 20M;
}
```

---

## 📊 Monitoring & Logging

### Application Logging

**Add to `backend/app.py`:**
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/genetic_ai.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Genetic Mutation AI startup')
```

### Health Monitoring

**Create monitoring script:**
```bash
#!/bin/bash
# monitor.sh
URL="http://localhost:5000/api/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $URL)

if [ $RESPONSE -eq 200 ]; then
    echo "$(date): ✅ Service is healthy"
else
    echo "$(date): ❌ Service is down (HTTP $RESPONSE)"
    # Restart service
    sudo systemctl restart genetic-ai
fi
```

**Add to crontab:**
```bash
*/5 * * * * /home/ubuntu/monitor.sh >> /var/log/genetic-ai-monitor.log
```

---

## 🚨 Troubleshooting

### Common Issues

**1. Models not found:**
```bash
# Ensure models are trained
jupyter notebook
# Run: notebooks/02_baseline_training.ipynb or 03_mlp_training.ipynb

# Check model paths
ls -la reports/results/checkpoints/
```

**2. Import errors:**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode  
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**3. Port conflicts:**
```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>

# Or use different port
export PORT=5001
```

**4. Permission errors:**
```bash
# Fix file permissions
chmod +x run.py run.bat
chmod -R 755 backend/ frontend/

# Fix data directory permissions  
mkdir -p data/uploads
chmod 777 data/uploads
```

### Log Locations

- **Application logs:** `logs/genetic_ai.log`
- **System logs:** `/var/log/syslog`
- **Nginx logs:** `/var/log/nginx/access.log`, `/var/log/nginx/error.log`
- **Service logs:** `journalctl -u genetic-ai -f`

---

## 📈 Performance Optimization

### GPU Acceleration

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch
pip uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Enable GPU in environment
export USE_GPU=true
```

### Memory Optimization

```python
# In backend/app.py, add memory management
import gc
import torch

@app.after_request 
def cleanup(response):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return response
```

### Load Balancing

**Multiple instances with Nginx:**
```nginx
upstream genetic_ai_backend {
    server 127.0.0.1:5001;
    server 127.0.0.1:5002; 
    server 127.0.0.1:5003;
}

server {
    location / {
        proxy_pass http://genetic_ai_backend;
        # ... other config
    }
}
```

---

**🧬 Ready for production deployment in clinical and research environments! 🚀**