# 🧬 Genetic Mutation Prioritization - Quick Launch Scripts

## Windows (.bat)

### run.bat - Start Web Application
```batch
@echo off
echo 🧬 Starting Genetic Mutation Prioritization Server...
cd /d "%~dp0"
python run.py
pause
```

### install.bat - Install Dependencies
```batch  
@echo off
echo 📦 Installing dependencies...
cd /d "%~dp0"
pip install -r requirements.txt
echo ✅ Installation complete!
pause
```

### activate.bat - Activate Virtual Environment (if using venv)
```batch
@echo off
if exist "venv\Scripts\activate.bat" (
    echo 🐍 Activating virtual environment...
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated!
) else (
    echo ⚠️  No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment created and activated!
)
cmd /k
```

## Linux/Mac (.sh)

### run.sh - Start Web Application
```bash
#!/bin/bash
echo "🧬 Starting Genetic Mutation Prioritization Server..."
cd "$(dirname "$0")"
python3 run.py
```

### install.sh - Install Dependencies
```bash
#!/bin/bash
echo "📦 Installing dependencies..."
cd "$(dirname "$0")"
pip3 install -r requirements.txt
echo "✅ Installation complete!"
```

### activate.sh - Activate Virtual Environment
```bash
#!/bin/bash
if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated!"
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated!"
fi
exec bash
```

## Usage

1. **Windows**: Double-click `run.bat` or `install.bat`
2. **Linux/Mac**: Make executable with `chmod +x *.sh`, then run `./run.sh`
3. **Any OS**: Run `python run.py` from project root

## Development

For development, use:
```bash
# Install in development mode
pip install -e .

# Run with auto-reload
cd backend
python app.py
```