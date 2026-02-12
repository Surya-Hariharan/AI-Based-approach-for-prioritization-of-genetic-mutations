#!/usr/bin/env python3
"""
Run script for Genetic Mutation Prioritization Web Application

This script can be run from the project root directory to start the web server.
"""

import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to Python path
    sys.path.insert(0, script_dir)
    
    # Change to backend directory and run the app
    backend_dir = os.path.join(script_dir, 'backend')
    app_path = os.path.join(backend_dir, 'app.py')
    
    if not os.path.exists(app_path):
        print(f"❌ Error: Flask app not found at {app_path}")
        print("Please make sure you're running this from the project root directory.")
        sys.exit(1)
    
    print("🧬 Starting Genetic Mutation Prioritization Server...")
    print(f"📁 Project root: {script_dir}")
    print(f"🖥️  Backend: {backend_dir}")
    print("━" * 60)
    
    try:
        # Change to backend directory and run the Flask app
        os.chdir(backend_dir)
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Flask app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Python not found. Please ensure Python is installed and in PATH.")
        sys.exit(1)

if __name__ == '__main__':
    main()