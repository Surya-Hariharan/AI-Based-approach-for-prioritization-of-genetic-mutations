#!/usr/bin/env python3
"""
Quick launch script for Genetic Mutation Prioritization System
Simple wrapper around main.py for easy access
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the application with sensible defaults"""
    print("🧬 Genetic Mutation Prioritization System")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python run.py web       # Launch web application (default)")
        print("  python run.py research  # Launch Jupyter research environment")
        print("  python run.py debug     # Launch web app in debug mode")
        print("  python run.py prod      # Launch production mode")
        print()
        # Default to web mode
        cmd = [sys.executable, "main.py", "--mode", "web"]
        print(f"Starting default mode: {' '.join(cmd[1:])}")
    else:
        mode = sys.argv[1].lower()
        
        if mode in ["web", "w"]:
            cmd = [sys.executable, "main.py", "--mode", "web"]
        elif mode in ["research", "r", "jupyter", "j"]:
            cmd = [sys.executable, "main.py", "--mode", "research"]
        elif mode in ["debug", "d", "dev"]:
            cmd = [sys.executable, "main.py", "--mode", "web", "--debug"]
        elif mode in ["production", "prod", "p"]:
            cmd = [sys.executable, "main.py", "--mode", "web", "--config", "production", "--host", "0.0.0.0"]
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: web, research, debug, prod")
            return
            
        print(f"Running: {' '.join(cmd[1:])}")
    
    print("=" * 50)
    
    # Run the main application
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()