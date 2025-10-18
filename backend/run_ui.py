#!/usr/bin/env python3
"""
Script to run the Multimodal RAG UI
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit UI"""
    ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'app.py')

    if not os.path.exists(ui_path):
        print(f"Error: UI app not found at {ui_path}")
        sys.exit(1)

    print("🚀 Starting Multimodal RAG UI...")
    print(f"📁 UI path: {ui_path}")
    print("🌐 Open your browser to http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', ui_path,
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down UI...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()