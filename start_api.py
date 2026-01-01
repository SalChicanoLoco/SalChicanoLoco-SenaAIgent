#!/usr/bin/env python3
"""
Script to start the SenaAIgent API server for local testing.
This script ensures the Python path is set correctly.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set PYTHONPATH environment variable for subprocess compatibility
os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')

# Get port from environment or use default
port = int(os.environ.get('PORT', 5000))
debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

print("=" * 60)
print("  Starting SenaAIgent API Server")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"Port: {port}")
print(f"Debug mode: {debug}")
print()
print("Available endpoints:")
print(f"  - Health check:     http://localhost:{port}/")
print(f"  - Water quality:    http://localhost:{port}/api/water")
print(f"  - Image generation: http://localhost:{port}/api/image")
print(f"  - Art analysis:     http://localhost:{port}/api/art")
print(f"  - Orchestrator:     http://localhost:{port}/api/orchestrator")
print(f"  - Dashboard:        http://localhost:{port}/dashboard")
print()
print("Press Ctrl+C to stop the server")
print("=" * 60)
print()

# Import and run the Flask app
try:
    from api.app import app
    app.run(host='0.0.0.0', port=port, debug=debug)
except ImportError as e:
    print(f"Error: Failed to import API application: {e}")
    print()
    print("Please ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)
