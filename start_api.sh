#!/bin/bash
# Script to start the SenaAIgent API server for local testing
# This script ensures the PYTHONPATH is set correctly

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to the project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Default port
PORT="${PORT:-5000}"

echo "============================================"
echo "  Starting SenaAIgent API Server"
echo "============================================"
echo "Project root: $SCRIPT_DIR"
echo "Port: $PORT"
echo ""
echo "Available endpoints:"
echo "  - Health check:     http://localhost:$PORT/"
echo "  - Water quality:    http://localhost:$PORT/api/water"
echo "  - Image generation: http://localhost:$PORT/api/image"
echo "  - Art analysis:     http://localhost:$PORT/api/art"
echo "  - Orchestrator:     http://localhost:$PORT/api/orchestrator"
echo "  - Dashboard:        http://localhost:$PORT/dashboard"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================"
echo ""

# Start the Flask development server
cd "$SCRIPT_DIR"
python api/app.py
