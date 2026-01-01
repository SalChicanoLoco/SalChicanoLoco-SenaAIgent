# API Backend Setup Guide

This guide will help you get the SenaAIgent API backend up and running for your testing needs.

## Problem Solved

Previously, the API backend was not accessible because:
1. Python dependencies were not installed
2. The PYTHONPATH was not configured correctly when running the API directly

This has been resolved with:
- Clear installation instructions
- Easy-to-use starter scripts that handle PYTHONPATH automatically
- Updated documentation

## Quick Setup (2 Minutes)

```bash
# 1. Navigate to the project directory
cd /path/to/SalChicanoLoco-SenaAIgent

# 2. Install dependencies (one-time setup)
pip install -r requirements.txt

# 3. Start the API server
python start_api.py
```

The API will be running at `http://localhost:5000`

## Available Starter Scripts

### Option 1: Python Script (Cross-Platform)

```bash
python start_api.py
```

**Features:**
- Works on Windows, macOS, and Linux
- Automatically sets up PYTHONPATH
- Shows available endpoints on startup
- Clear error messages if dependencies are missing

### Option 2: Bash Script (Unix/Linux/macOS)

```bash
./start_api.sh
```

**Features:**
- Native shell script for Unix-like systems
- Same automatic PYTHONPATH configuration
- Displays helpful information on startup

### Option 3: Manual Start

If you prefer to run the API manually with full control:

```bash
# Set PYTHONPATH to the project root
export PYTHONPATH=/path/to/SalChicanoLoco-SenaAIgent:$PYTHONPATH

# Run the Flask app
python api/app.py
```

On Windows (PowerShell):
```powershell
$env:PYTHONPATH = "C:\path\to\SalChicanoLoco-SenaAIgent;$env:PYTHONPATH"
python api/app.py
```

## Testing the API

Once the server is running, you can test the endpoints:

### 1. Health Check
```bash
curl http://localhost:5000/
```

### 2. Water Quality Analysis
```bash
curl -X POST http://localhost:5000/api/water \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.2,
    "turbidity": 2.0,
    "temperature": 22.0,
    "dissolved_oxygen": 8.0
  }'
```

### 3. Image Generation
```bash
curl -X POST http://localhost:5000/api/image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "width": 512,
    "height": 512,
    "style": "realistic"
  }'
```

### 4. Art Analysis
```bash
curl -X POST http://localhost:5000/api/art \
  -H "Content-Type: application/json" \
  -d '{
    "action": "analyze",
    "image_url": "https://example.com/image.jpg"
  }'
```

### 5. Orchestrator
```bash
curl -X POST http://localhost:5000/api/orchestrator \
  -H "Content-Type: application/json" \
  -d '{
    "action": "status"
  }'
```

## Running Tests

To verify everything is working correctly:

```bash
# Run all API tests
pytest tests/test_api.py -v

# Run all tests with coverage
pytest tests/ -v --cov=agents --cov=api --cov-report=html
```

## Customization

### Change the Port

```bash
# Set PORT environment variable
PORT=8080 python start_api.py
```

### Enable Debug Mode

```bash
# Set FLASK_DEBUG environment variable
FLASK_DEBUG=true python start_api.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "ModuleNotFoundError: No module named 'agents'"

**Solution:** Use one of the provided starter scripts or set PYTHONPATH manually
```bash
python start_api.py  # This handles PYTHONPATH automatically
```

### Issue: Port already in use

**Solution:** Either stop the other process or use a different port
```bash
PORT=8080 python start_api.py
```

### Issue: Permission denied when running bash script

**Solution:** Make the script executable
```bash
chmod +x start_api.sh
```

## What's Next?

Now that your API backend is accessible, you can:

1. **Integrate with your test application** - Use the API endpoints in your latest test app
2. **Explore the API** - Try different endpoints and parameters
3. **View the dashboard** - Visit `http://localhost:5000/dashboard` for a visual interface
4. **Deploy to production** - Use Docker or Gunicorn for production deployment

## Production Deployment

For production use, consider using Gunicorn:

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 api.app:app
```

Or use Docker:

```bash
docker build -t senaai-agent .
docker run -p 8000:8000 senaai-agent
```

## Support

If you encounter any issues:

1. Check the [README.md](README.md) for full documentation
2. Review the API endpoints at `http://localhost:5000/`
3. Run tests to verify the installation: `pytest tests/test_api.py -v`
4. Check the Flask logs for error messages

---

**Summary:** The API backend is now fully accessible and ready for your latest test application! Simply run `python start_api.py` and start making requests to `http://localhost:5000`.
