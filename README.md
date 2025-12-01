# SenaAIgent

A robust backend-focused scaffold for AI-powered analytics and image generation, built with Flask and designed for scalable deployment.

## Features

- **ML Analytics API**: Water quality analysis with predictive scoring
- **Image Generation API**: AI-powered image generation with customizable styles
- **Modular Agent Architecture**: Extensible agent classes for ML and image tasks
- **Production-Ready**: Docker containerization with Gunicorn
- **CI/CD Pipeline**: GitHub Actions for automated builds and deployments

## Project Structure

```
SenaAIgent/
├── agents/
│   ├── __init__.py
│   ├── model_agent.py      # ML analytics agent
│   └── image_agent.py      # Image generation agent
├── api/
│   ├── __init__.py
│   └── app.py              # Flask API application
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_model_agent.py
│   └── test_image_agent.py
├── .github/
│   └── workflows/
│       └── docker-build.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and service info |
| `/api/water` | GET | API documentation for water analysis |
| `/api/water` | POST | Analyze water quality parameters |
| `/api/image` | GET | API documentation and available styles |
| `/api/image` | POST | Generate image from prompt |

### Water Quality Analysis

**POST /api/water**

```json
{
  "ph": 7.2,
  "turbidity": 2.5,
  "temperature": 22.0,
  "dissolved_oxygen": 7.5
}
```

**Response:**

```json
{
  "success": true,
  "analysis": {
    "quality_score": 85.5,
    "quality_label": "Excellent",
    "parameters": { ... },
    "recommendations": ["Water quality is satisfactory"]
  }
}
```

### Image Generation

**POST /api/image**

```json
{
  "prompt": "A serene mountain landscape at sunset",
  "width": 1024,
  "height": 768,
  "style": "realistic"
}
```

**Response:**

```json
{
  "success": true,
  "mock": true,
  "image_id": "mock_abc123def",
  "prompt": "A serene mountain landscape at sunset",
  "dimensions": { "width": 1024, "height": 768 }
}
```

## Installation

### Prerequisites

- Python 3.12+
- pip

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/SalChicanoLoco/SalChicanoLoco-SenaAIgent.git
   cd SalChicanoLoco-SenaAIgent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server:
   ```bash
   python api/app.py
   ```

5. Access the API at `http://localhost:5000`

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=agents --cov=api --cov-report=html
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t senaai-agent .
```

### Run the Container

```bash
docker run -p 8000:8000 senaai-agent
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `FLASK_DEBUG` | Enable debug mode | false |
| `IMAGE_API_KEY` | External image API key | None |
| `IMAGE_API_ENDPOINT` | External image API URL | None |

### Docker Compose (Optional)

Create a `docker-compose.yml` for multi-container setups:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    restart: unless-stopped
```

## Deployment

### Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure build settings:
   - **Environment**: Docker
   - **Docker Command**: (leave blank to use Dockerfile CMD)
4. Add environment variables as needed
5. Deploy

### Manual Server Deployment

```bash
# With Gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 api.app:app

# Or with Docker
docker run -d -p 8000:8000 --name senaai senaai-agent
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/docker-build.yml`) automatically:

1. Runs tests on every push and pull request
2. Builds the Docker image
3. Pushes to GitHub Container Registry (on main branch)
4. Runs linting checks

### Manual Trigger

You can manually trigger the workflow from the Actions tab on GitHub.

## Extending the Agents

### Adding a New Agent

1. Create a new agent file in `agents/`:
   ```python
   # agents/my_agent.py
   class MyAgent:
       def __init__(self):
           pass

       def process(self, data):
           # Implementation
           pass
   ```

2. Export in `agents/__init__.py`:
   ```python
   from .my_agent import MyAgent
   __all__ = [..., "MyAgent"]
   ```

3. Add API endpoint in `api/app.py`

### Integrating External Services

The image agent supports external API integration. Set environment variables:

```bash
export IMAGE_API_KEY="your-api-key"
export IMAGE_API_ENDPOINT="https://api.example.com/v1/images"
```

## Future Scaling

- **Additional Agents**: Add new agent containers for specialized tasks
- **JIT Service Integration**: Connect to on-demand ML inference services
- **Load Balancing**: Deploy multiple API containers behind a load balancer
- **Message Queues**: Integrate with Redis/RabbitMQ for async processing

## License

MIT License - see [LICENSE](LICENSE) for details.
