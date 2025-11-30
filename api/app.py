"""
SenaAIgent API - Flask application with endpoints for ML analytics, image generation,
aesthetic analysis, and agent orchestration.
"""

import os
from flask import Flask, jsonify, request

from agents import ModelAgent, ImageAgent, ArtAgent, OrchestratorAgent, TaskPriority


def create_app():
    """Application factory for the Flask app."""
    app = Flask(__name__)

    # Initialize agents
    model_agent = ModelAgent()
    image_agent = ImageAgent()
    art_agent = ArtAgent()
    orchestrator = OrchestratorAgent()

    # Register agents with orchestrator
    orchestrator.register_agent("model", "model", model_agent, ["predict", "analyze"])
    orchestrator.register_agent("image", "image", image_agent, ["generate_image"])
    orchestrator.register_agent("art", "art", art_agent, ["analyze_aesthetics", "create_set"])

    @app.route("/", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "SenaAIgent",
            "version": "1.0.0",
            "endpoints": {
                "health": "/",
                "water_quality": "/api/water",
                "image_generation": "/api/image",
                "art_analysis": "/api/art",
                "orchestrator": "/api/orchestrator",
            },
        })

    @app.route("/api/water", methods=["GET", "POST"])
    def water_quality():
        """
        Water quality analysis endpoint.

        GET: Returns API documentation and example usage.
        POST: Analyzes water quality based on provided parameters.

        POST Body (JSON):
            - ph: float (optional, default: 7.0)
            - turbidity: float (optional, default: 1.0)
            - temperature: float (optional, default: 20.0)
            - dissolved_oxygen: float (optional, default: 8.0)

        Returns:
            JSON with water quality analysis results.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/water",
                "description": "Water quality analysis using ML analytics",
                "methods": ["GET", "POST"],
                "post_parameters": {
                    "ph": {
                        "type": "float",
                        "description": "pH level (0-14)",
                        "default": 7.0,
                    },
                    "turbidity": {
                        "type": "float",
                        "description": "Turbidity in NTU",
                        "default": 1.0,
                    },
                    "temperature": {
                        "type": "float",
                        "description": "Temperature in Celsius",
                        "default": 20.0,
                    },
                    "dissolved_oxygen": {
                        "type": "float",
                        "description": "Dissolved oxygen in mg/L",
                        "default": 8.0,
                    },
                },
                "example_request": {
                    "ph": 7.2,
                    "turbidity": 2.5,
                    "temperature": 22.0,
                    "dissolved_oxygen": 7.5,
                },
            })

        # POST request - analyze water quality
        try:
            data = request.get_json() or {}

            # Validate numeric types
            params = {}
            for key in ["ph", "turbidity", "temperature", "dissolved_oxygen"]:
                if key in data:
                    try:
                        params[key] = float(data[key])
                    except (TypeError, ValueError):
                        return jsonify({
                            "error": f"Invalid value for {key}: must be a number",
                        }), 400

            result = model_agent.predict_water_quality(params)
            return jsonify({
                "success": True,
                "analysis": result,
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
            }), 500

    @app.route("/api/image", methods=["GET", "POST"])
    def image_generation():
        """
        Image generation endpoint.

        GET: Returns API documentation and available styles.
        POST: Generates an image based on the provided prompt.

        POST Body (JSON):
            - prompt: str (required)
            - width: int (optional, default: 512)
            - height: int (optional, default: 512)
            - style: str (optional)

        Returns:
            JSON with image generation results.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/image",
                "description": "AI-powered image generation",
                "methods": ["GET", "POST"],
                "available_styles": image_agent.list_styles(),
                "post_parameters": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image",
                        "required": True,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width (64-2048)",
                        "default": 512,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height (64-2048)",
                        "default": 512,
                    },
                    "style": {
                        "type": "string",
                        "description": "Image style",
                        "required": False,
                    },
                },
                "example_request": {
                    "prompt": "A serene mountain landscape at sunset",
                    "width": 1024,
                    "height": 768,
                    "style": "realistic",
                },
            })

        # POST request - generate image
        try:
            data = request.get_json()

            if not data:
                return jsonify({
                    "success": False,
                    "error": "Request body is required",
                }), 400

            prompt = data.get("prompt")
            if not prompt:
                return jsonify({
                    "success": False,
                    "error": "Prompt is required",
                }), 400

            # Validate prompt
            validation = image_agent.validate_prompt(prompt)
            if not validation["valid"]:
                return jsonify({
                    "success": False,
                    "error": validation["error"],
                }), 400

            # Parse optional parameters
            width = data.get("width", 512)
            height = data.get("height", 512)
            style = data.get("style")

            try:
                width = int(width)
                height = int(height)
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "error": "Width and height must be integers",
                }), 400

            result = image_agent.generate_image(
                prompt=validation["processed_prompt"],
                width=width,
                height=height,
                style=style,
            )

            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
            }), 500

    @app.route("/api/art", methods=["GET", "POST"])
    def art_analysis():
        """
        Art and aesthetic analysis endpoint.

        GET: Returns API documentation and available styles.
        POST: Analyzes aesthetics or manages image sets.

        POST Body (JSON):
            - action: str (required) - "analyze", "create_set", "add_to_set", "export_context"
            - image_url: str (for analyze)
            - name: str (for create_set)
            - description: str (for create_set)
            - set_id: str (for add_to_set, export_context)

        Returns:
            JSON with analysis or operation results.
        """
        if request.method == "GET":
            return jsonify({
                "endpoint": "/api/art",
                "description": "Art and aesthetic analysis with scalable image sets",
                "methods": ["GET", "POST"],
                "available_styles": art_agent.list_available_styles(),
                "actions": {
                    "analyze": {
                        "description": "Analyze image aesthetics",
                        "params": ["image_url", "image_base64"],
                    },
                    "create_set": {
                        "description": "Create a new scalable image set",
                        "params": ["name", "description", "base_style", "target_count"],
                    },
                    "add_to_set": {
                        "description": "Add image to existing set",
                        "params": ["set_id", "image_url"],
                    },
                    "export_context": {
                        "description": "Export context for other agents",
                        "params": ["set_id"],
                    },
                    "style_transfer": {
                        "description": "Apply style transfer to image",
                        "params": ["image_url", "target_style", "intensity"],
                    },
                },
            })

        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "Request body required"}), 400

            action = data.get("action")
            if not action:
                return jsonify({"success": False, "error": "Action is required"}), 400

            if action == "analyze":
                result = art_agent.analyze_aesthetics({
                    "image_url": data.get("image_url"),
                    "image_base64": data.get("image_base64"),
                })
            elif action == "create_set":
                result = art_agent.create_image_set(
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    base_style=data.get("base_style"),
                    target_count=data.get("target_count", 10),
                    context=data.get("context"),
                )
            elif action == "add_to_set":
                result = art_agent.add_to_image_set(
                    set_id=data.get("set_id", ""),
                    image_data={"image_url": data.get("image_url")},
                )
            elif action == "export_context":
                result = art_agent.export_context(data.get("set_id", ""))
            elif action == "style_transfer":
                result = art_agent.apply_style_transfer(
                    source_image={"image_url": data.get("image_url")},
                    target_style=data.get("target_style", ""),
                    intensity=data.get("intensity", 0.7),
                )
            else:
                return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/orchestrator", methods=["GET", "POST"])
    def orchestration():
        """
        Agent orchestration endpoint.

        GET: Returns orchestrator status and documentation.
        POST: Create tasks, check status, or manage agent coordination.

        POST Body (JSON):
            - action: str (required) - "create_task", "get_task", "execute", "status", "analyze"
            - task_type: str (for create_task)
            - payload: dict (for create_task)
            - task_id: str (for get_task, execute)

        Returns:
            JSON with orchestration results.
        """
        if request.method == "GET":
            status = orchestrator.get_queue_status()
            agents = orchestrator.get_agent_status()
            return jsonify({
                "endpoint": "/api/orchestrator",
                "description": "ML-powered agent orchestration and task coordination",
                "methods": ["GET", "POST"],
                "queue_status": status,
                "registered_agents": agents.get("agents", []),
                "actions": {
                    "create_task": {
                        "description": "Create a new task",
                        "params": ["task_type", "payload", "priority"],
                    },
                    "get_task": {
                        "description": "Get task details",
                        "params": ["task_id"],
                    },
                    "execute": {
                        "description": "Execute a specific task",
                        "params": ["task_id"],
                    },
                    "process_queue": {
                        "description": "Process pending tasks",
                        "params": ["max_tasks"],
                    },
                    "status": {
                        "description": "Get queue and agent status",
                        "params": [],
                    },
                    "analyze": {
                        "description": "Analyze workload distribution",
                        "params": [],
                    },
                    "handoff": {
                        "description": "Hand off task to specific agent",
                        "params": ["task_id", "target_agent_id", "context"],
                    },
                },
            })

        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "Request body required"}), 400

            action = data.get("action")
            if not action:
                return jsonify({"success": False, "error": "Action is required"}), 400

            if action == "create_task":
                priority_str = data.get("priority", "MEDIUM").upper()
                priority = getattr(TaskPriority, priority_str, TaskPriority.MEDIUM)
                result = orchestrator.create_task(
                    task_type=data.get("task_type", ""),
                    payload=data.get("payload", {}),
                    priority=priority,
                    depends_on=data.get("depends_on"),
                    context=data.get("context"),
                )
            elif action == "get_task":
                result = orchestrator.get_task(data.get("task_id", ""))
            elif action == "execute":
                result = orchestrator.execute_task(data.get("task_id", ""))
            elif action == "process_queue":
                result = orchestrator.process_queue(data.get("max_tasks"))
            elif action == "status":
                queue = orchestrator.get_queue_status()
                agents = orchestrator.get_agent_status()
                result = {
                    "success": True,
                    "queue": queue,
                    "agents": agents,
                }
            elif action == "analyze":
                result = orchestrator.analyze_workload()
            elif action == "handoff":
                result = orchestrator.handoff_task(
                    task_id=data.get("task_id", ""),
                    target_agent_id=data.get("target_agent_id", ""),
                    additional_context=data.get("context"),
                )
            elif action == "cancel":
                result = orchestrator.cancel_task(data.get("task_id", ""))
            else:
                return jsonify({"success": False, "error": f"Unknown action: {action}"}), 400

            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 400

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        return jsonify({
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/api/water", "/api/image", "/api/art", "/api/orchestrator"],
        }), 404

    @app.errorhandler(500)
    def internal_error(e):
        """Handle 500 errors."""
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
        }), 500

    return app


# Create the app instance for Gunicorn
app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
