"""
SenaAIgent API - Flask application with endpoints for ML analytics and image generation.
"""

import os
from flask import Flask, jsonify, request

from agents import ModelAgent, ImageAgent


def create_app():
    """Application factory for the Flask app."""
    app = Flask(__name__)

    # Initialize agents
    model_agent = ModelAgent()
    image_agent = ImageAgent()

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

    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        return jsonify({
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/api/water", "/api/image"],
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
