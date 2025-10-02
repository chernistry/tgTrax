import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# Minimal Flask app hosting only the JSON API for Next.js UI
app = Flask(__name__)

# CORS: allow all in dev; restrict via CORS_ORIGINS/FRONTEND_ORIGIN in prod
origins_env = os.getenv('CORS_ORIGINS') or os.getenv('FRONTEND_ORIGIN') or '*'
origins = [o.strip() for o in origins_env.split(',')] if origins_env != '*' else '*'
CORS(app, resources={r"/api/*": {"origins": origins}})

# Register API blueprint
try:
    from tgTrax.web.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to register API blueprint: {e}")


@app.get('/')
def root():
    # UI lives at Next.js on a separate port; advertise API status
    return jsonify({
        "ok": True,
        "msg": "tgTrax API is running",
        "endpoints": [
            "/api/health",
            "/api/summary",
            "/api/pairs/significant",
            "/api/matrices",
            "/api/graph/combined",
            "/api/stack/*",
            "/api/auth/*"
        ]
    })

