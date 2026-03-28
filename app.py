"""
app.py
──────
Flask backend for OpsRAG.

Production-readiness features:
  - Port binds immediately — vectorstore loads in background thread
  - Structured JSON API with proper HTTP status codes
  - /health endpoint for Render's port scanner
  - Request validation with clear error messages
  - Graceful 503 while app is warming up
  - Request logging with timing
"""

import os
import time
import logging
import threading
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from vectorstore import get_vectorstore
from rag_chain import answer_question

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Lazy startup ──────────────────────────────────────────────────────────────
# WHY: Render scans for an open port immediately after starting gunicorn.
# If we load the vectorstore synchronously before binding, Render times out
# and kills the deploy — even though the app would eventually be ready.
#
# Fix: bind the port first (Flask does this instantly), then load resources
# in a background thread. Requests that arrive during loading get a 503
# with a "warming up" message instead of crashing.

VECTORSTORE  = None
GROQ_CLIENT  = None
_ready       = False   # flips to True once background loading finishes
_load_error  = None    # stores any startup error message


def _load_resources():
    global VECTORSTORE, GROQ_CLIENT, _ready, _load_error
    try:
        logger.info("Background: loading vectorstore...")
        VECTORSTORE = get_vectorstore()
        logger.info("Background: initialising Groq client...")
        GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
        _ready = True
        logger.info("OpsRAG ready — all resources loaded.")
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Failed to load resources: {e}", exc_info=True)


# Start loading in background — port binds immediately, this runs alongside
threading.Thread(target=_load_resources, daemon=True).start()

VALID_FOLDERS = {"engineering", "regulatory", "safety", "all"}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    """
    Health check endpoint — Render pings this to confirm the app is alive.
    Returns 200 immediately even while warming up, so Render does not kill
    the deploy during the vectorstore load.
    """
    if _load_error:
        return jsonify({"status": "error", "detail": _load_error}), 500

    if not _ready:
        return jsonify({"status": "warming_up", "detail": "Loading vectorstore..."}), 200

    return jsonify({
        "status":      "ok",
        "vectorstore": "loaded",
        "groq":        "connected",
    }), 200


@app.route("/api/query", methods=["POST"])
def query():
    """
    Main RAG endpoint.

    Expected JSON body:
      {
        "question": "What are inspection intervals for Class 1 piping?",
        "folder":   "all"   // engineering | regulatory | safety | all
      }
    """
    # Return 503 while resources are still loading
    if not _ready:
        return jsonify({
            "error": "App is warming up — please try again in 30 seconds."
        }), 503

    start_time = time.time()

    # ── Validate request ──
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data     = request.get_json()
    question = data.get("question", "").strip()
    folder   = data.get("folder", "all").strip().lower()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if len(question) > 1000:
        return jsonify({"error": "Question too long (max 1000 characters)"}), 400

    if folder not in VALID_FOLDERS:
        return jsonify({
            "error": f"Invalid folder '{folder}'. Must be one of: {', '.join(VALID_FOLDERS)}"
        }), 400

    folder_filter = None if folder == "all" else folder

    logger.info(f"Query | folder={folder} | question={question[:80]}...")

    # ── Run RAG pipeline ──
    try:
        result = answer_question(
            vectorstore=VECTORSTORE,
            client=GROQ_CLIENT,
            question=question,
            folder_filter=folder_filter,
        )
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        return jsonify({"error": "Internal error processing your question"}), 500

    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Response | domain={result['domain']} | latency={latency_ms}ms")

    return jsonify({
        "question":   result["question"],
        "answer":     result["answer"],
        "reasoning":  result["reasoning"],
        "sources":    result["sources"],
        "domain":     result["domain"],
        "latency_ms": latency_ms,
    }), 200


@app.route("/api/folders", methods=["GET"])
def folders():
    return jsonify({
        "folders": [
            {"id": "all",         "label": "All Documents"},
            {"id": "engineering", "label": "Engineering (API 570, Piping)"},
            {"id": "regulatory",  "label": "Regulatory (NUPRC, Nigerian Upstream)"},
            {"id": "safety",      "label": "Safety (IOGP 2024)"},
        ]
    }), 200


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled server error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_ENV", "production") == "development"
    port  = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)