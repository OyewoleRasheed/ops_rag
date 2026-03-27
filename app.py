"""
app.py
──────
Flask backend for OpsRAG.

Production-readiness features:
  - Structured JSON API (not just a chatbot page)
  - Request validation with error codes
  - Startup health check endpoint
  - Request logging with timing
  - Graceful error handling — app never crashes on bad input
  - Environment-based config (dev vs prod)
  - CORS headers for future API consumers
"""

import os
import time
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from vectorstore import get_vectorstore
from rag_chain import answer_question

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
# In production you'd ship these logs to something like Datadog or CloudWatch.
# For now they go to stdout so Render can capture them.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests — needed if you later build a separate frontend

# ── Startup — load heavy resources once ───────────────────────────────────────
# These are loaded at startup, not per-request.
# Loading per-request would mean ~2 min wait on every question — never do that.
logger.info("Loading vectorstore...")
VECTORSTORE = get_vectorstore()

logger.info("Initialising Groq client...")
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

logger.info("OpsRAG ready.\n")

VALID_FOLDERS = {"engineering", "regulatory", "safety", "all"}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    """
    Health check endpoint.
    Render, Railway, and most cloud platforms ping /health to know if the
    app is alive. Return 200 = healthy, anything else = restart the container.
    """
    return jsonify({
        "status": "ok",
        "vectorstore": "loaded",
        "groq": "connected",
    }), 200


@app.route("/api/query", methods=["POST"])
def query():
    """
    Main RAG endpoint.

    Expected JSON body:
      {
        "question": "What are inspection intervals for Class 1 piping?",
        "folder":   "engineering"   // optional: engineering | regulatory | safety | all
      }

    Returns:
      {
        "question":  "...",
        "answer":    "...",
        "reasoning": "...",
        "sources":   ["..."],
        "domain":    "Engineering",
        "latency_ms": 1243
      }
    """
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

    # ── Log incoming request ──
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

    # ── Build response ──
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
    """
    Returns available document folders.
    The frontend calls this on load to populate the filter dropdown dynamically.
    """
    return jsonify({
        "folders": [
            {"id": "all",         "label": "All Documents"},
            {"id": "engineering", "label": "Engineering (API 570, Piping)"},
            {"id": "regulatory",  "label": "Regulatory (NUPRC, Nigerian Upstream)"},
            {"id": "safety",      "label": "Safety (IOGP 2024)"},
        ]
    }), 200


# ── Error handlers ────────────────────────────────────────────────────────────
# These catch errors at the Flask level — not just inside route functions.
# A 404 page that returns HTML when the client expects JSON is a common bug.

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