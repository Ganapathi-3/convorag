"""
app.py - Flask chatbot server
"""
import json
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from rag_engine import RAGEngine
from chatbot import Chatbot

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Load indexes at startup ────────────────────────────────────────────────────
print("Loading RAG indexes and persona...")
rag = RAGEngine()
rag.load("checkpoints")

with open("persona/persona.json") as f:
    persona = json.load(f)

bot = Chatbot(rag, persona)
print("✅ Chatbot ready!")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    answer = bot.answer(query)
    return jsonify({"answer": answer, "query": query})


@app.route("/api/persona", methods=["GET"])
def get_persona():
    return jsonify(persona)


@app.route("/api/checkpoints", methods=["GET"])
def get_checkpoints():
    with open("checkpoints/topic_checkpoints.json") as f:
        topics = json.load(f)
    with open("checkpoints/hundred_checkpoints.json") as f:
        hundreds = json.load(f)
    return jsonify({
        "topic_count": len(topics),
        "hundred_count": len(hundreds),
        "topic_sample": topics[:10],
        "hundred_sample": hundreds[:5],
    })


@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 3))
    if not query:
        return jsonify({"error": "Empty query"}), 400
    results = rag.retrieve(query, top_k=top_k)
    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
