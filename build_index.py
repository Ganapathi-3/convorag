"""
build_index.py - One-time script to process conversations and build all indexes
Run this once before starting the server.
"""
import os
import json
import sys
import time

print("=" * 60)
print("RAG Chatbot - Index Builder")
print("=" * 60)

# ── Step 1: Load messages ──────────────────────────────────────────────────────
print("\n[1/4] Loading messages...")
t0 = time.time()
from data_loader import load_messages
messages = load_messages("data/conversations.csv")
print(f"  Loaded {len(messages):,} messages in {time.time()-t0:.1f}s")

# ── Step 2: Build checkpoints ──────────────────────────────────────────────────
print("\n[2/4] Building checkpoints (topic + 100-msg)...")
t0 = time.time()
from checkpoints import build_checkpoints
checkpoints = build_checkpoints(messages)
n_topics = len(checkpoints["topic_checkpoints"])
n_chunks = len(checkpoints["hundred_checkpoints"])
print(f"  Topic checkpoints: {n_topics}")
print(f"  100-msg checkpoints: {n_chunks}")
print(f"  Done in {time.time()-t0:.1f}s")

# Save checkpoints
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/topic_checkpoints.json", "w") as f:
    # Save summaries without raw messages for smaller file
    slim = [{k: v for k, v in cp.items() if k != "messages"} for cp in checkpoints["topic_checkpoints"]]
    json.dump(slim, f, indent=2)
with open("checkpoints/hundred_checkpoints.json", "w") as f:
    slim = [{k: v for k, v in cp.items() if k != "messages"} for cp in checkpoints["hundred_checkpoints"]]
    json.dump(slim, f, indent=2)
print(f"  Checkpoints saved to checkpoints/")

# Print sample topic checkpoints
print("\n  Sample topic checkpoints:")
for cp in checkpoints["topic_checkpoints"][:5]:
    print(f"    Topic: {cp['topic']:25s} | msgs {cp['start_global']:6d}-{cp['end_global']:6d} | {cp['message_count']} msgs")

# ── Step 3: Build RAG index ────────────────────────────────────────────────────
print("\n[3/4] Building RAG embeddings + FAISS index...")
t0 = time.time()
from rag_engine import RAGEngine
rag = RAGEngine()
rag.index_checkpoints(checkpoints)
rag.save("checkpoints")
print(f"  FAISS indexes built in {time.time()-t0:.1f}s")

# ── Step 4: Extract persona ────────────────────────────────────────────────────
print("\n[4/4] Extracting user persona...")
t0 = time.time()
from persona_extractor import extract_persona
persona = extract_persona(messages)
os.makedirs("persona", exist_ok=True)
with open("persona/persona.json", "w") as f:
    json.dump(persona, f, indent=2)
print(f"  Persona extracted in {time.time()-t0:.1f}s")
print(f"\n  Persona Summary: {persona.get('summary', '')[:200]}")
print(f"\n  Top habits: {list(persona['habits'].keys())[:5]}")
print(f"  Top traits: {list(persona['personality_traits'].keys())[:5]}")
print(f"  Top facts: {list(persona['personal_facts'].keys())[:5]}")

print("\n" + "=" * 60)
print("✅ Build complete! Run: python app.py")
print("=" * 60)
