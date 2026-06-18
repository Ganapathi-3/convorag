# ConvoRAG – Conversation Intelligence Chatbot

A RAG-powered chatbot that analyzes conversation history, detects topic shifts, builds semantic indexes, extracts a user persona, and answers natural language queries about the user.

---

## Architecture

```
conversations.csv
       │
       ▼
data_loader.py         ← Parse CSV → chronological Message objects
       │
       ├──▶ checkpoints.py    ← Build topic + 100-msg checkpoints
       │         │
       │         ├── Topic checkpoints (keyword-based topic shift detection)
       │         └── 100-message checkpoints (sliding window)
       │
       ├──▶ rag_engine.py     ← Embed summaries → FAISS index
       │         └── SentenceTransformer (all-MiniLM-L6-v2)
       │
       ├──▶ persona_extractor.py  ← Regex signal extraction → JSON persona
       │
       └──▶ chatbot.py        ← Intent detection + RAG retrieval + answer gen
                  │
                  ▼
             app.py           ← Flask REST API + web UI
```

---

## How Topic Change Detection Works

Topic detection uses a **sliding keyword-frequency window**:

1. Messages are processed **chronologically** (in conversation order).
2. Every **10 messages**, a window of recent text is analyzed.
3. The text is matched against 10 topic categories (food, travel, work, relationships, fitness, hobbies, technology, cars, shopping, general) using keyword dictionaries.
4. The category with the most keyword hits wins.
5. If the winning topic **differs from the current topic**, a **topic boundary is created**.
6. The previous segment is finalized with its message range and an extractive summary.

This produces output like:
```
Topic: travel          | msgs 0–89    | 90 messages  | "Moving to Portland, Oregon..."
Topic: general_chat    | msgs 90–179  | 90 messages  | "How are you, good..."
Topic: work_career     | msgs 180–279 | 100 messages | "Studying radiology, fulltime student..."
```

**Why this approach?** It is deterministic, fast, and works without LLM API calls, making it suitable for 190K+ messages without latency or cost issues.

---

## How Retrieval Works

The RAG engine builds **two FAISS indexes**:

### 1. Topic Index
- Each entry = a topic segment summary (e.g. "Travel (90 msgs): Moving to Portland, Oregon to pursue culinary dreams...")
- Encoded with `all-MiniLM-L6-v2` (384-dim vectors)
- L2-normalized → cosine similarity via inner product

### 2. Chunk Index  
- Each entry = a 100-message block summary
- Same encoding

### Query Flow
```
User Query
    │
    ▼ encode with SentenceTransformer
Query Vector (384-dim)
    │
    ├──▶ FAISS search Topic Index → top-3 topic segments
    └──▶ FAISS search Chunk Index → top-3 message blocks
              │
              ▼
       Intent detection (regex)
              │
       ┌──────┴──────┐
       ▼             ▼
  Persona answer   RAG-grounded answer
  (structured)     (with citations)
```

---

## How Persona Is Built

The persona extractor scans all **User 1** messages with:

| Layer | Method | Output |
|-------|--------|--------|
| **Habits** | 15 regex pattern groups (coffee_lover, late_sleeper, fitness_enthusiast, etc.) | Confidence % + evidence count |
| **Personal Facts** | 10 pattern groups (student, employed, has_pet, musician, etc.) | Confidence % |
| **Personality Traits** | 12 pattern groups (humorous, empathetic, curious, adventurous, etc.) | Confidence % |
| **Communication Style** | Computed stats: avg words/msg, emoji count, slang frequency, exclamation rate, question rate, formality markers | Categorical labels |
| **Interests** | Topic frequency from topic detection across all User 1 messages | Top-8 topics with counts |

All signals are evidence-based (regex matches on real text), never guessed.

**Output:** `persona/persona.json`

---

## Setup & Running

### Option 1: Local

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/convorag
cd convorag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data file
# Put conversations.csv in: data/conversations.csv

# 4. Build the indexes (one-time, ~2-5 minutes for 190K messages)
python build_index.py

# 5. Start the server
python app.py

# 6. Open browser
open http://localhost:5000
```

### Option 2: Docker

```bash
docker build -t convorag .
docker run -p 5000:5000 convorag
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web chatbot UI |
| POST | `/api/chat` | `{"query": "..."}` → answer |
| GET | `/api/persona` | Full persona JSON |
| GET | `/api/checkpoints` | Checkpoint summaries |
| POST | `/api/retrieve` | Raw RAG retrieval results |

---

## Example Questions

- "What kind of person is this user?"
- "What are their habits?"
- "How do they talk?"
- "What do they say about relationships?"
- "Are they a student or working?"
- "What topics come up most in their conversations?"

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Index | FAISS (IndexFlatIP) |
| Backend | Flask + Flask-CORS |
| Frontend | Vanilla HTML/CSS/JS |
| Persona | Regex pattern matching |
| Topic Detection | Keyword frequency windowing |

---

## Project Structure

```
convorag/
├── data/
│   └── conversations.csv       ← Input data
├── checkpoints/                ← Auto-generated FAISS indexes
│   ├── topic.index
│   ├── chunk.index
│   ├── topic_checkpoints.json
│   └── hundred_checkpoints.json
├── persona/
│   └── persona.json            ← Extracted user persona
├── static/
│   └── index.html              ← Web UI
├── data_loader.py              ← CSV parser
├── checkpoints.py              ← Topic + 100-msg checkpoint builder
├── rag_engine.py               ← FAISS embedding + retrieval
├── persona_extractor.py        ← Persona signal extraction
├── chatbot.py                  ← Query handler
├── app.py                      ← Flask server
├── build_index.py              ← One-time build script
├── requirements.txt
├── Dockerfile
└── README.md
```
