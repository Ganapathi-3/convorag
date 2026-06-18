"""
checkpoints.py - Build topic checkpoints and 100-message checkpoints
"""
import json
import re
from typing import List, Dict, Any
from data_loader import Message

# ── Simple keyword-based topic detection ──────────────────────────────────────
TOPIC_KEYWORDS = {
    "food_cooking": ["cook", "food", "eat", "recipe", "restaurant", "kitchen", "meal", "dinner", "lunch", "breakfast", "culinary", "dish", "chef", "bake", "grill"],
    "travel": ["travel", "trip", "visit", "city", "country", "move", "moving", "flight", "hotel", "vacation", "holiday", "abroad", "abroad", "portland", "tokyo", "london"],
    "work_career": ["job", "work", "career", "office", "boss", "colleague", "salary", "promotion", "company", "business", "startup", "profession", "study", "college", "university", "degree"],
    "relationships": ["friend", "family", "girlfriend", "boyfriend", "partner", "wife", "husband", "parents", "mom", "dad", "brother", "sister", "love", "date", "dating", "marriage"],
    "health_fitness": ["gym", "exercise", "run", "yoga", "workout", "health", "diet", "sleep", "tired", "sick", "doctor", "stress", "mental", "anxiety", "meditation"],
    "hobbies_entertainment": ["music", "band", "movie", "game", "book", "read", "hobby", "sport", "play", "guitar", "piano", "concert", "art", "draw", "paint", "photography"],
    "technology": ["phone", "computer", "app", "software", "code", "program", "tech", "ai", "internet", "social media", "twitter", "instagram", "tiktok"],
    "cars_vehicles": ["car", "drive", "vehicle", "impala", "truck", "road trip", "mechanic", "engine", "paint job"],
    "shopping_money": ["buy", "shop", "money", "price", "expensive", "cheap", "afford", "budget", "spend", "earn"],
    "general_chat": ["how are you", "good", "great", "nice", "cool", "thanks", "hello", "hi", "hey", "bye"],
}


def detect_topic(text: str) -> str:
    """Detect the dominant topic in a block of text using keyword matching."""
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "general_chat"
    return max(scores, key=scores.get)


def build_checkpoints(messages: List[Message], window: int = 10) -> Dict[str, Any]:
    """
    Build two kinds of checkpoints:
    1. Topic checkpoints: detect when topic shifts, split message range
    2. 100-message checkpoints: every 100 messages regardless of topic

    Returns a dict with both checkpoint lists.
    """
    topic_checkpoints = []
    hundred_checkpoints = []

    # ── Topic checkpoints ─────────────────────────────────────────────────────
    current_topic = None
    topic_start = 0
    topic_msgs = []

    for i, msg in enumerate(messages):
        # Evaluate topic every `window` messages
        if i % window == 0 or i == len(messages) - 1:
            window_text = " ".join(m.text for m in messages[max(0, i - window):i + 1])
            new_topic = detect_topic(window_text)

            if new_topic != current_topic:
                # Save previous segment
                if current_topic is not None and topic_msgs:
                    topic_checkpoints.append({
                        "topic": current_topic,
                        "start_global": topic_msgs[0].global_index,
                        "end_global": topic_msgs[-1].global_index,
                        "start_conv": topic_msgs[0].conv_index,
                        "end_conv": topic_msgs[-1].conv_index,
                        "message_count": len(topic_msgs),
                        "messages": [{"speaker": m.speaker, "text": m.text} for m in topic_msgs],
                        "summary": summarize_segment(topic_msgs, current_topic),
                    })
                current_topic = new_topic
                topic_msgs = []

        topic_msgs.append(msg)

    # flush last segment
    if topic_msgs and current_topic:
        topic_checkpoints.append({
            "topic": current_topic,
            "start_global": topic_msgs[0].global_index,
            "end_global": topic_msgs[-1].global_index,
            "start_conv": topic_msgs[0].conv_index,
            "end_conv": topic_msgs[-1].conv_index,
            "message_count": len(topic_msgs),
            "messages": [{"speaker": m.speaker, "text": m.text} for m in topic_msgs],
            "summary": summarize_segment(topic_msgs, current_topic),
        })

    # ── 100-message checkpoints ───────────────────────────────────────────────
    for i in range(0, len(messages), 100):
        chunk = messages[i: i + 100]
        hundred_checkpoints.append({
            "checkpoint_num": i // 100 + 1,
            "start_global": chunk[0].global_index,
            "end_global": chunk[-1].global_index,
            "message_count": len(chunk),
            "messages": [{"speaker": m.speaker, "text": m.text} for m in chunk],
            "summary": summarize_segment(chunk, "100-message block"),
        })

    return {
        "topic_checkpoints": topic_checkpoints,
        "hundred_checkpoints": hundred_checkpoints,
    }


def summarize_segment(msgs: List[Message], topic: str) -> str:
    """
    Generate a concise extractive summary of a message segment.
    Uses key sentences and frequency-based keyword extraction.
    """
    if not msgs:
        return ""

    all_text = " ".join(m.text for m in msgs)
    sentences = re.split(r"[.!?]+", all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # Score sentences by keyword overlap with topic
    topic_kws = TOPIC_KEYWORDS.get(topic, [])
    scored = []
    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for kw in topic_kws if kw in s_lower)
        # Penalize very short sentences
        score += len(s.split()) * 0.05
        scored.append((score, s))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:4]]

    speakers = list({m.speaker for m in msgs})
    summary = (
        f"[{topic.replace('_', ' ').title()}] "
        f"({len(msgs)} messages, convs {msgs[0].conv_index}–{msgs[-1].conv_index}): "
        + " | ".join(top[:3])
    )
    return summary[:600]  # cap length


if __name__ == "__main__":
    from data_loader import load_messages
    msgs = load_messages("data/conversations.csv")
    print(f"Building checkpoints for {len(msgs)} messages...")
    cp = build_checkpoints(msgs[:500])  # test on first 500
    print(f"Topic checkpoints: {len(cp['topic_checkpoints'])}")
    print(f"100-msg checkpoints: {len(cp['hundred_checkpoints'])}")
    for t in cp["topic_checkpoints"][:5]:
        print(f"  Topic: {t['topic']} | msgs {t['start_global']}-{t['end_global']}")
        print(f"  Summary: {t['summary'][:120]}")
