"""
chatbot.py - Query handler combining RAG retrieval and persona data
"""
import json
import re
from typing import Dict, Any
from rag_engine import RAGEngine


# ── Intent detection ───────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "persona_overview": [r"what kind of person", r"who is this", r"describe.*user", r"tell me about.*user"],
    "habits": [r"habit", r"routine", r"what.*they do", r"how.*they spend", r"what.*they like to do"],
    "communication_style": [r"how.*talk", r"communication", r"speak", r"write", r"message style", r"how.*they type"],
    "personality": [r"personalit", r"character", r"trait", r"nature", r"like as a person", r"kind of person"],
    "personal_facts": [r"personal fact", r"background", r"family", r"relationship", r"job", r"work", r"live"],
    "interests": [r"interest", r"topic", r"care about", r"passionate about", r"hobby", r"hobbies"],
    "topic_query": [],  # fallback — search RAG for topic content
}


def detect_intent(query: str) -> str:
    q_lower = query.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        if intent == "topic_query":
            continue
        for p in patterns:
            if re.search(p, q_lower):
                return intent
    return "topic_query"


class Chatbot:
    def __init__(self, rag: RAGEngine, persona: Dict[str, Any]):
        self.rag = rag
        self.persona = persona

    def answer(self, query: str) -> str:
        intent = detect_intent(query)

        if intent == "persona_overview":
            return self._answer_persona_overview()
        elif intent == "habits":
            return self._answer_habits()
        elif intent == "communication_style":
            return self._answer_communication()
        elif intent == "personality":
            return self._answer_personality()
        elif intent == "personal_facts":
            return self._answer_personal_facts()
        elif intent == "interests":
            return self._answer_interests()
        else:
            return self._answer_rag(query)

    # ── Persona-based answers ──────────────────────────────────────────────────

    def _answer_persona_overview(self) -> str:
        s = self.persona.get("summary", "")
        traits = list(self.persona.get("personality_traits", {}).keys())[:5]
        habits = list(self.persona.get("habits", {}).keys())[:5]
        interests = list(self.persona.get("topics_of_interest", {}).keys())[:5]
        cs = self.persona.get("communication_style", {})

        lines = [
            "## User Persona Overview",
            "",
            f"**Summary:** {s}",
            "",
            f"**Top personality traits:** {', '.join(traits) if traits else 'Not enough data'}",
            f"**Key habits:** {', '.join(h.replace('_', ' ') for h in habits) if habits else 'Not enough data'}",
            f"**Main interests:** {', '.join(i.replace('_', ' ') for i in interests) if interests else 'Not enough data'}",
            f"**Tone:** {cs.get('tone', 'N/A')} | **Message style:** {cs.get('message_length_style', 'N/A')}",
        ]
        return "\n".join(lines)

    def _answer_habits(self) -> str:
        habits = self.persona.get("habits", {})
        if not habits:
            return "No clear habit signals found in the conversations."

        sorted_habits = sorted(habits.items(), key=lambda x: -x[1]["confidence"])
        lines = ["## User Habits", ""]
        for habit, data in sorted_habits:
            conf_pct = int(data["confidence"] * 100)
            lines.append(f"- **{habit.replace('_', ' ').title()}** (confidence: {conf_pct}%, signals: {data['evidence_count']})")
        return "\n".join(lines)

    def _answer_communication(self) -> str:
        cs = self.persona.get("communication_style", {})
        lines = [
            "## Communication Style",
            "",
            f"- **Tone:** {cs.get('tone', 'N/A')}",
            f"- **Message length:** {cs.get('message_length_style', 'N/A')} (avg {cs.get('avg_message_length_words', 0)} words)",
            f"- **Emoji usage:** {cs.get('emoji_usage', 'N/A')}",
            f"- **Slang usage:** {cs.get('slang_usage', 'N/A')}",
            f"- **Formality:** {cs.get('formality', 'N/A')}",
            f"- **Exclamation rate:** {cs.get('exclamation_rate', 0)} per message",
            f"- **Question rate:** {cs.get('question_rate', 0)} per message",
            f"- **Caps usage:** {cs.get('caps_usage', 'N/A')}",
            f"- **Ellipsis usage:** {cs.get('ellipsis_usage', 'N/A')}",
            f"- **Total messages analyzed:** {cs.get('message_count', 0)}",
        ]
        return "\n".join(lines)

    def _answer_personality(self) -> str:
        traits = self.persona.get("personality_traits", {})
        if not traits:
            return "No strong personality signals detected."

        sorted_traits = sorted(traits.items(), key=lambda x: -x[1]["confidence"])
        lines = ["## Personality Traits", ""]
        for trait, data in sorted_traits:
            conf_pct = int(data["confidence"] * 100)
            lines.append(f"- **{trait.replace('_', ' ').title()}** (confidence: {conf_pct}%)")
        return "\n".join(lines)

    def _answer_personal_facts(self) -> str:
        facts = self.persona.get("personal_facts", {})
        if not facts:
            return "No personal facts confidently detected."

        sorted_facts = sorted(facts.items(), key=lambda x: -x[1]["confidence"])
        lines = ["## Personal Facts", ""]
        for fact, data in sorted_facts:
            conf_pct = int(data["confidence"] * 100)
            lines.append(f"- **{fact.replace('_', ' ').title()}** (confidence: {conf_pct}%)")
        return "\n".join(lines)

    def _answer_interests(self) -> str:
        interests = self.persona.get("topics_of_interest", {})
        if not interests:
            return "No strong topic interests detected."

        lines = ["## Topics of Interest", ""]
        for topic, count in sorted(interests.items(), key=lambda x: -x[1]):
            lines.append(f"- **{topic.replace('_', ' ').title()}** (mentioned in ~{count} messages)")
        return "\n".join(lines)

    # ── RAG-based answer ───────────────────────────────────────────────────────

    def _answer_rag(self, query: str) -> str:
        results = self.rag.retrieve(query, top_k=3)

        topic_hits = results.get("topic_results", [])
        chunk_hits = results.get("chunk_results", [])

        if not topic_hits and not chunk_hits:
            return "I couldn't find relevant information in the conversation history for that query."

        lines = [f"## Answer for: *{query}*", ""]

        if topic_hits:
            lines.append("### Relevant Topic Segments")
            for hit in topic_hits:
                score_pct = int(hit.get("score", 0) * 100)
                lines.append(f"**{hit.get('topic', '').replace('_', ' ').title()}** (relevance: {score_pct}%)")
                lines.append(f"> {hit.get('summary', '')}")
                lines.append("")

        if chunk_hits:
            lines.append("### Relevant Message Blocks")
            for hit in chunk_hits:
                score_pct = int(hit.get("score", 0) * 100)
                lines.append(f"**Block #{hit.get('checkpoint_num', '?')}** msgs {hit.get('start_global', 0)}–{hit.get('end_global', 0)} (relevance: {score_pct}%)")
                lines.append(f"> {hit.get('summary', '')}")
                lines.append("")

        return "\n".join(lines)
