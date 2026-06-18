"""
persona_extractor.py - Extract structured user persona from conversations
"""
import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any
from data_loader import Message


# ── Signal patterns ────────────────────────────────────────────────────────────

HABIT_PATTERNS = {
    "late_sleeper": [r"stay up late", r"up all night", r"can't sleep", r"insomnia", r"night owl", r"sleep late", r"tired all the time", r"wake up late", r"sleep in"],
    "early_riser": [r"wake up early", r"morning person", r"early bird", r"up at \d am", r"up at dawn"],
    "coffee_lover": [r"\bcoffee\b", r"espresso", r"latte", r"caffeine"],
    "tea_drinker": [r"\btea\b", r"chai", r"herbal tea"],
    "foodie": [r"love (?:to )?eat", r"favorite food", r"trying new restaurants", r"love cooking", r"food lover", r"foodie"],
    "fitness_enthusiast": [r"\bgym\b", r"work ?out", r"exercise", r"run(?:ning)?", r"yoga", r"fitness"],
    "gamer": [r"\bgam(?:e|ing)\b", r"video game", r"play(?:ing)? games", r"xbox", r"playstation", r"nintendo"],
    "reader": [r"\bread(?:ing)?\b", r"book(?:s)?\b", r"library", r"novel", r"literature"],
    "music_listener": [r"listen to music", r"favorite (?:band|song|artist)", r"music lover", r"playlist"],
    "traveler": [r"love to travel", r"been to", r"visited", r"traveling", r"trip to", r"vacation"],
    "smoker": [r"\bsmok(?:e|ing)\b", r"cigarette", r"vape"],
    "drinker": [r"\bdrink(?:ing)?\b alcohol", r"\bbeer\b", r"\bwine\b", r"go out for drinks"],
    "social_media_user": [r"instagram", r"twitter", r"tiktok", r"social media", r"post(?:ed)?"],
    "cook": [r"love (?:to )?cook", r"made (?:dinner|lunch|breakfast)", r"baking", r"recipe"],
    "procrastinator": [r"procrastinat", r"last minute", r"keep putting off", r"never get around to"],
}

PERSONALITY_PATTERNS = {
    "humorous": [r"haha", r"lol", r"lmao", r"😂", r"joke", r"funny", r"humor(?:ous)?"],
    "empathetic": [r"i understand", r"that must be hard", r"i'm sorry", r"feel for you", r"i feel you"],
    "curious": [r"i wonder", r"what do you think", r"how does", r"why do", r"tell me more", r"interesting"],
    "optimistic": [r"i'm sure it will", r"look on the bright side", r"it'll work out", r"positive", r"excited about"],
    "anxious_worried": [r"i'm worried", r"i'm nervous", r"stress(?:ed)?", r"anxious", r"overwhelm(?:ed)?"],
    "direct": [r"to be honest", r"honestly", r"straight up", r"i'll be real"],
    "romantic": [r"i love you", r"you mean a lot", r"my heart", r"miss you", r"can't stop thinking about you"],
    "intellectual": [r"philosophically", r"in theory", r"academically", r"literature", r"science", r"research"],
    "adventurous": [r"try new things", r"adventure", r"spontaneous", r"let's do it", r"take a risk"],
    "introverted": [r"stay home", r"prefer to be alone", r"not a people person", r"quiet night in", r"hate crowds"],
    "extroverted": [r"love hanging out", r"party", r"meet new people", r"social", r"love being around people"],
}

PERSONAL_FACT_PATTERNS = {
    "student": [r"\bstudent\b", r"studying\b", r"college\b", r"university\b", r"degree\b", r"class(?:es)?\b"],
    "employed": [r"\bjob\b", r"work(?:ing)? (?:at|for)\b", r"my (?:boss|colleague|coworker)"],
    "has_pet": [r"my (?:dog|cat|pet|puppy|kitten)", r"pet (?:dog|cat)", r"love animals"],
    "has_sibling": [r"my (?:brother|sister|sibling)", r"older (?:bro|sis)", r"younger (?:bro|sis)"],
    "has_partner": [r"my (?:boyfriend|girlfriend|partner|husband|wife|fiance)", r"we've been dating"],
    "has_children": [r"my (?:son|daughter|kid|child|baby)", r"being a (?:mom|dad|parent)"],
    "lives_alone": [r"live (?:alone|by myself)", r"my own (?:apartment|place)"],
    "lives_with_family": [r"live with (?:my|parents|family)", r"at home with"],
    "car_owner": [r"my (?:car|truck|vehicle|impala|honda|toyota)", r"drive to work", r"parking"],
    "musician": [r"play (?:guitar|piano|drums|bass|violin|music)", r"in a band", r"my band"],
}

COMMUNICATION_STYLE_SIGNALS = {
    "emoji_heavy": r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U00002702-\U000027B0]",
    "uses_slang": r"\b(gonna|wanna|gotta|kinda|sorta|ya|nah|yep|yup|tbh|idk|imo|lol|lmao|omg|bruh|dude)\b",
    "formal_language": r"\b(therefore|furthermore|consequently|nevertheless|regarding|henceforth)\b",
    "short_messages": None,  # computed from avg word count
    "long_messages": None,
    "uses_exclamations": r"!{1,}",
    "uses_questions": r"\?{1,}",
    "all_caps": r"\b[A-Z]{3,}\b",
    "ellipsis_user": r"\.{3,}",
}


def extract_persona(messages: List[Message]) -> Dict[str, Any]:
    """Extract a structured persona from all messages attributed to User 1."""
    # Focus on User 1 as "the user"
    user_msgs = [m for m in messages if m.speaker == "User 1"]
    all_user_text = " ".join(m.text for m in user_msgs).lower()
    full_text_raw = " ".join(m.text for m in user_msgs)

    persona = {
        "habits": {},
        "personal_facts": {},
        "personality_traits": {},
        "communication_style": {},
        "topics_of_interest": {},
        "raw_signals": defaultdict(list),
    }

    # ── Habits ────────────────────────────────────────────────────────────────
    for habit, patterns in HABIT_PATTERNS.items():
        matches = []
        for p in patterns:
            found = re.findall(p, all_user_text, re.IGNORECASE)
            matches.extend(found)
        if matches:
            persona["habits"][habit] = {
                "confidence": min(1.0, len(matches) / 3),
                "evidence_count": len(matches),
            }

    # ── Personal facts ────────────────────────────────────────────────────────
    for fact, patterns in PERSONAL_FACT_PATTERNS.items():
        matches = []
        for p in patterns:
            found = re.findall(p, all_user_text, re.IGNORECASE)
            matches.extend(found)
        if matches:
            persona["personal_facts"][fact] = {
                "confidence": min(1.0, len(matches) / 2),
                "evidence_count": len(matches),
            }

    # ── Personality traits ────────────────────────────────────────────────────
    for trait, patterns in PERSONALITY_PATTERNS.items():
        matches = []
        for p in patterns:
            found = re.findall(p, full_text_raw, re.IGNORECASE)
            matches.extend(found)
        if matches:
            persona["personality_traits"][trait] = {
                "confidence": min(1.0, len(matches) / 3),
                "evidence_count": len(matches),
            }

    # ── Communication style ───────────────────────────────────────────────────
    word_counts = [len(m.text.split()) for m in user_msgs]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    persona["communication_style"]["avg_message_length_words"] = round(avg_words, 1)
    persona["communication_style"]["message_count"] = len(user_msgs)

    if avg_words < 8:
        persona["communication_style"]["message_length_style"] = "very short"
    elif avg_words < 15:
        persona["communication_style"]["message_length_style"] = "concise"
    elif avg_words < 30:
        persona["communication_style"]["message_length_style"] = "moderate"
    else:
        persona["communication_style"]["message_length_style"] = "verbose"

    # Check specific style signals
    emoji_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["emoji_heavy"], full_text_raw))
    slang_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["uses_slang"], full_text_raw, re.IGNORECASE))
    formal_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["formal_language"], full_text_raw, re.IGNORECASE))
    exclamation_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["uses_exclamations"], full_text_raw))
    question_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["uses_questions"], full_text_raw))
    caps_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["all_caps"], full_text_raw))
    ellipsis_count = len(re.findall(COMMUNICATION_STYLE_SIGNALS["ellipsis_user"], full_text_raw))

    n = len(user_msgs) or 1
    persona["communication_style"]["emoji_usage"] = "heavy" if emoji_count / n > 0.5 else ("moderate" if emoji_count / n > 0.1 else "rare")
    persona["communication_style"]["slang_usage"] = "frequent" if slang_count / n > 1 else ("occasional" if slang_count / n > 0.2 else "rare")
    persona["communication_style"]["formality"] = "formal" if formal_count > 5 else "informal"
    persona["communication_style"]["exclamation_rate"] = round(exclamation_count / n, 2)
    persona["communication_style"]["question_rate"] = round(question_count / n, 2)
    persona["communication_style"]["caps_usage"] = "frequent" if caps_count / n > 0.5 else "occasional" if caps_count / n > 0.1 else "rare"
    persona["communication_style"]["ellipsis_usage"] = "frequent" if ellipsis_count / n > 0.3 else "rare"

    if exclamation_count / n > 0.8:
        persona["communication_style"]["tone"] = "enthusiastic"
    elif question_count / n > 0.8:
        persona["communication_style"]["tone"] = "inquisitive"
    elif formal_count > 5:
        persona["communication_style"]["tone"] = "formal"
    else:
        persona["communication_style"]["tone"] = "casual"

    # ── Topic interests (from topic checkpoint word frequency) ─────────────────
    from checkpoints import detect_topic, TOPIC_KEYWORDS
    topic_counter = Counter()
    for m in user_msgs:
        t = detect_topic(m.text)
        if t != "general_chat":
            topic_counter[t] += 1
    persona["topics_of_interest"] = dict(topic_counter.most_common(8))

    # ── Remove internal raw_signals from output ────────────────────────────────
    del persona["raw_signals"]

    # ── Summary paragraph ─────────────────────────────────────────────────────
    persona["summary"] = _generate_persona_summary(persona)

    return persona


def _generate_persona_summary(p: Dict) -> str:
    parts = []

    # Personality
    traits = [t for t, v in p["personality_traits"].items() if v["confidence"] > 0.3]
    if traits:
        parts.append(f"Personality: {', '.join(traits[:4])}")

    # Habits
    habits = [h for h, v in p["habits"].items() if v["confidence"] > 0.2]
    if habits:
        parts.append(f"Habits: {', '.join(habits[:5])}")

    # Facts
    facts = [f for f, v in p["personal_facts"].items() if v["confidence"] > 0.3]
    if facts:
        parts.append(f"Facts: {', '.join(facts[:5])}")

    # Communication
    cs = p["communication_style"]
    style_desc = f"{cs.get('tone', 'casual')} tone, {cs.get('message_length_style', 'moderate')} messages"
    if cs.get("slang_usage") == "frequent":
        style_desc += ", uses slang"
    if cs.get("emoji_usage") in ("heavy", "moderate"):
        style_desc += ", uses emojis"
    parts.append(f"Communication: {style_desc}")

    # Interests
    interests = list(p.get("topics_of_interest", {}).keys())[:4]
    if interests:
        parts.append(f"Interests: {', '.join(i.replace('_', ' ') for i in interests)}")

    return " | ".join(parts)


if __name__ == "__main__":
    from data_loader import load_messages
    msgs = load_messages("data/conversations.csv")
    print(f"Extracting persona from {len(msgs)} messages...")
    persona = extract_persona(msgs)
    print(json.dumps(persona, indent=2))
