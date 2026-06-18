"""
data_loader.py - Parse conversations CSV into chronological messages
"""
import csv
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    global_index: int        # overall message # across all conversations
    conv_index: int          # which conversation (day)
    msg_index: int           # position within conversation
    speaker: str             # "User 1" or "User 2"
    text: str                # message text


def load_messages(csv_path: str) -> List[Message]:
    """Load all conversations in order, flatten into a chronological message list."""
    messages = []
    global_idx = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for conv_idx, row in enumerate(reader):
            if not row:
                continue
            raw = row[0].strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip()]

            for msg_idx, line in enumerate(lines):
                # Parse "User 1: text" or "User 2: text"
                m = re.match(r"^(User \d+):\s*(.+)$", line)
                if m:
                    speaker = m.group(1)
                    text = m.group(2).strip()
                else:
                    # Continuation or unlabeled — attach to previous
                    if messages and messages[-1].conv_index == conv_idx:
                        messages[-1].text += " " + line
                    continue

                messages.append(Message(
                    global_index=global_idx,
                    conv_index=conv_idx,
                    msg_index=msg_idx,
                    speaker=speaker,
                    text=text,
                ))
                global_idx += 1

    return messages


if __name__ == "__main__":
    msgs = load_messages("data/conversations.csv")
    print(f"Loaded {len(msgs)} messages from conversations")
    print("Sample:", msgs[0])
    print("Sample:", msgs[100])
