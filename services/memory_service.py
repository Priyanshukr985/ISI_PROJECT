from collections import defaultdict, deque
from copy import deepcopy
from uuid import uuid4


class MemoryService:
    """Stores short session history and resolves simple follow-up prompts."""

    FOLLOW_UP_PHRASES = {
        "example",
        "examples",
        "real life example",
        "give example",
        "give examples",
        "why",
        "how",
        "next",
        "solve this",
        "continue",
        "more",
        "more details",
        "deeper explanation",
    }

    def __init__(self, max_messages=8):
        self.max_messages = max_messages
        self.sessions = defaultdict(lambda: deque(maxlen=self.max_messages))

    def ensure_session(self, session_id=None):
        session_key = (session_id or "").strip() or str(uuid4())
        self.sessions[session_key]
        return session_key

    def get_history(self, session_id):
        session_key = self.ensure_session(session_id)
        return deepcopy(list(self.sessions[session_key]))

    def add_turn(self, session_id, user_content, assistant_content):
        session_key = self.ensure_session(session_id)
        history = self.sessions[session_key]
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})

    def is_follow_up(self, message):
        normalized = " ".join((message or "").strip().lower().split())
        if not normalized:
            return False
        if normalized in self.FOLLOW_UP_PHRASES:
            return True
        return len(normalized.split()) <= 4 and any(
            phrase in normalized for phrase in self.FOLLOW_UP_PHRASES
        )

    def _last_user_topic(self, history):
        for item in reversed(history):
            if item.get("role") == "user" and item.get("content"):
                return item["content"].strip()
        return ""

    def resolve_user_query(self, message, history):
        current_message = (message or "").strip()
        if not current_message:
            return {
                "is_follow_up": False,
                "resolved_question": "",
                "follow_up_instruction": "",
            }

        if not history or not self.is_follow_up(current_message):
            return {
                "is_follow_up": False,
                "resolved_question": current_message,
                "follow_up_instruction": "",
            }

        prior_topic = self._last_user_topic(history)
        if not prior_topic:
            return {
                "is_follow_up": False,
                "resolved_question": current_message,
                "follow_up_instruction": "",
            }

        normalized = current_message.lower()
        if "example" in normalized:
            resolved = f"Give real-life examples of {prior_topic}"
        elif normalized == "why":
            resolved = f"Why does {prior_topic} matter in mathematical statistics?"
        elif normalized == "how":
            resolved = f"How does {prior_topic} work in mathematical statistics?"
        elif normalized == "next":
            resolved = f"What should I learn next after {prior_topic}?"
        elif "solve this" in normalized:
            resolved = f"Solve this in the context of {prior_topic}"
        else:
            resolved = f"In the context of {prior_topic}, {current_message}"

        instruction = (
            "This is a follow-up question. Use the conversation history to continue naturally. "
            "Answer only the specific follow-up request and avoid repeating the full earlier explanation."
        )
        return {
            "is_follow_up": True,
            "resolved_question": resolved,
            "follow_up_instruction": instruction,
        }
