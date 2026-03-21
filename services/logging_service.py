from __future__ import annotations

import json
import os
from threading import Lock


class LoggingService:
    """Appends compare-mode records to a local JSON log file."""

    _lock = Lock()

    def __init__(self, log_path="logs/rag_logs.json"):
        self.log_path = log_path
        self._ensure_log_file()

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as file:
                json.dump([], file, ensure_ascii=True, indent=2)

    def log_interaction(self, data: dict):
        self._ensure_log_file()
        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as file:
                    current_logs = json.load(file)
                    if not isinstance(current_logs, list):
                        current_logs = []
            except Exception:
                current_logs = []

            current_logs.append(data)

            with open(self.log_path, "w", encoding="utf-8") as file:
                json.dump(current_logs, file, ensure_ascii=True, indent=2)
