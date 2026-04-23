from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class MemoryWriteDecision:
    capture: bool
    salience: float
    reason: str


class MemoryWritePolicy:
    """Small rule-based gate that prevents obvious chat noise from becoming memory."""

    explicit_patterns = [
        r"\bremember\b",
        r"\bkeep in mind\b",
        r"\bnote that\b",
        r"\bstore\b",
        r"\bmerk dir\b",
        r"\bnotiere\b",
        r"\bspeichere\b",
    ]
    preference_patterns = [
        r"\bprefer\b",
        r"\bprefers\b",
        r"\blike\b",
        r"\blikes\b",
        r"\bmag\b",
        r"\bbevorzuge\b",
        r"\bbevorzugt\b",
        r"\bmein name ist\b",
        r"\bmy name is\b",
        r"\balways\b",
        r"\bnever\b",
        r"\bimmer\b",
        r"\bnie\b",
    ]
    correction_patterns = [
        r"\bcorrection\b",
        r"\bactually\b",
        r"\binstead\b",
        r"\bkorrektur\b",
        r"\bstimmt nicht\b",
        r"\bstattdessen\b",
    ]
    low_information = {
        "ok",
        "okay",
        "ja",
        "nein",
        "no",
        "yes",
        "danke",
        "thanks",
        "thank you",
        "hi",
        "hello",
        "hey",
    }

    def evaluate(self, user_message: str, assistant_answer: str) -> MemoryWriteDecision:
        text = user_message.strip().lower()
        if not text:
            return MemoryWriteDecision(False, 0.0, "empty message")
        if text in self.low_information:
            return MemoryWriteDecision(False, 0.0, "low information chat turn")
        if self._matches(text, self.explicit_patterns):
            return MemoryWriteDecision(True, 0.85, "explicit memory request")
        if self._matches(text, self.correction_patterns):
            return MemoryWriteDecision(True, 0.8, "correction or contradiction")
        if self._matches(text, self.preference_patterns):
            return MemoryWriteDecision(True, 0.75, "stable preference or identity signal")
        if len(text) >= 180 and not text.endswith("?"):
            return MemoryWriteDecision(True, 0.45, "substantial statement")
        return MemoryWriteDecision(False, 0.0, "no durable memory signal")

    def _matches(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

