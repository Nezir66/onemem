from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .runtime import MemoryRuntime


class ChatClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass(slots=True)
class OpenAIResponsesClient:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenAIResponsesClient":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for chat")
        return cls(
            api_key=api_key,
            model=model or os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    def complete(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "input": prompt}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        return extract_output_text(data)


class MemoryChatbot:
    def __init__(
        self,
        runtime: MemoryRuntime,
        client: ChatClient,
        *,
        memory_limit: int = 8,
        auto_flush: bool = True,
    ) -> None:
        self.runtime = runtime
        self.client = client
        self.memory_limit = memory_limit
        self.auto_flush = auto_flush
        self.history: list[tuple[str, str]] = []

    def ask(self, user_message: str) -> str:
        memories = self.runtime.retrieve(user_message, limit=self.memory_limit)
        prompt = build_prompt(user_message, memories.context(), self.history[-6:])
        answer = self.client.complete(prompt).strip()
        self.history.append(("user", user_message))
        self.history.append(("assistant", answer))
        self.runtime.capture(
            f"User asked: {user_message}\nAssistant answered: {answer}",
            source="chatbot",
            session="chat",
            salience=0.55,
        )
        if self.auto_flush:
            self.runtime.flush()
        return answer


def build_prompt(user_message: str, memory_context: str, history: list[tuple[str, str]]) -> str:
    history_text = "\n".join(f"{role}: {content}" for role, content in history) or "No prior turns."
    memory_text = memory_context or "No relevant memory found."
    return f"""You are a helpful chatbot connected to OneMem.
Use the memory context when it is relevant. Do not claim the memory says something it does not say.
If memory is missing or uncertain, answer normally and say what would need to be remembered.

Memory context:
{memory_text}

Recent conversation:
{history_text}

User message:
{user_message}
"""


def extract_output_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    if parts:
        return "\n".join(parts)
    raise RuntimeError("OpenAI response did not contain text output")

