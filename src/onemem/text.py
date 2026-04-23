from __future__ import annotations

import hashlib
import re
import unicodedata

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "auf",
    "aus",
    "bei",
    "bin",
    "bist",
    "dem",
    "den",
    "das",
    "der",
    "die",
    "dies",
    "diese",
    "dieser",
    "du",
    "ein",
    "eine",
    "einen",
    "einer",
    "es",
    "for",
    "from",
    "fur",
    "haben",
    "hat",
    "habe",
    "ich",
    "im",
    "in",
    "ist",
    "meine",
    "meinem",
    "meinen",
    "mein",
    "mit",
    "name",
    "not",
    "oder",
    "sich",
    "sie",
    "sind",
    "that",
    "the",
    "this",
    "to",
    "und",
    "wir",
    "von",
    "was",
    "what",
    "wie",
    "will",
    "with",
    "you",
    "your",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", normalized.lower()).strip()


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()[:length]


def slugify(value: str, fallback: str = "memory") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(value)).strip("-")
    return slug[:64] or fallback


def tokenize(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", normalize_text(value))
        if token not in STOPWORDS
    ]


def split_sentences(value: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", value.strip())
    return [part.strip(" -\t\r\n") for part in parts if len(part.strip()) >= 8]


def title_from_body(value: str, max_len: int = 72) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "..."
