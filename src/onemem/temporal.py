from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_LONGMEMEVAL_HEADER = re.compile(
    r"(\d{4})/(\d{1,2})/(\d{1,2})\s*\(\w{3}\)\s*(\d{1,2}):(\d{2})"
)
_ISO_DATETIME = re.compile(
    r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s](\d{1,2}):(\d{2}))?"
)
_SLASH_DATE = re.compile(r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b")
_MONTH_NAME = re.compile(
    r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
    re.IGNORECASE,
)

_EARLIEST_MARKERS = (
    "first",
    "earliest",
    "initially",
    "at the start",
    "for the first time",
)
_LATEST_MARKERS = (
    "last",
    "latest",
    "most recent",
    "most recently",
    "recently",
    "just now",
)
_BEFORE_MARKERS = ("before ", "earlier than", "prior to", "ahead of")
_AFTER_MARKERS = ("after ", "later than", "following ", "since ")
_OTHER_TEMPORAL = (
    "how many days",
    "how long ago",
    "how many weeks",
    "how many months",
    "when did",
    "when was",
)


@dataclass(frozen=True, slots=True)
class TemporalIntent:
    is_temporal: bool
    prefer_earliest: bool
    prefer_latest: bool
    before: bool
    after: bool


def parse_event_date(text: str | None, *, default_year: int | None = None) -> str | None:
    """Return ISO-Z timestamp for the first recognizable date in `text`, else None."""
    if not text:
        return None
    match = _LONGMEMEVAL_HEADER.search(text)
    if match:
        year, month, day, hour, minute = (int(part) for part in match.groups())
        return _iso(year, month, day, hour, minute)
    match = _ISO_DATETIME.search(text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        hour = int(match.group(4)) if match.group(4) else 0
        minute = int(match.group(5)) if match.group(5) else 0
        return _iso(year, month, day, hour, minute)
    match = _SLASH_DATE.search(text)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return _iso(year, month, day)
    match = _MONTH_NAME.search(text)
    if match:
        month = MONTHS[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else default_year
        if year is not None:
            return _iso(year, month, day)
    return None


def detect_temporal_intent(query: str) -> TemporalIntent:
    text = query.lower()
    prefer_earliest = any(marker in text for marker in _EARLIEST_MARKERS)
    prefer_latest = any(marker in text for marker in _LATEST_MARKERS)
    before = any(marker in text for marker in _BEFORE_MARKERS)
    after = any(marker in text for marker in _AFTER_MARKERS)
    other = any(marker in text for marker in _OTHER_TEMPORAL)
    is_temporal = prefer_earliest or prefer_latest or before or after or other
    return TemporalIntent(
        is_temporal=is_temporal,
        prefer_earliest=prefer_earliest,
        prefer_latest=prefer_latest,
        before=before,
        after=after,
    )


def _iso(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> str:
    return (
        datetime(year, month, day, hour, minute, tzinfo=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
