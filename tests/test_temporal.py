from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from onemem.runtime import MemoryRuntime
from onemem.temporal import detect_temporal_intent, parse_event_date


class ParseEventDateTest(unittest.TestCase):
    def test_parses_longmemeval_header(self) -> None:
        self.assertEqual(parse_event_date("2023/04/10 (Mon) 17:50"), "2023-04-10T17:50:00Z")

    def test_parses_iso_date(self) -> None:
        self.assertEqual(parse_event_date("2024-02-03"), "2024-02-03T00:00:00Z")

    def test_parses_plain_slash_date(self) -> None:
        self.assertEqual(parse_event_date("2022/12/31"), "2022-12-31T00:00:00Z")

    def test_parses_month_name_with_year(self) -> None:
        self.assertEqual(parse_event_date("March 15, 2023"), "2023-03-15T00:00:00Z")

    def test_returns_none_for_junk(self) -> None:
        self.assertIsNone(parse_event_date("no date here"))
        self.assertIsNone(parse_event_date(None))
        self.assertIsNone(parse_event_date(""))


class TemporalIntentTest(unittest.TestCase):
    def test_earliest_marker(self) -> None:
        intent = detect_temporal_intent("What was the first issue I had with my car?")
        self.assertTrue(intent.is_temporal)
        self.assertTrue(intent.prefer_earliest)
        self.assertFalse(intent.prefer_latest)

    def test_latest_marker(self) -> None:
        intent = detect_temporal_intent("When did I last run the 5K?")
        self.assertTrue(intent.is_temporal)
        self.assertTrue(intent.prefer_latest)

    def test_before_after(self) -> None:
        self.assertTrue(detect_temporal_intent("did X happen before Y").before)
        self.assertTrue(detect_temporal_intent("what was said after Monday").after)

    def test_non_temporal_query(self) -> None:
        intent = detect_temporal_intent("What is my favorite color?")
        self.assertFalse(intent.is_temporal)


class TemporalRetrievalIntegrationTest(unittest.TestCase):
    def _seed(self, runtime: MemoryRuntime) -> None:
        runtime.capture(
            "I bought a new car on February 10th, 2023.",
            source="test",
            session="car_timeline",
            event_date="2023-02-10",
            salience=0.8,
        )
        runtime.capture(
            "I took the car to the dealer for a GPS fix on March 22nd, 2023.",
            source="test",
            session="car_timeline",
            event_date="2023-03-22",
            salience=0.8,
        )
        runtime.capture(
            "I got the car detailed on May 4th, 2023.",
            source="test",
            session="car_timeline",
            event_date="2023-05-04",
            salience=0.8,
        )

    def test_earliest_query_prefers_earliest_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            self._seed(runtime)

            result = runtime.retrieve(
                "What was the first event with my car?",
                limit=5,
                reference_date="2023-06-01",
            )
            episodes = [memory for memory in result.memories if memory.kind == "episode"]
            self.assertGreaterEqual(len(episodes), 2)
            self.assertIn("February", episodes[0].body)

    def test_latest_query_prefers_latest_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            self._seed(runtime)

            result = runtime.retrieve(
                "What was the most recent event with my car?",
                limit=5,
                reference_date="2023-06-01",
            )
            episodes = [memory for memory in result.memories if memory.kind == "episode"]
            self.assertGreaterEqual(len(episodes), 2)
            self.assertIn("May", episodes[0].body)

    def test_non_temporal_query_ignores_reference_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            self._seed(runtime)

            result = runtime.retrieve("what car did I buy?", limit=5)
            episodes = [memory for memory in result.memories if memory.kind == "episode"]
            self.assertTrue(episodes)
            # With no temporal intent, temporal_score is zero for every candidate.
            for memory in episodes:
                self.assertEqual(memory.debug["temporal_score"], 0.0)
                self.assertFalse(memory.debug["temporal_query"])


if __name__ == "__main__":
    unittest.main()
