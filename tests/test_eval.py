from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onemem.eval import EvalRunner, format_report, run_write_filter_eval


class EvalRunnerTest(unittest.TestCase):
    def test_runs_basic_eval_file(self) -> None:
        report = EvalRunner().run_file(Path("evals/basic_memory.json"))

        self.assertTrue(report.passed, format_report(report))
        self.assertEqual(report.summary()["cases"], 3)
        self.assertGreaterEqual(report.summary()["mrr"], 0.5)

    def test_reports_failed_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fail.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "missing",
                                "episodes": ["OneMem stores memories."],
                                "queries": [
                                    {
                                        "query": "What does Nezir prefer?",
                                        "must_contain": ["short answers"],
                                        "top_k": 3,
                                    }
                                ],
                                "check_rebuild": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = EvalRunner().run_file(path)

            self.assertFalse(report.passed)
            self.assertIn("FAIL", format_report(report))

    def test_summary_includes_separated_metric_keys(self) -> None:
        report = EvalRunner().run_file(Path("evals/basic_memory.json"))
        summary = report.summary()

        for key in (
            "answer_recall",
            "answer_mrr",
            "evidence_recall_at_5",
            "evidence_recall_at_10",
            "evidence_mrr",
            "context_pollution",
            "evidence_queries",
        ):
            self.assertIn(key, summary)

    def test_temporal_eval_prefers_earliest_with_reference_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporal.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "car_timeline",
                                "reference_date": "2023-06-01",
                                "episodes": [
                                    {
                                        "text": "I bought a new car on February 10th, 2023.",
                                        "source": "bench:car:feb",
                                        "session": "car_timeline",
                                        "event_date": "2023-02-10",
                                    },
                                    {
                                        "text": "I took the car to the dealer for a GPS fix on March 22nd, 2023.",
                                        "source": "bench:car:mar",
                                        "session": "car_timeline",
                                        "event_date": "2023-03-22",
                                    },
                                    {
                                        "text": "I got the car detailed on May 4th, 2023.",
                                        "source": "bench:car:may",
                                        "session": "car_timeline",
                                        "event_date": "2023-05-04",
                                    },
                                ],
                                "queries": [
                                    {
                                        "query": "What was the first event with my car?",
                                        "expected_source_refs": ["bench:car:feb"],
                                        "score": "evidence",
                                        "top_k": 5,
                                    }
                                ],
                                "check_rebuild": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = EvalRunner().run_file(path)

            self.assertTrue(report.passed, format_report(report))
            self.assertEqual(report.cases[0].queries[0].evidence_rank, 1)

    def test_write_filter_eval_scores_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.json"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"text": "Remember that I prefer short answers.", "capture": True},
                            {"text": "ok", "capture": False},
                            {"text": "Mein Name ist Nezir.", "capture": True},
                            {"text": "danke", "capture": False},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = run_write_filter_eval(path)

            self.assertEqual(report.total, 4)
            self.assertEqual(report.correct, 4)
            self.assertEqual(report.mistakes, [])

    def test_scores_evidence_refs_independently_from_answer_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "evidence",
                                "episodes": [
                                    {
                                        "text": "The workshop was on Monday. The team meeting was the next Monday.",
                                        "source": "benchmark:q1:s1",
                                        "session": "q1",
                                    }
                                ],
                                "queries": [
                                    {
                                        "query": "How many days before the team meeting was the workshop?",
                                        "must_contain": ["7 days"],
                                        "expected_source_refs": ["benchmark:q1:s1"],
                                        "score": "evidence",
                                        "top_k": 3,
                                    }
                                ],
                                "check_rebuild": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = EvalRunner().run_file(path)
            query = report.cases[0].queries[0]

            self.assertTrue(report.passed, format_report(report))
            self.assertIsNone(query.rank)
            self.assertEqual(query.evidence_rank, 1)
            self.assertEqual(query.missing, ["7 days"])


if __name__ == "__main__":
    unittest.main()
