from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onemem.eval import EvalRunner, format_report


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


if __name__ == "__main__":
    unittest.main()
