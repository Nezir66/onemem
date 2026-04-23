from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onemem.eval import LongMemEvalImporter


class LongMemEvalImportTest(unittest.TestCase):
    def test_imports_longmemeval_oracle_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "longmemeval_oracle.json"
            out = Path(tmp) / "eval.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "question_id": "q1",
                            "question_type": "information-extraction",
                            "question": "What does Nezir prefer?",
                            "answer": "short technical answers",
                            "question_date": "2026/04/24",
                            "haystack_dates": ["2026/04/23"],
                            "haystack_session_ids": ["s1"],
                            "haystack_sessions": [
                                [
                                    {
                                        "role": "user",
                                        "content": "Nezir prefers short technical answers.",
                                        "has_answer": True,
                                    }
                                ]
                            ],
                            "answer_session_ids": ["s1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = LongMemEvalImporter().import_file(source, out, limit=1, top_k=7)
            imported = json.loads(out.read_text(encoding="utf-8"))

            self.assertEqual(result["cases"], 1)
            self.assertEqual(imported["cases"][0]["name"], "q1")
            self.assertEqual(imported["cases"][0]["queries"][0]["top_k"], 7)
            self.assertEqual(imported["cases"][0]["queries"][0]["must_contain"], ["short technical answers"])
            self.assertIn("user: Nezir prefers short technical answers.", imported["cases"][0]["episodes"][0]["text"])


if __name__ == "__main__":
    unittest.main()

