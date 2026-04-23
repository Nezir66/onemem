from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from onemem.env import load_env_file


class EnvLoaderTest(unittest.TestCase):
    def test_loads_env_without_overriding_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_API_KEY=from-file",
                        'OPENAI_MODEL="gpt-test"',
                        "EMPTY=",
                    ]
                ),
                encoding="utf-8",
            )
            old_key = os.environ.get("OPENAI_API_KEY")
            old_model = os.environ.get("OPENAI_MODEL")
            old_empty = os.environ.get("EMPTY")
            try:
                os.environ["OPENAI_API_KEY"] = "already-set"
                os.environ.pop("OPENAI_MODEL", None)
                os.environ.pop("EMPTY", None)

                loaded = load_env_file(path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "already-set")
                self.assertEqual(os.environ["OPENAI_MODEL"], "gpt-test")
                self.assertEqual(os.environ["EMPTY"], "")
                self.assertEqual(loaded, {"OPENAI_MODEL": "gpt-test", "EMPTY": ""})
            finally:
                _restore_env("OPENAI_API_KEY", old_key)
                _restore_env("OPENAI_MODEL", old_model)
                _restore_env("EMPTY", old_empty)


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()

