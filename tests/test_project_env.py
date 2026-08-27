import os
import tempfile
import unittest
from pathlib import Path

from project_env import find_project_root, load_project_env, require_env


class ProjectEnvironmentTests(unittest.TestCase):
    def tearDown(self):
        for name in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
            os.environ.pop(name, None)

    def test_root_env_file_overrides_an_existing_shell_value(self):
        os.environ["OPENAI_API_KEY"] = "old-shell-value"
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("OPENAI_API_KEY=root-value\n", encoding="utf-8")

            load_project_env(env_file)

            self.assertEqual("root-value", require_env("OPENAI_API_KEY"))

    def test_missing_required_value_mentions_root_env_file(self):
        with self.assertRaisesRegex(RuntimeError, r"root.*\.env.*TAVILY_API_KEY"):
            require_env("TAVILY_API_KEY")

    def test_finds_root_from_a_nested_practice_path(self):
        nested_file = Path(__file__).resolve().parents[1] / "삼성-rag-1일차" / "no-comments" / "실습1-1-로컬 LLM 기반의 RAG 챗봇 구현.py"

        self.assertEqual(Path(__file__).resolve().parents[1], find_project_root(nested_file))


if __name__ == "__main__":
    unittest.main()
