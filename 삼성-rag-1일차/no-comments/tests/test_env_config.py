import os
import unittest

from env_config import ROOT_DIR, load_environment, require_env


class EnvironmentConfigTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("TAVILY_API_KEY", None)

    def test_loads_project_root_env_file(self):
        self.assertEqual(ROOT_DIR / ".env", load_environment())

    def test_required_value_has_beginner_friendly_error(self):
        os.environ.pop("TAVILY_API_KEY", None)

        with self.assertRaisesRegex(RuntimeError, r"\.env.*TAVILY_API_KEY"):
            require_env("TAVILY_API_KEY")



if __name__ == "__main__":
    unittest.main()
