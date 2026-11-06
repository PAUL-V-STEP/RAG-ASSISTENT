import os
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant_api.settings import SPECS, ConfigError, load_settings, setting_names


class SettingsTests(unittest.TestCase):
    def test_names_match_env_example(self):
        root = Path(__file__).resolve().parents[2]
        text = (root / "env.example").read_text(encoding="utf-8")
        env_vars = {line.split("=", 1)[0] for line in text.splitlines() if line and line[0].isupper()}
        self.assertEqual(set(setting_names()), env_vars)

    def test_bad_enum(self):
        with self.assertRaises(ConfigError):
            load_settings(cli_overrides={"TRACE_MODE": "verbose"})

    def test_cli_priority(self):
        s = load_settings(cli_overrides={"TRACE_MODE": "full"})
        self.assertEqual(s.TRACE_MODE, "full")
        self.assertEqual(s.source_of("TRACE_MODE"), "cli")


if __name__ == "__main__":
    unittest.main()
