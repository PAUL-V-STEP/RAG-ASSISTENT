import unittest
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant_api.normalization.input_fix import fix_input
from assistant_api.settings import load_settings


class InputFixFixtures(unittest.TestCase):
    def test_fixtures(self):
        path = Path(__file__).resolve().parents[1] / "fixtures" / "input_fix.yaml"
        items = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = load_settings()
        for item in items:
            got = fix_input(item["raw"], settings=settings, glossary_terms=["ТО-1", "ЕО", "СО", "Магос", "Унидем"])
            exp = item["expect"]
            self.assertEqual(got.action, exp["action"], item["id"])
            if exp.get("text"):
                self.assertEqual(got.text, exp["text"], item["id"])


if __name__ == "__main__":
    unittest.main()
