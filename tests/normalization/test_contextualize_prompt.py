import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant_api.normalization.contextualizer import (
    PROMPT_PATH,
    render_contextualize_prompt,
)


class ContextualizePromptTests(unittest.TestCase):
    def test_json_example_survives_substitution(self):
        prompt = render_contextualize_prompt(
            PROMPT_PATH.read_text(encoding="utf-8"),
            state='{"node": null}',
            glossary_terms="ТО-1, СО",
            history="user: А для СО?",
            question="А по часам?",
        )
        self.assertIn('"standalone"', prompt)
        self.assertIn("Вопрос: А по часам?", prompt)
        self.assertIn("Термины глоссария: ТО-1, СО", prompt)
        self.assertNotIn("{question}", prompt)
        self.assertNotIn("{state}", prompt)


if __name__ == "__main__":
    unittest.main()
