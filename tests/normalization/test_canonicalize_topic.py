import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant_api.normalization.canonicalizer import canonicalize


class TopicCanonicalTests(unittest.TestCase):
    def test_what_is_to_not_same_as_al(self):
        to_q = canonicalize("Что такое ТО?")
        al_q = canonicalize("Что такое АЛ?")
        self.assertNotEqual(to_q["canonical"], al_q["canonical"])
        self.assertIn("то", to_q["canonical"].lower().split("|")[-1])
        self.assertNotEqual(to_q["cache_key"], al_q["cache_key"])


if __name__ == "__main__":
    unittest.main()
