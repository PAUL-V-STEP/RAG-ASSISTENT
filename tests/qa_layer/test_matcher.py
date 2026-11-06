import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant_api.normalization.schema import Entities, InputFix, NormalizedQuery
from assistant_api.qa_layer.matcher import decide


class MatcherTieBreakTests(unittest.TestCase):
    def test_same_canonical_prefers_lexical_overlap(self):
        can = "parameter_value|-|СО|-|-|трудоемкость|-|-|-|-"
        nq = NormalizedQuery(
            raw="Сколько времени на подтяжку резьбовых соединений при сезонном обслуживании?",
            cleaned="Сколько времени на подтяжку резьбовых соединений при сезонном обслуживании?",
            input_fix=InputFix(text="Сколько времени на подтяжку резьбовых соединений при сезонном обслуживании?"),
            standalone="Сколько времени на подтяжку резьбовых соединений при сезонном обслуживании?",
            canonical=can,
            intent="parameter_value",
            entities=Entities(maintenance_level="СО", parameter="трудоемкость"),
        )
        records = [
            {
                "id": "qa008",
                "question": "Какова трудоемкость сезонного технического обслуживания (СО)?",
                "paraphrases": ["Сколько часов занимает СО?"],
                "intent": "parameter_value",
                "maintenance_level": "СО",
                "verified_by": "draft-ai",
                "_canonical": can,
            },
            {
                "id": "qa026",
                "question": "Какова трудоемкость проверки затяжки резьбовых соединений при СО?",
                "paraphrases": ["Сколько времени занимает подтяжка резьбовых соединений при сезонном ТО?"],
                "intent": "parameter_value",
                "maintenance_level": "СО",
                "verified_by": "draft-ai",
                "_canonical": can,
            },
        ]
        settings = SimpleNamespace(
            QA_STOP_RULES_ENABLED=False,
            QA_DIRECT_MAX_DISTANCE=0.12,
            QA_HYBRID_MAX_DISTANCE=0.25,
            QA_MIN_MARGIN=0.03,
            QA_REQUIRE_VERIFIED=False,
            QA_MODE="full",
            TRACE_SHOW_CANDIDATES=10,
        )
        dec = decide(nq, records, {}, settings)
        self.assertEqual(dec.qa_id, "qa026")
        self.assertEqual(dec.d1, 0.0)


if __name__ == "__main__":
    unittest.main()
