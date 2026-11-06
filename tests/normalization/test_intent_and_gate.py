import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from assistant_api.normalization.history_gate import needs_history
from assistant_api.normalization.intent_rules import detect_intent
from assistant_api.normalization.period import normalize_period
from assistant_api.normalization.input_fix import to_cyrillic, to_latin
from assistant_api.normalization.state import DialogState, topic_switch_by_slots
from assistant_api.normalization.schema import Entities


class IntentTests(unittest.TestCase):
    def test_q25_parameter(self):
        i, _ = detect_intent("Какую температуру рабочей жидкости гидросистемы нельзя превышать?")
        self.assertEqual(i, "parameter_value")

    def test_q18_safety(self):
        i, _ = detect_intent("При каком уровне масла в баке запрещено эксплуатировать автолифт?")
        self.assertEqual(i, "safety")

    def test_can_last_token(self):
        i, _ = detect_intent("А без страхующих упоров платформу смотреть можно?")
        self.assertEqual(i, "safety")

    def test_nuzhno_li(self):
        i, _ = detect_intent("Нужно ли смывать поверхность после промывки?")
        self.assertEqual(i, "procedure")

    def test_narabotka_periodicity(self):
        i, _ = detect_intent("Какая наработка гидросистемы до ТО-1?")
        self.assertEqual(i, "periodicity")


class GateTests(unittest.TestCase):
    def test_empty_history(self):
        self.assertEqual(needs_history("А как его проверить?", [])[0], False)

    def test_first_a(self):
        self.assertTrue(needs_history("А для СО?", [{"role": "user", "content": "x"}])[0])

    def test_ih_not_i(self):
        ok, rule = needs_history("Их нужно снимать?", [{"role": "user", "content": "x"}])
        self.assertTrue(ok)
        self.assertEqual(rule, 3)

    def test_interval_not_i(self):
        # без предметного термина правило 4, с ТО-1 — 5
        ok, rule = needs_history("Интервал ТО-1?", [{"role": "user", "content": "x"}])
        self.assertFalse(ok)
        self.assertEqual(rule, 5)

    def test_list_to_types_standalone(self):
        ok, rule = needs_history(
            "Перечислите виды технического обслуживания автолифта.",
            [{"role": "user", "content": "x"}],
        )
        self.assertFalse(ok)
        self.assertEqual(rule, 5)


class PeriodTests(unittest.TestCase):
    def test_freq(self):
        self.assertEqual(normalize_period("два раза в год"), "2/год")
        self.assertEqual(normalize_period("2/год"), "2/год")
        self.assertEqual(normalize_period("2–3 раза в неделю"), "2-3/неделя")
        self.assertIsNone(normalize_period("как часто"))


class LayoutTableTests(unittest.TestCase):
    def test_roundtrip(self):
        s = "qwertyuiop[]asdfghjkl;'zxcvbnm,./`"
        self.assertEqual(to_latin(to_cyrillic(s)), s)


class TopicSwitchTests(unittest.TestCase):
    def test_empty_state(self):
        st = DialogState()
        ent = Entities(maintenance_level="ТО-1")
        self.assertFalse(topic_switch_by_slots(st, ent))


if __name__ == "__main__":
    unittest.main()
