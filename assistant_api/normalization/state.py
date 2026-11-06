"""DialogState и topic_switch по слотам."""

from __future__ import annotations

from assistant_api.normalization.schema import DialogState, Entities

SUBJECT_SLOTS = ("node", "maintenance_level", "operation", "part")


def topic_switch_by_slots(state: DialogState, ent: Entities) -> bool:
    prev = {k: getattr(state, k) for k in SUBJECT_SLOTS if getattr(state, k)}
    if not prev:
        return False
    cur = {k: getattr(ent, k) for k in SUBJECT_SLOTS if getattr(ent, k)}
    shared = any(prev.get(k) == v for k, v in cur.items())
    return not shared


def update_state(state: DialogState, ent: Entities, topic_switch: bool, turn: int) -> DialogState:
    if topic_switch:
        state = DialogState(pending_input_fix=state.pending_input_fix)
    data = state.model_dump()
    for slot in SUBJECT_SLOTS:
        val = getattr(ent, slot)
        if val:
            data[slot] = val
    data["updated_turn"] = turn
    return DialogState(**data)
