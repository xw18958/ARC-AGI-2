import pytest

from arcagi2.prompts import STUDENT_UNIVERSAL_METHOD, build_messages, load_reasoning_method
from arcagi2.types import Pair


def test_original_prompt_remains_default_and_unchanged():
    messages = build_messages([Pair([[1]], [[2]])], [[3]])
    system = messages[0]["content"]
    assert STUDENT_UNIVERSAL_METHOD.startswith("# ARC Solving Guide\n")
    assert system.count(STUDENT_UNIVERSAL_METHOD) == 1
    assert system.count("# ARC Solving Guide") == 1


def test_v1_and_v2_are_selectable_and_distinct():
    v1 = load_reasoning_method("v1")
    v2 = load_reasoning_method("v2")
    assert v1 != v2
    assert "categorical colors/symbols" in v2
    assert "Obstacle-constrained connection" in v2
    assert "Symbolic quadrant compression" in v2


def test_build_messages_uses_requested_prompt_only():
    support = Pair([[1]], [[2]])
    query = [[3]]
    v1_messages = build_messages([support], query, prompt_method="v1")
    v2_messages = build_messages([support], query, prompt_method="v2")
    assert v1_messages[0]["content"] != v2_messages[0]["content"]
    assert "categorical colors/symbols" in v2_messages[0]["content"]
    assert v1_messages[1]["content"] == v2_messages[1]["content"]
    assert v2_messages[1]["content"].endswith("Infer the rule and solve the test input.")


def test_unknown_prompt_method_fails_fast():
    with pytest.raises(ValueError, match="Unknown prompt method"):
        load_reasoning_method("not-a-prompt")
