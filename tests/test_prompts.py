from arcagi2.prompts import STUDENT_UNIVERSAL_METHOD, build_messages
from arcagi2.types import Pair


def test_episode_system_message_contains_packaged_universal_method():
    messages = build_messages([Pair([[1]], [[2]])], [[3]])
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert STUDENT_UNIVERSAL_METHOD.startswith("# ARC Solving Guide\n")
    assert system.count(STUDENT_UNIVERSAL_METHOD) == 1
    assert system.count("# ARC Solving Guide") == 1
    assert (
        "After reasoning, return exactly one final grid inside <answer>...</answer>."
        in system
    )
    assert (
        "The answer must be a JSON list of equally sized rows containing only integers 0 through 9."
        in system
    )
    assert user.endswith("Infer the rule and solve the test input.")
