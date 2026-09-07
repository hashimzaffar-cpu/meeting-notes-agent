import re

from .schema import Output


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z']+", text.lower()))


def ground_output(output: Output, notes: str, attendees: list[str]) -> Output:
    """Demote any action whose owner can't be traced back to the input text.

    The model's output is schema-valid JSON, but the schema has no way to
    constrain "owner" to real names — that set only exists at run time, inside
    the notes and attendees. So the check happens here, after generation:
    an owner is kept only if every word in it also appears in the notes or
    the attendees list. Anything else is treated as a hallucination and its
    task is moved to `unassigned`.
    """
    source_words = _words(notes) | _words(" ".join(attendees))

    grounded_actions = []
    unassigned = list(output.unassigned)

    for action in output.actions:
        owner_words = _words(action.owner) if action.owner else None
        if owner_words and owner_words.issubset(source_words):
            grounded_actions.append(action)
        else:
            unassigned.append(action.task)

    return Output(actions=grounded_actions, unassigned=unassigned)
