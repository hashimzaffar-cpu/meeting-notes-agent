import re

from .schema import Output


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z']+", text.lower()))


def is_owner_grounded(name: str, notes: str, attendees: list[str]) -> bool:
    """True only if every word in `name` appears in the notes or attendees.

    Shared by `ground_output` (the post-generation safety net) and the
    agent's `verify_owner` tool (a pre-generation self-check) — both need the
    identical definition of "a real name", since the set of real names only
    exists at run time and can't be baked into the schema ahead of time.
    """
    if not name:
        return False
    source_words = _words(notes) | _words(" ".join(attendees))
    return _words(name).issubset(source_words)


def ground_output(output: Output, notes: str, attendees: list[str]) -> Output:
    """Demote any action whose owner can't be traced back to the input text.

    The model's output is schema-valid JSON, but the schema has no way to
    constrain "owner" to real names — that set only exists at run time, inside
    the notes and attendees. So the check happens here, after generation:
    an owner is kept only if every word in it also appears in the notes or
    the attendees list. Anything else is treated as a hallucination and its
    task is moved to `unassigned`.
    """
    grounded_actions = []
    unassigned = list(output.unassigned)

    for action in output.actions:
        if is_owner_grounded(action.owner, notes, attendees):
            grounded_actions.append(action)
        else:
            unassigned.append(action.task)

    # The model sometimes lists the same task in both "actions" (with a null
    # owner) and "unassigned" already -- demoting the former then duplicates
    # the latter. dict.fromkeys dedupes while preserving first-seen order.
    unassigned = list(dict.fromkeys(unassigned))

    return Output(actions=grounded_actions, unassigned=unassigned)
