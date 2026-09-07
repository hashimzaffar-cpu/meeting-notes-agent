import json

from action_items.extract import extract
from action_items.pipeline import run
from action_items.validate import ground_output
from action_items.schema import Output


class FakeClient:
    """Test double: returns a canned JSON string instead of calling a model."""

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._raw


def test_owner_named_in_notes_but_not_in_attendees_is_kept():
    # Spec test: notes mention a name not in attendees. The owner must not be
    # invented -- but it also must not be discarded just for being off-roster.
    notes = "Sam will review the PR. Ask Marcus to send the deck by Friday."
    attendees = ["Aiza", "Sam"]

    client = FakeClient(
        {
            "actions": [
                {
                    "task": "review the PR",
                    "owner": "Sam",
                    "due": "unspecified",
                    "confidence": "explicit",
                },
                {
                    "task": "send the deck",
                    "owner": "Marcus",
                    "due": "this_week",
                    "confidence": "explicit",
                },
            ],
            "unassigned": [],
        }
    )

    result = run(notes, attendees, client)

    owners = {a.task: a.owner for a in result.actions}
    assert owners["review the PR"] == "Sam"
    assert owners["send the deck"] == "Marcus"
    assert result.unassigned == []


def test_hallucinated_owner_is_demoted_to_unassigned():
    # The model claims an owner that appears nowhere in notes or attendees.
    notes = "Someone should update the wiki at some point."
    attendees = ["Aiza", "Sam"]

    client = FakeClient(
        {
            "actions": [
                {
                    "task": "update the wiki",
                    "owner": "Priya",  # not in notes, not in attendees
                    "due": "unspecified",
                    "confidence": "inferred",
                }
            ],
            "unassigned": [],
        }
    )

    result = run(notes, attendees, client)

    assert result.actions == []
    assert result.unassigned == ["update the wiki"]


def test_null_owner_goes_to_unassigned():
    notes = "We need to figure out the budget eventually."
    attendees = ["Aiza"]

    client = FakeClient(
        {
            "actions": [
                {
                    "task": "figure out the budget",
                    "owner": None,
                    "due": "unspecified",
                    "confidence": "inferred",
                }
            ],
            "unassigned": [],
        }
    )

    result = run(notes, attendees, client)

    assert result.actions == []
    assert result.unassigned == ["figure out the budget"]


def test_ground_output_directly_partial_name_match_fails():
    # Owner "Marcus Lee" only half-grounded in the notes should not be trusted.
    notes = "Ask Marcus to send the deck."
    output = Output(
        actions=[
            {
                "task": "send the deck",
                "owner": "Marcus Lee",
                "due": "unspecified",
                "confidence": "explicit",
            }
        ],
        unassigned=[],
    )

    result = ground_output(output, notes, attendees=[])

    assert result.actions == []
    assert result.unassigned == ["send the deck"]


def test_extract_parses_model_json_into_schema():
    client = FakeClient(
        {
            "actions": [
                {
                    "task": "send the deck",
                    "owner": "Sam",
                    "due": "today",
                    "confidence": "explicit",
                }
            ],
            "unassigned": ["update the wiki"],
        }
    )

    output = extract("Sam will send the deck today.", ["Sam"], client)

    assert isinstance(output, Output)
    assert output.actions[0].owner == "Sam"
    assert output.unassigned == ["update the wiki"]
