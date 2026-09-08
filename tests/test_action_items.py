import json

from action_items.agent import extract
from action_items.pipeline import run
from action_items.tools import build_verify_owner_tool
from action_items.validate import ground_output
from action_items.schema import Output


class FakeAgentRunner:
    """Test double: returns a canned JSON string instead of running a real agent."""

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload)

    def run(self, notes: str, attendees: list[str]) -> str:
        return self._raw


def test_owner_named_in_notes_but_not_in_attendees_is_kept():
    # Spec test: notes mention a name not in attendees. The owner must not be
    # invented -- but it also must not be discarded just for being off-roster.
    notes = "Sam will review the PR. Ask Marcus to send the deck by Friday."
    attendees = ["Aiza", "Sam"]

    runner = FakeAgentRunner(
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

    result = run(notes, attendees, runner)

    owners = {a.task: a.owner for a in result.actions}
    assert owners["review the PR"] == "Sam"
    assert owners["send the deck"] == "Marcus"
    assert result.unassigned == []


def test_hallucinated_owner_is_demoted_to_unassigned():
    # The agent claims an owner that appears nowhere in notes or attendees.
    notes = "Someone should update the wiki at some point."
    attendees = ["Aiza", "Sam"]

    runner = FakeAgentRunner(
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

    result = run(notes, attendees, runner)

    assert result.actions == []
    assert result.unassigned == ["update the wiki"]


def test_null_owner_goes_to_unassigned():
    notes = "We need to figure out the budget eventually."
    attendees = ["Aiza"]

    runner = FakeAgentRunner(
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

    result = run(notes, attendees, runner)

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


def test_ground_output_dedupes_task_listed_both_ways():
    # A model sometimes lists the same task as a null-owner action AND
    # already in "unassigned" -- demoting the former shouldn't duplicate it.
    notes = "I'll follow up with legal today."
    output = Output(
        actions=[
            {
                "task": "follow up with legal",
                "owner": None,
                "due": "today",
                "confidence": "inferred",
            }
        ],
        unassigned=["follow up with legal"],
    )

    result = ground_output(output, notes, attendees=[])

    assert result.actions == []
    assert result.unassigned == ["follow up with legal"]


def test_extract_parses_agent_json_into_schema():
    runner = FakeAgentRunner(
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

    output = extract("Sam will send the deck today.", ["Sam"], runner)

    assert isinstance(output, Output)
    assert output.actions[0].owner == "Sam"
    assert output.unassigned == ["update the wiki"]


def test_verify_owner_tool_matches_ground_output_rules():
    # The agent's self-check tool must agree with the post-hoc grounding
    # check -- same definition of "a real name" on both sides.
    notes = "Ask Marcus to send the deck."
    verify_owner = build_verify_owner_tool(notes, attendees=["Aiza", "Sam"])

    assert verify_owner(name="Marcus") is True
    assert verify_owner(name="Sam") is True
    assert verify_owner(name="Priya") is False
    assert verify_owner(name="Marcus Lee") is False
