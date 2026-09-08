from smolagents import Tool, tool

from .validate import is_owner_grounded


def build_verify_owner_tool(notes: str, attendees: list[str]) -> Tool:
    """Build a per-request tool the agent can call to self-check a candidate owner.

    The tool is built fresh for each request because it closes over that
    request's specific `notes` and `attendees` — the set of real names is
    only known once the notes arrive, so the check can't be a fixed,
    stateless tool shared across requests.
    """

    @tool
    def verify_owner(name: str) -> bool:
        """Check whether a candidate owner name actually appears in this meeting's notes or attendee list.

        Call this before assigning a task to anyone whose name you are not
        certain is genuinely present in the notes or attendees. Returns
        False for any name that would be a hallucination.

        Args:
            name: The candidate owner name to check, exactly as you intend to use it.
        """
        return is_owner_grounded(name, notes, attendees)

    return verify_owner
