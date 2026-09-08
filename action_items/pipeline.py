from .agent import AgentRunner, extract
from .schema import Output
from .validate import ground_output


def run(notes: str, attendees: list[str], runner: AgentRunner) -> Output:
    output = extract(notes, attendees, runner)
    return ground_output(output, notes, attendees)
