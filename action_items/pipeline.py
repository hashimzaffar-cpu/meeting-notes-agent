from .extract import ChatClient, extract
from .schema import Output
from .validate import ground_output


def run(notes: str, attendees: list[str], client: ChatClient) -> Output:
    output = extract(notes, attendees, client)
    return ground_output(output, notes, attendees)
