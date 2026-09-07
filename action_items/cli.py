from __future__ import annotations

import argparse
import json
import os
import sys

from .extract import DEFAULT_MODEL, HFClient
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract owned action items from raw meeting notes using a Hugging Face model."
    )
    parser.add_argument(
        "notes_file",
        nargs="?",
        help="Path to a text file with the meeting notes. Reads stdin if omitted.",
    )
    parser.add_argument(
        "--attendees",
        required=True,
        help='Comma-separated attendee names, e.g. "Aiza,Sam"',
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (defaults to $HF_TOKEN)",
    )
    args = parser.parse_args(argv)

    if not args.token:
        print(
            "Error: no Hugging Face token found. Set HF_TOKEN or pass --token.\n"
            "Get one at https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    notes = open(args.notes_file).read() if args.notes_file else sys.stdin.read()
    attendees = [a.strip() for a in args.attendees.split(",") if a.strip()]

    client = HFClient(model=args.model, token=args.token)
    output = run(notes, attendees, client)
    print(json.dumps(output.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
