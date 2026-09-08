from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_HF_MODEL,
    DEFAULT_LOCAL_MODEL,
    GroqAgentRunner,
    HFAgentRunner,
    LocalAgentRunner,
)
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract owned action items from raw meeting notes using an AI agent."
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
    parser.add_argument(
        "--backend",
        choices=["groq", "local", "hf"],
        default="groq",
        help="'groq' calls a Groq-hosted model (free tier, needs GROQ_API_KEY, default). "
        "'local' runs a small model on-device via MLX (free, no key, but lower quality). "
        "'hf' calls a Hugging Face-hosted model (needs HF_TOKEN with Inference Providers credits).",
    )
    parser.add_argument("--model", default=None, help="Override the default model id for the chosen backend")
    parser.add_argument(
        "--token",
        default=None,
        help="API key/token for the chosen backend (defaults to $GROQ_API_KEY or $HF_TOKEN)",
    )
    args = parser.parse_args(argv)

    if args.backend == "groq":
        api_key = args.token or os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(
                "Error: no Groq API key found. Set GROQ_API_KEY or pass --token.\n"
                "Get one free at https://console.groq.com/keys",
                file=sys.stderr,
            )
            return 1
        runner = GroqAgentRunner(model=args.model or DEFAULT_GROQ_MODEL, api_key=api_key)
    elif args.backend == "hf":
        token = args.token or os.environ.get("HF_TOKEN")
        if not token:
            print(
                "Error: no Hugging Face token found. Set HF_TOKEN or pass --token.\n"
                "Get one at https://huggingface.co/settings/tokens",
                file=sys.stderr,
            )
            return 1
        runner = HFAgentRunner(model=args.model or DEFAULT_HF_MODEL, token=token)
    else:
        runner = LocalAgentRunner(model=args.model or DEFAULT_LOCAL_MODEL)

    notes = open(args.notes_file).read() if args.notes_file else sys.stdin.read()
    attendees = [a.strip() for a in args.attendees.split(",") if a.strip()]

    output = run(notes, attendees, runner)
    print(json.dumps(output.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
