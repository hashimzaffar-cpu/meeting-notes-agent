from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from action_items.agent import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_HF_MODEL,
    DEFAULT_LOCAL_MODEL,
    GroqAgentRunner,
    HFAgentRunner,
    LocalAgentRunner,
)
from action_items.pipeline import run

app = Flask(__name__)

# "groq" (default) calls a Groq-hosted model -- free tier, fast, needs
# GROQ_API_KEY. Set AGENT_BACKEND=local for a fully offline on-device model
# (no key, lower quality), or AGENT_BACKEND=hf for a Hugging Face-hosted
# model (needs HF_TOKEN with Inference Providers credits).
_BACKEND = os.environ.get("AGENT_BACKEND", "groq")
_MODEL_ID = {"groq": DEFAULT_GROQ_MODEL, "hf": DEFAULT_HF_MODEL, "local": DEFAULT_LOCAL_MODEL}[_BACKEND]
_runner = None


def _get_runner():
    # Built lazily and reused across requests: constructing a LocalAgentRunner
    # loads model weights into memory, which is too expensive to redo on
    # every single request.
    global _runner
    if _runner is not None:
        return _runner
    if _BACKEND == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("AGENT_BACKEND=groq but no GROQ_API_KEY is set.")
        _runner = GroqAgentRunner(api_key=api_key)
    elif _BACKEND == "hf":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("AGENT_BACKEND=hf but no HF_TOKEN is set.")
        _runner = HFAgentRunner(token=token)
    else:
        _runner = LocalAgentRunner()
    return _runner


_BACKEND_LABEL = {
    "groq": "Groq",
    "hf": "Hugging Face Inference Providers",
    "local": "on-device via MLX",
}[_BACKEND]


@app.get("/")
def index():
    return render_template("index.html", default_model=_MODEL_ID, backend_label=_BACKEND_LABEL)


@app.post("/api/extract")
def api_extract():
    payload = request.get_json(force=True)
    notes = (payload.get("notes") or "").strip()
    attendees_raw = payload.get("attendees") or ""
    attendees = [a.strip() for a in attendees_raw.split(",") if a.strip()]

    if not notes:
        return jsonify({"error": "Notes are required."}), 400

    try:
        runner = _get_runner()
        output = run(notes, attendees, runner)
    except Exception as exc:  # surface model/network errors to the UI
        return jsonify({"error": str(exc)}), 502

    return jsonify(output.model_dump())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
