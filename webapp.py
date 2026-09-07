from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from action_items.extract import DEFAULT_MODEL, HFClient
from action_items.pipeline import run

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL)


@app.post("/api/extract")
def api_extract():
    payload = request.get_json(force=True)
    notes = (payload.get("notes") or "").strip()
    attendees_raw = payload.get("attendees") or ""
    attendees = [a.strip() for a in attendees_raw.split(",") if a.strip()]

    if not notes:
        return jsonify({"error": "Notes are required."}), 400

    token = os.environ.get("HF_TOKEN")
    if not token:
        return jsonify({"error": "Server has no HF_TOKEN set."}), 500

    try:
        client = HFClient(token=token)
        output = run(notes, attendees, client)
    except Exception as exc:  # surface model/network errors to the UI
        return jsonify({"error": str(exc)}), 502

    return jsonify(output.model_dump())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
