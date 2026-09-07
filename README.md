# Meeting Notes → Action Items

Turn messy, raw meeting notes into a clean list of action items — task, owner,
due date, and how confident we are about the owner — using a Hugging Face AI
model, with a safety check that stops the AI from inventing people who were
never mentioned.

This README explains **everything**, from scratch: what the project does, what
tools it's built with, how every file fits together, and what happens
step-by-step when you use it.

---

## 1. What problem does this solve?

Meetings produce messy notes like:

> "Sam will review the PR. Ask Marcus to send the deck by Friday. I'll follow
> up with legal today."

Buried in there are real to-dos, owners, and deadlines. Pulling that out by
hand is boring and easy to get wrong. This project automates exactly that one
step:

```
raw notes + attendee names  →  { task, owner, due date, confidence }[]
```

## 2. The one idea this project is built around

You can ask an AI model to always answer in a fixed shape — e.g. "always give
me `{task, owner, due, confidence}`". That's called a **schema**, and it's
easy to enforce.

What you **can't** enforce with a schema is *which names are real*. The AI
could confidently write `"owner": "John"` even if no John was ever mentioned
in this specific meeting — a hallucination. The list of real names (the
attendees, plus anyone named in the notes) is different every single time you
run this, so it can't be baked into a fixed schema ahead of time.

So the rule "don't invent an owner" has to be enforced in **two** places:

1. **In the prompt** — politely instructing the model not to invent names.
2. **In plain code, after the model answers** — a dumb, reliable check that
   looks at the actual words in the notes/attendees and refuses to trust any
   owner name that doesn't really appear there.

Step 2 is the part that makes this trustworthy instead of "probably fine."
Prompts are requests; code checks are guarantees.

## 3. What's actually used here (the tech stack)

| Piece | What it is | Why it's here |
|---|---|---|
| **Python** | Programming language | Everything is written in it |
| **Pydantic** | A Python library for defining data "shapes" (schemas) | Forces the AI's JSON answer into a strict, typed structure |
| **Hugging Face `huggingface_hub`** | Python client for calling AI models hosted by Hugging Face | Lets us send notes to a real AI model over the internet |
| **Qwen2.5-7B-Instruct** | The actual AI model doing the reading/extracting | It's the "brain" — free-tier friendly and good at following instructions |
| **Flask** | A small Python web server framework | Serves the web page and handles the "Extract" button's request |
| **HTML / CSS / JavaScript** | The web page itself | What you actually see and click in the browser |
| **pytest** | Python testing tool | Runs automated checks on the safety-net logic, without needing to call the real AI every time |

## 4. Project layout

```
meeting-notes-agent/
├── action_items/
│   ├── schema.py     ← defines the exact shape of a valid answer
│   ├── prompt.py      ← the instructions + examples given to the AI
│   ├── extract.py     ← talks to the Hugging Face model, parses its answer
│   ├── validate.py     ← the safety-net check (catches invented owners)
│   ├── pipeline.py      ← glues extract + validate together
│   └── cli.py            ← run everything from a terminal
├── webapp.py               ← the Flask web server
├── templates/
│   └── index.html           ← the web page (form + results + JavaScript)
├── tests/
│   └── test_action_items.py  ← automated tests for the safety-net logic
└── requirements.txt            ← list of Python packages this needs
```

## 5. What every file actually does, in plain words

### `action_items/schema.py` — the "form template"
Defines two shapes using Pydantic:
- `Action`: one task — its text, its owner (or nothing), a due date (must be
  one of `today` / `this_week` / `this_month` / `unspecified` — nothing else
  is allowed), and a confidence label (`explicit` or `inferred`).
- `Output`: a list of `Action`s, plus a separate list of `unassigned` tasks
  (tasks nobody could be confidently tied to).

If the AI's answer doesn't fit this shape, Pydantic raises an error instead
of silently accepting garbage.

### `action_items/prompt.py` — the instructions for the AI
Plain English text sent to the model before your notes. It explains:
- the exact JSON shape expected,
- that an owner can be someone **not** in the attendee list, as long as
  they're actually named in the notes,
- that owners must **never** be invented,
- what `explicit` vs `inferred` confidence means,
- two worked examples showing correct input → output.

This is the file you'd tweak first if the AI's answers need to change.

### `action_items/extract.py` — talking to the AI
- Defines `HFClient`, a small wrapper around Hugging Face's
  `InferenceClient`. It sends your notes + the prompt to the model
  (`Qwen/Qwen2.5-7B-Instruct`, via the `featherless-ai` hosting provider) and
  asks for a JSON-only reply.
- `extract()` takes the raw text the model returns, parses it as JSON, and
  validates it against the `Output` schema from `schema.py`.

### `action_items/validate.py` — the safety net
This is the part with no AI involved at all — just word matching:
1. Collect every word that appears in the notes and in the attendee list.
2. For every action the AI returned, check whether its owner's name is fully
   made up of words that actually appear there.
3. If yes → keep the action as-is.
4. If no (or there's no owner) → move that task into `unassigned` instead of
   keeping a possibly-invented name.

This is why the agent can be trusted: even if the AI hallucinates, this step
catches it before you ever see it.

### `action_items/pipeline.py` — the glue
Two lines: call `extract()` (talk to the AI), then call `ground_output()`
(run the safety check), and return the final, cleaned result.

### `action_items/cli.py` — terminal version
Lets you run the whole thing from a terminal instead of a browser:
```bash
python3 -m action_items.cli notes.txt --attendees "Aiza,Sam"
```

### `webapp.py` — the web server
A small Flask app with two routes:
- `GET /` — serves the web page (`templates/index.html`).
- `POST /api/extract` — receives `{notes, attendees}` as JSON from the
  browser, runs the pipeline, and sends back the final result as JSON.

### `templates/index.html` — the web page
A single-file page with:
- an attendees field and a notes textarea,
- a few one-click example scenarios,
- an "Extract action items" button (with a loading spinner while the AI
  thinks),
- results rendered as cards (task, owner badge, due-date badge,
  confidence badge),
- an "Unassigned" section for anything without a trustworthy owner,
- a "Download CSV" button to export the results,
- a collapsible "View raw JSON" panel for the exact data underneath.

### `tests/test_action_items.py` — automated tests
These tests **don't** call the real AI (that would be slow and cost API
calls every time you test). Instead, they use a `FakeClient` that returns a
pre-written fake AI answer, so the tests can focus purely on checking that
`validate.py` correctly:
- keeps an owner named in the notes even if they're not an attendee,
- demotes a hallucinated owner to `unassigned`,
- demotes a `null` owner to `unassigned`,
- rejects a partially-matching name like "Marcus Lee" when only "Marcus" was
  actually mentioned.

## 6. End-to-end walkthrough with a real example

Say you open the web page, and type:

- **Attendees:** `Aiza, Sam`
- **Notes:** *"Sam will review the PR. Ask Marcus to send the deck by
  Friday."*

Then click **Extract action items**. Here's exactly what happens, in order:

1. **Browser → server.** The page's JavaScript sends
   `{"notes": "...", "attendees": "Aiza, Sam"}` to `webapp.py`'s
   `/api/extract` route, without reloading the page.
2. **Server → pipeline.** `webapp.py` calls
   `pipeline.run(notes, attendees, client)`.
3. **Pipeline → AI.** `extract.py` builds one message: the system prompt
   (rules + examples) plus your specific notes and attendees, and sends it
   to the Qwen model hosted on Hugging Face's servers.
4. **AI replies** with something like:
   ```json
   {
     "actions": [
       {"task": "review the PR", "owner": "Sam", "due": "unspecified", "confidence": "explicit"},
       {"task": "send the deck", "owner": "Marcus", "due": "this_week", "confidence": "explicit"}
     ],
     "unassigned": []
   }
   ```
5. **Pipeline → safety check.** `validate.py` checks each owner:
   - "Sam" → appears in the attendee list. Kept.
   - "Marcus" → **not** an attendee, but *is* named directly in the notes.
     Still kept, because the rule is "must appear somewhere in the input,"
     not "must be an attendee."
   - (If the AI had instead said `"owner": "Priya"` — a name appearing
     nowhere — that task would be moved into `unassigned` here.)
6. **Server → browser.** The final JSON goes back to the page.
7. **Browser renders it** as two task cards with colored badges, plus a
   "Download CSV" option and a raw-JSON view.

## 7. How to run it yourself

### Install dependencies
```bash
pip3 install -r requirements.txt
```

### Get a Hugging Face token
1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens → **Create new token**
3. Make sure **"Make calls to Inference Providers"** permission is enabled
4. Add it to your shell so you don't have to type it every time:
   ```bash
   echo 'export HF_TOKEN=hf_your_token_here' >> ~/.zshrc
   source ~/.zshrc
   ```

### Run the automated tests (no AI calls, instant)
```bash
python3 -m pytest tests/ -v
```

### Run it in a browser
```bash
python3 webapp.py
```
Then open **http://127.0.0.1:7860**.

### Run it from a terminal instead
```bash
echo "Sam will review the PR by Friday." | python3 -m action_items.cli --attendees "Aiza,Sam"
```

## 8. The one thing worth remembering

**A schema tells the AI what shape to answer in. It cannot tell the AI which
names are real, because that set of real names only exists once you actually
see this specific meeting's notes.** That's why the grounding check in
`validate.py` exists as ordinary, boring, reliable code — running *after*
the AI answers — instead of trying to make the AI promise to behave.
