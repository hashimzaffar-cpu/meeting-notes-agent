from __future__ import annotations

import ast
import json
import re
import time
from typing import Protocol

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schema import Output
from .tools import build_verify_owner_tool

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
MAX_GENERATION_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
MAX_MALFORMED_ANSWER_RETRIES = 2
# smolagents defaults to 20 steps. This task is a single extraction with a
# handful of verify_owner calls at most -- left uncapped, a model that gets
# stuck (e.g. repeatedly failing to format a tool call) can spiral through
# dozens of steps and tens of thousands of tokens before giving up.
MAX_AGENT_STEPS = 8


class AgentRunner(Protocol):
    """Anything that can turn (notes, attendees) into a raw JSON string."""

    def run(self, notes: str, attendees: list[str]) -> str: ...


def _run_tool_calling_agent(model, notes: str, attendees: list[str]) -> str:
    """Build a ToolCallingAgent around `model` and run it, with retries.

    Shared by every backend (local or hosted): they all get the same
    `verify_owner` tool, the same step cap, and the same retry-on-transient-
    generation-error behavior. Only how `model` talks to the underlying LLM
    differs between backends.
    """
    from smolagents import ToolCallingAgent
    from smolagents.utils import AgentGenerationError

    verify_owner = build_verify_owner_tool(notes, attendees)
    agent = ToolCallingAgent(tools=[verify_owner], model=model, max_steps=MAX_AGENT_STEPS)
    task = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(notes, attendees)}"

    last_error: AgentGenerationError | None = None
    for attempt in range(MAX_GENERATION_RETRIES):
        try:
            return agent.run(task)
        except AgentGenerationError as e:
            # smolagents doesn't retry these itself -- they're almost always
            # a transient hiccup (e.g. "503 model is busy"), not a real
            # failure, so retry with backoff before giving up.
            last_error = e
            if attempt < MAX_GENERATION_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error


class LocalAgentRunner:
    """Runs a smolagents ToolCallingAgent on a model loaded locally via MLX.

    This is the default backend: it needs no API token, no per-call billing,
    and no external service to be up -- the model runs on-device (Apple
    Silicon, via `mlx-lm`). The first call downloads the model from the
    Hugging Face Hub once and caches it; every call after that is fully
    offline.
    """

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        from smolagents import MLXModel

        self._model_id = model
        self._model = MLXModel(model_id=model, max_tokens=2000)

    def run(self, notes: str, attendees: list[str]) -> str:
        return _run_tool_calling_agent(self._model, notes, attendees)


class GroqAgentRunner:
    """Runs a smolagents ToolCallingAgent on a Groq-hosted model.

    Groq's API is OpenAI-compatible, so this reuses smolagents' generic
    `OpenAIServerModel` pointed at Groq's endpoint -- no Groq-specific SDK
    needed. Groq's free tier is generous and its hosted models are large
    enough (e.g. 70B) to handle text-based tool calling reliably, unlike the
    small model `LocalAgentRunner` runs on-device.
    """

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, api_key: str | None = None):
        from smolagents import OpenAIServerModel

        self._model_id = model
        self._model = OpenAIServerModel(
            model_id=model,
            api_base=GROQ_API_BASE,
            api_key=api_key,
            client_kwargs={"timeout": 120.0},
            # Without an explicit cap, the request's *possible* output size
            # (not what the model actually generates) gets checked against
            # Groq's free-tier output-tokens-per-minute limit, which is easy
            # to exceed even though our actual answer is a short JSON blob.
            max_tokens=900,
        )

    def run(self, notes: str, attendees: list[str]) -> str:
        return _run_tool_calling_agent(self._model, notes, attendees)


class HFAgentRunner:
    """Runs a smolagents ToolCallingAgent, backed by a Hugging Face model.

    Unlike a single chat-completion call, this agent can call `verify_owner`
    (see tools.py) mid-reasoning to self-check a candidate name against the
    notes before committing to it. That tool only reduces how often a bad
    owner is proposed in the first place — it is not the guarantee. The
    pipeline's `ground_output` still runs on whatever the agent returns,
    because a tool the agent can choose not to use, or misuse, isn't a
    substitute for a deterministic check.

    Requires an HF_TOKEN with Inference Providers credits. Prefer
    `LocalAgentRunner` if you don't have (or have exhausted) those credits.
    """

    def __init__(
        self,
        model: str = DEFAULT_HF_MODEL,
        token: str | None = None,
        provider: str = "featherless-ai",
    ):
        from smolagents import InferenceClientModel

        self._model_id = model
        self._model = InferenceClientModel(model_id=model, token=token, provider=provider)

    def run(self, notes: str, attendees: list[str]) -> str:
        return _run_tool_calling_agent(self._model, notes, attendees)


def _parse_final_answer(raw: str):
    """Parse the agent's final answer, tolerating small-model formatting drift.

    A 7B model driven through text-based tool calling doesn't get a strict
    JSON grammar the way a single `response_format="json_object"` completion
    would — it sometimes emits Python-repr syntax (single quotes, bare
    `null`) instead of JSON when passing a nested object through a tool
    call's string argument. Try strict JSON first, then fall back to
    parsing it as a Python literal.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    normalized = re.sub(r"\bnull\b", "None", raw)
    normalized = re.sub(r"\btrue\b", "True", normalized)
    normalized = re.sub(r"\bfalse\b", "False", normalized)
    return ast.literal_eval(normalized)


def extract(notes: str, attendees: list[str], runner: AgentRunner) -> Output:
    """Run the agent and parse its answer, retrying on a garbled response.

    A small model doesn't reliably produce a well-formed final answer on
    every single run (occasionally it cuts itself off mid-object). That's a
    different failure than a hallucinated owner -- there's nothing to ground,
    the response just isn't usable -- so the fix is the same shape as the
    grounding check itself: don't trust the first answer blindly, verify it,
    and retry when it fails the check.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_MALFORMED_ANSWER_RETRIES):
        raw = runner.run(notes, attendees)
        try:
            data = _parse_final_answer(raw)
            if isinstance(data, list):
                # The model occasionally answers with just the actions list,
                # dropping the "unassigned" wrapper it was asked for.
                data = {"actions": data, "unassigned": []}
            return Output.model_validate(data)
        except Exception as e:
            last_error = e
    raise last_error
