"""The five LangGraph nodes: understand, plan, execute, verify, complete.

Loop: Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete.
The LLM decides which tool to call at every step; nothing here is task-specific.
"""

import asyncio
import json
import os
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agent.state import AgentState
from agent.tools import TOOL_DESCRIPTIONS, TOOLS, call_tool, is_error, read_file, run_python

# Provider is chosen by which API key is set, checked in this order.
# xAI and Groq both expose OpenAI-compatible endpoints.
PROVIDERS: dict[str, dict[str, str]] = {
    "xai": {"key": "XAI_API_KEY", "model": "grok-4.7", "base_url": "https://api.x.ai/v1"},
    "groq": {"key": "GROQ_API_KEY", "model": "openai/gpt-oss-20b", "base_url": "https://api.groq.com/openai/v1"},
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": "claude-sonnet-5-5", "base_url": ""},
}
MAX_ATTEMPTS_PER_STEP = 3
MAX_PLAN_STEPS = 8
MAX_REPLANS = 1
OBSERVATION_CHAR_LIMIT = 6000
EVIDENCE_CHAR_LIMIT = 8000
PREVIEW_LINES = 6


def _resolve() -> tuple[str, str] | None:
    """(provider, api_key) to use: TASKFORGE_PROVIDER if set, else the first provider whose API key is present.

    A Groq key ("gsk_...") pasted under XAI_API_KEY is still routed to Groq, since the two names are easy to mix up.
    """
    forced = os.environ.get("TASKFORGE_PROVIDER", "").strip().lower()
    for name, cfg in PROVIDERS.items():
        key = os.environ.get(cfg["key"], "").strip()
        if forced and name != forced:
            continue
        if key:
            if not forced and name == "xai" and key.startswith("gsk_"):
                return "groq", key
            return name, key
    return None


def active_provider() -> str | None:
    """Name of the provider that will be used, or None if no API key is set."""
    resolved = _resolve()
    return resolved[0] if resolved else None


def active_model() -> str:
    """Model ID that will be used: TASKFORGE_MODEL if set, else the provider's default."""
    provider = active_provider()
    return os.environ.get("TASKFORGE_MODEL") or (PROVIDERS[provider]["model"] if provider else "")


@lru_cache(maxsize=2)
def get_llm(json_mode: bool = True) -> BaseChatModel:
    """Build the chat model for whichever provider has an API key in the environment (cached).

    json_mode only affects the OpenAI-compatible providers: it constrains the reply to a JSON
    object, and must be off when the model is given native tools to call.
    """
    resolved = _resolve()
    if resolved is None:
        raise RuntimeError(f"No API key found. Set one of: {', '.join(c['key'] for c in PROVIDERS.values())}")
    provider, api_key = resolved
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=active_model(), max_tokens=16000, output_config={"effort": "medium"})
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=active_model(),
        api_key=api_key,
        base_url=PROVIDERS[provider]["base_url"],
        max_tokens=8000,
        # Free tiers have low tokens-per-minute limits; back off and retry on 429s.
        max_retries=8,
        model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
    )


def _clip(text: str, limit: int) -> str:
    """Shorten long tool output for the prompt, saying how much was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... truncated {len(text) - limit} more chars]"


def _extract_json(text: str) -> Any:
    """Parse the first JSON object/array found in an LLM reply (tolerates fences and prose)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model reply: {text[:200]!r}")


JSON_ONLY_NUDGE = (
    "Your previous reply could not be used. Reply with one plain JSON value as text and nothing else. "
    "Do not invoke tools or functions yourself; name the tool inside the JSON."
)
LLM_ATTEMPTS = 3


async def ask_json(system: str, user: str) -> Any:
    """Ask the model for a JSON reply and parse it, re-asking if the reply is rejected or is not valid JSON."""
    messages: list[tuple[str, str]] = [("system", system), ("human", user)]
    last_error: Exception | None = None
    for _ in range(LLM_ATTEMPTS):
        try:
            response = await get_llm().ainvoke(messages)
        except Exception as exc:
            # Open-weight models sometimes emit a native tool call, which the API rejects with a 400.
            if "tool_use_failed" not in str(exc) and "json_validate_failed" not in str(exc):
                raise
            last_error = exc
        else:
            if response.response_metadata.get("stop_reason") == "refusal":
                raise RuntimeError("The model declined this request (stop_reason=refusal).")
            try:
                return _extract_json(response.text)
            except ValueError as exc:
                last_error = exc
        messages = [("system", system), ("human", f"{user}\n\n{JSON_ONLY_NUDGE}")]
    raise RuntimeError(f"Model did not return valid JSON after {LLM_ATTEMPTS} attempts: {last_error}")


def _as_list(value: Any) -> list[str]:
    """Coerce an LLM-provided field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _history(state: AgentState) -> str:
    """All observations so far, as prompt text."""
    observations = state.get("observations", [])
    return "\n\n".join(observations) if observations else "(nothing has been executed yet)"


UNDERSTAND_SYSTEM = """\
You are the "understand" stage of an autonomous task worker. Read the user's task and extract what must be achieved.
Do not solve the task. Reply with JSON only, in this shape:
{
  "goal": "one sentence stating the outcome the user wants",
  "inputs": ["every input the task depends on: files, values, facts to look up"],
  "input_files": ["paths of files that must be read"],
  "expected_output": "what the finished result looks like",
  "output_files": ["paths of every file the task may create; if it creates one of several alternatives depending on a condition, list all alternatives"],
  "success_criteria": ["concrete, checkable conditions that prove the task was done correctly"]
}
Success criteria must be checkable by re-reading the outputs later, e.g. "output.csv has one data row per data row in amounts.csv",
or "exactly one of approved.txt / rejected.txt exists and it states the reason".
Do not demand more precision than the task asks for: money amounts rounded to 2 decimal places are correct."""


async def understand_node(state: AgentState) -> dict[str, Any]:
    """UNDERSTAND: turn the raw task into a structured goal, inputs, expected output and success criteria.

    First step of the loop. The success criteria written here are what verify_node
    checks the real outputs against at the end, so the agent is graded on the goal
    it committed to before doing any work.
    """
    understanding = await ask_json(UNDERSTAND_SYSTEM, f"Task: {state['task']}")
    if not isinstance(understanding, dict):
        raise RuntimeError("understand_node expected a JSON object from the model")
    for key in ("inputs", "input_files", "output_files", "success_criteria"):
        understanding[key] = _as_list(understanding.get(key))
    return {
        "understanding": understanding,
        "plan": [],
        "steps_taken": [],
        "observations": [],
        "current_step": 0,
        "attempts": 0,
        "replans": 0,
        "verified": False,
        "verify_reason": "",
    }


PLAN_SYSTEM = f"""\
You are the "plan" stage of an autonomous task worker. Break the task into an ordered sequence of tool calls.

Available tools:
{TOOL_DESCRIPTIONS}

Rules:
- One tool call per step, at most {MAX_PLAN_STEPS} steps, as few as the task needs.
- You do not know the contents of files or the results of lookups yet, so describe what each step must achieve
  instead of inventing values. The exact tool arguments are chosen later, once earlier results are known.
- web_search often returns only snippets without an exact number. For live data (exchange rates, weather),
  plan the lookup as one step and expect the executor to fall back to run_python calling a free no-key JSON API.
- Do not add a verification step; a separate stage re-reads the outputs.
- Do not create intermediate files. Values pass between steps through each step's printed result.
- The final step must produce the output the task asks for (e.g. actually write the output file).

Reply with JSON only:
{{"steps": [{{"tool": "<tool name>", "action": "what this step does", "success": "what a successful result looks like"}}]}}"""


async def plan_node(state: AgentState) -> dict[str, Any]:
    """PLAN: have the LLM choose the ordered tool sequence for this task.

    The plan fixes the intent and success condition of each step but not the exact
    arguments, because those depend on results that do not exist yet (file contents,
    a fetched rate). execute_node fills them in step by step.

    Also the ADAPT entry point after a failed verification: it is then given the
    execution log and the verifier's findings, and plans only the steps needed to repair the result.
    """
    user = f"Task: {state['task']}\n\nUnderstanding:\n{json.dumps(state['understanding'], indent=2, ensure_ascii=False)}"
    repairing = bool(state.get("verify_reason")) and not state.get("verified")
    if repairing:
        user += (
            f"\n\nA first attempt was executed but FAILED verification.\n"
            f"Execution log:\n{_history(state)}\n\n"
            f"Verifier findings: {state['verify_reason']}\n\n"
            "Plan only the steps still needed to fix this. Reuse results already in the log; do not repeat steps that worked."
        )
    raw = await ask_json(PLAN_SYSTEM, user)
    if isinstance(raw, dict):
        raw = raw.get("steps") or raw.get("plan") or []
    plan: list[str] = []
    for step in raw[:MAX_PLAN_STEPS]:
        if isinstance(step, dict):
            if not str(step.get("action", "")).strip():
                continue
            plan.append(f"[{step.get('tool', '?')}] {step.get('action', '')} | success: {step.get('success', '')}")
        elif str(step).strip():
            plan.append(str(step))
    if not plan:
        raise RuntimeError("plan_node got an empty plan from the model")
    update: dict[str, Any] = {"plan": plan, "current_step": 0, "attempts": 0}
    if repairing:
        update["replans"] = state.get("replans", 0) + 1
        update["observations"] = state.get("observations", []) + [
            f"--- Verification failed ({state['verify_reason']}). Starting repair pass {update['replans']}; step numbers restart. ---"
        ]
    return update


_EXECUTE_RULES = f"""\
You are the "execute" stage of an autonomous task worker. Choose the single tool call that carries out the current plan step.

Available tools:
{TOOL_DESCRIPTIONS}

Rules:
- Use real values from the observations (file contents, fetched numbers). Never invent data.
- If an earlier attempt at this step failed or returned nothing usable, try a different approach, not the same call.
  If web_search did not give an exact value, use run_python with urllib to call a free no-key JSON API, for example
  https://open.er-api.com/v6/latest/USD for exchange rates or https://wttr.in/<city>?format=j1 for weather.
- A step whose purpose was already fully achieved by an earlier step can be skipped.
- run_python starts fresh each time: re-read files and re-define values inside the code, and print the results you need.
- When writing a file, write the complete final content.
"""

EXECUTE_SYSTEM = _EXECUTE_RULES + """
Reply with JSON only:
{"tool": "<tool name>", "args": {...}, "reason": "why this call"}
or, to skip the step:
{"tool": "skip", "reason": "why it is unnecessary"}"""

EXECUTE_SYSTEM_NATIVE = _EXECUTE_RULES + """
Call exactly one tool for the current step. To skip the step, call the skip tool."""


def _tool_schema(name: str, description: str, params: dict[str, str]) -> dict[str, Any]:
    """OpenAI-style function tool definition with all-string, all-required parameters."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool_schema("web_search", "Search the web with DuckDuckGo; returns an instant answer or result snippets.", {"query": "search query"}),
    _tool_schema("read_file", "Return the text contents of a file.", {"path": "file path"}),
    _tool_schema("write_file", "Write text to a file, creating parent directories.", {"path": "file path", "content": "complete file content"}),
    _tool_schema("run_python", "Run Python 3 code in a fresh subprocess (10s timeout, stdlib only) and return stdout.", {"code": "Python source code"}),
    _tool_schema("skip", "Skip the current step because it is unnecessary.", {"reason": "why it is unnecessary"}),
]


async def choose_tool(user: str) -> dict[str, Any]:
    """Have the LLM pick the tool call for the current step. Returns {"tool", "args", "reason"}.

    Claude replies with the call as JSON text. The OpenAI-compatible models (Grok, Groq) are
    given the tools natively instead, because they tend to emit a native call regardless.
    """
    if active_provider() in (None, "anthropic"):
        decision = await ask_json(EXECUTE_SYSTEM, user)
        return decision if isinstance(decision, dict) else {"tool": "invalid", "args": {}}

    llm = get_llm(json_mode=False).bind_tools(TOOL_SCHEMAS, tool_choice="required")
    messages: list[tuple[str, str]] = [("system", EXECUTE_SYSTEM_NATIVE), ("human", user)]
    last_error = "model made no tool call"
    for _ in range(LLM_ATTEMPTS):
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            # A malformed call (e.g. arguments that are not valid JSON) is rejected by the API with a 400.
            if "tool_use_failed" not in str(exc):
                raise
            last_error = str(exc)
        else:
            if response.tool_calls:
                call = response.tool_calls[0]
                args = dict(call["args"])
                if call["name"] == "skip":
                    return {"tool": "skip", "args": {}, "reason": args.get("reason", "")}
                return {"tool": call["name"], "args": args}
        messages = [
            ("system", EXECUTE_SYSTEM_NATIVE),
            ("human", f"{user}\n\nYour previous tool call was malformed. Call one tool with valid JSON arguments; every argument is a JSON string."),
        ]
    # Surface it as a failed attempt so the step-level retry and the final report see it.
    return {"tool": "invalid", "args": {}, "error": _clip(last_error, 500)}


STEP_CHECK_SYSTEM = """\
You check one step of an autonomous task worker. Given the step (with its success criterion) and the tool result,
decide whether the result actually meets the criterion. Judge only what the result contains: a search result that
does not contain the needed value, or code output that lacks what the step was supposed to produce, does not meet it.
Do not check arithmetic.

Reply with JSON only: {"met": true or false, "reason": "one short sentence"}"""

STEP_CHECK_RESULT_LIMIT = 2000
# read_file and write_file results are deterministic (contents / confirmation), so only these are judged.
CHECKED_TOOLS = ("web_search", "run_python")


async def _criterion_met(step: str, tool: str, result: str) -> tuple[bool, str]:
    """Ask the LLM whether a successful tool result meets the plan step's own success criterion."""
    user = f"Step: {step}\n\nTool called: {tool}\n\nResult:\n{_clip(result, STEP_CHECK_RESULT_LIMIT)}"
    try:
        check = await ask_json(STEP_CHECK_SYSTEM, user)
    except RuntimeError:
        return True, ""
    if isinstance(check, dict) and check.get("met") is False:
        return False, str(check.get("reason", "")).strip() or "result does not meet the step's success criterion"
    return True, ""


async def execute_node(state: AgentState) -> dict[str, Any]:
    """EXECUTE + OBSERVE + ADAPT: run one tool call for the current plan step and record what happened.

    Each invocation makes exactly one attempt. The LLM picks the concrete tool and
    arguments from everything observed so far. A failed attempt leaves current_step
    unchanged so the graph loops back here and the LLM retries with a different
    approach; after MAX_ATTEMPTS_PER_STEP the step is recorded as failed and the
    agent moves on, leaving verify_node to judge the consequences.

    A tool call that runs without error but does not meet the step's own success
    criterion (e.g. a search that returns no usable value) also counts as a failed
    attempt, recorded with status "unmet", so the fallback is visible in the trace.
    """
    index = state["current_step"]
    attempt = state.get("attempts", 0) + 1
    plan = state["plan"]
    numbered_plan = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    user = (
        f"Task: {state['task']}\n\n"
        f"Plan:\n{numbered_plan}\n\n"
        f"Observations so far:\n{_history(state)}\n\n"
        f"Current step: {index + 1} of {len(plan)} (attempt {attempt} of {MAX_ATTEMPTS_PER_STEP}):\n{plan[index]}"
    )
    decision = await choose_tool(user)
    tool = str(decision.get("tool", ""))
    args = decision.get("args") if isinstance(decision.get("args"), dict) else {}

    note = ""
    if tool == "skip":
        result = f"Skipped: {decision.get('reason', 'no reason given')}"
        status = "ok"
    elif decision.get("error"):
        result = f"ERROR: model produced an unusable tool call: {decision['error']}"
        status = "error"
    else:
        result = await asyncio.to_thread(call_tool, tool, args)
        status = "error" if is_error(result) else "ok"
        if status == "ok" and tool in CHECKED_TOOLS:
            met, note = await _criterion_met(plan[index], tool, result)
            if not met:
                status = "unmet"
    ok = status == "ok"

    record = {
        "step": index + 1, "attempt": attempt, "tool": tool, "args": args,
        "result": result, "ok": ok, "status": status, "note": note,
    }
    shown_args = json.dumps(args, ensure_ascii=False)
    label = {"ok": "Result", "error": "FAILED", "unmet": f"SUCCESS CRITERION NOT MET ({note}). Result was"}[status]
    observation = (
        f"Step {index + 1}, attempt {attempt}: {tool}({_clip(shown_args, 1500)})\n"
        f"{label}: {_clip(result, OBSERVATION_CHAR_LIMIT)}"
    )

    advance = ok or attempt >= MAX_ATTEMPTS_PER_STEP
    if not ok and advance:
        observation += f"\nGave up on step {index + 1} after {attempt} attempts; moving on."
    return {
        "steps_taken": state.get("steps_taken", []) + [record],
        "observations": state.get("observations", []) + [observation],
        "current_step": index + 1 if advance else index,
        "attempts": 0 if advance else attempt,
    }


VERIFY_CHECK_SYSTEM = """\
You are preparing the "verify" stage of an autonomous task worker. Write a short Python 3 script (standard library only,
no network) that checks the success criteria against the files on disk: it re-reads the input and output files itself,
recomputes any numbers (using looked-up figures from the execution log, such as an exchange rate, as constants), and
prints one line per criterion starting with PASS or FAIL plus the actual values it found. It must not modify any file.
Allow for rounding to the precision used in the output.

Reply with JSON only: {"code": "<the script>"} or {"code": ""} if nothing can be checked by a script."""

VERIFY_SYSTEM = """\
You are the "verify" stage of an autonomous task worker. You are given the task, its success criteria, the execution log,
and the files as they exist on disk right now (re-read after execution). Decide whether the task was really completed.

Judge from the re-read files, not from what the execution log claims. Check every success criterion. Where the task
transforms a file, compare input and output (e.g. the same number of data rows, values computed correctly from the
inputs and the looked-up figures in the log). Never do arithmetic or counting in your head: where a check script's output is provided,
take its PASS/FAIL lines as the result for numbers and row counts, unless the script itself crashed.
A missing or empty required output, a mismatch, or a step that was given
up on and that the result depends on, means not verified.

Reply with JSON only:
{"verified": true or false, "reason": "what you checked and what you found, with concrete numbers", "issues": ["anything missing or wrong"]}"""


def _file_report(paths: list[str], label: str) -> str:
    """Re-read files from disk and format them for the verifier and for evidence."""
    blocks: list[str] = []
    for path in paths:
        content = read_file(path)
        if is_error(content):
            blocks.append(f"{label} {path}: (does not exist or is unreadable)")
        else:
            lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
            blocks.append(f"{label} {path} ({lines} lines, {len(content)} chars):\n{_clip(content, EVIDENCE_CHAR_LIMIT)}")
    return "\n\n".join(blocks)


async def verify_node(state: AgentState) -> dict[str, Any]:
    """VERIFY: re-read the real outputs from disk and check them against the success criteria.

    Does not trust the execution log. It reads every output file the task could
    have produced (plus anything written by write_file) and the input files, then
    asks the LLM to check each criterion from understand_node against that ground
    truth. Sets verified and verify_reason, and stores what it read as evidence.
    """
    understanding = state["understanding"]
    written = [
        str(s["args"].get("path"))
        for s in state.get("steps_taken", [])
        if s["tool"] == "write_file" and s["ok"] and s["args"].get("path")
    ]
    output_files = list(dict.fromkeys(understanding.get("output_files", []) + written))
    input_files = [p for p in understanding.get("input_files", []) if p not in output_files]

    if output_files:
        evidence = await asyncio.to_thread(_file_report, output_files, "Output file")
    else:
        successes = [s for s in state.get("steps_taken", []) if s["ok"] and s["tool"] in TOOLS]
        last = successes[-1]["result"] if successes else "(no successful tool call)"
        evidence = f"No output file expected. Last successful result:\n{_clip(last, EVIDENCE_CHAR_LIMIT)}"
    inputs_report = await asyncio.to_thread(_file_report, input_files, "Input file") if input_files else "(none)"

    user = (
        f"Task: {state['task']}\n\n"
        f"Expected output: {understanding.get('expected_output', '')}\n"
        f"Success criteria:\n" + "\n".join(f"- {c}" for c in understanding.get("success_criteria", [])) + "\n\n"
        f"Execution log:\n{_history(state)}\n\n"
        f"Input files on disk:\n{inputs_report}\n\n"
        f"Outputs on disk now:\n{evidence}"
    )
    # Arithmetic and row counts are checked by running code, not by the LLM reading numbers.
    try:
        check = await ask_json(VERIFY_CHECK_SYSTEM, user)
    except RuntimeError:
        check = {}
    code = str(check.get("code", "")).strip() if isinstance(check, dict) else ""
    if code:
        check_output = await asyncio.to_thread(run_python, code)
        user += f"\n\nCheck script output:\n{_clip(check_output, OBSERVATION_CHAR_LIMIT)}"
        evidence += f"\n\nCheck script output:\n{_clip(check_output, EVIDENCE_CHAR_LIMIT)}"

    verdict = await ask_json(VERIFY_SYSTEM, user)
    if not isinstance(verdict, dict):
        verdict = {"verified": False, "reason": "verifier returned an unusable reply"}
    reason = str(verdict.get("reason", ""))
    issues = _as_list(verdict.get("issues"))
    if issues:
        reason += "\nIssues: " + "; ".join(issues)
    return {"verified": verdict.get("verified") is True, "verify_reason": reason, "evidence": evidence}


COMPLETE_SYSTEM = """\
You are the "complete" stage of an autonomous task worker. Write the final report line for the user.
State what was done and the result. Every number you write (rates, amounts, counts) must be copied character for
character from the evidence or the execution log; never compute, round or restate a number yourself. Do not list
per-row values: the evidence section shows the file itself. If verification failed, say plainly what is wrong or
missing. Do not claim anything the log does not show.

Reply with JSON only: {"summary": "2-3 sentences", "caveats": ["limits or assumptions the user should know, if any"]}"""

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Small whole numbers (step numbers, counts of rows/files) are not treated as factual claims to ground.
_UNGROUNDED_EXEMPT_MAX = 31


def _numbers(text: str) -> dict[Decimal, str]:
    """Every number in the text, normalised (commas and trailing zeros removed) -> as first written."""
    found: dict[Decimal, str] = {}
    for token in _NUMBER_RE.findall(text):
        try:
            value = Decimal(token.replace(",", "").rstrip(".")).normalize()
        except InvalidOperation:
            continue
        found.setdefault(value, token.rstrip(",."))
    return found


def ungrounded_numbers(text: str, sources: str) -> list[str]:
    """Numbers stated in LLM-written text that appear nowhere in the sources it was written from."""
    known = _numbers(sources)
    return [
        written
        for value, written in _numbers(text).items()
        if value not in known and not (value == value.to_integral_value() and value <= _UNGROUNDED_EXEMPT_MAX)
    ]


def _grounding_sources(state: AgentState) -> str:
    """Everything a report may legitimately quote numbers from: the task, real tool results, re-read evidence."""
    results = "\n".join(str(s["result"]) for s in state.get("steps_taken", []))
    return f"{state['task']}\n{results}\n{state.get('evidence', '')}"


def _fallback_summary(state: AgentState) -> str:
    """Summary built only from state, used when the LLM's summary cannot be grounded."""
    steps = state.get("steps_taken", [])
    wrote = [str(s["args"].get("path")) for s in steps if s["tool"] == "write_file" and s["ok"]]
    files = list(dict.fromkeys(state.get("understanding", {}).get("output_files", []) + wrote))
    outcome = "passed verification" if state.get("verified") else "did not pass verification"
    target = f" Output files: {', '.join(files)}." if files else ""
    return f"Ran {len(steps)} tool calls; the result {outcome}.{target} The exact values are in the evidence below."


def _preview(evidence: str) -> str:
    """First few lines of each evidence block, for the final report."""
    blocks: list[str] = []
    for block in evidence.split("\n\n"):
        lines = block.splitlines()
        shown = lines[: PREVIEW_LINES + 1]
        if len(lines) > len(shown):
            shown.append(f"  ... ({len(lines) - len(shown)} more lines)")
        blocks.append("\n".join(f"  {line}" for line in shown))
    return "\n".join(blocks)


async def complete_node(state: AgentState) -> dict[str, Any]:
    """COMPLETE: assemble the final answer: summary, steps taken, verification verdict, evidence and caveats.

    Last step of the loop. The summary is written by the LLM, then checked: every
    number in it must appear in the tool results or the re-read evidence. An
    ungrounded summary is re-asked once and otherwise replaced by one built from
    state, so the narration cannot contradict the files under a "verified" verdict.
    The step list, verdict and evidence preview are assembled from state.
    """
    steps = state.get("steps_taken", [])
    sources = _grounding_sources(state)
    user = (
        f"Task: {state['task']}\n\n"
        f"Execution log:\n{_history(state)}\n\n"
        f"Evidence (re-read from disk):\n{state.get('evidence', '')}\n\n"
        f"Verification: {'PASSED' if state.get('verified') else 'FAILED'} - {state.get('verify_reason', '')}"
    )
    caveats: list[str] = []
    summary = ""
    prompt = user
    for _ in range(2):
        report = await ask_json(COMPLETE_SYSTEM, prompt)
        if not isinstance(report, dict):
            report = {"summary": str(report), "caveats": []}
        summary = str(report.get("summary", ""))
        invented = ungrounded_numbers(summary, sources)
        if not invented:
            break
        prompt = (
            f"{user}\n\nYour previous summary stated numbers that are not in the evidence or the log: "
            f"{', '.join(invented)}. Rewrite it using only numbers copied exactly from the evidence, or no numbers."
        )
    else:
        summary = _fallback_summary(state)
        caveats.append(
            f"The model's own summary was discarded because it stated numbers not found in the evidence ({', '.join(invented)})."
        )

    # Caveats are LLM-written too: keep only those whose numbers are grounded.
    caveats += [c for c in _as_list(report.get("caveats")) if not ungrounded_numbers(c, sources)]
    for failed in (s for s in steps if not s["ok"]):
        if failed.get("status") == "unmet":
            caveats.append(
                f"Step {failed['step']} attempt {failed['attempt']} ({failed['tool']}) ran but did not meet "
                f"the step's success criterion: {failed.get('note', '')}"
            )
        else:
            first_line = (failed["result"].splitlines() or [""])[0]
            caveats.append(f"Step {failed['step']} attempt {failed['attempt']} ({failed['tool']}) failed: {first_line}")

    labels = {"ok": "ok   ", "error": "FAIL ", "unmet": "UNMET"}
    step_lines = [
        f"  {s['step']}. {labels[s.get('status', 'ok' if s['ok'] else 'error')]} {s['tool']}"
        + (f" (attempt {s['attempt']})" if s["attempt"] > 1 else "")
        + (f" - {s['note']}" if s.get("note") else "")
        for s in steps
    ]
    verified = bool(state.get("verified"))
    verify_reason = state.get("verify_reason", "")
    unbacked = ungrounded_numbers(verify_reason, sources)
    if unbacked:
        verify_reason += (
            f"\n(The verifier's wording quotes numbers not found in the evidence: {', '.join(unbacked)}. "
            "Rely on the check script output and the files below.)"
        )
    sections = [
        "TASK COMPLETE - VERIFIED" if verified else "TASK FINISHED - NOT VERIFIED",
        f"Summary: {summary}",
        "Steps taken:\n" + "\n".join(step_lines),
        f"Verification: {'passed' if verified else 'failed'}. {verify_reason}",
        "Evidence (re-read from disk):\n" + _preview(state.get("evidence", "")),
    ]
    if caveats:
        sections.append("Caveats:\n" + "\n".join(f"  - {c}" for c in caveats))
    return {"final_output": "\n\n".join(sections)}
