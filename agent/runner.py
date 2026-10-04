"""Runs a task through the graph and yields progress. Shared by the CLI and the web server."""

from collections.abc import AsyncIterator
from typing import Any

from agent.graph import build_graph
from agent.state import AgentState

# Two passes (initial + one repair) of plan + up to 8 steps x 3 attempts + verify, with headroom.
RECURSION_LIMIT = 100


def _brief(result: str, limit: int = 160) -> str:
    """First line of a tool result, plus how much more there is, so a multi-line result does not look empty."""
    lines = result.splitlines() or [""]
    more = f" (+{len(lines) - 1} more lines)" if len(lines) > 1 else ""
    return f"{lines[0][:limit]}{more}"


def describe(node: str, update: dict[str, Any]) -> str:
    """One short human-readable description of what a node just did."""
    if node == "understand":
        return f"goal: {update['understanding'].get('goal', '')}"
    if node == "plan":
        return "\n".join(["plan:"] + [f"    {i + 1}. {s}" for i, s in enumerate(update["plan"])])
    if node == "execute":
        step = update["steps_taken"][-1]
        head = f"step {step['step']} attempt {step['attempt']}: {step['tool']}"
        status = step.get("status", "ok" if step["ok"] else "error")
        if status == "unmet":
            return f"{head} -> CRITERION NOT MET: {step.get('note', '')} | got: {_brief(step['result'])}"
        return f"{head} -> {'ok' if status == 'ok' else 'FAILED'}: {_brief(step['result'])}"
    if node == "verify":
        reason = " ".join(update.get("verify_reason", "").split())[:300]
        return f"verified={update['verified']} - {reason}"
    return "assembling final output"


async def stream_task(task: str) -> AsyncIterator[tuple[str, dict[str, Any], AgentState]]:
    """Run one task, yielding (node, that node's update, state so far) after every node."""
    state: AgentState = {"task": task}
    async for chunk in build_graph().astream(state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update)
            yield node, update, state
