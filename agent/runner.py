"""Runs a task through the graph and yields progress. Shared by the CLI and the web server."""

from collections.abc import AsyncIterator
from typing import Any

from agent.graph import build_graph
from agent.state import AgentState

# Two passes (initial + one repair) of plan + up to 8 steps x 3 attempts + verify, with headroom.
RECURSION_LIMIT = 100


def describe(node: str, update: dict[str, Any]) -> str:
    """One short human-readable description of what a node just did."""
    if node == "understand":
        return f"goal: {update['understanding'].get('goal', '')}"
    if node == "plan":
        return "\n".join(["plan:"] + [f"    {i + 1}. {s}" for i, s in enumerate(update["plan"])])
    if node == "execute":
        step = update["steps_taken"][-1]
        first_line = (step["result"].splitlines() or [""])[0][:160]
        return f"step {step['step']} attempt {step['attempt']}: {step['tool']} -> {'ok' if step['ok'] else 'FAILED'}: {first_line}"
    if node == "verify":
        if update["verified"]:
            return "verified=True"
        return f"verified=False - {(update['verify_reason'].splitlines() or [''])[0][:200]}"
    return "assembling final output"


async def stream_task(task: str) -> AsyncIterator[tuple[str, dict[str, Any], AgentState]]:
    """Run one task, yielding (node, that node's update, state so far) after every node."""
    state: AgentState = {"task": task}
    async for chunk in build_graph().astream(state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update)
            yield node, update, state
