"""TaskForge CLI: python main.py "<task>" """

import asyncio
import sys
from typing import Any

from dotenv import load_dotenv

from agent.graph import build_graph
from agent.nodes import PROVIDERS, active_model, active_provider
from agent.state import AgentState

# Two passes (initial + one repair) of plan + up to 8 steps x 3 attempts + verify, with headroom.
RECURSION_LIMIT = 100


def _progress(node: str, update: dict[str, Any]) -> None:
    """Print a one-line trace of what each node just did (to stderr, so stdout is only the final answer)."""
    if node == "understand":
        line = f"goal: {update['understanding'].get('goal', '')}"
    elif node == "plan":
        line = "\n".join(["plan:"] + [f"    {i + 1}. {s}" for i, s in enumerate(update["plan"])])
    elif node == "execute":
        step = update["steps_taken"][-1]
        first_line = (step["result"].splitlines() or [""])[0][:160]
        line = f"step {step['step']} attempt {step['attempt']}: {step['tool']} -> {'ok' if step['ok'] else 'FAILED'}: {first_line}"
    elif node == "verify":
        line = f"verified={update['verified']}" + ("" if update["verified"] else f" - {update['verify_reason'].splitlines()[0][:200]}")
    else:
        line = "assembling final output"
    print(f"[{node}] {line}", file=sys.stderr, flush=True)


async def run_task(task: str) -> AgentState:
    """Run one task through the graph, printing progress, and return the final state."""
    graph = build_graph()
    state: AgentState = {"task": task}
    async for chunk in graph.astream(state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node, update in chunk.items():
            state.update(update)
            _progress(node, update)
    return state


def main() -> int:
    """CLI entry point. Exit code 0 if the task was verified, 1 otherwise."""
    load_dotenv()
    task = " ".join(sys.argv[1:]).strip() or input("Task: ").strip()
    if not task:
        print("No task given.", file=sys.stderr)
        return 2
    provider = active_provider()
    if provider is None:
        keys = ", ".join(cfg["key"] for cfg in PROVIDERS.values())
        print(f"No API key found. Copy .env.example to .env and set one of: {keys}", file=sys.stderr)
        return 2
    print(f"[llm] {provider}: {active_model()}", file=sys.stderr)
    try:
        state = asyncio.run(run_task(task))
    except Exception as exc:
        print(f"TaskForge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(state.get("final_output", "(no output)"))
    return 0 if state.get("verified") else 1


if __name__ == "__main__":
    sys.exit(main())
