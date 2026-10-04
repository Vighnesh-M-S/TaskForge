"""TaskForge CLI: python main.py "<task>" """

import asyncio
import sys

from dotenv import load_dotenv

from agent.nodes import PROVIDERS, active_model, active_provider
from agent.runner import describe, stream_task
from agent.state import AgentState


async def run_task(task: str) -> AgentState:
    """Run one task through the graph, printing progress to stderr, and return the final state."""
    state: AgentState = {"task": task}
    async for node, update, state in stream_task(task):
        print(f"[{node}] {describe(node, update)}", file=sys.stderr, flush=True)
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
