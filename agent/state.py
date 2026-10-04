"""Shared state passed between the LangGraph nodes."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Everything the agent knows about one task run.

    Nodes return partial dicts; LangGraph merges them into this state.
    """

    # The natural language task exactly as the user gave it.
    task: str
    # understand_node output: {"goal", "inputs", "expected_output", "output_files", "success_criteria"}.
    understanding: dict[str, Any]
    # plan_node output: ordered step descriptions (what to do + what success looks like).
    plan: list[str]
    # One record per tool call attempt: {"step", "attempt", "tool", "args", "result", "ok"}.
    steps_taken: list[dict[str, Any]]
    # Index into plan of the step execute_node will run next.
    current_step: int
    # How many times the current step has already been attempted (reset when the step advances).
    attempts: int
    # Human-readable log of what each attempt produced; fed back to the LLM so it can adapt.
    observations: list[str]
    # How many times verify_node has sent the run back to plan_node for a repair pass.
    replans: int
    # Final answer shown to the user.
    final_output: str
    # Set by verify_node after re-reading the real outputs.
    verified: bool
    # Why verify_node passed or failed the run.
    verify_reason: str
    # Raw proof: the re-read output files / results that the verdict was based on.
    evidence: str
