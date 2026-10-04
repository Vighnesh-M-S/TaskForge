"""End-to-end test of the graph loop with a scripted LLM. The tools are real; only the model replies are stubbed."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agent import nodes
from agent.graph import build_graph


def scripted_llm(monkeypatch: pytest.MonkeyPatch, tool_calls: list[dict[str, Any]], verified: bool) -> None:
    """Replace nodes.ask_json with canned replies; execute replies are consumed in order."""
    remaining = list(tool_calls)
    # No provider key, so execute_node takes the JSON path and goes through ask_json too.
    for cfg in nodes.PROVIDERS.values():
        monkeypatch.delenv(cfg["key"], raising=False)
    monkeypatch.delenv("TASKFORGE_PROVIDER", raising=False)

    async def fake_ask_json(system: str, user: str) -> Any:
        if system is nodes.UNDERSTAND_SYSTEM:
            return {"goal": "double the amounts", "input_files": [], "output_files": [], "success_criteria": ["out has 2 rows"]}
        if system is nodes.PLAN_SYSTEM:
            return [
                {"tool": "read_file", "action": "read input", "success": "contents returned"},
                {"tool": "write_file", "action": "write output", "success": "file written"},
            ]
        if system is nodes.EXECUTE_SYSTEM:
            return remaining.pop(0)
        if system is nodes.VERIFY_SYSTEM:
            return {"verified": verified, "reason": "row counts match" if verified else "output missing"}
        return {"summary": "Doubled 2 rows.", "caveats": []}

    monkeypatch.setattr(nodes, "ask_json", fake_ask_json)


def run(task: str) -> dict[str, Any]:
    return asyncio.run(build_graph().ainvoke({"task": task}, {"recursion_limit": 60}))


def test_happy_path_runs_every_step_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("n\n1\n2\n")
    scripted_llm(
        monkeypatch,
        [
            {"tool": "read_file", "args": {"path": str(src)}},
            {"tool": "write_file", "args": {"path": str(out), "content": "n\n2\n4\n"}},
        ],
        verified=True,
    )
    state = run("double it")

    assert state["current_step"] == 2
    assert [s["tool"] for s in state["steps_taken"]] == ["read_file", "write_file"]
    assert out.read_text() == "n\n2\n4\n"
    assert state["verified"] is True
    # Evidence is the output file re-read from disk, even though understand listed no output files.
    assert "n\n2\n4" in state["evidence"]
    assert state["final_output"].startswith("TASK COMPLETE - VERIFIED")


def test_failed_step_is_retried_with_a_new_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("n\n1\n")
    scripted_llm(
        monkeypatch,
        [
            {"tool": "read_file", "args": {"path": str(tmp_path / "wrong.csv")}},
            {"tool": "read_file", "args": {"path": str(src)}},
            {"tool": "write_file", "args": {"path": str(out), "content": "n\n2\n"}},
        ],
        verified=True,
    )
    state = run("double it")

    assert [(s["step"], s["attempt"], s["ok"]) for s in state["steps_taken"]] == [(1, 1, False), (1, 2, True), (2, 1, True)]
    assert "Step 1 attempt 1 (read_file) failed" in state["final_output"]


def test_failed_verification_gets_one_repair_pass_then_reports_not_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"tool": "read_file", "args": {"path": str(tmp_path / "missing.csv")}}
    skip = {"tool": "skip", "reason": "nothing to write"}
    # First pass: step 1 fails every attempt, step 2 skipped. Repair pass: both steps skipped.
    scripted_llm(monkeypatch, [bad] * nodes.MAX_ATTEMPTS_PER_STEP + [skip, skip, skip], verified=False)
    state = run("double it")

    assert len(state["steps_taken"]) == nodes.MAX_ATTEMPTS_PER_STEP + 3
    assert state["replans"] == nodes.MAX_REPLANS
    assert any("Starting repair pass 1" in o for o in state["observations"])
    assert state["verified"] is False
    assert state["final_output"].startswith("TASK FINISHED - NOT VERIFIED")


def test_repair_pass_can_fix_a_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, out = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("n\n1\n")
    scripted_llm(
        monkeypatch,
        [
            {"tool": "read_file", "args": {"path": str(src)}},
            {"tool": "skip", "reason": "forgot to write"},
            {"tool": "read_file", "args": {"path": str(src)}},
            {"tool": "write_file", "args": {"path": str(out), "content": "n\n2\n"}},
        ],
        verified=False,
    )
    verdicts = iter([False, True])
    scripted = nodes.ask_json

    async def ask(system: str, user: str) -> Any:
        if system is nodes.VERIFY_SYSTEM:
            ok = next(verdicts)
            return {"verified": ok, "reason": "ok" if ok else "out.csv missing"}
        return await scripted(system, user)

    monkeypatch.setattr(nodes, "ask_json", ask)
    state = run("double it")

    assert state["replans"] == 1
    assert state["verified"] is True
    assert out.read_text() == "n\n2\n"


def test_extract_json_tolerates_fences_and_prose() -> None:
    reply = 'Here is the plan:\n```json\n{"tool": "read_file", "args": {"path": "a.csv"}}\n```'
    assert nodes._extract_json(reply) == {"tool": "read_file", "args": {"path": "a.csv"}}
    assert nodes._extract_json(json.dumps([1, 2])) == [1, 2]
    with pytest.raises(ValueError):
        nodes._extract_json("no json here")
