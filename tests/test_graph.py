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


# --- grounding of the narration, and enforcement of step success criteria -----

EVIDENCE = "Name,USD_Amount,EUR_Amount\nMeera Nair,89.99,79.98\nDaniel Kim,3000,2666.36\nPriya Menon,725.25,644.59\nrate 0.888786"


def test_ungrounded_numbers_flags_values_that_are_not_in_the_evidence() -> None:
    summary = "Converted 3 rows at 0.888786: Meera Nair 80.02, Daniel Kim 2668.36, Priya Menon 645.21."
    assert nodes.ungrounded_numbers(summary, EVIDENCE) == ["80.02", "2668.36", "645.21"]


def test_ungrounded_numbers_accepts_exact_values_in_other_formats() -> None:
    assert nodes.ungrounded_numbers("Total ₹67,500.00 exceeds ₹50,000; 3 rows, 2,666.36 EUR.", "TOTAL 67500, limit 50000\n2666.36") == []


def scripted_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, summaries: list[str], checks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run a one-step task whose run_python step prints 79.98 and writes out.txt; summaries/checks are consumed in order."""
    out = tmp_path / "out.txt"
    code = f"open({str(out)!r}, 'w').write('79.98'); print('79.98')"
    scripted_llm(monkeypatch, [{"tool": "run_python", "args": {"code": code}}] * 3, verified=True)
    base = nodes.ask_json
    remaining_summaries, remaining_checks = list(summaries), list(checks or [])

    async def ask(system: str, user: str) -> Any:
        if system is nodes.PLAN_SYSTEM:
            return {"steps": [{"tool": "run_python", "action": "compute", "success": "prints the amount"}]}
        if system is nodes.STEP_CHECK_SYSTEM:
            return remaining_checks.pop(0) if remaining_checks else {"met": True}
        if system is nodes.COMPLETE_SYSTEM:
            return {"summary": remaining_summaries.pop(0), "caveats": ["Rate was 0.5 yesterday."]}
        return await base(system, user)

    monkeypatch.setattr(nodes, "ask_json", ask)
    return run("convert it")


def test_summary_with_invented_numbers_is_reasked_then_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = scripted_run(monkeypatch, tmp_path, ["The amount is 80.02.", "The amount is 79.98."])
    assert "Summary: The amount is 79.98." in state["final_output"]
    assert "80.02" not in state["final_output"]
    # The caveat quoting a number that is nowhere in the evidence is dropped as well.
    assert "0.5" not in state["final_output"]


def test_summary_that_stays_ungrounded_is_replaced_and_disclosed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = scripted_run(monkeypatch, tmp_path, ["The amount is 80.02.", "The amount is 80.02."])
    assert "Summary: Ran 1 tool calls; the result passed verification." in state["final_output"]
    assert "discarded because it stated numbers not found in the evidence (80.02)" in state["final_output"]


def test_code_output_that_reports_an_error_is_retried_without_asking_the_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out.txt"
    scripted_llm(
        monkeypatch,
        [
            {"tool": "run_python", "args": {"code": "print(\"Error: 'rates'\")"}},
            {"tool": "run_python", "args": {"code": f"open({str(out)!r}, 'w').write('79.98'); print('79.98')"}},
            {"tool": "skip", "reason": "already written at plan step 1"},
        ],
        verified=True,
    )
    base = nodes.ask_json

    async def ask(system: str, user: str) -> Any:
        assert system is not nodes.STEP_CHECK_SYSTEM, "run_python results must not cost an LLM call to check"
        return await base(system, user)

    monkeypatch.setattr(nodes, "ask_json", ask)
    state = run("convert it")

    assert [(s["step"], s["attempt"], s["status"]) for s in state["steps_taken"]] == [(1, 1, "unmet"), (1, 2, "ok"), (2, 1, "ok")]
    assert "1. UNMET run_python (plan called for read_file) - the output reports an error (Error: 'rates')" in state["final_output"]


def test_search_result_checks() -> None:
    import asyncio as aio

    step = "[web_search] find the USD to EUR rate | success: snippet containing the numeric rate"
    # No digit at all: rejected without an LLM call (ask_json is not patched here, so a call would raise).
    assert aio.run(nodes._criterion_met(step, "web_search", "- Free, fast currency converter. Updated hourly.")) == (False, "the search result contains no number")
    assert aio.run(nodes._criterion_met(step, "run_python", "(no output)"))[0] is False
    # File tools are never checked.
    assert aio.run(nodes._criterion_met(step, "read_file", "anything")) == (True, "")
    # Code output that is not an error is accepted even if its format differs from the criterion's wording.
    assert aio.run(nodes._criterion_met(step, "run_python", "Name,USD_Amount,EUR_Amount\nA,1,0.89")) == (True, "")


def test_call_label_shows_target_source_and_deviation_from_plan() -> None:
    fetch = {"tool": "run_python", "args": {"code": "import urllib.request\nurllib.request.urlopen('https://wttr.in/Bangalore?format=j1')"}, "planned_tool": "web_search"}
    assert nodes.call_label(fetch) == "run_python [fetches wttr.in] (plan called for web_search)"
    assert nodes.call_label({"tool": "write_file", "args": {"path": "demo/weather.txt", "content": "x"}, "planned_tool": "run_python"}) == "write_file demo/weather.txt (plan called for run_python)"
    assert nodes.call_label({"tool": "web_search", "args": {"query": "weather Bangalore"}, "planned_tool": "web_search"}) == 'web_search "weather Bangalore"'
    assert nodes.call_label({"tool": "run_python", "args": {"code": "print(1)"}, "planned_tool": "run_python"}) == "run_python"


def test_report_explains_a_step_done_with_a_different_tool_and_a_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out.csv"
    # Plan is read_file then write_file; the agent writes at plan step 1 and skips plan step 2.
    scripted_llm(
        monkeypatch,
        [{"tool": "write_file", "args": {"path": str(out), "content": "n\n2\n"}}, {"tool": "skip", "reason": "already written at plan step 1"}],
        verified=True,
    )
    state = run("double it")

    assert f"1. ok    write_file {out} (plan called for read_file)" in state["final_output"]
    assert "2. ok    skip - already written at plan step 1" in state["final_output"]
