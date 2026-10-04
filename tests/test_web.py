"""Tests for the web server and the workspace confinement it relies on. The LLM is scripted; the tools are real."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent import nodes, tools
from agent.tools import WORKSPACE, is_error, read_file, run_python, write_file
from web.app import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose 'LLM' reads the demo CSV and writes demo/out.txt."""
    for cfg in nodes.PROVIDERS.values():
        monkeypatch.delenv(cfg["key"], raising=False)
    monkeypatch.delenv("TASKFORGE_PROVIDER", raising=False)
    monkeypatch.delenv("TASKFORGE_PASSWORD", raising=False)
    calls = [
        {"tool": "read_file", "args": {"path": "demo/amounts.csv"}},
        {"tool": "write_file", "args": {"path": "demo/out.txt", "content": "5 rows"}},
    ]

    async def fake_ask_json(system: str, user: str) -> Any:
        if system is nodes.UNDERSTAND_SYSTEM:
            return {"goal": "count rows", "output_files": ["demo/out.txt"], "success_criteria": ["out.txt says 5 rows"]}
        if system is nodes.PLAN_SYSTEM:
            return {"steps": [{"tool": "read_file", "action": "read"}, {"tool": "write_file", "action": "write"}]}
        if system is nodes.EXECUTE_SYSTEM:
            return calls.pop(0)
        if system is nodes.VERIFY_SYSTEM:
            return {"verified": True, "reason": "ok"}
        return {"summary": "Counted 5 rows.", "caveats": []}

    monkeypatch.setattr(nodes, "ask_json", fake_ask_json)
    # /api/run refuses to start without a configured provider; the scripted LLM stands in for one.
    monkeypatch.setattr("web.app.active_provider", lambda: "scripted")
    return TestClient(app)


def events(response: Any) -> list[dict[str, Any]]:
    return [json.loads(part[6:]) for part in response.text.split("\n\n") if part.startswith("data: ")]


def test_index_page_loads(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "TaskForge" in response.text


def test_run_streams_progress_and_returns_produced_files(client: TestClient) -> None:
    response = client.post("/api/run", json={"task": "count the rows"})
    assert response.status_code == 200
    received = events(response)

    assert [e["node"] for e in received if e["type"] == "progress"] == ["understand", "plan", "execute", "execute", "verify", "complete"]
    done = received[-1]
    assert done["type"] == "done"
    assert done["verified"] is True
    # Only the file the run created is returned, not the demo inputs.
    assert done["files"] == {"demo/out.txt": "5 rows"}
    # The run happened in a throwaway workspace, not in the repository.
    assert not Path("demo/out.txt").exists()


def test_password_is_enforced_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASKFORGE_PASSWORD", "open-sesame")
    assert client.post("/api/run", json={"task": "x"}).status_code == 401
    assert client.post("/api/run", json={"task": "x"}, headers={"X-TaskForge-Password": "wrong"}).status_code == 401
    assert client.post("/api/run", json={"task": "x"}, headers={"X-TaskForge-Password": "open-sesame"}).status_code == 200
    assert client.get("/api/info").json()["password_required"] is True


def test_empty_and_oversized_tasks_are_rejected(client: TestClient) -> None:
    assert client.post("/api/run", json={"task": ""}).status_code == 422
    assert client.post("/api/run", json={"task": "x" * 1001}).status_code == 422


def test_file_tools_cannot_leave_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    token = WORKSPACE.set(workspace)
    try:
        assert is_error(read_file("../secret.txt"))
        assert is_error(read_file(str(secret)))
        assert is_error(write_file("../escaped.txt", "x"))
        assert not (tmp_path / "escaped.txt").exists()
        assert not is_error(write_file("sub/ok.txt", "fine"))
        assert read_file("sub/ok.txt") == "fine"
        assert Path(run_python("import os; print(os.getcwd())").strip()).resolve() == workspace.resolve()
    finally:
        WORKSPACE.reset(token)


def test_run_python_does_not_see_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_should_not_leak")
    monkeypatch.setenv("TASKFORGE_PASSWORD", "also-secret")
    output = run_python("import os; print(sorted(os.environ))")
    assert "GROQ_API_KEY" not in output
    assert "TASKFORGE_PASSWORD" not in output
    assert "PATH" in output
    assert tools._child_env().get("PATH")
