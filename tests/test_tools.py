"""Unit tests for the four tools. web_search is tested against a stubbed HTTP layer so the suite runs offline."""

from pathlib import Path

import httpx
import pytest

from agent import tools
from agent.tools import call_tool, is_error, read_file, run_python, web_search, write_file


class FakeResponse:
    def __init__(self, json_data: dict | None = None, text: str = "") -> None:
        self._json = json_data or {}
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json


# --- read_file / write_file -------------------------------------------------


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.txt"
    result = write_file(str(path), "hello ₹67,500")
    assert not is_error(result)
    assert "13 chars" in result
    assert read_file(str(path)) == "hello ₹67,500"


def test_read_missing_file_returns_error(tmp_path: Path) -> None:
    result = read_file(str(tmp_path / "missing.txt"))
    assert is_error(result)
    assert "missing.txt" in result


def test_write_to_directory_returns_error(tmp_path: Path) -> None:
    assert is_error(write_file(str(tmp_path), "x"))


# --- run_python ---------------------------------------------------------------


def test_run_python_returns_stdout() -> None:
    assert run_python("print(6 * 7)").strip() == "42"


def test_run_python_exception_returns_error() -> None:
    result = run_python("raise ValueError('boom')")
    assert is_error(result)
    assert "boom" in result


def test_run_python_no_output_is_not_error() -> None:
    assert run_python("x = 1") == "(no output)"


def test_run_python_timeout_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "PYTHON_TIMEOUT_SECONDS", 1)
    result = run_python("import time; time.sleep(5)")
    assert is_error(result)
    assert "timed out" in result


# --- web_search ---------------------------------------------------------------


def test_web_search_uses_instant_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda url, **kw: FakeResponse({"Answer": "42 is the answer"}))
    assert web_search("meaning of life") == "42 is the answer"


def test_web_search_falls_back_to_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    page = '<a class="result__snippet" href="x">1 <b>USD</b> = 83.4 INR &amp; rising</a>'

    def fake_get(url: str, **kw: object) -> FakeResponse:
        if "api.duckduckgo.com" in url:
            return FakeResponse({"Answer": "", "AbstractText": "", "RelatedTopics": []})
        return FakeResponse(text=page)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert web_search("usd to inr") == "- 1 USD = 83.4 INR & rising"


def test_web_search_network_failure_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kw: object) -> FakeResponse:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", fake_get)
    result = web_search("anything")
    assert is_error(result)
    assert "offline" in result


def test_web_search_no_results_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda url, **kw: FakeResponse({}, text="<html></html>"))
    assert is_error(web_search("zzzz"))


def test_web_search_empty_query_returns_error() -> None:
    assert is_error(web_search("  "))


# --- call_tool dispatch -------------------------------------------------------


def test_call_tool_dispatches(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    assert not is_error(call_tool("write_file", {"path": str(path), "content": "hi"}))
    assert call_tool("read_file", {"path": str(path)}) == "hi"


def test_call_tool_unknown_tool_returns_error() -> None:
    assert is_error(call_tool("delete_everything", {}))


def test_call_tool_bad_arguments_returns_error() -> None:
    assert is_error(call_tool("read_file", {"wrong": "arg"}))
