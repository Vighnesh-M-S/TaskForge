"""The four real tools the agent can call. Nothing is mocked.

Contract: every tool takes keyword string arguments and returns a string.
Tools never raise; a failure is returned as a string starting with ``ERROR:``
so the agent can observe it and adapt.
"""

import html
import os
import re
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

import httpx

ERROR_PREFIX = "ERROR:"
PYTHON_TIMEOUT_SECONDS = 10
HTTP_TIMEOUT_SECONDS = 10
MAX_SNIPPETS = 5

_USER_AGENT = "Mozilla/5.0 (compatible; TaskForge/1.0)"
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


# When set (by the web server), file tools are confined to this directory and run_python starts in it.
# Unset (the CLI), paths are relative to the current directory as usual.
WORKSPACE: ContextVar[Path | None] = ContextVar("taskforge_workspace", default=None)

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _resolve(path: str) -> Path:
    """Map a tool path to a real path, refusing anything outside the workspace when one is set."""
    workspace = WORKSPACE.get()
    if workspace is None:
        return Path(path)
    root = workspace.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError("path is outside the task workspace")
    return target


def _child_env() -> dict[str, str]:
    """Environment for run_python: the parent's, minus anything that looks like a credential."""
    return {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in _SECRET_MARKERS)}


def is_error(result: str) -> bool:
    """True if a tool result string signals failure."""
    return result.startswith(ERROR_PREFIX)


def _instant_answer(query: str) -> str:
    """DuckDuckGo Instant Answer API (no key). Returns '' when it has no answer."""
    r = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        headers={"User-Agent": _USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    r.raise_for_status()
    data = r.json()
    direct = data.get("Answer") or data.get("AbstractText") or data.get("Definition")
    if direct:
        return str(direct)
    topics = [t.get("Text", "") for t in data.get("RelatedTopics", []) if isinstance(t, dict)]
    return "\n".join(t for t in topics[:MAX_SNIPPETS] if t)


def _result_snippets(query: str) -> str:
    """DuckDuckGo HTML results page (no key). Returns the top result snippets."""
    r = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
    )
    r.raise_for_status()
    snippets = [html.unescape(_TAG_RE.sub("", s)).strip() for s in _SNIPPET_RE.findall(r.text)]
    return "\n".join(f"- {s}" for s in snippets[:MAX_SNIPPETS] if s)


def web_search(query: str) -> str:
    """Search the web with DuckDuckGo (no API key).

    Tries the Instant Answer API first, then falls back to result snippets
    from the HTML endpoint, since Instant Answer is empty for many queries.
    """
    try:
        if not query or not query.strip():
            return f"{ERROR_PREFIX} empty search query"
        failures: list[str] = []
        for source in (_instant_answer, _result_snippets):
            try:
                text = source(query)
            except Exception as exc:
                failures.append(f"{source.__name__}: {type(exc).__name__}: {exc}")
                continue
            if text:
                return text
        if failures:
            return f"{ERROR_PREFIX} web search failed ({'; '.join(failures)})"
        return f"{ERROR_PREFIX} no results for query: {query!r}"
    except Exception as exc:
        return f"{ERROR_PREFIX} web search failed: {type(exc).__name__}: {exc}"


def read_file(path: str) -> str:
    """Return the text contents of a file."""
    try:
        return _resolve(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"{ERROR_PREFIX} could not read {path}: {type(exc).__name__}: {exc}"


def write_file(path: str, content: str) -> str:
    """Write text to a file, creating parent directories as needed."""
    try:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written to {path} ({len(content)} chars)"
    except Exception as exc:
        return f"{ERROR_PREFIX} could not write {path}: {type(exc).__name__}: {exc}"


def run_python(code: str) -> str:
    """Run Python code in a subprocess (10s timeout) and return stdout, or the error."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT_SECONDS,
            cwd=WORKSPACE.get(),
            env=_child_env(),
        )
    except subprocess.TimeoutExpired:
        return f"{ERROR_PREFIX} python timed out after {PYTHON_TIMEOUT_SECONDS}s"
    except Exception as exc:
        return f"{ERROR_PREFIX} could not run python: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return f"{ERROR_PREFIX} python exited with code {result.returncode}\n{result.stderr or result.stdout}"
    return result.stdout or "(no output)"


TOOLS: dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "run_python": run_python,
}

TOOL_DESCRIPTIONS = """\
- web_search(query: str): DuckDuckGo search. Returns an instant answer or the top result snippets as text.
- read_file(path: str): Returns the text contents of a file.
- write_file(path: str, content: str): Writes text to a file (creates parent directories). Returns a confirmation.
- run_python(code: str): Runs Python 3 code in a fresh subprocess with a 10 second timeout and returns stdout.
  Standard library only. It has file and network access, and nothing persists between calls, so print what you need."""


def call_tool(name: str, args: dict[str, object]) -> str:
    """Dispatch a tool call by name. Never raises; bad names/arguments come back as ERROR strings."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"{ERROR_PREFIX} unknown tool {name!r}; available: {', '.join(TOOLS)}"
    try:
        return tool(**{k: str(v) for k, v in args.items()})
    except TypeError as exc:
        return f"{ERROR_PREFIX} bad arguments for {name}: {exc}"
    except Exception as exc:
        return f"{ERROR_PREFIX} {name} failed: {type(exc).__name__}: {exc}"
