"""TaskForge web server: one page, and one endpoint that streams a task run as server-sent events.

Run locally:  uvicorn web.app:app --reload
"""

import asyncio
import hmac
import json
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.nodes import active_model, active_provider
from agent.runner import describe, stream_task
from agent.tools import WORKSPACE

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = Path(__file__).with_name("index.html")
DEMO_INPUTS = ("amounts.csv", "invoice.txt")
MAX_TASK_CHARS = 1000
MAX_FILE_CHARS = 20_000
KEEPALIVE_SECONDS = 15

app = FastAPI(title="TaskForge")
# The free-tier model limits cannot serve parallel runs, so only one task runs at a time.
_run_lock = asyncio.Lock()


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=MAX_TASK_CHARS)


def _check_password(supplied: str | None) -> None:
    """Reject the request unless it carries TASKFORGE_PASSWORD (when one is configured)."""
    expected = os.environ.get("TASKFORGE_PASSWORD", "")
    if expected and not hmac.compare_digest((supplied or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Wrong or missing access password.")


def _new_workspace() -> Path:
    """A fresh temp directory holding copies of the demo inputs; each run is confined to its own."""
    workspace = Path(tempfile.mkdtemp(prefix="taskforge-"))
    (workspace / "demo").mkdir()
    for name in DEMO_INPUTS:
        shutil.copy(ROOT / "demo" / name, workspace / "demo" / name)
    return workspace


def _produced_files(workspace: Path) -> dict[str, str]:
    """Text of every file the run created in its workspace (the demo inputs are left out)."""
    inputs = {f"demo/{name}" for name in DEMO_INPUTS}
    files: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace).as_posix()
        if not path.is_file() or relative in inputs:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = "(not a text file)"
        files[relative] = text[:MAX_FILE_CHARS]
    return files


def _friendly_error(exc: Exception) -> str:
    """Error text for the page. Model quota errors get a plain explanation instead of the raw API payload."""
    text = str(exc)
    if "rate_limit" in text or "Error code: 429" in text:
        wait = re.search(r"try again in ([0-9hms.]*[hms])", text)
        scope = "daily" if "per day" in text else "per-minute"
        return (
            f"The model's {scope} free-tier quota is used up, so this run stopped before it could be verified."
            + (f" Try again in about {wait.group(1)}." if wait else " Try again later.")
        )
    return f"{type(exc).__name__}: {text[:500]}"


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _run(task: str, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
    """Run the task in its own workspace and push progress events onto the queue (None marks the end)."""
    workspace = _new_workspace()
    WORKSPACE.set(workspace)
    try:
        state: dict[str, Any] = {}
        async for node, update, state in stream_task(task):
            await queue.put({"type": "progress", "node": node, "text": describe(node, update)})
        await queue.put(
            {
                "type": "done",
                "verified": bool(state.get("verified")),
                "final_output": state.get("final_output", ""),
                "files": _produced_files(workspace),
            }
        )
    except Exception as exc:
        await queue.put({"type": "error", "message": _friendly_error(exc)})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        await queue.put(None)


async def _events(task: str) -> AsyncIterator[str]:
    """Server-sent events for one run, with keepalive comments so proxies do not drop a quiet connection."""
    async with _run_lock:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        runner = asyncio.create_task(_run(task, queue))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield _sse(event)
        finally:
            # Client disconnected or run finished: make sure the run stops and its workspace is removed.
            runner.cancel()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/api/info")
async def info() -> dict[str, Any]:
    """What the page needs to know before a run."""
    return {
        "provider": active_provider(),
        "model": active_model(),
        "password_required": bool(os.environ.get("TASKFORGE_PASSWORD")),
        "busy": _run_lock.locked(),
    }


@app.post("/api/run")
async def run(request: RunRequest, x_taskforge_password: str | None = Header(default=None)) -> StreamingResponse:
    """Run a task and stream its progress, then the final report and the files it produced."""
    _check_password(x_taskforge_password)
    if active_provider() is None:
        raise HTTPException(status_code=503, detail="No model API key is configured on the server.")
    if _run_lock.locked():
        raise HTTPException(status_code=429, detail="Another task is running. Try again in a minute.")
    return StreamingResponse(
        _events(request.task.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
