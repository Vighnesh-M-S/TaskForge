# TaskForge

Give it a natural language task involving a file and a web lookup. It plans the steps, executes them using real tools, verifies the outcome, and returns evidence.

```
$ python main.py "Read demo/amounts.csv, find today's USD to INR exchange rate, convert each USD amount to INR, and write the results to demo/output.csv"
```

TaskForge works out that this is a currency conversion on a file, plans the tool calls, fetches the rate, reads the CSV, computes the amounts, writes `demo/output.csv`, re-reads it to check the row count against the input, and reports what it did with a preview of the output.

## The loop

**Goal → Understand → Plan → Execute → Observe → Adapt → Verify → Complete**

| Stage | Where | What happens |
|---|---|---|
| Goal | `main.py` | The task arrives as plain text. |
| Understand | `understand_node` | The LLM extracts the goal, inputs, expected output, the files involved, and checkable success criteria as JSON. |
| Plan | `plan_node` | The LLM chooses an ordered sequence of tool calls. Each step says what to do and what success looks like. |
| Execute | `execute_node` | One tool call per pass. The LLM picks the exact tool and arguments using everything observed so far. |
| Observe | `execute_node` | The tool result (or error) is appended to the state and shown to the LLM on the next pass. |
| Adapt | `execute_node`, `plan_node` + routers | A failed step is retried with a different approach, up to 3 attempts, then recorded as failed. A step that is no longer needed can be skipped. If verification fails, the run goes back to Plan once with the verifier's findings for a repair pass. |
| Verify | `verify_node` | Output and input files are re-read from disk and checked against the success criteria from Understand. The LLM writes a check script that recomputes numbers and row counts, the script is run, and the verdict is based on its output. The execution log is not trusted. |
| Complete | `complete_node` | Summary, steps taken, verdict, evidence preview and caveats (including every failed attempt). |

Nothing in the code is specific to a task. The same graph handles all three demo tasks; the tool sequence comes from the LLM.

## Architecture

```
START
  │
understand ──► plan ──► execute ◄──┐
                ▲          │       │  current_step < len(plan)
                │          ├───────┘  (next step, or retry of a failed one)
                │          │
   not verified │          ▼  all steps done
   (one repair  └─────── verify ──► complete ──► END
    pass)
```

```
main.py            CLI entry point. Runs the graph, prints progress to stderr and the final answer to stdout.
agent/state.py     AgentState TypedDict shared by all nodes.
agent/tools.py     The four tools.
agent/nodes.py     The five async node functions and their prompts.
agent/graph.py     LangGraph StateGraph wiring and the execute-loop router.
demo/              Sample inputs and run_demo.sh.
tests/             Tool unit tests and an end-to-end graph test with a scripted LLM.
```

### Tools

All four are real. Each returns a string and never raises; failures come back as `ERROR: ...` so the agent can see them and react.

| Tool | What it does |
|---|---|
| `web_search(query)` | DuckDuckGo, no API key. Tries the Instant Answer API, then falls back to result snippets from the HTML endpoint. |
| `read_file(path)` | Returns the file's text. |
| `write_file(path, content)` | Writes text, creating parent directories. |
| `run_python(code)` | Runs code in a subprocess with a 10 second timeout and returns stdout. |

Web search without a key rarely returns an exact live number (an exchange rate, a temperature). When the search result is not usable, the agent falls back to `run_python` and calls a free no-key JSON API such as `open.er-api.com` or `wttr.in`. The first attempt and the fallback both appear in the final report.

### Model

The provider is picked by whichever API key is set. The first match wins.

| Key in `.env` | Provider | Default model |
|---|---|---|
| `XAI_API_KEY` | xAI Grok | `grok-4.7` |
| `GROQ_API_KEY` | Groq | `openai/gpt-oss-20b` |
| `ANTHROPIC_API_KEY` | Anthropic Claude | `claude-sonnet-5-5` |

Set `TASKFORGE_MODEL` to use a different model, and `TASKFORGE_PROVIDER` to force a provider when more than one key is set.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your XAI_API_KEY (Grok) in .env
```

Run a task:

```bash
python main.py "Find the current weather in Bangalore and write a one-line summary to demo/weather.txt"
```

With no argument, it prompts for the task. Exit code is `0` if the result was verified, `1` if not, `2` for a usage error.

Run the tests (no API key or network needed):

```bash
python -m pytest
```

## Demo tasks

```bash
./demo/run_demo.sh
```

Live values (rate, weather) change from run to run, and the LLM may choose a different tool sequence each time. What stays fixed is described below.

**1. File + web lookup + calculation**

> Read demo/amounts.csv, find today's USD to INR exchange rate, convert each USD amount to INR, and write the results to demo/output.csv with columns Name, USD_Amount, INR_Amount.

Expected: `demo/output.csv` with a header and 5 data rows, one per row of `demo/amounts.csv`, each `INR_Amount` equal to `USD_Amount` times the fetched rate. The report states the rate used and shows the first rows.

**2. Web lookup + file write**

> Find the current weather in Bangalore and write a one-line summary to demo/weather.txt

Expected: `demo/weather.txt` containing a single line with the current conditions and temperature.

**3. Multi-step with a conditional output**

> Read demo/invoice.txt, extract the total amount, check if it exceeds ₹50,000, and write demo/approved.txt if it does not exceed it or demo/rejected.txt if it does, with the reason.

Expected: the invoice total is ₹67,500, which exceeds ₹50,000, so `demo/rejected.txt` is written with the reason and `demo/approved.txt` is not created.

## Limits

- `run_python` executes LLM-written code on your machine with your user's permissions. It has a timeout but no sandbox. Run tasks you trust, in a directory you are comfortable with.
- File paths are relative to the directory you run `main.py` from.
- Plans are capped at 8 steps, each step at 3 attempts, and a run at one repair pass.
- Small open-weight models (e.g. `openai/gpt-oss-20b`) make more mistakes per step than Grok or Claude; expect more retries and repair passes. Free-tier rate limits are retried automatically, which can add pauses.
- Verification is an LLM verdict over the re-read files and the output of an LLM-written check script. It catches missing files, row count mismatches and wrong values, but it is a check, not a proof.
