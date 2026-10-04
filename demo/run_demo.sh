#!/usr/bin/env bash
# Runs the three demo tasks in sequence. Run from anywhere: ./demo/run_demo.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="python3"
[ -x .venv/bin/python ] && PYTHON=".venv/bin/python"

# Start clean so every output below is produced by this run.
rm -f demo/output.csv demo/weather.txt demo/approved.txt demo/rejected.txt

run() {
  echo
  echo "=================================================================="
  echo "$1"
  echo "TASK: $2"
  echo "=================================================================="
  "$PYTHON" main.py "$2"
  echo "(exit code: $?)"
}

run "DEMO 1 - file + web lookup + calculation" \
  "Read demo/amounts.csv, find today's USD to INR exchange rate, convert each USD amount to INR, and write the results to demo/output.csv with columns Name, USD_Amount, INR_Amount."

run "DEMO 2 - web lookup + file write" \
  "Find the current weather in Bangalore and write a one-line summary to demo/weather.txt"

run "DEMO 3 - multi-step with a conditional output" \
  "Read demo/invoice.txt, extract the total amount, check if it exceeds ₹50,000, and write demo/approved.txt if it does not exceed it or demo/rejected.txt if it does, with the reason."

echo
echo "Files produced:"
ls -l demo/output.csv demo/weather.txt demo/approved.txt demo/rejected.txt 2>/dev/null
