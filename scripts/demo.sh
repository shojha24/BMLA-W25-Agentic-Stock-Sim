#!/usr/bin/env bash
# End-to-end demo. No API key, no network beyond free Yahoo endpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
SIM="Agent persona/src/simulate.py"

echo "== 1. build the local headline store (skipped if present) =="
[ -f dataset/headline_store.sqlite ] || $PY rag_prep/build_headline_store.py

echo "== 2. retrieval status =="
$PY "Agent persona/src/main.py" --task rag-status

echo "== 3. one offline cycle: 3 personas, roundtable, consensus =="
$PY "Agent persona/src/main.py" --task roundtable --mode mock --quiet

echo "== 4. backtest 2016 H1 against the index and naive models =="
$PY "$SIM" backtest --start 2016-01-04 --end 2016-06-30 --mode mock --quiet

echo "== 5. score the run =="
$PY "$SIM" score

echo "== 6. ablations: panel vs no-communication vs single agent =="
$PY "$SIM" ablate --start 2016-01-04 --end 2016-01-29 --mode mock --quiet

echo "== 7. one live cycle from Yahoo RSS =="
$PY "$SIM" live --feed yahoo --cycles 1 --mode mock --horizon 15m --quiet

echo "done. run logs in 'Agent persona/data/runs/'"
