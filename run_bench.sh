#!/usr/bin/env bash
# Run agent.py against ridges-bench db-engineering tasks with the real
# verifier: live PostgreSQL, real tests, real reward.
#
#   ./run_bench.sh                       # all six tasks
#   ./run_bench.sh pg-netbox-vlangroup-utilization-001 [more...]
set -uo pipefail

RIDGES=/home/ajh/Documents/ridges
BENCH=/home/ajh/Documents/ridges-bench
AGENT=/home/ajh/Documents/ridges-db-agent/agent.py
export PATH="$HOME/.local/bin:$PATH"

tasks=("$@")
if [ ${#tasks[@]} -eq 0 ]; then
  mapfile -t tasks < <(cd "$BENCH/db-engineering" && ls -d pg-* ch-* 2>/dev/null)
fi

for task in "${tasks[@]}"; do
  echo "############################################################"
  echo "### $task"
  echo "############################################################"
  ( cd "$BENCH" && uv run --project "$RIDGES" ridges miner run-local \
      --task-path "./db-engineering/$task" \
      --agent-path "$AGENT" ) 2>&1 | tail -40
  echo
done
