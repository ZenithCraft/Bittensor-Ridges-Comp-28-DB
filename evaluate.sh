#!/bin/bash
# Pass rate per task: run the agent, grade the patch the way a validator would,
# repeat. Scoring needs unanimity across validators, so the number that predicts
# a score is the pass RATE, not whether any single run passed.
#
#   ./evaluate.sh                     all six tasks, 5 repeats
#   REPEATS=3 ./evaluate.sh           fewer repeats
#   TASKS="pg-netbox-vlangroup-utilization-001" ./evaluate.sh    one task
#
# Budget roughly 12 minutes and $0.03 per run: 6 tasks x 5 repeats is ~6 hours
# and ~$0.80.
set -uo pipefail          # deliberately NOT -e; see below

DEFAULT_TASKS="pg-netbox-bulk-tag-assignment-001
pg-netbox-cached-value-index-001
pg-netbox-contact-group-counts-001
pg-netbox-ipaddress-device-filter-001
pg-netbox-prefix-hierarchy-annotations-001
pg-netbox-vlangroup-utilization-001"

read -r -a TASKS <<< "$(echo "${TASKS:-$DEFAULT_TASKS}" | tr '\n' ' ')"
REPEATS="${REPEATS:-5}"
OUT=~/.ridges/evaluation_results
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

# `set -e` is off on purpose. validate.py exits non-zero when a task fails and
# grade.py exits non-zero when the reward is not 1 -- which is exactly the case
# this script exists to count. With -e the run aborts on the first failure and
# reports a pass rate computed from the runs that happened before it, which is
# both wrong and silently wrong.
for TASK in "${TASKS[@]}"; do
    echo "===== $TASK ====="
    REWARDS=()
    for ((i = 1; i <= REPEATS; i++)); do
        echo "--- run $i/$REPEATS  $TASK ---"

        ./validate.py "$TASK" || echo "  (validate.py exited non-zero -- continuing)"

        RUN_DIR=$(ls -td ~/.ridges/runs/"$TASK"__*/ 2>/dev/null | head -1)
        if [ -z "$RUN_DIR" ] || [ -z "$(find "$RUN_DIR" -name patch.diff -print -quit 2>/dev/null)" ]; then
            echo "  no patch produced -- scoring 0 for this run"
            REWARDS+=("0.0")
            continue
        fi
        echo "  run: $RUN_DIR"

        ./grade.py "$TASK" --patch-from-run "$RUN_DIR" || true

        GRADE_DIR=$(ls -td "$HOME"/.ridges/grades/"$TASK"__*/ 2>/dev/null | head -1)
        REWARD=$(jq -r '.reward // 0' "$GRADE_DIR/result.json" 2>/dev/null || echo 0)
        REWARDS+=("$REWARD")
        echo "  reward: $REWARD"
    done

    # grep -c exits 1 when it matches nothing, which is a legitimate result
    # here (a task that never passed), so count without letting that leak out.
    PASSES=$(printf '%s\n' "${REWARDS[@]}" | grep -c '^1\(\.0\)\?$' || true)
    PASSES=${PASSES:-0}
    PASS_RATE=$((PASSES * 100 / REPEATS))
    echo "=== $TASK: $PASSES/$REPEATS passed (${PASS_RATE}%) ==="
    echo "$TASK: $PASSES/$REPEATS (${PASS_RATE}%)  rewards: ${REWARDS[*]}" >> "$SUMMARY"
    printf '%s\n' "${REWARDS[@]}" > "$OUT/${TASK}_rewards.txt"
done

echo
echo "===== summary ====="
cat "$SUMMARY"
echo
