#!/usr/bin/env python3
"""Summarise repeated live runs per task: pass rate, cost spread, and path.

    python3 consistency_report.py [<runs-dir>]   (default ~/.ridges/runs)

Reads each run's runtime.log and result.json. "Path" is the sequence of models
that produced edit attempts, whether reasoning was switched off, how many
context rounds and repair rounds were used -- so variance is attributable.
"""
import json, re, statistics, sys
from collections import defaultdict
from pathlib import Path

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".ridges/runs"
SINCE = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def read_run(job: Path) -> dict | None:
    logs = list(job.rglob("runtime.log"))
    if not logs:
        return None
    text = logs[0].read_text(errors="replace")
    lines = [l for l in text.splitlines() if "[db-agent" in l]
    if not lines:
        return None
    if not list(job.rglob("result.json")):
        return None                                    # still running
    # real checks from junit
    passed = failed = 0
    for junit in job.rglob("junit.xml"):
        xml = junit.read_text(errors="replace")
        for case in re.finditer(r"<testcase[^>]*>(.*?)</testcase>|<testcase[^>]*/>", xml, re.S):
            body = case.group(1) or ""
            if "/opt/task/" in body:
                continue                               # local artifact
            if "<failure" in body or "<error" in body:
                failed += 1
            else:
                passed += 1
    cost = re.search(r"cost \$([\d.]+)", text)
    cached = re.search(r"\((\d+) cached\)", text)
    reasoning = re.search(r"\((\d+) reasoning\)", text)
    attempts = re.findall(r"attempt (\d) via (\S+)", text)
    return {
        "task": job.name.split("__")[0],
        "mtime": job.stat().st_mtime,
        "pass": failed == 0 and passed > 0,
        "checks": f"{passed}/{passed + failed}",
        "cost": float(cost.group(1)) if cost else None,
        "cached": int(cached.group(1)) if cached else 0,
        "reasoning": int(reasoning.group(1)) if reasoning else 0,
        "context_rounds": len(re.findall(r"model requested context", text)),
        "attempts": [m.split("/")[-1][:14] for _, m in attempts],
        "reasoning_off": "reasoning disabled" in text,
        "slips": len(attempts) - len({n for n, _ in attempts}),   # same attempt number repeated = gate slip
        "clean": "all checks passed" in text,
    }


def main() -> None:
    rows = [r for r in (read_run(j) for j in RUNS.iterdir() if j.is_dir()) if r]
    if SINCE:
        cutoff = SINCE.stat().st_mtime
        rows = [r for r in rows if r["mtime"] >= cutoff]
    by_task = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["mtime"]):
        by_task[r["task"]].append(r)
    for task, runs in by_task.items():
        costs = [r["cost"] for r in runs if r["cost"] is not None]
        print(f"\n{task}: {sum(r['pass'] for r in runs)}/{len(runs)} pass"
              + (f", cost mean ${statistics.mean(costs):.3f} min ${min(costs):.3f} max ${max(costs):.3f}" if costs else ""))
        for r in runs:
            print(f"  {'PASS' if r['pass'] else 'fail'} {r['checks']:6} ${r['cost'] or 0:.3f}  ctx={r['context_rounds']} "
                  f"attempts={r['attempts']} slips={r['slips']} reasoning_off={r['reasoning_off']} "
                  f"cached={r['cached']} reasoning_tok={r['reasoning']}")


if __name__ == "__main__":
    main()
