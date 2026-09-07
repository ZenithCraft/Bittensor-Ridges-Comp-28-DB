#!/usr/bin/env python3
"""Per-run post-mortem of everything under ~/.ridges/runs.

For each run: what the harness did, what the agent did, what it cost, what the
verifier checked, and -- when it failed -- where exactly it broke.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUNS = Path.home() / ".ridges/runs"
G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"

# These fail for every agent locally, including the task's gold solution: the
# verifier runs inside the environment container, which never had
# create_source_manifest.py run in it.
LOCAL_ARTIFACTS = {
    "source_tree_conservation_before_tests",
    "source_tree_conservation_after_tests",
}
ARTIFACT_HINT = re.compile(r"/opt/task/(SOURCE_REVISION|original-|source-manifest)")


def read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def job_verdict(job: Path) -> dict:
    data = json.loads(read(job / "result.json") or "{}")
    out = {"reward": None, "error": None, "state": None}
    for trial in data.get("trial_results") or []:
        out["state"] = trial.get("state")
        if trial.get("reward") is not None:
            out["reward"] = trial["reward"]
        info = trial.get("exception_info") or {}
        out["error"] = out["error"] or info.get("exception_type")
    for stats in (data.get("stats", {}).get("evals") or {}).values():
        for metric in stats.get("metrics") or []:
            if out["reward"] is None and metric.get("mean") is not None:
                out["reward"] = metric["mean"]
        for name in stats.get("exception_stats") or {}:
            out["error"] = out["error"] or name
    out["started"] = data.get("started_at", "")[:19].replace("T", " ")
    out["finished"] = data.get("finished_at", "")[:19].replace("T", " ")
    return out


def agent_facts(job: Path) -> dict:
    """Everything the agent told us about itself, from its own log lines."""
    lines: list[str] = []
    for path in job.rglob("*"):
        if path.is_file():
            lines += [l for l in read(path).splitlines() if "[db-agent" in l]
    seen, ordered = set(), []
    for line in lines:
        body = line.split("]", 1)[-1].strip()
        if body not in seen:
            seen.add(body)
            ordered.append(line.strip())
    facts = {"log": ordered}
    for line in ordered:
        if m := re.search(r"model=(\S+)", line):
            facts["model"] = m.group(1)
        if m := re.search(r"kind=(\S+) engine=(\S+) single_method=(\S+)", line):
            facts.update(kind=m.group(1), engine=m.group(2), single=m.group(3))
        if m := re.search(r"database: (\S+) (\S+)", line):
            facts.setdefault("db", m.group(2))
        if m := re.search(r"evidence bundle: (\d+)", line):
            facts["evidence"] = int(m.group(1))
        if m := re.search(r"cost \$([0-9.]+) \([^)]+\) over (\d+) call\(s\), (\d+) prompt \+ (\d+)", line):
            facts.update(cost=float(m.group(1)), calls=int(m.group(2)),
                         prompt=int(m.group(3)), completion=int(m.group(4)))
        if m := re.search(r"attempt \d+ via \S+: (.*)", line):
            facts.setdefault("diagnosis", m.group(1))
        if "returning patch" in line:
            facts["patched"] = True
        if "no patch produced" in line:
            facts["patched"] = False
    return facts


def verifier_checks(job: Path) -> list[tuple[str, bool, str]]:
    for path in job.rglob("junit.xml"):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        return [(c.get("name") or "?", c.find("failure") is None,
                 (c.find("failure").get("message") or "") if c.find("failure") is not None else "")
                for c in root.iter("testcase")]
    return []


def report(job: Path) -> None:
    verdict = job_verdict(job)
    agent = agent_facts(job)
    checks = verifier_checks(job)
    task = job.name.rsplit("__", 1)[0]

    print(f"\n{'=' * 96}")
    print(f"{B}{task}{X}   {D}{job.name.rsplit('__',1)[1]}  {verdict['started']} -> {verdict['finished']}{X}")
    print("=" * 96)

    # --- what the agent did
    if agent.get("model"):
        print(f"  model        {agent['model']}")
        print(f"  classified   kind={agent.get('kind')} engine={agent.get('engine')} "
              f"single_method={agent.get('single')}")
        print(f"  database     {agent.get('db', '-')}")
        print(f"  evidence     {agent.get('evidence', '-')} chars")
        if "cost" in agent:
            print(f"  cost         ${agent['cost']:.5f} over {agent['calls']} call(s)"
                  f"  ({agent['prompt']} prompt + {agent['completion']} completion tokens)")
        if agent.get("diagnosis"):
            print(f"  diagnosis    {agent['diagnosis'][:150]}")
        print(f"  patch        {'produced' if agent.get('patched') else R + 'NONE' + X}")
    else:
        print(f"  {Y}the agent never ran{X}")

    # --- where it stopped
    if verdict["error"]:
        stage = {"EnvironmentStartTimeoutError": "environment build (docker) timed out",
                 "MinerRuntimeError": "the agent itself raised",
                 "CancelledError": "run was cancelled / killed"}.get(
                     verdict["error"], verdict["error"])
        print(f"  {R}stopped at{X}   {stage}")

    # --- verifier
    if checks:
        real = [(n, ok, m) for n, ok, m in checks
                if not (n in LOCAL_ARTIFACTS or ARTIFACT_HINT.search(m or ""))]
        passed = sum(1 for _, ok, _ in real if ok)
        colour = G if passed == len(real) else R
        print(f"\n  verifier     {colour}{passed}/{len(real)} real checks{X}"
              f"  {D}({len(checks) - len(real)} local-only checks excluded){X}")
        for name, ok, message in checks:
            artifact = name in LOCAL_ARTIFACTS or ARTIFACT_HINT.search(message or "")
            mark = (f"{G}pass{X}" if ok else (f"{Y}n/a {X}" if artifact else f"{R}FAIL{X}"))
            note = ""
            if not ok:
                note = f"  {D}{(message or '').splitlines()[0][:110]}{X}"
            print(f"    [{mark}] {name}{note}")
    elif agent.get("patched"):
        stdout = ""
        for path in job.rglob("test-stdout.txt"):
            stdout = read(path).strip()
        print(f"\n  verifier     {R}never produced results{X}"
              + (f"\n    {D}{stdout[:200]}{X}" if stdout else ""))

    print(f"\n  reward       {verdict['reward']}")


def main() -> int:
    jobs = sorted((p for p in RUNS.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not jobs:
        print("no runs found")
        return 1
    print(f"{len(jobs)} run(s) under {RUNS}")
    for job in jobs:
        report(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
