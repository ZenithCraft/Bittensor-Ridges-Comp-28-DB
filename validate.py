#!/usr/bin/env python3
"""Validate agent.py against ridges-bench tasks using the real verifier.

Each task runs through `ridges miner run-local`, which builds the task's
container, runs the agent against the live database, applies the resulting
patch to a pristine checkout and runs the task's own tests -- the same verifier
a subnet validator uses.

    ./validate.py                          preflight + all tasks
    ./validate.py --preflight              environment checks only
    ./validate.py --costs                  spend state and past runs
    ./validate.py <task> [<task> ...]      named tasks
    ./validate.py --model qwen/qwen3-coder-next
    ./validate.py --all-models <task>      compare every allowed model
    ./validate.py --quiet                  suppress the streamed harness log
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

RIDGES = Path("/home/ajh/Documents/ridges")
BENCH = Path("/home/ajh/Documents/ridges-bench/db-engineering")
FAST = Path(__file__).resolve().parent / "fast-tasks"
AGENT = Path(__file__).resolve().parent / "agent.py"
ENV_MINER = Path.home() / ".ridges/.env.miner"
RESULTS = Path.home() / ".ridges/runs"
UV = Path.home() / ".local/bin/uv"
VARIANTS = Path("/tmp/ridges-agent-variants")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent                                                 # noqa: E402  (path set above)
import selftest                                              # noqa: E402

# Per-task wall clock and verifier budgets, straight from each task.toml. The
# agent container never receives that file, so the harness reads it instead.
TASK_BUDGETS = selftest.task_budgets()

# Set from --trace before any task runs; run_task has no options argument.
TRACING = False

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m")

COST_POLL_SECONDS = 20


# ---------------------------------------------------------------------------
# Account / cost
# ---------------------------------------------------------------------------

def read_key() -> str:
    if not ENV_MINER.exists():
        return ""
    for line in ENV_MINER.read_text().splitlines():
        if line.startswith("RIDGES_OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _openrouter(path: str) -> dict:
    key = read_key()
    if not key:
        return {}
    request = urllib.request.Request(f"https://openrouter.ai/api/v1/{path}",
                                     headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read()).get("data") or {}


def account_state() -> dict:
    try:
        credits = _openrouter("credits")
        key_info = _openrouter("key")
    except Exception as exc:
        return {"error": str(exc)}
    if not credits and not key_info:
        return {"error": "no key configured"}
    state = {**credits, **{k: key_info.get(k) for k in
                           ("label", "limit", "limit_remaining", "usage", "is_free_tier")}}
    state["remaining"] = float(state.get("total_credits") or 0) - float(state.get("total_usage") or 0)
    return state


def money(value) -> str:
    return f"${value:.5f}" if isinstance(value, (int, float)) else "-"


def print_account(state: dict, heading: str = "OpenRouter account") -> None:
    print(heading)
    if state.get("error"):
        print(f"  {RED}{state['error']}{RESET}\n")
        return
    remaining = state.get("remaining", 0.0)
    colour = GREEN if remaining > 0.5 else RED
    tier = "paid" if state.get("is_free_tier") is False else "free tier"
    print(f"  key            {state.get('label')}   {DIM}({tier}){RESET}")
    print(f"  balance        {colour}${remaining:.4f}{RESET}   {DIM}(credits "
          f"${float(state.get('total_credits') or 0):.2f} - usage "
          f"${float(state.get('total_usage') or 0):.2f}){RESET}")
    print(f"  this key used  ${float(state.get('usage') or 0):.4f}")
    limit = state.get("limit")
    print(f"  per-key limit  {('$%.2f' % limit) if limit else YELLOW + 'none set' + RESET}\n")


class CostWatcher(threading.Thread):
    """Poll the OpenRouter ledger during a run and report spend as it happens."""

    def __init__(self, label: str, quiet: bool = False):
        super().__init__(daemon=True)
        self.label = label
        self.quiet = quiet
        self._stop = threading.Event()
        self.baseline: float | None = None
        self.latest: float | None = None

    def run(self) -> None:
        state = account_state()
        self.baseline = self.latest = state.get("remaining")
        started = time.time()
        while not self._stop.wait(COST_POLL_SECONDS):
            state = account_state()
            remaining = state.get("remaining")
            if remaining is None or self.baseline is None:
                continue
            self.latest = remaining
            spent = self.baseline - remaining
            if not self.quiet:
                print(f"{BLUE}[cost t+{time.time() - started:4.0f}s] balance ${remaining:.4f}"
                      f"  spent this run {money(spent)}{RESET}", flush=True)

    def stop(self) -> float | None:
        self._stop.set()
        state = account_state()
        remaining = state.get("remaining")
        if remaining is not None:
            self.latest = remaining
        if self.baseline is None or self.latest is None:
            return None
        return self.baseline - self.latest


# ---------------------------------------------------------------------------
# Harness invocation
# ---------------------------------------------------------------------------

def allowed_models() -> list[str]:
    """The model slugs agent.py knows about, read from its own table."""
    tree = ast.parse(AGENT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "MODELS":
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    return []


def agent_variant(model: str | None, agent_timeout: float | None = None,
                  tracing: "str | bool" = False) -> Path:
    """A throwaway agent copy pinned to one model, budget and/or trace setting.

    Nothing reaches the task container except the file itself: `ridges miner
    run-local` sets AGENT_TIMEOUT only when its caller passes
    `agent_timeout_sec`, and the CLI has no flag that plumbs one, so the host
    environment cannot carry either choice in. Both are therefore appended to
    a copy of the agent.

    Only append names the agent actually reads. `ROUTING_MODE` was baked here
    after agent.py stopped defining it, which changed the artifact's bytes
    without changing its behaviour.
    """
    if not model and agent_timeout is None and not tracing:
        return AGENT
    VARIANTS.mkdir(parents=True, exist_ok=True)
    parts = [re.sub(r"[^a-z0-9]+", "_", (model or "ladder").lower())]
    lines = [AGENT.read_text()]
    if model:
        lines.append(f'\n\nFORCE_MODEL = "{model}"\n')
    if agent_timeout is not None:
        # Production always injects AGENT_TIMEOUT (engine.py: min(spec, max)),
        # so DEFAULT_AGENT_TIMEOUT is dead there and live only here. Left at
        # the agent's own fallback, a local run paces itself against
        # {agent.DEFAULT_AGENT_TIMEOUT:.0f}s while the graded run gets the
        # task.toml budget -- so an early stop measured locally would not have
        # happened in the container we are trying to predict.
        parts.append(f"t{int(agent_timeout)}")
        lines.append(f"\n\nDEFAULT_AGENT_TIMEOUT = {float(agent_timeout)!r}\n")
    if tracing:
        # RIDGES_TRACE is read from the environment, and the agent's
        # environment is the task container's, not this shell's -- the harness
        # forwards nothing. Baking the flag is the only way in, the same
        # problem AGENT_TIMEOUT has.
        parts.append({"stages": "trace", "all": "traceall"}.get(tracing, "tracevars"))
        lines.append("\n\nTRACE = True\n")
        if tracing == "all":
            lines.append("TRACE_ALL = True\n")
        elif tracing == "vars":
            lines.append("TRACE_VARS = True\n")
    path = VARIANTS / f"agent_{'_'.join(parts)}.py"
    path.write_text("".join(lines))
    return path


def stream(command: str, quiet: bool, prefix: str = "") -> tuple[int, str]:
    """Run a command, echoing output live while capturing it."""
    # The CLI does `from validator.config import ...` at report time, which
    # resolves only when the ridges repo root is the working directory.
    process = subprocess.Popen(["sg", "docker", "-c", command], cwd=RIDGES,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        if not quiet:
            sys.stdout.write(f"{DIM}{prefix}{RESET}{line}")
            sys.stdout.flush()
    process.wait()
    return process.returncode, "".join(lines)


def read_result_json(job_dir: Path | None) -> dict:
    if job_dir is None or not (job_dir / "result.json").exists():
        return {}
    try:
        data = json.loads((job_dir / "result.json").read_text())
    except (OSError, ValueError):
        return {}
    out: dict = {}
    for trial in data.get("trial_results") or []:
        out.setdefault("state", trial.get("state"))
        if trial.get("reward") is not None:
            out["reward"] = float(trial["reward"])
        info = trial.get("exception_info") or {}
        if info.get("exception_type"):
            out["error"] = info["exception_type"]
    if "reward" not in out:
        for stats in (data.get("stats", {}).get("evals") or {}).values():
            for metric in stats.get("metrics") or []:
                if metric.get("mean") is not None:
                    out["reward"] = float(metric["mean"])
            for exception_type in stats.get("exception_stats") or {}:
                out.setdefault("error", exception_type)
    return out


# `ridges miner run-local` has no separate verifier container, so verify.py runs
# inside the environment image. Each task's environment/Dockerfile now stages
# /opt/task itself (a LOCAL ONLY layer mirroring tests/Dockerfile), which is what
# lets the bounded-method check run for real.
#
# So bounded_*_method is deliberately NOT excluded any more. It grades the patch
# -- byte-identity above and below the method, signature, size, AST node count,
# forbidden constructs -- and a failure there is a real contract violation. It
# used to be excluded by name whenever it failed, which after staging would have
# hidden exactly the violations this harness exists to catch.
#
# source_tree_conservation still cannot pass locally, and no agent can fix it:
# the harness runs `git init` in /app before the agent starts, and verify.py's
# rglob carries no .git exclusion, so the live tree holds hundreds of entries no
# image-time manifest can contain. In production the verifier is a separate,
# pristine container that never sees them. Excluded by NAME, because the message
# differs by cause -- a missing fixture before staging, source drift after.
LOCAL_ONLY_CHECKS = re.compile(r"source_tree_conservation")
# Still failing on a missing fixture means the image predates the staging layer.
# Excluded too, but it means the image wants rebuilding rather than that the
# check is impossible.
MISSING_FIXTURE = re.compile(r"/opt/task/(SOURCE_REVISION|original-|source-manifest)")


def is_local_artifact(name: str, message: str) -> bool:
    return bool(LOCAL_ONLY_CHECKS.search(name or "")
                or MISSING_FIXTURE.search(message or ""))


def junit_checks(job_dir: Path | None) -> list[tuple[str, bool, str]]:
    """Per-check results from the verifier's junit.xml."""
    if job_dir is None:
        return []
    import xml.etree.ElementTree as ET
    for path in job_dir.rglob("junit.xml"):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        out = []
        for case in root.iter("testcase"):
            failure = case.find("failure")
            out.append((case.get("name") or "?", failure is None,
                        (failure.get("message") or "")[:200] if failure is not None else ""))
        return out
    return []


def newest_job_dir(task: str, since: float) -> Path | None:
    candidates = [p for p in RESULTS.glob(f"{task}__*")
                  if p.is_dir() and p.stat().st_mtime >= since - 5]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def agent_log(job_dir: Path | None) -> tuple[float | None, int | None, list[str]]:
    """The agent's own cost line and log, from the harbor job tree."""
    if job_dir is None:
        return None, None, []
    cost = calls = None
    interesting: list[str] = []
    for path in job_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in re.finditer(r"cost \$([0-9.]+) \([^)]+\) over (\d+) call", text):
            cost, calls = float(match.group(1)), int(match.group(2))
        interesting += [ln for ln in text.splitlines() if "[db-agent" in ln]
    return cost, calls, interesting


def parse_stdout(output: str) -> dict:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
    result: dict = {}
    state = re.search(r"^\s*(SUCCEEDED|FAILED|ERRORED|TIMED_OUT|CANCELLED)\s*$", plain, re.MULTILINE)
    if state:
        result["state"] = state.group(1)
    reward = re.search(r"reward:\s*([0-9.]+)", plain)
    if reward:
        result["reward"] = float(reward.group(1))
    tests = re.search(r"tests:\s*(\d+)\s*total\s*\((\d+)\s*passed,\s*(\d+)\s*failed", plain)
    if tests:
        result["tests"] = tuple(int(g) for g in tests.groups())
    return result


def run_task(task: str, model: str | None, quiet: bool) -> dict:
    tag = f"  [{model}]" if model else ""
    print(f"\n{'=' * 78}\n{task}{tag}\n{'=' * 78}", flush=True)

    agent_path = agent_variant(model, agent_timeout=TASK_BUDGETS.get(task, {}).get("agent"),
                               tracing=TRACING)
    watcher = CostWatcher(task, quiet=quiet)
    watcher.start()
    started = time.time()

    command = (f"{UV} run --project {RIDGES} ridges miner run-local "
               f"--task-path {BENCH / task} --agent-path {agent_path}")
    code, output = stream(command, quiet, prefix="| ")

    elapsed = time.time() - started
    ledger_delta = watcher.stop()

    job_dir = newest_job_dir(task, started)
    result = {"task": task, "model": model or "ladder", "elapsed": elapsed,
              "ledger_delta": ledger_delta, "returncode": code}
    result.update(parse_stdout(output))
    result.update({k: v for k, v in read_result_json(job_dir).items() if v is not None})
    cost, calls, agent_lines = agent_log(job_dir)
    result["cost"], result["calls"] = cost, calls

    checks = junit_checks(job_dir)
    if checks:
        real = [(n, ok, m) for n, ok, m in checks if not is_local_artifact(n, m)]
        result["checks"] = checks
        result["real_passed"] = sum(1 for _, ok, _ in real if ok)
        result["real_total"] = len(real)
        print(f"\n{DIM}--- verifier checks ---{RESET}")
        for name, ok, message in checks:
            artifact = is_local_artifact(name, message)
            mark = (GREEN + "pass" if ok else (YELLOW + "n/a " if artifact else RED + "FAIL")) + RESET
            note = f"  {DIM}{message}{RESET}" if not ok else ""
            print(f"  [{mark}] {name}{'  (local artifact)' if artifact and not ok else ''}{note}")

    if agent_lines and not quiet:
        print(f"\n{DIM}--- agent log ---{RESET}")
        for line in agent_lines[-40:]:
            print(f"  {line}")

    if result.get("real_total"):
        verdict = f"verifier {result['real_passed']}/{result['real_total']} real checks"
    else:
        verdict = f"reward={result.get('reward')} state={result.get('state')}"
    if result.get("error"):
        verdict += f" {RED}error={result['error']}{RESET}"
    print(f"\n-> {verdict} tests={result.get('tests')} agent_cost={money(cost)} "
          f"ledger={money(ledger_delta)} in {elapsed:.0f}s", flush=True)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def scorecard(results: list[dict]) -> int:
    print(f"\n{'=' * 88}\nSCORECARD\n{'=' * 88}")
    print(f"{'task':<34}{'model':<24}{'reward':>8}{'tests':>8}{'agent $':>10}{'secs':>7}")
    print("-" * 88)
    total = passed = 0.0, 0
    total_cost = 0.0
    passed = 0
    for r in results:
        reward = r.get("reward")
        passed += 1 if reward is not None and reward >= 1.0 else 0
        if r.get("real_total"):
            tests = f"{r['real_passed']}/{r['real_total']}"
            ok = r["real_passed"] == r["real_total"]
        else:
            tests = f"{r['tests'][1]}/{r['tests'][0]}" if r.get("tests") else "-"
            ok = reward is not None and reward >= 1.0
        passed += -1 if (reward is not None and reward >= 1.0) else 0   # recount below
        passed += 1 if ok else 0
        total_cost += r.get("cost") or 0.0
        colour = GREEN if ok else RED
        label = r["task"][:32] + ("*" if r.get("error") else "")
        print(f"{label:<34}{r['model'][:22]:<24}{colour}{str(reward):>8}{RESET}"
              f"{tests:>8}{money(r.get('cost')):>10}{r.get('elapsed', 0):>7.0f}")
    print("-" * 88)
    n = len(results)
    print(f"{'TOTAL':<58}{passed}/{n:<6}{money(total_cost):>10}")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n{YELLOW}* infrastructure error -- the agent never ran:{RESET}")
        for r in errors:
            print(f"    {r['task']} [{r['model']}]: {r['error']}")
    if n:
        mean = total_cost / n
        target, cap = agent.COST_TARGET_USD, agent.DEFAULT_MAX_COST_USD
        verdict = (GREEN + "within target" if mean <= target else
                   YELLOW + f"over ${target:.2f} target" if mean <= cap
                   else RED + f"OVER THE ${cap:.2f} CAP")
        print(f"\nmean agent cost/task {money(mean)}   {verdict}{RESET}"
              f"   {DIM}(target ${target:.2f}, cap ${cap:.2f} -- both read from agent.py){RESET}")
        print_clock_headroom(results)
    print_account(account_state(), "\nOpenRouter account after run")
    return 0 if passed == n and n else 1


def print_clock_headroom(results: list[dict]) -> None:
    """How much of each task's own budget the run actually spent.

    Two different clocks, and only one of them is ours. `[agent] timeout_sec`
    bounds this run; `[verifier] timeout_sec` bounds the separate container
    that re-runs the tests against our patch, and those are NOT uniform across
    the bench (measured 2026-09-07: task.toml asks 1800 for the agent everywhere,
    capped to the platform's 25-minute grant; verifier 900-1800).
    A patch that makes the suite slower is graded against the tighter of the
    two, and nothing in the agent's own run would reveal that.
    """
    rows = []
    for r in results:
        budget = TASK_BUDGETS.get(r["task"], {})
        agent_budget, verifier_budget = budget.get("agent"), budget.get("verifier")
        if not agent_budget:
            continue
        used = r.get("elapsed", 0.0)
        rows.append((r["task"], used, agent_budget, verifier_budget))
    if not rows:
        return
    worst = max(rows, key=lambda row: row[1] / row[2])
    task, used, agent_budget, verifier_budget = worst
    share = used / agent_budget
    colour = GREEN if share < 0.6 else YELLOW if share < 0.85 else RED
    print(f"slowest run {colour}{share:.0%}{RESET} of its agent budget"
          f"   {DIM}({task}: {used:.0f}s of {agent_budget:.0f}s){RESET}")
    tight = sorted({v for _, _, _, v in rows if v is not None})
    if tight:
        print(f"{DIM}verifier budgets among these tasks: "
              f"{', '.join(f'{v:.0f}s' for v in tight)} -- the verifier re-runs the suite "
              f"in its own container, so a slower suite is graded against that clock{RESET}")

def model_matrix(results: list[dict]) -> None:
    models = sorted({r["model"] for r in results})
    tasks = sorted({r["task"] for r in results})
    print(f"\n{'=' * 88}\nMODEL COMPARISON\n{'=' * 88}")
    print(f"{'model':<26}" + "".join(f"{t.replace('pg-netbox-', '')[:12]:>13}" for t in tasks)
          + f"{'pass':>7}{'cost':>10}")
    print("-" * 88)
    for model in models:
        rows = [r for r in results if r["model"] == model]
        cells = ""
        wins = 0
        cost = 0.0
        for task in tasks:
            match = next((r for r in rows if r["task"] == task), None)
            reward = match.get("reward") if match else None
            wins += 1 if reward is not None and reward >= 1.0 else 0
            cost += (match or {}).get("cost") or 0.0
            colour = GREEN if reward is not None and reward >= 1.0 else RED
            cells += f"{colour}{str(reward) if reward is not None else '-':>13}{RESET}"
        print(f"{model[:24]:<26}{cells}{wins:>7}{money(cost):>10}")
    print("-" * 88)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def status(ok: bool | None, label: str, detail: str = "") -> bool:
    mark = {True: f"{GREEN}PASS{RESET}", False: f"{RED}FAIL{RESET}", None: f"{YELLOW}WARN{RESET}"}[ok]
    print(f"  [{mark}] {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    return ok is not False


def preflight() -> bool:
    print("Preflight\n")
    ok = True
    docker = subprocess.run(["sg", "docker", "-c", "docker info --format {{.ServerVersion}}"],
                            capture_output=True, text=True)
    ok &= status(docker.returncode == 0, "docker reachable", docker.stdout.strip()[:40])

    # What actually decides whether the next build is minutes or half an hour is
    # the layer cache, not any particular image: `docker rmi` frees images and
    # leaves the cache untouched, so a cold image list with a warm cache still
    # rebuilds in a couple of minutes. This used to look for a tag named
    # `warm-main` that nothing in this repo ever creates, so it reported a
    # 30-minute build every single time, including immediately after a build.
    listed = subprocess.run(["sg", "docker", "-c", "docker images --format '{{.Repository}}'"],
                            capture_output=True, text=True)
    images = [line for line in (listed.stdout or "").splitlines() if line.strip().endswith("-main")]
    df = subprocess.run(["sg", "docker", "-c", "docker system df --format '{{.Type}}\t{{.Size}}'"],
                        capture_output=True, text=True)
    cache = next((row.split("\t")[-1] for row in (df.stdout or "").splitlines()
                  if row.startswith("Build Cache")), "0B")
    warm_cache = not cache.startswith("0")
    status(bool(images) or warm_cache, "netbox build cache",
           f"{len(images)} task image(s) built, {cache} of layer cache" if (images or warm_cache)
           else "cold: expect a ~30min build on the first task")

    ok &= status(UV.exists(), "uv installed")
    cli = subprocess.run([str(UV), "run", "--project", str(RIDGES), "ridges", "--version"],
                         capture_output=True, text=True, cwd=RIDGES)
    ok &= status(cli.returncode == 0, "ridges CLI runnable", cli.stdout.strip()[:40])
    ok &= status(AGENT.exists(), "agent.py present")

    state = account_state()
    if not state.get("error"):
        remaining = state.get("remaining", 0.0)
        status(True if remaining > 0.5 else None, "OpenRouter credit", f"${remaining:.4f}")
    else:
        ok &= status(False, "OpenRouter credit", state["error"])

    load = os.getloadavg()[0]
    status(True if load < 2.0 else None, "machine load",
           f"{load:.2f} on {os.cpu_count()} cores"
           + ("" if load < 2.0 else "  <-- builds may time out; stop other work"))
    print()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--model", help="pin one model instead of the escalation ladder")
    parser.add_argument("--all-models", action="store_true", help="run each allowed model in turn")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--costs", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="do not stream harness output")
    parser.add_argument("--trace", action="store_true",
                        help="per-module input/output lines in the agent log")
    parser.add_argument("--trace-all", action="store_true",
                        help="every function call, its arguments, the locals it computed and "
                             "the state it changed. Loud and slow; use on one task")
    parser.add_argument("--trace-vars", action="store_true",
                        help="every variable change, statement by statement, with the value "
                             "before and after and the line that changed it. The loudest mode")
    parser.add_argument("--tasks-dir", help="task directory (default: fast-tasks if present)")
    args = parser.parse_args()
    global TRACING
    TRACING = ("vars" if args.trace_vars else
               "all" if args.trace_all else
               "stages" if args.trace else False)

    global BENCH
    if args.tasks_dir:
        BENCH = Path(args.tasks_dir).resolve()
    elif FAST.is_dir() and any(FAST.iterdir()):
        BENCH = FAST

    if args.costs:
        print_account(account_state())
        print("Past local runs")
        rows = sorted(RESULTS.glob("*__*"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
        for job_dir in rows or []:
            verdict = read_result_json(job_dir)
            when = time.strftime("%m-%d %H:%M", time.localtime(job_dir.stat().st_mtime))
            cost, _, _ = agent_log(job_dir)
            note = verdict.get("error") or f"reward {verdict.get('reward')}"
            print(f"  {when}  {job_dir.name[:50]:<52}{note:<30}{money(cost)}")
        if not rows:
            print(f"  {DIM}none yet{RESET}")
        return 0

    if not preflight() and not args.tasks:
        print(f"{RED}preflight failed{RESET}")
        return 2
    if args.preflight:
        return 0

    tasks = args.tasks or sorted(p.name for p in BENCH.iterdir() if p.is_dir())
    models = allowed_models() if args.all_models else [args.model]
    if args.all_models:
        print(f"Comparing {len(models)} models over {len(tasks)} task(s): {', '.join(models)}\n")

    results = [run_task(task, model, args.quiet) for model in models for task in tasks]
    code = scorecard(results)
    if args.all_models:
        model_matrix(results)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
