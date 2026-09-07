#!/usr/bin/env python3
"""Pre-upload qualification for the Ridges DB-query agent.

Screener 1 and Screener 2 use mutually exclusive problem sets and the
validators draw a random mix of both, so passing the six public samples proves
very little. This harness checks the properties that generalise, in layers, and
refuses to report readiness for anything it could not actually verify.

    ./qualify.py                 static + offline layers (fast, free)
    ./qualify.py --live          also run real tasks (needs a working model)
    ./qualify.py --live --model deepseek/deepseek-v4-pro-0813
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent                                                # noqa: E402  (path set above)
import selftest                                             # noqa: E402

AGENT = HERE / "agent.py"
NETBOX = selftest.default_checkout()
BENCH = selftest.BENCH
RUNS = Path.home() / ".ridges/runs"

G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"

# Read from the artifact rather than restated here: a budget this harness
# believes in but the agent does not is a check that passes while the upload
# fails. These moved once already (the cap was raised to $0.29) and the copy
# here went stale.
COST_BUDGET = agent.COST_TARGET_USD          # our soft target
COST_CAP = agent.DEFAULT_MAX_COST_USD        # the hard per-problem cap
TIME_BUDGET = agent.DEFAULT_AGENT_TIMEOUT    # the agent's fallback when AGENT_TIMEOUT is unset


class Layer:
    def __init__(self, number: int, name: str):
        self.number, self.name = number, name
        self.results: list[tuple[str, str, str]] = []   # (status, check, detail)

    def add(self, ok: bool | None, check: str, detail: str = "") -> None:
        self.results.append(("pass" if ok else ("skip" if ok is None else "FAIL"), check, detail))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.results if s == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for s, _, _ in self.results if s == "skip")

    def render(self) -> None:
        head = f"Layer {self.number} — {self.name}"
        print(f"\n{B}{head}{X}\n{'-' * len(head)}")
        for status, check, detail in self.results:
            colour = {"pass": G, "FAIL": R, "skip": Y}[status]
            print(f"  [{colour}{status:>4}{X}] {check}" + (f"  {D}{detail}{X}" if detail else ""))


# ---------------------------------------------------------------------------
# Layer 1 — agent contract
# ---------------------------------------------------------------------------

def layer_contract() -> Layer:
    layer = Layer(1, "Agent contract")
    source = AGENT.read_text()
    tree = ast.parse(source)

    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "agent_main"), None)
    layer.add(fn is not None, "agent_main is a top-level function")
    if fn:
        args = [a.arg for a in fn.args.args]
        layer.add(len(args) == 1, "takes exactly one argument", f"args={args}")
        layer.add(not isinstance(fn, ast.AsyncFunctionDef), "is synchronous")

    # Loads the way the Harbor runtime loads it (no sys.modules registration).
    import importlib.util
    spec = importlib.util.spec_from_file_location("ridges_miner_agent", AGENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("ridges_miner_agent", None)
    try:
        spec.loader.exec_module(module)
        layer.add(True, "imports under the Harbor runtime loader")
    except Exception as exc:
        layer.add(False, "imports under the Harbor runtime loader", repr(exc))
        return layer

    # Never crash, always return a string -- a raised exception is scored as an
    # agent crash, not a zero.
    for label, payload in (("empty dict", {}),
                           ("missing key", {"unexpected": "x"}),
                           ("None value", {"problem_statement": None}),
                           ("empty statement", {"problem_statement": ""})):
        try:
            out = module.agent_main(payload)
            layer.add(isinstance(out, str), f"survives malformed input: {label}",
                      f"returned {type(out).__name__}")
        except Exception as exc:
            layer.add(False, f"survives malformed input: {label}", repr(exc))

    # Patch machinery.
    ok = subprocess.run([sys.executable, str(HERE / "test_units.py")],
                        capture_output=True, text=True, cwd=HERE)
    count = re.search(r"Ran (\d+) tests", ok.stdout + ok.stderr)
    layer.add(ok.returncode == 0, "unit suite (patch format, edits, verifier)",
              f"{count.group(1) if count else '?'} tests")
    return layer


# ---------------------------------------------------------------------------
# Layer 2 — sandbox portability
# ---------------------------------------------------------------------------

STDLIB = set(sys.stdlib_module_names)

# Identifiers from the public sample repo/tasks. Any of these baked into agent
# logic means the agent is fitted to tasks it will never see again.
TASK_SPECIFIC = re.compile(
    r"\b(netbox|contactgroup|contact_group|tenancy|vlangroup|vlan_count|ipaddress"
    r"|cachedvalue|cached_value|annotate_contacts|annotate_utilization|filter_device"
    r"|annotate_hierarchy|tree_id|rght|mptt|taggit|dcim|ridges-bench)\b",
    re.IGNORECASE)


def layer_portability() -> Layer:
    layer = Layer(2, "Sandbox portability")
    source = AGENT.read_text()
    tree = ast.parse(source)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    third_party = sorted(imported - STDLIB)
    layer.add(not third_party, "imports only the standard library",
              f"third-party: {third_party}" if third_party else "")

    # Absolute paths that only exist on this machine.
    LOCAL_ROOTS = ("/home/", "/Users/", "/mnt/", "/media/", "/root/", "/var/home/")
    local_paths = sorted({
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and n.value.startswith(LOCAL_ROOTS)})
    layer.add(not local_paths, "no machine-specific absolute paths", str(local_paths[:3]) if local_paths else "")

    # Task-specific identifiers in executable code (not comments/docstrings).
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node is not getattr(node, "_docstring", None) and TASK_SPECIFIC.search(node.value):
                hits.append(f"L{getattr(node, 'lineno', '?')}: {node.value[:60]}")
        elif isinstance(node, ast.Name) and TASK_SPECIFIC.search(node.id):
            hits.append(f"L{node.lineno}: name {node.id}")
    docstrings = {ast.get_docstring(n) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))}
    hits = [h for h in hits if not any(h.split(": ", 1)[-1][:40] in (d or "") for d in docstrings)]
    layer.add(not hits, "no sample-repo identifiers in agent logic",
              "; ".join(hits[:3]) if hits else "")

    # Budget and timeout awareness.
    layer.add("RIDGES_MAX_COST_USD" in source, "reads the injected cost budget")
    layer.add("AGENT_TIMEOUT" in source, "reads the injected agent timeout")
    layer.add("revert_all" in source, "restores the working tree before exiting")

    # task.toml is not among the four files uploaded to the agent container
    # (agent.py, _stdlib_contract.py, ridges_miner_runtime.py, instruction.md).
    # An agent that reached for it would read nothing in production and the
    # wrong thing locally, where /opt/task exists only for the verifier.
    reaches = [f"L{n.lineno}: {n.value!r}" for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and ("task.toml" in n.value or "/opt/task" in n.value)]
    layer.add(not reaches, "does not reach for task.toml (never uploaded to the agent container)",
              "; ".join(reaches[:3]))

    budgets = selftest.task_budgets()
    granted = sorted({b["agent"] for b in budgets.values() if "agent" in b})
    if granted:
        layer.add(TIME_BUDGET <= granted[0],
                  "wall-clock fallback fits the tightest task budget",
                  f"agent fallback {TIME_BUDGET:.0f}s vs [agent] timeout_sec "
                  + (f"{granted[0]:.0f}s across {len(budgets)} tasks" if len(granted) == 1
                     else f"{granted[0]:.0f}-{granted[-1]:.0f}s"))
        # Not a failure: production always injects AGENT_TIMEOUT, so the
        # fallback is dead there. It is live locally, where the CLI plumbs no
        # timeout -- validate.py bakes the real one into its agent copy.
        layer.add(None if TIME_BUDGET == granted[0] else True,
                  "local runs are paced like production",
                  f"validate.py bakes DEFAULT_AGENT_TIMEOUT={granted[0]:.0f} "
                  f"(the agent's own fallback is {TIME_BUDGET:.0f})"
                  if TIME_BUDGET != granted[0] else "fallback already matches")
    else:
        layer.add(None, "no task.toml budgets found", str(BENCH))
    return layer


# ---------------------------------------------------------------------------
# Layer 3 — target localization
# ---------------------------------------------------------------------------

def layer_localization() -> Layer:
    layer = Layer(3, "Target localization")
    if not NETBOX.exists():
        layer.add(None, "needs a pinned netbox checkout", str(NETBOX))
        return layer
    proc = subprocess.run([sys.executable, str(HERE / "selftest.py"), str(NETBOX)],
                          capture_output=True, text=True, cwd=HERE)
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        if m := re.match(r"\[(PASS|FAIL)\] (\S+)", line):
            layer.add(m.group(1) == "PASS", f"locate+slice+patch: {m.group(2)}")
    summary = re.search(r"(\d)/6 tasks located", text)
    if summary:
        layer.add(summary.group(1) == "6", "all six samples resolved", summary.group(0))

    # Explicit vs implicit target: the samples cover both, and the implicit one
    # is the case the competition warns about.
    layer.add("bulk-tag-assignment" in text, "covers an implicit-target task",
              "instruction names no file")
    return layer


# ---------------------------------------------------------------------------
# Layer 4 — solve-loop behaviour
# ---------------------------------------------------------------------------

def layer_loop() -> Layer:
    layer = Layer(4, "Solve loop and self-correction")
    if not NETBOX.exists():
        layer.add(None, "needs a pinned netbox checkout")
        return layer
    proc = subprocess.run([sys.executable, str(HERE / "test_e2e.py"), str(NETBOX)],
                          capture_output=True, text=True, cwd=HERE)
    text = proc.stdout + proc.stderr
    for name, label in (
        ("test_single_round_produces_applicable_patch", "produces an applicable patch"),
        ("test_broken_edit_is_repaired_on_retry", "repairs a broken edit on retry"),
        ("test_out_of_scope_edit_is_rejected_and_reported", "rejects out-of-scope edits"),
        ("test_unparseable_reply_is_challenged", "recovers from an unparseable reply"),
        ("test_working_tree_is_restored_after_run", "leaves the checkout pristine"),
        ("test_need_context_round_then_edit", "can request more evidence first"),
        ("test_failing_checks_still_yield_best_effort_patch", "keeps a best-effort patch"),
    ):
        layer.add(f"{name} (__main__.E2E) ... ok" in text or f"{name}" in text and "OK" in text, label)
    return layer


# ---------------------------------------------------------------------------
# Layers 5-7 — correctness, DB work, cost (need live runs)
# ---------------------------------------------------------------------------

def read_run_artifact(path: Path) -> str:
    """One file from a previous run, or "" if it cannot be read.

    The harness writes some verifier artifacts from inside the container as
    root, so a plain read raises PermissionError on the host and used to abort
    the whole qualification before any layer printed. A run this harness cannot
    read is one run's evidence missing, not a reason to report nothing.
    """
    try:
        return path.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""


def layer_live(model: str | None, enabled: bool) -> tuple[Layer, Layer]:
    correctness = Layer(5, "Correctness on real tasks (live database)")
    economics = Layer(7, "Cost and runtime")

    history = []
    for job in sorted(RUNS.glob("*__*"), key=lambda p: p.stat().st_mtime, reverse=True):
        text = "".join(read_run_artifact(p) for p in job.rglob("*") if p.is_file())
        cost = re.findall(r"cost \$([0-9.]+) \([^)]+\) over (\d+) call", text)
        checks = re.findall(r'testcase name="([^"]+)"', text)
        history.append({"job": job.name, "cost": float(cost[-1][0]) if cost else None,
                        "checks": checks, "text": text})

    scored = [h for h in history if h["checks"]]
    if not scored:
        correctness.add(None, "no scored run found in ~/.ridges/runs",
                        "run ./validate.py to produce one")
    # These three fail for every agent locally -- including the task's own gold
    # solution -- because the verifier runs in the environment container, which
    # never had create_source_manifest.py applied.
    LOCAL_ONLY = re.compile(r"source_tree_conservation|bounded_\w+_method")
    for h in scored[:4]:
        real = [c for c in h["checks"] if not LOCAL_ONLY.search(c)]
        failed_names = re.findall(r'testcase name="([^"]+)"[^>]*>\s*<failure', h["text"])
        real_failed = [f for f in failed_names if not LOCAL_ONLY.search(f)]
        task = h["job"].rsplit("__", 1)[0]
        correctness.add(not real_failed, task,
                        f"{len(real) - len(real_failed)}/{len(real)} real checks"
                        + (f"; failed: {real_failed[:2]}" if real_failed else ""))

    costs = [h["cost"] for h in history if h["cost"] is not None]
    if costs:
        worst, mean = max(costs), sum(costs) / len(costs)
        # The cap is the only hard line: one run over it is one problem scored
        # zero, and under unanimity that is the whole problem lost across all
        # three validators. The target is a ranking preference, so it is judged
        # on the mean -- judged on the worst run it could never pass again once
        # a single expensive run entered the history, and reported NOT READY
        # for something that blocks nothing.
        economics.add(worst <= COST_CAP, "never exceeded the hard cap",
                      f"worst ${worst:.5f} of ${COST_CAP:.2f} over {len(costs)} run(s)")
        economics.add(mean <= COST_BUDGET, f"mean within the ${COST_BUDGET:.2f} target",
                      f"mean ${mean:.5f} over {len(costs)} run(s)")
        near = [c for c in costs if c > COST_CAP * 0.75]
        economics.add(None if near else True, "runs approaching the cap",
                      f"{len(near)} run(s) above ${COST_CAP * 0.75:.3f}; worst ${worst:.5f}"
                      if near else "none within 25% of the cap")
    else:
        economics.add(None, "no cost data recorded yet")

    if not enabled:
        correctness.add(None, "live sweep not requested", "re-run with --live")
    return correctness, economics


def run_live_sweep(model: str | None, tasks: list[str]) -> None:
    """Actually execute the tasks, so Layers 5 and 7 have fresh data to read."""
    command = [sys.executable, str(HERE / "validate.py")]
    if model:
        command += ["--model", model]
    command += tasks
    print(f"{D}running: {' '.join(command)}{X}\n")
    subprocess.run(command, cwd=HERE)


def layer_db_work() -> Layer:
    layer = Layer(6, "Database-work measurement")
    source = AGENT.read_text()
    layer.add("EXPLAIN (ANALYZE, BUFFERS" in source, "can EXPLAIN with buffers (PostgreSQL)")
    layer.add("EXPLAIN indexes = 1" in source, "can EXPLAIN on ClickHouse")
    layer.add("information_schema.columns" in source, "reads live column metadata")
    layer.add("pg_indexes" in source, "reads existing indexes")
    layer.add("SHOW CREATE TABLE" in source, "reads ClickHouse DDL")
    layer.add("clickhouse" in source.lower() and "8123" in source, "reaches ClickHouse over HTTP")
    layer.add("measure_query_scaling" in source, "measures query-count scaling itself",
              "CaptureQueriesContext at two selection sizes")
    layer.add("system.query_log" in source, "reads ClickHouse read_rows/read_bytes")
    layer.add("check_protected" in source, "refuses to edit tests, fixtures and migrations")
    return layer


def verdict(layers: list[Layer]) -> int:
    print(f"\n{'=' * 72}")
    total_failed = sum(l.failed for l in layers)
    total_skipped = sum(l.skipped for l in layers)
    for l in layers:
        state = (f"{R}{l.failed} failed{X}" if l.failed
                 else (f"{Y}{l.skipped} unverified{X}" if l.skipped else f"{G}ok{X}"))
        print(f"  Layer {l.number}  {l.name:<44} {state}")
    print("=" * 72)

    if total_failed:
        print(f"{R}NOT READY{X} — {total_failed} failing check(s). Fix these before uploading.")
        return 1
    if total_skipped:
        print(f"{Y}NOT PROVEN{X} — every check that ran passed, but {total_skipped} could not be "
              f"verified.\n            Readiness for a 60% screener gate is UNKNOWN until those run.")
        return 2
    print(f"{G}READY{X} — all layers verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("tasks", nargs="*", help="task names for --live (default: all six)")
    parser.add_argument("--preflight", action="store_true",
                        help="offline layers only: no Docker, no inference, no cost")
    args = parser.parse_args()

    print(f"{B}Ridges DB-agent qualification{X}   {D}{time.strftime('%Y-%m-%d %H:%M')}{X}")
    print(f"{D}agent: {AGENT}{X}")

    layers = [layer_contract(), layer_portability(), layer_localization(), layer_loop()]
    if args.preflight:
        for l in layers:
            l.render()
        return verdict(layers)
    if args.live:
        if any(l.failed for l in layers):
            print(f"\n{R}offline layers failed — fix those before spending money on live runs{X}")
            for l in layers:
                l.render()
            return verdict(layers + list(layer_live(args.model, False)) + [layer_db_work()])
        print(f"\n{B}Running live tasks (this takes ~5-7 minutes each){X}")
        run_live_sweep(args.model, args.tasks)
    correctness, economics = layer_live(args.model, args.live)
    layers += [correctness, layer_db_work(), economics]
    for l in layers:
        l.render()
    return verdict(layers)


if __name__ == "__main__":
    raise SystemExit(main())
