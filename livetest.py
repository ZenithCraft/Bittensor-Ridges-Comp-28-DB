"""Run agent_main against a staged sample task using REAL inference.

No Docker and no database: the task's own test commands are stubbed out, so
this exercises the transport, prompt, edit protocol and patch generation, and
reports what a task actually costs.

    python3 livetest.py <task-name> [checkout]
"""
import os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

key = ""
for line in Path("/home/ajh/.ridges/.env.miner").read_text().splitlines():
    if line.startswith("RIDGES_OPENROUTER_API_KEY="):
        key = line.split("=", 1)[1].strip()
os.environ["OPENROUTER_API_KEY"] = key
os.environ.pop("SANDBOX_PROXY_URL", None)
import agent, selftest

BENCH = selftest.BENCH
# The task name comes first now; the checkout is resolved. Accept the old
# <checkout> <task> order too rather than staging the wrong thing silently.
args = sys.argv[1:]
if args and (BENCH / args[0]).is_dir():
    task_name, rest = args[0], args[1:]
elif len(args) > 1:
    task_name, rest = args[1], args[:1]
else:
    raise SystemExit(f"usage: livetest.py <task-name> [checkout]\n"
                     f"tasks live in {BENCH}")
pristine = selftest.default_checkout(rest[0] if rest else None)
if pristine is None:
    raise SystemExit(f"no pinned checkout: pass one, set $RIDGES_NETBOX, or place it at "
                     f"{selftest.CHECKOUT_CACHE}")
task = BENCH / task_name

# Pace the run the way production does. Every bench task grants the agent
# [agent] timeout_sec = 1800, and the runner passes that through as
# AGENT_TIMEOUT; the agent's 1500 fallback only applies when nothing sets it.
# At 900 this harness measured the agent against half the clock it will
# actually get, so a run that stopped early here would not have stopped there.
os.environ["AGENT_TIMEOUT"] = os.getenv("AGENT_TIMEOUT") or str(selftest.PRODUCTION_AGENT_TIMEOUT)

tmp = tempfile.TemporaryDirectory()
root = selftest.stage(pristine, task, Path(tmp.name))
agent.workdir = lambda: root
agent.DatabaseProbe._discover = lambda self: None          # no live database here
agent.Verifier.run_task_commands = lambda self: [          # no django/postgres here
    agent.CheckResult("$ task checks", True, "skipped (no database in livetest)")]

def account_totals():
    """OpenRouter's own ledger, for cross-checking the agent's accounting."""
    import json, urllib.request
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["data"]
    except Exception as exc:
        return {"error": str(exc)}

before = account_totals()
started = time.monotonic()
patch = agent.agent_main({"problem_statement": (task / "instruction.md").read_text()})
elapsed = time.monotonic() - started

after = account_totals()
print("\n" + "=" * 72)
print(f"TASK      {task_name}")
print(f"ELAPSED   {elapsed:.0f}s")
print(f"LEDGER    total_usage {before.get('total_usage')} -> {after.get('total_usage')}")
print("=" * 72)
print(patch if patch.strip() else "*** NO PATCH PRODUCED ***")
print("=" * 72)

gold = task / "solution/solve.sh"
if gold.exists():
    print("\n--- reference solution (for comparison) ---")
    print(gold.read_text()[:2500])
tmp.cleanup()
