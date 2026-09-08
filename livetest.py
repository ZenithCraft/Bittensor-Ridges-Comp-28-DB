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

# Pace the run the way production does. Bench tasks ask for [agent]
# timeout_sec = 1800, but the platform caps every run at 25 minutes and
# engine.py pins it to min(spec, cap) -- so the graded budget is 1500 and
# selftest applies that cap. Pacing this harness against the 1800 the task file
# states would let a run finish here 300s past the point where the graded run
# is killed.
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

# Save it. This run costs real money and produces the one artifact that can be
# graded independently -- printing it to a terminal and throwing it away means
# paying again to get it back. grade.py takes the file directly, which makes
# this the cheap path to a true verdict: no Docker for the agent, no database,
# ~$0.03, and the real verifier still passes judgement on the diff.
if patch.strip():
    patches = Path.home() / ".ridges/patches"
    patches.mkdir(parents=True, exist_ok=True)
    saved = patches / f"{task_name}__{time.strftime('%Y%m%d-%H%M%S')}.diff"
    saved.write_text(patch)
    print(f"\nsaved: {saved}")
    print(f"grade it: ./grade.py {task_name} --patch {saved}")

gold = task / "solution/solve.sh"
if gold.exists():
    print("\n--- reference solution (for comparison) ---")
    print(gold.read_text()[:2500])
tmp.cleanup()
