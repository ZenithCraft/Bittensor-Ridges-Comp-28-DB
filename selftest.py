"""Offline checks for agent.py's deterministic half.

Reconstructs each sample task's starting state (pinned checkout + the task's
own baseline_mutation.py) and exercises instruction parsing, target location,
slicing, edit application and patch generation.  No inference, no database.

    python3 selftest.py <path-to-pinned-netbox-checkout>
"""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agent

BENCH = Path(os.getenv("RIDGES_BENCH") or "/home/ajh/Documents/ridges-bench/db-engineering")

# Where the pinned application checkout lives. Every harness here needs it, so
# they resolve it the same way instead of each taking a positional argument --
# `python -m unittest test_e2e` used to read the module name as the path and
# fail with a FileNotFoundError three frames deep.
CHECKOUT_CACHE = Path.home() / ".cache/ridges-db-agent/netbox"


def task_budgets(bench: Path | None = None) -> dict[str, dict[str, float]]:
    """`{task: {"agent": sec, "verifier": sec}}` read from each task.toml.

    task.toml is NOT one of the four files uploaded to the agent container
    (agent.py, _stdlib_contract.py, ridges_miner_runtime.py, instruction.md),
    so the agent itself can never read it -- it only sees AGENT_TIMEOUT, and
    only when the runner passes one. The local runner does not, so without
    this the agent falls back to DEFAULT_AGENT_TIMEOUT and every local run is
    paced against a different clock than the graded one. The harnesses read
    the file so the agent does not have to.
    """
    budgets: dict[str, dict[str, float]] = {}
    for toml in sorted((bench or BENCH).glob("*/task.toml")):
        try:
            text = toml.read_text()
        except OSError:
            continue
        found: dict[str, float] = {}
        for section in ("agent", "verifier"):
            m = re.search(rf"\[{section}\][^\[]*?timeout_sec\s*=\s*([\d.]+)", text, re.S)
            if m:
                found[section] = float(m.group(1))
        if found:
            budgets[toml.parent.name] = found
    return budgets


def production_agent_timeout(bench: Path | None = None, fallback: float = 1800.0) -> float:
    """The smallest `[agent] timeout_sec` any bench task grants.

    The smallest, not the mean: a local run must not be paced against more
    clock than the tightest graded task allows. Measured 2026-09-07 -- all 56
    tasks state 1800.0, so this is currently a constant, but it is read rather
    than pinned because a single differing task would otherwise go unnoticed.
    """
    values = [b["agent"] for b in task_budgets(bench).values() if "agent" in b]
    return min(values) if values else fallback


PRODUCTION_AGENT_TIMEOUT = production_agent_timeout()


def default_checkout(argv_path: str | None = None) -> Path | None:
    """The pinned checkout: an explicit path, then $RIDGES_NETBOX, then the cache.

    Returns None rather than raising when there is none, so a harness can skip
    the layers that need it and still report the ones that do not.
    """
    for candidate in (argv_path, os.getenv("RIDGES_NETBOX"), CHECKOUT_CACHE):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        # A bare word ("test_e2e") is unittest's argument, not a checkout.
        if path.is_dir() and (path / "netbox").is_dir():
            return path.resolve()
    return None
EXPECTED = {
    "pg-netbox-bulk-tag-assignment-001": "netbox/extras/managers.py",
    "pg-netbox-cached-value-index-001": "netbox/extras/migrations/0107_cachedvalue_extras_cachedvalue_object.py",
    "pg-netbox-contact-group-counts-001": "netbox/tenancy/models/contacts.py",
    "pg-netbox-ipaddress-device-filter-001": "netbox/ipam/filtersets.py",
    "pg-netbox-prefix-hierarchy-annotations-001": "netbox/ipam/querysets.py",
    "pg-netbox-vlangroup-utilization-001": "netbox/ipam/querysets.py",
}
EXPECTED_METHOD = {
    "pg-netbox-bulk-tag-assignment-001": "add",
    "pg-netbox-contact-group-counts-001": "annotate_contacts",
    "pg-netbox-ipaddress-device-filter-001": "filter_device",
    "pg-netbox-prefix-hierarchy-annotations-001": "annotate_hierarchy",
    "pg-netbox-vlangroup-utilization-001": "annotate_utilization",
}

def stage(pristine: Path, task: Path, into: Path) -> Path:
    root = into / "app"
    shutil.copytree(pristine, root, ignore=shutil.ignore_patterns(".git"))
    shutil.copy(task / "environment/configuration.py", root / "netbox/netbox/configuration.py")
    # The task's mutation script hardcodes /app; retarget it at our staging copy.
    mutation = (task / "environment/baseline_mutation.py").read_text().replace("/app", str(root))
    script = into / "mutate.py"
    script.write_text(mutation)
    result = subprocess.run([sys.executable, str(script)], cwd=root,
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"baseline mutation failed for {task.name}: {result.stdout}{result.stderr}")
    return root

def main() -> int:
    pristine = default_checkout(sys.argv[1] if len(sys.argv) > 1 else None)
    if pristine is None:
        print(f"no pinned checkout: pass one, set $RIDGES_NETBOX, or place it at {CHECKOUT_CACHE}")
        return 2
    failures = 0
    # Only the six PostgreSQL samples have the netbox layout this harness stages.
    for task in sorted(p for p in BENCH.iterdir() if p.is_dir() and p.name in EXPECTED):
        with tempfile.TemporaryDirectory() as tmp:
            root = stage(pristine, task, Path(tmp))
            text = (task / "instruction.md").read_text()
            ins = agent.parse_instruction(text, root)
            repo = agent.Repository(root)
            targets = agent.locate_targets(repo, ins)
            ranked = [path for path, _ in targets]
            want = EXPECTED[task.name]
            rank = ranked.index(want) + 1 if want in ranked else 0
            # When the instruction names the file, rank 1 is required. When it
            # does not, the top candidates all reach the model, so presence in
            # the shortlist is the honest bar.
            ok_file = rank == 1 if (ins.edit_only or ins.lint_paths) else 0 < rank <= 3

            slices = agent.slice_around(repo, want, dict(targets).get(want, []), ins) if want in ranked else []
            labels = " / ".join(s.label for s in slices)
            want_method = EXPECTED_METHOD.get(task.name)
            ok_slice = (want_method is None) or any(want_method in s.label for s in slices) \
                       or any(s.label in ("whole file",) for s in slices)

            # Patch round trip: rewrite the target method region and diff it.
            ok_patch = False
            if want in ranked and slices:
                original = repo.read(want)
                repo.write(want, original.rstrip("\n") + "\n# ridges-selftest\n")
                patch = agent.build_patch(repo, [want])
                applies, detail = agent.verify_patch_applies(patch, repo)
                ok_patch = applies and patch.startswith(f"diff --git a/{want}")
                repo.revert_all()
                assert repo.read(want) == original, "revert did not restore the file"

            status = "PASS" if (ok_file and ok_slice and ok_patch) else "FAIL"
            failures += status == "FAIL"
            print(f"[{status}] {task.name}")
            print(f"        target rank : {rank or 'MISSING'} of {len(ranked)}  ({ranked[:3]})")
            print(f"        slices      : {labels[:110] or '-'}")
            print(f"        patch       : {'applies' if ok_patch else 'FAILED'}")
    print(f"\n{6 - failures}/6 tasks located, sliced and patched cleanly")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
