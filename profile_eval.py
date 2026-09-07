#!/usr/bin/env python3
"""Offline check of the deterministic problem profiler -- no inference, no database.

Scores every bench task and prints the tier it would route to, so weights and
thresholds can be tuned against real tasks rather than intuition.

    python3 profile_eval.py <pinned-netbox-checkout> [--json out.json]
"""
import argparse, collections, json, os, sys, tempfile
from pathlib import Path

os.environ.setdefault("RIDGES_PROBE_NO_PING", "1")
sys.path.insert(0, str(Path(__file__).parent))
import agent, routing_lab as routing, selftest

BENCH = Path("/home/ajh/Documents/ridges-bench/db-engineering")


def profile_one(root: Path, text: str) -> routing.TaskProfile:
    instruction = agent.parse_instruction(text, root)
    repo = agent.Repository(root)
    targets = agent.locate_targets(repo, instruction)
    probe = agent.DatabaseProbe(repo, instruction)
    return routing.profile_task(instruction, repo, targets, probe)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pristine", nargs="?",
                        help="pinned checkout (default: $RIDGES_NETBOX or the local cache)")
    parser.add_argument("--json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.verbose:
        agent.log = lambda message: None

    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in sorted(p for p in BENCH.iterdir() if p.is_dir()):
            if task.name in selftest.EXPECTED:
                args.pristine = args.pristine or selftest.default_checkout()
                if not args.pristine:
                    continue
                root = selftest.stage(args.pristine, task, Path(tmp) / task.name)
            elif task.name.endswith("-test"):
                root = task / "environment" / "app"
            else:
                continue
            profile = profile_one(root, (task / "instruction.md").read_text())
            rows[task.name] = profile

    for name, p in sorted(rows.items(), key=lambda kv: -kv[1].overall_risk):
        route = routing.select_initial_route(p)
        print(f"{p.tier:10} {p.overall_risk:5.1f}  s{p.structural_risk:4.1f} m{p.semantic_risk:4.1f} "
              f"q{p.query_risk:4.1f} c{p.correctness_risk:4.1f} u{p.uncertainty:4.1f}  "
              f"{route.models[0].split('/')[-1][:14]:14} {name}")
        if args.verbose:
            print("     ", "; ".join(p.reasons[:6]))
    tiers = collections.Counter(p.tier for p in rows.values())
    print(f"\n{len(rows)} tasks: " + "  ".join(f"{t}={tiers[t]}" for t in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")))
    if args.json:
        Path(args.json).write_text(json.dumps({k: v.as_dict() for k, v in rows.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
