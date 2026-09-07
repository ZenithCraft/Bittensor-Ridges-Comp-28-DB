#!/usr/bin/env python3
"""Database-discovery harness: does DatabaseProbe find the connection on every
bench app without a live database? (RIDGES_PROBE_NO_PING=1 skips the ping.)

    RIDGES_PROBE_NO_PING=1 python3 probe_eval.py [<pinned-netbox-checkout>]
"""
import os, sys, tempfile
from pathlib import Path

os.environ.setdefault("RIDGES_PROBE_NO_PING", "1")
sys.path.insert(0, str(Path(__file__).parent))
import agent, selftest

CHECKOUT = selftest.default_checkout(sys.argv[1] if len(sys.argv) > 1 else None)

BENCH = Path("/home/ajh/Documents/ridges-bench/db-engineering")


def main() -> int:
    agent.log = lambda message: None
    rows, ok = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for task in sorted(p for p in BENCH.iterdir() if p.is_dir()):
            if task.name.endswith("-test"):
                root = task / "environment" / "app"
            elif task.name in selftest.EXPECTED and CHECKOUT:
                root = selftest.stage(CHECKOUT, task, Path(tmp) / task.name)
            else:
                continue
            want_engine = "clickhouse" if task.name.startswith("ch-") else "postgresql"
            repo = agent.Repository(root)
            ins = agent.parse_instruction((task / "instruction.md").read_text(), root)
            probe = agent.DatabaseProbe(repo, ins)
            t = probe.targets[0] if probe.targets else None
            good = bool(t) and t.engine == want_engine and bool(t.host) and bool(t.user) and bool(t.database)
            ok += good
            shown = f"{t.engine} {t.user}@{t.host}:{t.port}/{t.database} (p{t.priority})" if t else "nothing"
            rows.append(f"{'ok  ' if good else 'MISS'} {task.name:48} {shown}")
    print("\n".join(rows))
    print(f"discovered with engine+host+user+database on {ok}/{len(rows)} apps")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
