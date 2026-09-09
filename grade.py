#!/usr/bin/env python3
"""Grade a patch the way a real validator does, in a separate verifier container.

`ridges miner run-local` has no separate verifier environment: harbor builds one
image from `environment/`, copies `tests/` to `/tests`, and runs verify.py inside
the agent's own container. Two consequences make its verdict weaker than a
validator's -- the verifier sees the miner-visible database password instead of
the hidden admin one, and `source_tree_conservation` can never pass, because the
harness runs `git init` in /app and verify.py's rglob has no .git exclusion.

task.toml asks for what harbor does not implement:

    [verifier] environment_mode = "separate"

This builds it by hand. `tests/Dockerfile` is self-contained and `tests/
docker-compose.yaml` ships its own postgres and redis with the frozen admin
credentials, so the production topology is reproducible: a pristine container
with /opt/task baked in, no .git, a fresh database, and the real verify.py.

    ./grade.py <task> --patch out.diff          # grade a patch
    ./grade.py <task> --solution                # grade the task's own answer
    ./grade.py <task> --patch-from-run <dir>    # grade what validate.py produced
    ./grade.py --all --solution                 # calibrate: every task must score 1

Calibration is not optional. A grader nobody checked is another thing that might
be lying, and `solution/solve.sh` is the one patch whose verdict is known in
advance: it must score 1. If it does not, this script is wrong, not the solution.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selftest

BENCH = selftest.BENCH
FAST = Path(__file__).resolve().parent / "fast-tasks"
# Beside ~/.ridges/runs, which is where validate.py's artifacts land. Everything
# the verifier produced is copied out here before the containers are torn down:
# /logs/verifier lives inside a container this script deletes, so anything not
# saved at that moment is gone, and a grade nobody can re-read afterwards is a
# grade you have to pay to reproduce.
GRADES = Path.home() / ".ridges/grades"
G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"

# The verifier service harbor would have built from [verifier.environment].
# `main` is deliberately not reused: that name belongs to the agent's container,
# and the whole point here is that they are different images.
OVERLAY_HEAD = """\
services:
  verifier:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["sh", "-c", "sleep infinity"]
"""

# Top-level service names in a compose file, and whether each declares a
# healthcheck. Two indentation levels of a mapping is all this needs to know,
# so it stays a regex rather than a YAML dependency this harness would
# otherwise carry for one field.
_SERVICE = re.compile(r"^  ([A-Za-z0-9][\w.-]*):\s*$")
_HEALTHCHECK = re.compile(r"^    healthcheck:\s*$")


def sidecars(compose: Path) -> dict[str, bool]:
    """The task's own sidecar services, mapped to whether they report health.

    The bench is not one stack: the PostgreSQL tasks ship postgres and redis,
    the ClickHouse tasks ship a single clickhouse. Hardcoding the first set
    meant `grade.py` produced a compose file referring to services that do not
    exist on a ClickHouse task, so the true grader could not be run on a third
    of the corpus. Reading the names out of the task's own file needs no list
    to keep up to date.

    Health matters because `depends_on` and `--wait` only mean "started" for a
    service that declares no healthcheck, and waiting on health that is never
    reported hangs until the timeout.
    """
    found: dict[str, bool] = {}
    current: str | None = None
    inside_services = False
    for line in compose.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            inside_services = line.startswith("services:")
            current = None
            continue
        if not inside_services:
            continue
        named = _SERVICE.match(line)
        if named:
            current = named.group(1)
            found.setdefault(current, False)
        elif current and _HEALTHCHECK.match(line):
            found[current] = True
    found.pop("verifier", None)
    return found


def overlay_for(services: dict[str, bool]) -> str:
    """The verifier service, waiting on whatever the task actually ships."""
    if not services:
        return OVERLAY_HEAD
    rows = "".join(
        f"      {name}:\n        condition: "
        f"{'service_healthy' if healthy else 'service_started'}\n"
        for name, healthy in sorted(services.items()))
    return OVERLAY_HEAD + "    depends_on:\n" + rows


def docker(*args: str, timeout: float = 3600, check: bool = False,
           cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a docker command, through `sg docker` when the group is not active."""
    command = ["docker", *args]
    if not _direct_docker():
        command = ["sg", "docker", "-c", " ".join(_quote(part) for part in command)]
    return subprocess.run(command, cwd=cwd, text=True, timeout=timeout, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _quote(part: str) -> str:
    return part if re.fullmatch(r"[\w@%+=:,./-]+", part) else "'" + part.replace("'", "'\\''") + "'"


_DIRECT: bool | None = None


def _direct_docker() -> bool:
    """Whether this shell can reach the docker socket without `sg docker`."""
    global _DIRECT
    if _DIRECT is None:
        probe = subprocess.run(["docker", "info"], text=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        _DIRECT = probe.returncode == 0
    return _DIRECT


def project_name(task: Path) -> str:
    """A compose project unique to this invocation.

    Not just the task name: two graders on one task would share the project,
    and `down -v` in either tears down the other's containers mid-run -- which
    surfaces as "no running container for service 'verifier'" in whichever
    process was slower. The suffix costs nothing and makes concurrent grading,
    or a grade alongside a validate.py run, safe.
    """
    stem = re.sub(r"[^a-z0-9]+", "-", task.name.lower())[:36]
    return f"ridges-grade-{stem}-{os.getpid():d}"


def resolve_task(argument: str, tasks_dir: Path) -> Path:
    for candidate in (Path(argument), tasks_dir / argument):
        if (candidate / "tests" / "test.sh").is_file():
            return candidate.resolve()
    raise SystemExit(f"no task with tests/test.sh at {argument!r} or under {tasks_dir}")


def verifier_timeout(task: Path) -> float:
    """The clock the graded run actually gets, from task.toml.

    Not the agent's budget: the verifier re-runs the suite in its own container
    against `[verifier] timeout_sec`, which is 900-1800 across the bench and is
    the tighter of the two on most tasks. A patch that makes the suite slower is
    graded against this, and nothing in the agent's own run reveals it.
    """
    budgets = selftest.task_budgets(task.parent)
    return budgets.get(task.name, {}).get("verifier", 900.0)


class Verifier:
    """The compose project: postgres, redis, and a pristine verifier container."""

    def __init__(self, task: Path, *, keep: bool = False, verbose: bool = False) -> None:
        self.task = task
        self.tests = task / "tests"
        self.project = project_name(task)
        self.keep = keep
        self.verbose = verbose
        self.overlay = self.tests / ".grade-verifier.yaml"
        self.sidecars = sidecars(self.tests / "docker-compose.yaml")

    # -- lifecycle -------------------------------------------------------
    def _compose(self, *args: str, timeout: float = 3600) -> subprocess.CompletedProcess:
        return docker("compose", "-p", self.project,
                      "-f", "docker-compose.yaml", "-f", self.overlay.name,
                      *args, cwd=self.tests, timeout=timeout)

    def __enter__(self) -> "Verifier":
        self.overlay.write_text(overlay_for(self.sidecars))
        # Down first: a previous run's database holds the schema this run's
        # conservation check hashes, so a reused one is a different test.
        self._compose("down", "-v", "--remove-orphans", timeout=300)
        say(f"{D}building the verifier image (cached layers are reused){X}")
        built = self._compose("build", "verifier")
        if built.returncode:
            raise SystemExit(f"verifier image build failed:\n{built.stdout[-3000:]}")
        if self.sidecars:
            names = sorted(self.sidecars)
            say(f"{D}starting {', '.join(names)}{X}")
            # --wait only where health is reported; asking for it otherwise
            # waits out the full timeout on a container that is already up.
            waitable = [n for n in names if self.sidecars[n]]
            up = self._compose("up", "-d", *(["--wait"] if waitable else []),
                               *names, timeout=900)
            if up.returncode:
                raise SystemExit(f"sidecars did not become healthy:\n{up.stdout[-3000:]}")
        started = self._compose("up", "-d", "--force-recreate", "verifier", timeout=600)
        if started.returncode:
            raise SystemExit(f"verifier container did not start:\n{started.stdout[-3000:]}")
        self.stage_pristine()
        return self

    def stage_pristine(self) -> None:
        """Put the pristine copy of the application where the grader looks.

        The two halves of the bench check source-tree conservation differently.
        The PostgreSQL tasks hash a manifest baked into the image; the
        ClickHouse ones diff /app against a pristine tree at /tests/app -- and
        their verifier Dockerfile copies the test scripts into /tests without
        copying that tree, so every file in /app reads as newly added and the
        reference solution scores zero.

        In a real run the verifier has the task's whole tests directory
        available, so the tree is simply there. Here the image is built from
        that directory and keeps only what the Dockerfile asked for, so it has
        to be put back. Copied after every container creation, because
        recreating the verifier is what gives each patch a fresh /app and it
        takes the staged tree with it.
        """
        pristine = self.tests / "app"
        if not pristine.is_dir():
            return                       # a manifest-based task; nothing to stage
        self.copy_in(pristine, "/tests/app")

    def __exit__(self, *exc) -> None:
        if self.keep:
            say(f"{D}left running: docker compose -p {self.project} ... down -v{X}")
        else:
            self._compose("down", "-v", "--remove-orphans", timeout=300)
        self.overlay.unlink(missing_ok=True)

    # -- container access ------------------------------------------------
    def container(self, service: str = "verifier") -> str:
        found = self._compose("ps", "-q", service)
        name = [line for line in (found.stdout or "").strip().splitlines() if line.strip()]
        if not name:
            raise SystemExit(
                f"no running container for service {service!r} in project {self.project}.\n"
                f"  Another grade.py or a `docker compose down` may have removed it; "
                f"re-run, and check `docker ps -a` if it repeats.")
        return name[-1]

    def exec(self, script: str, *, timeout: float = 3600,
             service: str = "verifier") -> subprocess.CompletedProcess:
        return docker("exec", self.container(service), "bash", "-lc", script, timeout=timeout)

    def copy_in(self, source: Path, destination: str) -> None:
        done = docker("cp", str(source), f"{self.container()}:{destination}")
        if done.returncode:
            raise SystemExit(f"could not copy {source} into the container:\n{done.stdout}")

    def copy_out(self, source: str, destination: Path) -> bool:
        return docker("cp", f"{self.container()}:{source}", str(destination)).returncode == 0

    def recreate(self) -> None:
        """A fresh /app. Grading mutates it, so each patch needs its own."""
        self._compose("up", "-d", "--force-recreate", "verifier", timeout=600)
        self.stage_pristine()

    # -- the two things this script does ---------------------------------
    def patch_from_solution(self) -> str:
        """The reference answer as a unified diff, produced inside the image.

        Generated rather than shipped: solve.sh edits files in place, and the
        patch has to describe the same tree verify.py will hash -- the one the
        verifier image built, not a local checkout of it.
        """
        solve = self.task / "solution" / "solve.sh"
        if not solve.is_file():
            raise SystemExit(f"no solution/solve.sh for {self.task.name}")
        self.copy_in(solve, "/tmp/solve.sh")
        # git is only a diff engine here; .git is removed before verify.py ever
        # walks the tree, because source_tree_conservation would count it.
        result = self.exec(
            "set -e; cd /app; rm -rf .git; git init -q .;"
            " git config user.email g@example.invalid; git config user.name grade;"
            " git add -A; git commit -qm base;"
            " bash /tmp/solve.sh;"
            " git diff > /tmp/solution.diff; rm -rf .git /tmp/solve.sh;"
            " wc -c < /tmp/solution.diff", timeout=900)
        if result.returncode:
            raise SystemExit(f"could not build a patch from solve.sh:\n{result.stdout[-3000:]}")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "solution.diff"
            if not self.copy_out("/tmp/solution.diff", out):
                raise SystemExit("solve.sh produced no patch")
            return out.read_text()

    def grade(self, patch: str, *, timeout: float) -> dict:
        """Apply a patch in a pristine container and run the real test.sh."""
        artifacts = GRADES / f"{self.task.name}__{time.strftime('%Y%m%d-%H%M%S')}"
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "patch.diff"
            staged.write_text(patch)
            self.exec("mkdir -p /logs/agent /logs/verifier /logs/artifacts")
            self.copy_in(staged, "/logs/agent/patch.diff")
            started = time.monotonic()
            run = self.exec("/tests/test.sh", timeout=timeout + 60)
            elapsed = time.monotonic() - started
            reward = self.exec("cat /logs/verifier/reward.txt 2>/dev/null || echo -")
            checks = []
            junit = Path(tmp) / "junit.xml"
            if self.copy_out("/logs/verifier/junit.xml", junit):
                checks = read_junit(junit)
                artifacts.mkdir(parents=True, exist_ok=True)
                shutil.copy2(junit, artifacts / "junit.xml")
        text = (reward.stdout or "").strip()
        result = {"reward": float(text) if re.fullmatch(r"[\d.]+", text) else None,
                  "checks": checks, "seconds": elapsed, "returncode": run.returncode,
                  "output": run.stdout or "", "timed_out": elapsed > timeout}
        # Written now, not at the end: the verifier's own logs go with the
        # container, and a failure here is exactly when you want them.
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "graded.patch").write_text(patch)
        (artifacts / "test-output.txt").write_text(run.stdout or "")
        (artifacts / "result.json").write_text(json.dumps(
            {k: v for k, v in result.items() if k != "output"}, indent=2, default=str))
        self.artifacts = artifacts
        return result


def read_junit(path: Path) -> list[tuple[str, bool, str]]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    out = []
    for case in root.iter("testcase"):
        failure = case.find("failure")
        out.append((case.get("name") or "?", failure is None,
                    (failure.get("message") or "")[:300] if failure is not None else ""))
    return out


def say(message: str) -> None:
    print(message, flush=True)


def report(task: str, result: dict, *, patch_bytes: int, budget: float) -> bool:
    passed = result["reward"] == 1.0
    say(f"\n{'=' * 78}\n{task}\n{'=' * 78}")
    for name, ok, message in result["checks"]:
        say(f"   [{G}pass{X}] {name}" if ok else f"   [{R}FAIL{X}] {name}\n        {D}{message}{X}")
    colour = G if passed else R
    say(f"\n   reward {colour}{result['reward']}{X}"
        f"   {sum(1 for _, ok, _ in result['checks'] if ok)}/{len(result['checks'])} checks"
        f"   {patch_bytes:,} patch bytes"
        f"   {result['seconds']:.0f}s of {budget:.0f}s")
    if result["timed_out"]:
        say(f"   {R}over the verifier budget: a graded run would be killed here{X}")
    if result.get("artifacts"):
        say(f"   {D}artifacts: {result['artifacts']}{X}")
    if not result["checks"] and result["output"]:
        say(f"{D}{result['output'][-1500:]}{X}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", nargs="?", help="task name or path")
    parser.add_argument("--all", action="store_true", help="every task in the task directory")
    parser.add_argument("--tasks-dir", help=f"where tasks live (default: fast-tasks, else {BENCH})")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--patch", help="a unified diff to grade")
    source.add_argument("--solution", action="store_true",
                        help="grade the task's own solution/solve.sh -- must score 1")
    source.add_argument("--patch-from-run", help="a ~/.ridges/runs job directory")
    parser.add_argument("--keep", action="store_true", help="leave the containers running")
    parser.add_argument("--json", help="write every verdict here as well")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks_dir = (Path(args.tasks_dir).resolve() if args.tasks_dir
                 else (FAST if FAST.is_dir() and any(FAST.iterdir()) else BENCH))
    if args.all:
        tasks = sorted(p for p in tasks_dir.iterdir() if (p / "tests" / "test.sh").is_file())
        if not tasks:
            parser.error(f"no tasks with tests/test.sh under {tasks_dir}")
    elif args.task:
        tasks = [resolve_task(args.task, tasks_dir)]
    else:
        parser.error("give a task name or --all")

    verdicts: dict[str, bool] = {}
    collected: dict[str, dict] = {}
    for task in tasks:
        with Verifier(task, keep=args.keep, verbose=args.verbose) as verifier:
            if args.solution:
                patch = verifier.patch_from_solution()
                verifier.recreate()          # solve.sh mutated /app; grade on a clean one
            elif args.patch:
                patch = Path(args.patch).read_text()
            else:
                found = list(Path(args.patch_from_run).rglob("patch.diff"))
                if not found:
                    raise SystemExit(f"no patch.diff under {args.patch_from_run}")
                patch = found[0].read_text()
            budget = verifier_timeout(task)
            result = verifier.grade(patch, timeout=budget)
            result["artifacts"] = str(getattr(verifier, "artifacts", ""))
            verdicts[task.name] = report(task.name, result, patch_bytes=len(patch), budget=budget)
            collected[task.name] = {k: v for k, v in result.items() if k != "output"}

    if args.json:
        Path(args.json).write_text(json.dumps(collected, indent=2, default=str))
        say(f"\n{D}wrote {args.json}{X}")
    if len(verdicts) > 1 or args.solution:
        say(f"\n{'-' * 78}")
        say(f"{D}per-task artifacts under {GRADES}{X}")
        won = sum(verdicts.values())
        say(f"{won}/{len(verdicts)} scored reward 1")
        if args.solution and won != len(verdicts):
            say(f"{R}CALIBRATION FAILED{X} — the reference solution must score 1 on every task. "
                f"Fix this script before trusting any other verdict it gives.")
            return 1
    return 0 if all(verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
