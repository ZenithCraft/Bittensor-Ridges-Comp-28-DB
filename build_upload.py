#!/usr/bin/env python3
"""Produce the upload artifact: agent.py with its comments and docstrings gone.

    ./build_upload.py                 -> dist/agent.py
    ./build_upload.py --verify        -> also run the full suite against it

This exists because `/retrieval/agent-code` serves an uploaded agent to anyone
once it has finished evaluating -- unless it is scoring above the code-hiding
cutoff. So the code is public precisely when it is NOT winning, which is when
the reasoning inside it is most worth copying. The comments in agent.py record
what was measured and why (why reasoning stays on, why `For` is not forbidden,
why the node budget is derived rather than fixed); the running code does not
need any of it.

Nothing here is an optimisation. Comments cost nothing at runtime, agent.py is
never sent to a model, and at 253 KB it uses 12% of the 2 MB upload limit. The
only thing being bought is that a competitor reading the artifact gets the
what and not the why.

agent.py itself is never modified. Treat dist/agent.py the way you would treat
a minified bundle: build it, upload it, and keep editing the original.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "agent.py"
DIST = HERE / "dist"


# The tracing subsystem. It exists for reading a run locally and is switched on
# by RIDGES_TRACE, which no graded run sets -- the environment the agent runs in
# is the container's, and the harness forwards nothing. So in production every
# one of these is dead: `trace()` returns on its first line, the tracers are
# never installed, and `ledger()` is never called.
#
# Dead code would be harmless if it were ordinary, but this is 178 lines of
# sys.setprofile, sys.settrace, frame.f_locals and frame.f_globals -- machinery
# that reads every local variable in every frame. Read without its docstrings
# by someone deciding whether an agent is doing what it claims, that is the
# least explicable code in the file, and it buys the graded run nothing.
TRACE_NAMES = {"TRACE", "TRACE_ALL", "TRACE_VARS", "TRACE_SKIP"}
TRACE_FUNCTIONS = {"install_call_tracer", "install_variable_tracer",
                   "_render", "_state_of", "_brief", "trace", "_traced", "ledger"}


class Strip(ast.NodeTransformer):
    """Remove docstrings and the tracing subsystem.

    Comments need no rule of their own: ast keeps no record of them, so
    unparsing a parsed file cannot reproduce one.
    """

    def _clean(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
        if not node.body:                       # a body emptied by the above
            node.body = [ast.Pass()]
        return node

    visit_Module = visit_ClassDef = _clean
    visit_AsyncFunctionDef = _clean

    def visit_FunctionDef(self, node):
        if node.name in TRACE_FUNCTIONS:
            return None
        return self._clean(node)

    def visit_Assign(self, node):
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        return None if targets & TRACE_NAMES else node

    def visit_AnnAssign(self, node):
        name = getattr(node.target, "id", None)
        return None if name in TRACE_NAMES else node

    def visit_Expr(self, node):
        # A bare `trace(...)` or `_traced(...)` statement.
        called = getattr(getattr(node.value, "func", None), "id", None)
        return None if called in TRACE_FUNCTIONS else node

    def visit_If(self, node):
        # `if TRACE:`, `if TRACE and total > 0:`, `if TRACE_VARS: ... elif TRACE_ALL:`
        # and the module's `if __name__ == "__main__":` entry point, which the
        # runtime never takes -- it imports the module and calls agent_main.
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if names & TRACE_NAMES:
            return None
        if (isinstance(node.test, ast.Compare)
                and getattr(node.test.left, "id", None) == "__name__"
                and any(getattr(c, "value", None) == "__main__" for c in node.test.comparators)):
            return None
        self.generic_visit(node)
        return node if node.body else None

    def visit_For(self, node):
        self.generic_visit(node)
        return node if node.body else None      # a loop whose only call was traced

    def visit_While(self, node):
        self.generic_visit(node)
        return node if node.body else None

    def visit_Try(self, node):
        # LLM.complete wraps its call in `try: ... finally: trace(...)`, so
        # removing the trace empties the finally and Python rejects a try with
        # neither handler nor finally. With nothing left to guard, the guard
        # goes and the body stays.
        self.generic_visit(node)
        if not node.handlers and not node.finalbody:
            return node.body
        return node


def strip(tree: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(Strip().visit(tree))



# The build strips every comment, which is right for provenance notes -- "measured
# on the netbox samples", "this cost four clauses before it moved" -- and wrong for
# the handful of constructs that are surprising on sight. A reader deciding what
# this agent does should not have to infer why TLS verification is switched off, or
# why a dict of regexes is matched against the problem statement.
#
# So a short reason goes back above each of those. Anchors are matched exactly once
# and the build fails loudly if one stops matching, because a rename that silently
# dropped the explanation would leave exactly the construct that needed it bare.
ANNOTATIONS: tuple[tuple[str, str], ...] = (
    ("class Allowance:",
     "The run's clock and money. Durations are measured on the monotonic clock, which is\n"
     "what the harness times the run on; the wall clock can be stepped under us and is only\n"
     "used to compare file mtimes. sync() adopts the proxy's own cost total when it is above\n"
     "ours, since the proxy is what refuses the next call once the budget is gone."),
    ("SEAT_CACHE_TERMS = ",
     "List prices per model, used only to pace the run when the endpoint does not quote a\n"
     "cost. They never choose a model or a fix; the graded cost comes from the proxy."),
    ("HISTORY_GIT = ",
     "The patch is read straight off the working tree at the end, so a git command that\n"
     "moves or discards changes loses the work. Read-only git is allowed."),
    ("NETWORK_COMMAND = ",
     "Nothing the task needs is outside the container. A network command from the shell\n"
     "would only burn time waiting on an unreachable host."),
    ("class Tree:",
     "The application folder, snapshotted at start without git: some task images have no\n"
     "git binary and the repo ships without its history. The patch is a difflib diff of\n"
     "snapshot versus disk in the format git apply reads, and restore writes the snapshot\n"
     "back so the harness applies the patch to a clean tree."),
    ("def diff_lines(",
     "The shared head and tail of a file are left out of the comparison so a one-line edit\n"
     "to a long file costs a few lines of work; a middle too large to compare in bounded\n"
     "time is replaced whole, which git applies just the same."),
    ("SCOPE_FILE_RES = ",
     "Phrasing that names the file an instruction confines the change to. Read out so the\n"
     "edit gate and the final revert know the scope; every path is verified on disk first\n"
     "and nothing here selects a fix."),
    ("class Warden:",
     "The instruction's own conditions, checked before hand-in: only the named files\n"
     "change, no test files are touched, no definitions are dropped, no suppressions are\n"
     "added, and the check commands it names exit clean."),
    ("def harness_lead(",
     "The harness writes the instruction file the moment its timer starts, then commits a\n"
     "git baseline of the whole tree before this process runs. The file's age is how far\n"
     "the harness clock is ahead of ours; both deadlines are moved by it."),
    ("def agent_main(",
     "Entry point. Reads the scope out of the instruction, locates and plans only when the\n"
     "instruction names no file, drives the edit, then always builds the patch from the\n"
     "tree, restores it, and checks that the patch applies. There is no path that returns a\n"
     "patch from anywhere but this run's edits."),
)

HEADER = '''"""Ridges miner agent for the database query engineering category.

    def agent_main(input: dict) -> str   # returns a unified diff

Runs inside the task container with the application repository at the workdir. Only
the unified diff returned travels any further: the harness applies it to the tree and
the verifier runs elsewhere. Stages: the scope is read out of the instruction; a
locator and a planner run only when it names no file; a driver edits, runs the checks
the instruction names, and submits. The patch is a diff of the working tree against a
snapshot taken at start, so no git and no network are needed.

Standard library only: an arbitrary application container is not guaranteed to have
anything else. Built from agent.py by build_upload.py; edit that, not this.
"""
'''


def annotate(source: str) -> str:
    """Put the explanations back above the constructs that need one."""
    lines = source.splitlines()
    for anchor, comment in ANNOTATIONS:
        hits = [i for i, line in enumerate(lines) if line.lstrip().startswith(anchor)]
        if len(hits) != 1:
            raise SystemExit(f"annotation anchor matched {len(hits)} times, expected 1: {anchor!r}")
        index = hits[0]
        indent = " " * (len(lines[index]) - len(lines[index].lstrip()))
        lines[index:index] = [f"{indent}# {part}" for part in comment.split("\n")]
    return HEADER + "\n".join(lines)


def build() -> Path:
    source = SOURCE.read_text()
    artifact = annotate(ast.unparse(strip(ast.parse(source))))
    DIST.mkdir(exist_ok=True)
    out = DIST / "agent.py"
    out.unlink(missing_ok=True)            # the previous build is read-only
    out.write_text(artifact + "\n")
    # Read-only for the same reason as the staged tools: this is what gets
    # uploaded, and an edit made here is an edit lost on the next build. Edit
    # agent.py.
    out.chmod(0o444)
    before, after = len(source.encode()), len(out.read_bytes())
    print(f"agent.py      {before:>9,} bytes  {len(source.splitlines()):>6,} lines")
    print(f"dist/agent.py {after:>9,} bytes  {len(artifact.splitlines()):>6,} lines"
          f"   ({(after - before) / before:+.0%})")
    return out


def stage_harnesses() -> list[str]:
    """Put the validation tools beside the artifact.

    Every harness resolves the agent as `Path(__file__).parent / "agent.py"` and
    inserts its own directory on sys.path, so a copy living in dist/ finds
    dist/agent.py with no flag, no environment variable and no change to the
    tool. Python's own import rules do the work that a --dist switch would
    otherwise have to fake -- and faking it is awkward, because `import agent`
    runs at module load, before argparse could have seen a flag.

    What this buys is the thing static equivalence cannot give: the artifact
    can be run. It is proved equal to the source on parsing and prompt bytes,
    but that proof stops at the prompt -- it says nothing about the transport,
    the JSON round trip, or patch assembly in the file actually being uploaded.

    The copies are made read-only. They are build output, and an edit made here
    is an edit lost on the next build; better to refuse the write than to
    discard it silently later.
    """
    staged: list[str] = []
    for source in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh")):
        if source.name in ("agent.py", Path(__file__).name):
            continue                       # generated; and the builder itself
        target = DIST / source.name
        target.unlink(missing_ok=True)     # the previous copy is read-only
        shutil.copy2(source, target)
        target.chmod(0o555 if source.read_text().startswith("#!") else 0o444)
        staged.append(source.name)
    # The task corpora stay where they are; dist just needs to see them.
    for name in ("fast-tasks",):
        link = DIST / name
        if (HERE / name).is_dir() and not link.exists():
            link.symlink_to(HERE / name, target_is_directory=True)
    return staged


def verify() -> int:
    """Run the real suite in dist/, against the artifact.

    A stripped file that imports is not a stripped file that works, and this is
    the only thing standing between a mechanical rewrite and a silently
    different agent. It runs where you would run it by hand, so a failure is
    reproducible with the same command.
    """
    done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py", "-q"],
                          cwd=DIST, capture_output=True, text=True)
    tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
    print("\nfull suite in dist/, against dist/agent.py:")
    for line in tail:
        print(f"   {line}")
    return done.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="run the test suite against the artifact")
    args = parser.parse_args()
    build()
    staged = stage_harnesses()
    print(f"staged {len(staged)} validation tool(s) into dist/  "
          f"{', '.join(n for n in staged if n in ('validate.py','livetest.py','grade.py','preprocessing.py','qualify.py'))}")
    code = verify() if args.verify else 0
    print("\nvalidate the artifact from inside dist/ -- `import agent` resolves there:")
    print("    cd dist && python3 livetest.py <task>          # real inference, ~$0.03")
    print("    cd dist && ./grade.py <task> --patch <diff>    # true reward")
    print("    cd dist && ./validate.py <task>                # full container run")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
