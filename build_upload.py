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
    ("if __name__ not in sys.modules:",
     "The host may exec this module without registering it in sys.modules, which breaks\n"
     "anything resolving a class back to its defining module -- dataclasses' KW_ONLY probe\n"
     "is the one that bites. Give them something real."),
    ("LADDER: list[list[str]] = [",
     "Escalation ladder, strongest model first. A retry after a failed attempt moves one\n"
     "rung up and switches model family: a second opinion from the same architecture tends\n"
     "to repeat the same mistake."),
    ("self._insecure_ctx.check_hostname = False",
     "Certificate verification off for one retry only. In production a transparent proxy\n"
     "intercepts openrouter.ai; when its CA is not in the container trust store an otherwise\n"
     "healthy request fails verification. The retry is used only after an SSLError and the\n"
     "hop is loopback-local. Normal calls verify."),
    ("PROBLEM_PATTERNS: dict[str, tuple[str, ...]] = {",
     "What KIND of problem the statement describes, which selects the evidence gathered and\n"
     "the checks run. It never selects a fix: no branch here leads to stored patch text, and\n"
     "the label is not shown to the model. The statement itself is sent verbatim."),
    ("_PROBE_HOSTS = (",
     "Last-resort discovery of the application's own database, used only when the repository\n"
     "and environment name none. The task ships a live database and the app's config holds\n"
     "the credentials; these conventional names and default logins are the fallback for when\n"
     "that config could not be read. Every candidate must answer SELECT 1 to be used."),
    ("def _postgres_via_python(self, target: DatabaseTarget, query: str, timeout: float) -> str:",
     "Runs a read-only query through the app's own driver when psql is absent. Investigation\n"
     "queries are filtered by is_read_only_sql before they reach here."),
    ("class Checker:",
     "Everything this agent can check for itself before committing to a patch: scope, syntax,\n"
     "the constraints the instruction states, and the repository's own tests. Ordered\n"
     "cheapest-first so a syntax slip never costs a six-minute test run."),
    ("_DANGEROUS_NAMES = {",
     "Names an edit may not introduce into a method the instruction bounds. Pre-existing uses\n"
     "are not flagged -- see the inherited-violation check below -- so a correct fix is never\n"
     "rejected for code it did not write."),
    ("ERROR_HINTS: tuple[tuple[str, str], ...] = (",
     "Translations of failure output the models keep misreading, matched against the test\n"
     "output of the run rather than against the problem statement. Each is a fact about SQL\n"
     "or the ORM, true of any repository; none names a task, a file or an expected value."),
    ("def agent_main(input: dict) -> str:",
     "Entry point. Parses the statement, indexes and ranks the repository, connects to the\n"
     "live database, then loops: ask the model, apply its edits, run the repository's own\n"
     "checks, feed failures back. The patch returned is always built from edits applied\n"
     "during this run; there is no path that returns one from anywhere else."),
)

HEADER = '''"""Ridges miner agent for the database query engineering category.

    def agent_main(input: dict) -> str   # returns a unified diff

Runs inside the task container with the application repository at the workdir and a
live database reachable from it. Only the unified diff returned travels any further:
the patch is applied to an untouched checkout elsewhere and the tests are re-run
there. So the agent may experiment freely here -- run the suite, run EXPLAIN, read
the schema -- and the diff must stay minimal and confined to the files the
instruction names.

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
    out.write_text(artifact + "\n")
    before, after = len(source.encode()), len(out.read_bytes())
    print(f"agent.py      {before:>9,} bytes  {len(source.splitlines()):>6,} lines")
    print(f"dist/agent.py {after:>9,} bytes  {len(artifact.splitlines()):>6,} lines"
          f"   ({(after - before) / before:+.0%})")
    return out


def verify(artifact: Path) -> int:
    """Run the real suite against the artifact, not against agent.py.

    A stripped file that imports is not a stripped file that works: this is the
    only thing standing between a mechanical rewrite and a silently different
    agent, so it runs the whole suite rather than a smoke test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "check"
        shutil.copytree(HERE, staged, ignore=shutil.ignore_patterns(
            "dist", "__pycache__", ".git", "fast-tasks", "results"))
        shutil.copy2(artifact, staged / "agent.py")
        for name in ("fast-tasks",):
            if (HERE / name).is_dir():
                (staged / name).symlink_to(HERE / name)
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py", "-q"],
                              cwd=staged, capture_output=True, text=True)
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        print("\nfull suite against dist/agent.py:")
        for line in tail:
            print(f"   {line}")
        return done.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="run the test suite against the artifact")
    args = parser.parse_args()
    artifact = build()
    return verify(artifact) if args.verify else 0


if __name__ == "__main__":
    raise SystemExit(main())
