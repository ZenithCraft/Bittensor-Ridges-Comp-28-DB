"""Ridges miner agent for the database query engineering category.

Contract (see ridges/docs/sandbox.md and ridges_harbor/ridges_miner_runtime.py):

    def agent_main(input: dict) -> str   # returns a unified diff

The agent runs as root inside the task's `main` container, with the application
repository at the task workdir (usually /app) and a *live* database reachable
from that container.  The verifier runs in a separate, pristine container: it
copies out only /logs/agent/patch.diff, `git apply`s it to an untouched
checkout, and re-runs the tests.  Two consequences drive the whole design:

  1. We may experiment freely in our own container -- run the test suite, run
     EXPLAIN, probe the schema -- because none of that reaches the verifier.
  2. Only the diff is graded, and verifiers in this category hash every file
     they did not authorise us to touch.  The diff must therefore be minimal
     and confined to the files the instruction names.

The repository checkout has had .git removed, so patches are produced from an
in-memory snapshot with difflib rather than by shelling out to git.
"""

import ast
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The host runtime may exec this module without registering it in sys.modules,
# which breaks anything that resolves a class back to its defining module
# (dataclasses' KW_ONLY probe is the one that bites). Give them something real.
if __name__ not in sys.modules:  # pragma: no cover - depends on the loader
    sys.modules[__name__] = types.ModuleType(__name__)

AGENT_START = time.monotonic()

# Reserve a slice of the wall clock for patch emission; never spend it all on
# inference.
#
# AGENT_TIMEOUT is injected by the runtime from the task's `[agent]
# timeout_sec`. The task.toml stating it is NOT one of the four files uploaded
# to this container -- those are agent.py, _stdlib_contract.py,
# ridges_miner_runtime.py and instruction.md -- and /opt/task exists only in
# the verifier's container, so this process cannot read the budget itself and
# must not try. The env var is the whole channel.
#
# Production always sets it (engine.py: min(spec_timeout, max_agent_timeout_sec)),
# so the fallback below is dead there and live only in local runs, where
# `ridges miner run-local` plumbs no timeout. Every bench task grants 1800s;
# validate.py bakes that into its local agent copy so a local run is paced like
# a graded one instead of against this shorter fallback.
DEFAULT_AGENT_TIMEOUT = 1500.0   # local-only fallback; AGENT_TIMEOUT overrides in production
TIMEOUT_SAFETY_MARGIN = 120.0

# Budget.  RIDGES_MAX_COST_USD is injected by the runtime; the proxy also
# exposes live usage.  Ranking rewards cheap runs, so we aim far below the cap.
DEFAULT_MAX_COST_USD = 0.29   # production per-problem inference budget
COST_TARGET_USD = 0.05  # soft target: stop escalating models past this

USAGE_URL = "http://sandbox-proxy:80/api/v1/usage"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model roster.  These are the OpenRouter slugs the Ridges inference gateway
# whitelists (inference_gateway/providers/openrouter.py).  Prices are indicative
# only -- the proxy's usage endpoint is authoritative for spend.
@dataclass(frozen=True)
class ModelSpec:
    slug: str
    usd_per_m_in: float
    usd_per_m_out: float
    context: int


MODELS: dict[str, ModelSpec] = {
    "deepseek/deepseek-v4-pro-0813": ModelSpec("deepseek/deepseek-v4-pro-0813", 1.12, 3.36, 1048576),  # coding 69, agentic 50, ~$0.0114/task
    "qwen/qwen3.8-27b": ModelSpec("qwen/qwen3.8-27b", 0.42, 3.00, 1000000),  # coding 68, agentic 51, ~$0.0147/task
    "moonshotai/kimi-k2.6": ModelSpec("moonshotai/kimi-k2.6", 0.95, 4.00, 262144),  # coding 62, agentic 31, ~$0.0230/task
    "moonshotai/kimi-k2.7-code": ModelSpec("moonshotai/kimi-k2.7-code", 0.66, 3.40, 262144),  # coding 61, agentic 30, ~$0.0183/task
    "minimax/minimax-m3": ModelSpec("minimax/minimax-m3", 0.30, 1.20, 1048576),  # coding 59, agentic 36, ~$0.0070/task
    "qwen/qwen3.6-35b-a3b": ModelSpec("qwen/qwen3.6-35b-a3b", 0.10, 0.90, 262144),  # coding 42, agentic 22, ~$0.0042/task
    "google/gemma-4-31b-it": ModelSpec("google/gemma-4-31b-it", 0.09, 0.34, 262144),  # coding 43, agentic 14, ~$0.0020/task
}

# Escalation ladder.  Tier 0 handles the great majority of tasks; each retry
# after a *verified* failure moves one rung up.  Entries are tried left to
# right if the gateway rejects a slug.
# Escalation ladder, ordered by measured capability rather than price.
#
# Passing the screener is a *success-rate* gate (40%, then 60%); price only
# moves ranking afterwards. So tier 0 is the strongest agentic model that still
# fits the budget, not the cheapest model overall. Artificial Analysis indices
# via OpenRouter, cost projected from the token profile of real runs:
#
#   glm-4.7            coding 45  agentic 26   $0.0099/task   <- best agentic
#   kimi-k2.5          coding 47  agentic 22   $0.0122
#   qwen3.5-397b       coding 48  agentic 20   $0.0177        <- best coding
#   glm-4.6            coding 46  agentic 19   $0.0129
#   qwen3-coder-next   coding 36  agentic  9   $0.0040        <- cheapest, weakest agentically
#
# This agent is agentic: it runs a context round, applies edits, reads real test
# failures and retries. Agentic score therefore matters more than raw coding.
LADDER: list[list[str]] = [
    # Score is the gate: screeners threshold on success rate, and a validator
    # counts a problem only when every validator solved it. So each tier leads
    # with the most capable model available, and escalation switches model
    # *family* rather than just spending more -- a second opinion from the same
    # architecture tends to repeat the same mistake.
    #
    # deepseek-v4-flash is excluded entirely despite benchmarking well
    # (coding 69, agentic 48, $0.0012). On two real tasks it returned no content
    # even with the cap removed, burning 551s to produce nothing -- it loses on
    # score, runtime and, once you count the wasted calls, price as well.
    ["deepseek/deepseek-v4-pro-0813", "qwen/qwen3.8-27b", "minimax/minimax-m3"],
    ["qwen/qwen3.8-27b", "moonshotai/kimi-k2.6", "deepseek/deepseek-v4-pro-0813"],
    ["moonshotai/kimi-k2.6", "moonshotai/kimi-k2.7-code", "qwen/qwen3.8-27b"],
]

MAX_REPAIR_ROUNDS = 3        # verified wrong answers (tests failed) before giving up
MAX_GATE_SLIPS = 4           # cheap protocol slips (scope/syntax/contract) -- no tests were run
MAX_CONTEXT_ROUNDS = 6       # absolute ceiling on investigation, whatever the dynamic budget says

# Guard against runaway generation. The edit protocol needs a short JSON object;
# anything beyond this is reasoning we are paying for and do not use.
# Reasoning tokens bill as completion and are emitted *before* any content, so
# too low a cap yields an empty reply and a wasted call. Observed: a 6000 cap on
# deepseek-v4-flash produced two empty completions before succeeding at 16000.
COMPLETION_CAP = 12000                   # a JSON edit is ~2-4k tokens; the rest is reasoning
MIN_COMPLETION_CAP = 6000
PRICE_SAFETY = 1.3                       # providers bill above the listed rate
NUDGE = ("Your previous reply contained no answer text -- the reasoning used up the "
         "whole token budget. Answer now with the JSON object only.")

# When set, every round uses this model instead of the escalation ladder.
# Set via RIDGES_DB_AGENT_MODEL, or by appending an assignment to a copy of
# this file (see validate.py --model).
FORCE_MODEL = os.getenv("RIDGES_DB_AGENT_MODEL", "").strip()

SOURCE_SUFFIXES = {
    ".py", ".sql", ".go", ".rb", ".java", ".kt", ".ts", ".tsx", ".js", ".jsx",
    ".rs", ".php", ".cs", ".scala", ".ex", ".exs", ".c", ".cpp", ".h", ".hpp",
    ".yml", ".yaml", ".toml", ".json", ".xml", ".hql", ".erb",
    ".ini", ".cfg", ".properties", ".prisma", ".env",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist",
    "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    ".tox", "target", "vendor", "coverage", ".next", ".idea", ".cache",
}

MAX_INDEXED_FILES = 20_000


def log(message: str) -> None:
    elapsed = time.monotonic() - AGENT_START
    print(f"[db-agent {elapsed:7.1f}s] {message}", flush=True)


# Per-module tracing: what went into each stage and what came out, so a run can
# be read stage by stage instead of inferred from its result. Off unless
# RIDGES_TRACE is set, because a graded run should not pay for it and its
# operator never reads it -- set it locally when inspecting a run.
TRACE = bool(os.getenv("RIDGES_TRACE"))


def _brief(value, limit: int = 96) -> str:
    """A value at a glance: size first, then as much content as fits.

    Sizes matter more than contents when the question is where the tokens go,
    so a long string reports its length and a list reports its count before
    either shows anything.
    """
    if isinstance(value, str):
        flat = " ".join(value.split())
        return f"{len(value)}c" if len(flat) > limit else repr(flat)
    if isinstance(value, dict):
        return f"{{{len(value)}}}" + (f" {sorted(value)[:6]}" if value else "")
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        shown = ", ".join(_brief(item, 32) for item in items[:4])
        return f"[{len(items)}]" + (f" {shown}" + (" ..." if len(items) > 4 else "") if items else "")
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# Full call tracing: every function this module enters, with the arguments it
# was given, and every value it returns. RIDGES_TRACE=all turns it on.
#
# The curated trace() points below say what a stage decided; this says what
# every function did, which is what you want when the question is "where did
# this value come from" rather than "was this stage right". It is slow and
# loud, so it is a deliberate mode rather than a default.
TRACE_ALL = (os.getenv("RIDGES_TRACE") or "").lower() in ("all", "calls", "full")

# Called thousands of times each and carrying nothing worth reading one call at
# a time. Excluded by name so the output stays legible; the curated trace lines
# still report what they produced. `log`/`trace`/`_brief` must be here whatever
# else changes -- tracing the logger from inside the logger does not terminate.
TRACE_SKIP = frozenset({
    "log", "trace", "_brief", "_render", "_call_tracer",
    "read", "read_source", "write_source", "snapshot", "original",
    "query_density", "_looks_like_path", "_path_tokens", "_token_match",
    "truncate", "remaining_seconds", "score", "query_weight",
})


def _render(value, limit: int = 72) -> str:
    """Any value, short, and never raising -- a repr that throws would take the
    run down with it, which is not a trade a debugging aid gets to make."""
    try:
        if isinstance(value, (str, bytes)):
            body = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
            flat = " ".join(body.split())
            return repr(flat) if len(flat) <= limit else f"<{len(body)} chars>"
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            # Length first: a repr built only to be measured and thrown away is
            # the whole cost of this function under per-line tracing, and no
            # container of more than a dozen items fits in `limit` anyway.
            if len(value) > 12:
                return f"<{type(value).__name__} of {len(value)}>"
            shown = repr(value)
            return shown if len(shown) <= limit else f"<{type(value).__name__} of {len(value)}>"
        if isinstance(value, Path):
            return repr(str(value))
        if isinstance(value, (int, float, bool, type(None))):
            return repr(value)
        shown = repr(value)
        return shown if len(shown) <= limit else f"<{type(value).__name__}>"
    except Exception:
        return "<unreprable>"


def _state_of(subject) -> dict:
    """A shallow snapshot of an object's attributes, for diffing.

    Values are rendered immediately rather than held: a list captured by
    reference and mutated in place would compare equal to itself and the
    change -- the whole point of the snapshot -- would be invisible.
    """
    try:
        attributes = vars(subject)
    except TypeError:
        return {}                                    # no __dict__: nothing to diff
    snapshot: dict = {}
    for key, value in attributes.items():
        if key.startswith("__"):
            continue
        snapshot[key] = _render(value, 160)
        # One level deeper for objects the caller mutates through: the parser
        # writes to self.parsed.kinds, not to self.parsed, so a snapshot that
        # stopped here would render the same object twice and report that
        # nothing changed -- which is the opposite of what happened.
        try:
            nested = vars(value)
        except TypeError:
            continue
        for inner, held in nested.items():
            if not inner.startswith("__"):
                snapshot[f"{key}.{inner}"] = _render(held, 160)
    return snapshot


def install_call_tracer() -> None:
    """Log every call into this module: arguments in, locals computed, state
    changed, value returned.

    Most functions here mutate rather than return -- the parser's rules all
    return None and write to `self.parsed` -- so a tracer that reported only
    return values would say "None" twelve times and show nothing. What changed
    is the interesting part, so each frame is snapshotted on the way in and
    diffed on the way out.

    Consecutive repeats of one function collapse into a count: a loop calling
    one helper four hundred times is one line saying so.
    """
    frames: dict[int, dict] = {}
    state = {"depth": 0, "last": None, "repeats": 0}

    def flush() -> None:
        if state["repeats"] > 1:
            log(f"{'  ' * state['depth']}   ... {state['last']} x{state['repeats']}")
        state["last"], state["repeats"] = None, 0

    def hook(frame, event, arg):
        if event not in ("call", "return"):
            return
        if frame.f_globals.get("__name__") != __name__:
            return                                   # this module only, not the stdlib
        code = frame.f_code
        name = code.co_name
        if name in TRACE_SKIP or name.startswith("<"):
            return
        subject = frame.f_locals.get("self")

        if event == "call":
            if name == state["last"]:                # a repeat: count, do not print
                state["repeats"] += 1
                state["depth"] += 1
                return
            flush()
            state["last"], state["repeats"] = name, 1
            arguments = code.co_varnames[: code.co_argcount]
            shown = ", ".join(f"{n}={_render(frame.f_locals.get(n))}"
                              for n in arguments if n not in ("self", "cls"))
            log(f"{'  ' * state['depth']}-> {name}({shown})")
            frames[id(frame)] = {"before": _state_of(subject) if subject is not None else {},
                                 "args": set(arguments)}
            state["depth"] += 1
            return

        state["depth"] = max(0, state["depth"] - 1)
        entry = frames.pop(id(frame), None)
        if state["repeats"] > 1:                     # inside a collapsed run
            return
        pad = "  " * (state["depth"] + 1)
        if entry:
            # What the function computed: locals that are not its arguments.
            for key, value in frame.f_locals.items():
                if key in entry["args"] or key.startswith("_") or key in ("self", "cls"):
                    continue
                log(f"{pad}. {key} = {_render(value)}")
            # What it changed on the object it was called on.
            after = _state_of(subject) if subject is not None else {}
            for key, value in after.items():
                if entry["before"].get(key) != value:
                    log(f"{pad}~ self.{key}: {entry['before'].get(key, '<new>')} -> {value}")
        log(f"{'  ' * state['depth']}<- {name} = {_render(arg)}")

    sys.setprofile(hook)


# Line-level variable tracing: every assignment, as it happens, with the value
# before it, the value after it, and the statement that caused the change.
# RIDGES_TRACE=vars turns it on.
#
# Call tracing above reports a function's locals once, at return. That is one
# value per name per call: a variable assigned three times inside a loop shows
# only what it held at the end, and the statement that produced each value is
# not visible at all. When the question is "at which line did this stop being
# right", that is not enough.
#
# sys.settrace fires between statements rather than around calls, so each frame
# can be re-read after every line and the difference attributed to the line
# that made it. Nothing is inferred: a name appears here only because its
# rendered value actually changed.
#
# It re-reads every frame after every statement, so it is far slower and far
# louder than either mode above. It is for reading one task, never for a graded
# run.
TRACE_VARS = (os.getenv("RIDGES_TRACE") or "").lower() in ("vars", "lines", "values", "everything")


def install_variable_tracer() -> None:
    """Log every variable change in this module, statement by statement.

    Four things change inside a function and all four are reported:

      * locals -- a new name, or a name whose value is not what it was;
      * mutation in place -- values are rendered to text on capture, so a list
        appended to differs from itself a statement earlier, which a tracer
        holding the object by reference could never see;
      * attributes of `self`, one level deep, which is where this agent's
        parsers and probes actually keep their results;
      * module globals, checked on return, for the few the code mutates.

    Output is one line naming the statement and one line per change:

        L1102| self.parsed.kinds = [kind for _, kind in sorted(scores, reverse=True)]
            scores: [] -> [(3, 'result_correctness'), (2, 'index_or_plan')]
            self.parsed.kinds: [] -> ['result_correctness', 'index_or_plan']
    """
    import linecache

    # Module state the code mutates rather than rebinds -- MODELS gains live
    # prices from discover_models, for one. Diffed on return rather than per
    # line: rendering them costs more than a local does and they change rarely.
    watched = {name: value for name, value in globals().items()
               if isinstance(value, (dict, list, set)) and not name.startswith("_")}
    frames: dict[int, dict] = {}
    depth = {"n": 0}

    def locals_of(frame) -> dict:
        try:
            items = dict(frame.f_locals)
        except Exception:                            # a frame mid-teardown
            return {}
        return {key: _render(value, 200) for key, value in items.items()
                if not key.startswith("__")}

    def globals_now() -> dict:
        return {name: _render(value, 200) for name, value in watched.items()}

    def statement(frame, lineno: int) -> str:
        text = linecache.getline(frame.f_code.co_filename, lineno).strip()
        return text if len(text) <= 110 else text[:107] + "..."

    def report(frame, record, *, include_globals: bool = False) -> None:
        """What the statement just executed changed, or silence if nothing."""
        after = locals_of(frame)
        before = record["locals"]
        changes = [f"{key} = {value}" if key not in before else f"{key}: {before[key]} -> {value}"
                   for key, value in after.items() if before.get(key) != value]
        record["locals"] = after

        subject = record["self"]
        if subject is not None:
            state = _state_of(subject)
            changes += [f"self.{key}: {record['state'].get(key, '<new>')} -> {value}"
                        for key, value in state.items() if record["state"].get(key) != value]
            record["state"] = state
        if include_globals:
            module = globals_now()
            changes += [f"{key} (module global): {record['globals'].get(key, '<new>')} -> {value}"
                        for key, value in module.items() if record["globals"].get(key) != value]
            record["globals"] = module

        if not changes:
            return
        pad = "  " * (record["depth"] + 1)
        log(f"{pad}L{record['line']}| {statement(frame, record['line'])}")
        for change in changes:
            log(f"{pad}    {change}")

    def local_hook(frame, event, arg):
        record = frames.get(id(frame))
        if record is None:
            return None
        if event == "line":
            report(frame, record)
            record["line"] = frame.f_lineno          # attribute the next diff to this line
        elif event == "exception":
            kind, value, _ = arg
            log(f"{'  ' * (record['depth'] + 1)}!! line {frame.f_lineno}: "
                f"{getattr(kind, '__name__', kind)}: {_render(value)}")
        elif event == "return":
            report(frame, record, include_globals=True)
            frames.pop(id(frame), None)
            depth["n"] = record["depth"]
            log(f"{'  ' * depth['n']}<- {frame.f_code.co_name} = {_render(arg)}")
        return local_hook

    def hook(frame, event, arg):
        if event != "call":
            return None
        if frame.f_globals.get("__name__") != __name__:
            return None                              # this module only, not the stdlib
        code = frame.f_code
        name = code.co_name
        if name in TRACE_SKIP or name.startswith("<"):
            return None
        arguments = code.co_varnames[: code.co_argcount]
        shown = ", ".join(f"{n}={_render(frame.f_locals.get(n))}"
                          for n in arguments if n not in ("self", "cls"))
        log(f"{'  ' * depth['n']}-> {name}({shown})")
        subject = frame.f_locals.get("self")
        frames[id(frame)] = {
            "depth": depth["n"],
            "line": frame.f_lineno,
            "self": subject,
            "locals": locals_of(frame),
            "state": _state_of(subject) if subject is not None else {},
            "globals": globals_now(),
        }
        depth["n"] += 1
        return local_hook

    sys.settrace(hook)


def trace(stage: str, direction: str = "", **fields) -> None:
    """One line at a module boundary. `direction` is "in", "out" or "".

    Deliberately flat key=value rather than a nested dump: the point is to scan
    a whole run for the stage where a number stops making sense.
    """
    if not TRACE:
        return
    rendered = "  ".join(f"{key}={_brief(value)}" for key, value in fields.items())
    log(f"  {direction:<3} {stage:<24} {rendered}")


def remaining_seconds() -> float:
    try:
        budget = float(os.getenv("AGENT_TIMEOUT") or DEFAULT_AGENT_TIMEOUT)
    except ValueError:
        budget = DEFAULT_AGENT_TIMEOUT
    return budget - TIMEOUT_SAFETY_MARGIN - (time.monotonic() - AGENT_START)


PROJECT_MARKERS = ("manage.py", "pyproject.toml", "setup.py", "requirements.txt", "package.json",
                   "go.mod", "Gemfile", "pom.xml", "build.gradle", "composer.json", "Cargo.toml",
                   "mix.exs", "Makefile", "schema", "src")


def _looks_like_project(path: Path) -> bool:
    try:
        return any((path / marker).exists() for marker in PROJECT_MARKERS)
    except OSError:
        return False


def workdir() -> Path:
    """The application repository root inside the task container.

    The explicit override wins; then the working directory, but only when it
    holds a project (a runtime started from `/` must not index the whole
    filesystem); then the conventional mount points.
    """
    override = os.getenv("RIDGES_WORKDIR")
    if override and Path(override).is_dir():
        return Path(override).resolve()
    cwd = Path(os.getcwd())
    if cwd.is_dir() and cwd.resolve() not in (Path("/"), Path("/installed-agent")) and _looks_like_project(cwd):
        return cwd.resolve()
    for candidate in ("/app", "/repo", "/workspace", "/src"):
        path = Path(candidate)
        if path.is_dir() and _looks_like_project(path):
            trace("workdir", "out", root=str(path.resolve()), source="conventional mount")
            return path.resolve()
    chosen = cwd.resolve() if cwd.is_dir() and cwd.resolve() != Path("/") else Path("/app")
    trace("workdir", "out", root=str(chosen), source="fallback")
    return chosen


def app_python(root: Path) -> str:
    """The interpreter the application itself runs under.

    `sys.executable` is the agent's Python. When the application lives in a
    virtualenv, asking Django for its settings with the wrong interpreter fails
    silently on the first import -- so look for the app's own first.
    """
    manage = root / "manage.py"
    if manage.is_file():
        try:
            first = manage.read_text(errors="replace").splitlines()[:1]
        except OSError:
            first = []
        if first and first[0].startswith("#!"):
            exe = first[0][2:].split()[-1]
            if exe.startswith("/") and Path(exe).exists() and "env" not in Path(exe).name:
                return exe
    for candidate in (root / ".venv/bin/python", root / "venv/bin/python",
                      Path("/opt/venv/bin/python"), Path("/app/.venv/bin/python")):
        if candidate.exists():
            trace("app_python", "out", interpreter=str(candidate), source="virtualenv")
            return str(candidate)
    trace("app_python", "out", interpreter=sys.executable, source="the agent's own")
    return sys.executable


def run_command(
    command: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    timeout: float = 300.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, capturing merged output, never raising on failure."""
    shell = isinstance(command, str)
    merged = os.environ.copy()
    merged["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged.update(env)
    shown = command if shell else " ".join(str(part) for part in command)
    trace("run_command", "in", cmd=shown, timeout=timeout)
    started = time.monotonic()
    try:
        done = subprocess.run(
            command,
            shell=shell,
            cwd=str(cwd or workdir()),
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        done = subprocess.CompletedProcess(command, 124, output + f"\n[timeout after {timeout}s]")
    except (OSError, ValueError) as exc:
        done = subprocess.CompletedProcess(command, 127, f"[could not run: {exc}]")
    # Subprocesses are where the wall clock goes -- a test run is minutes while
    # every other stage is milliseconds -- so each one reports what it spent.
    trace("run_command", "out", cmd=shown, rc=done.returncode,
          seconds=time.monotonic() - started, output=len(done.stdout or ""))
    return done


def truncate(text: str, limit: int, *, head_ratio: float = 0.4) -> str:
    """Keep the head and tail of long output -- errors live at both ends."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * head_ratio)
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters elided] ...\n{text[-tail:]}"


# ---------------------------------------------------------------------------
# Inference transport
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class LLM:
    """Inference client that works in production and in `ridges miner run-local`.

    Three transports, tried in order of availability:

      * the sandbox proxy's /api/inference (older runtime, still deployed);
      * OpenRouter directly -- in production a transparent MITM proxy
        intercepts openrouter.ai and enforces the model allowlist and budget;
      * a provider base URL from the local-testing env vars the miner CLI sets.

    Only the standard library is used: an arbitrary task container is not
    guaranteed to have `requests`, let alone the `openrouter` SDK.
    """

    def __init__(self) -> None:
        self.run_id = os.getenv("EVALUATION_RUN_ID") or os.getenv("RUN_ID") or ""
        self.sandbox_proxy = (os.getenv("SANDBOX_PROXY_URL") or "").rstrip("/")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY") or ""
        self.local_key = os.getenv("RIDGES_INFERENCE_API_KEY") or ""
        self.local_base = (os.getenv("RIDGES_INFERENCE_BASE_URL") or "").rstrip("/")
        try:
            self.max_cost = float(os.getenv("RIDGES_MAX_COST_USD") or DEFAULT_MAX_COST_USD)
        except ValueError:
            self.max_cost = DEFAULT_MAX_COST_USD
        self.spent_estimate = 0.0
        self.calls = 0
        self.unsupported: set[str] = set()
        self._usage_unavailable = False
        self.blocked: set[str] = set()   # permanently refused for this key
        # Models that spent a whole completion cap on reasoning and said nothing.
        # For the rest of the run they answer with reasoning switched off: on
        # this prompt family their thinking does not terminate, and we pay for
        # every token of it.
        self.no_reasoning: set[str] = set()
        self._second_pass = False
        self.discovered: list[str] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0          # prompt tokens the provider served from its prefix cache
        self.reasoning_tokens = 0       # completion tokens spent thinking, not answering
        self.per_model: dict[str, float] = {}
        self._insecure_ctx = ssl.create_default_context()
        self._insecure_ctx.check_hostname = False
        self._insecure_ctx.verify_mode = ssl.CERT_NONE
        # OpenAI-style routes in order of preference. The documented production
        # contract is `{SANDBOX_PROXY_URL}/api/v1/chat/completions` with no
        # open internet; the direct URL is what local runs reach through the
        # transparent proxy. A route that cannot be reached is dropped for the
        # run, so a wrong first guess costs one failed connection, not the task.
        self.dead_routes: set[str] = set()
        self.effort = "low"             # reasoning effort sent with every call

    def routes(self) -> list[str]:
        found: list[str] = []
        if self.sandbox_proxy and self.openrouter_key:
            found.append(f"{self.sandbox_proxy}/api/v1/chat/completions")
        if self.openrouter_key:
            found.append(OPENROUTER_URL)
        if self.local_base and self.local_key:
            found.append(f"{self.local_base}/chat/completions")
        return found

    # -- roster discovery -------------------------------------------------
    def discover_models(self) -> list[str]:
        """Ask the platform which models it will actually accept.

        The allowed list is operational config, not a repo constant -- the docs
        point at Discord for it, and it changes. Asking beats assuming: whatever
        comes back is authoritative for this run, and MODELS is only the
        fallback for when nothing answers.
        """
        endpoints = []
        if self.sandbox_proxy:
            endpoints += [f"{self.sandbox_proxy}/api/inference-models",
                          f"{self.sandbox_proxy}/api/v1/models"]
        endpoints += ["http://sandbox-proxy:80/api/inference-models",
                      "http://sandbox-proxy:80/api/v1/models"]

        for url in endpoints:
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(request, timeout=8) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                continue
            rows = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(rows, list) or not rows:
                continue
            names: list[str] = []
            for row in rows:
                if isinstance(row, str):
                    names.append(row)
                elif isinstance(row, dict):
                    # /api/inference-models uses `name`; OpenAI-shaped lists use `id`.
                    name = row.get("name") or row.get("id") or row.get("external_name")
                    if name:
                        names.append(name)
                        spec = MODELS.get(name)
                        cin = row.get("cost_usd_per_million_input_tokens")
                        cout = row.get("cost_usd_per_million_output_tokens")
                        if cin is not None and cout is not None:
                            MODELS[name] = ModelSpec(name, float(cin), float(cout),
                                                     int(row.get("max_input_tokens")
                                                         or (spec.context if spec else 128000)))
            if names:
                log(f"discovered {len(names)} allowed model(s) from {url}")
                trace("LLM.discover_models", "out", source=url, allowed=names)
                self.discovered = names
                return names
        trace("LLM.discover_models", "out", allowed=[],
              note="nothing answered; the built-in roster stands")
        return []

    def roster(self, preferred: Sequence[str]) -> list[str]:
        """Preferred models first, then anything else the platform allows."""
        allowed = self.discovered
        if allowed:
            ordered = [m for m in preferred if m in allowed]
            ordered += sorted(m for m in allowed if m not in ordered)      # fixed order for the tail
            return ordered or sorted(allowed)
        return list(preferred) + [name for name in MODELS if name not in preferred]

    # -- budget ----------------------------------------------------------
    def spent(self) -> float:
        """Authoritative spend from the proxy, falling back to our estimate."""
        reported = self._usage()
        if reported is not None:
            return reported
        return self.spent_estimate

    def _usage(self) -> float | None:
        if self._usage_unavailable:
            return None
        if not self.sandbox_proxy:
            # The proxy's usage endpoint only exists behind the sandbox proxy.
            # Elsewhere the lookup blocks on DNS for seconds per call; the
            # provider-reported figure is exact anyway.
            self._usage_unavailable = True
            return None
        try:
            request = urllib.request.Request(USAGE_URL, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            value = payload.get("total_cost_usd")
            return float(value) if value is not None else None
        except Exception:
            self._usage_unavailable = True
            return None

    def headroom(self) -> float:
        return max(0.0, self.max_cost - self.spent())

    # -- transport -------------------------------------------------------
    def _post(self, url: str, payload: dict, headers: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except ssl.SSLError:
                # The production proxy MITMs openrouter.ai; if its CA is not in the
                # container trust store, verification fails on an otherwise healthy
                # request.  Retry without verification -- the hop is loopback-local.
                with urllib.request.urlopen(request, timeout=timeout,
                                            context=self._insecure_ctx) as response:
                    return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise InferenceError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise InferenceError(f"transport failure: {exc.reason}") from exc
        except (OSError, ValueError) as exc:
            # Connection reset, remote disconnect, socket timeout, or a body
            # that is not JSON. All of these are one bad round trip, not a
            # reason to abandon the task: report them as retryable.
            raise InferenceError(f"transport failure: {exc.__class__.__name__}: {exc}") from exc

    def _call_via_routes(self, model: str, messages: list[dict], temperature: float,
                         timeout: float, cap: int, reasoning: bool) -> str:
        """Try each OpenAI-style route in order; a route that cannot be reached
        at all (connection refused, unknown host, 404 on the path) is retired
        for the run. Model-level errors propagate unchanged so the caller's
        per-model handling still applies."""
        live = [r for r in self.routes() if r not in self.dead_routes]
        last: InferenceError | None = None
        for url in live:
            key = self.local_key if url.startswith(self.local_base or "\0") else self.openrouter_key
            try:
                return self._call_openai_style(url, key, model, messages, temperature, timeout, cap,
                                               reasoning=reasoning)
            except InferenceError as exc:
                text = str(exc)
                unreachable = ("transport failure" in text
                               or ("HTTP 404" in text and "model" not in text.lower())
                               or "HTTP 502" in text or "HTTP 503" in text)
                if unreachable and len(live) > 1:
                    self.dead_routes.add(url)
                    log(f"inference route unreachable, retired for this run: {url} ({truncate(text, 120)})")
                    last = exc
                    continue
                raise
        if self.sandbox_proxy:
            # Legacy sandbox schema: no max_tokens or reasoning control, but an answer.
            return self._call_sandbox_proxy(model, messages, temperature, timeout)
        raise last or InferenceError("no inference transport configured")

    def _call_sandbox_proxy(self, model: str, messages: list[dict], temperature: float, timeout: float) -> str:
        payload = {
            "run_id": self.run_id,
            "evaluation_run_id": self.run_id,
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        data = self._post(
            f"{self.sandbox_proxy}/api/inference",
            payload,
            {"Content-Type": "application/json"},
            timeout,
        )
        if isinstance(data, str):
            return data
        return data.get("content") or ""

    def _call_openai_style(self, url: str, key: str, model: str, messages: list[dict],
                           temperature: float, timeout: float,
                           max_tokens: int | None = None, reasoning: bool = True) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            # Reasoning tokens bill as completion. This agent wants a short JSON
            # edit, not an essay; providers that ignore the field are unaffected.
            # Measured on deepseek-v4-pro: effort=low still thinks 1k tokens for
            # a one-line question, and on some prompts never stops. enabled=false
            # yields 0 reasoning tokens and the answer in the content stream.
            "reasoning": {"effort": self.effort} if reasoning else {"enabled": False},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        data = self._post(
            url,
            payload,
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            timeout,
        )
        self._record_usage(model, data.get("usage") or {})
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"malformed completion: {truncate(json.dumps(data), 300)}") from exc
        if not content.strip() and choice.get("finish_reason") == "length":
            # The cap cut the reply off before any content survived; say so
            # clearly so the caller can retry with more room.
            raise InferenceError("completion truncated by max_tokens before any content")
        return content

    def _record_usage(self, model: str, usage: dict) -> None:
        prompt = float(usage.get("prompt_tokens") or 0)
        completion = float(usage.get("completion_tokens") or 0)
        self.prompt_tokens += int(prompt)
        self.completion_tokens += int(completion)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens += int(details.get("cached_tokens") or 0)
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        self.reasoning_tokens += int(reasoning)

        reported = usage.get("cost")
        if reported is not None:
            # The provider's own figure: exact, and free of a stale price table.
            cost = float(reported)
        else:
            spec = MODELS.get(model)
            if not spec:
                return
            cost = (prompt / 1e6) * spec.usd_per_m_in + (completion / 1e6) * spec.usd_per_m_out

        self.spent_estimate += cost
        self.per_model[model] = self.per_model.get(model, 0.0) + cost

    def ledger(self) -> str:
        """Where the tokens went, and how much of it was avoidable.

        Three numbers decide whether a run was expensive for a good reason.
        Cached prompt tokens are prompt this agent re-sent and the provider
        served from its prefix cache -- cheap, and evidence the message stack
        is stable. Reasoning tokens bill as completion and are never read by
        anything: they are pure loss. Prompt tokens per call is what falls
        when the prompt gets shorter, and it is multiplied by every call the
        loop makes, so a long first prompt is paid again on every retry.
        """
        if not self.calls:
            return "no inference calls"
        billed = self.prompt_tokens - self.cached_tokens
        rows = [
            f"calls              {self.calls}",
            f"prompt tokens      {self.prompt_tokens:>8,}   {self.prompt_tokens // self.calls:>7,}/call",
            f"  served by cache  {self.cached_tokens:>8,}   {self.cached_tokens / max(1, self.prompt_tokens):>7.0%}",
            f"  billed in full   {billed:>8,}",
            f"completion tokens  {self.completion_tokens:>8,}   {self.completion_tokens // self.calls:>7,}/call",
            f"  spent reasoning  {self.reasoning_tokens:>8,}   "
            f"{self.reasoning_tokens / max(1, self.completion_tokens):>7.0%} of completion, never read",
        ]
        return "token ledger\n  " + "\n  ".join(rows)

    def report(self) -> str:
        """A human-readable account of what this run cost."""
        rows = [f"  {model:32} ${cost:.5f}" for model, cost in
                sorted(self.per_model.items(), key=lambda item: -item[1])]
        source = "proxy" if self._usage() is not None else "provider-reported"
        return (
            f"cost ${self.spent():.5f} ({source}) over {self.calls} call(s), "
            f"{self.prompt_tokens} prompt ({self.cached_tokens} cached) + "
            f"{self.completion_tokens} completion ({self.reasoning_tokens} reasoning) tokens"
            + ("\n" + "\n".join(rows) if rows else "")
        )

    def complete(
        self,
        candidates: Sequence[str],
        messages: list[dict],
        *,
        temperature: float = 0.0,
        timeout: float = 240.0,
    ) -> tuple[str, str]:
        """Return (content, model_used), trying each candidate slug in turn."""
        if self.headroom() <= 0.005:
            raise BudgetExhausted(f"cost cap reached (~${self.spent():.4f} of ${self.max_cost:.2f})")

        ordered = list(candidates) if FORCE_MODEL else self.roster(candidates)
        usable = [name for name in ordered if name not in self.unsupported]
        if not usable:
            raise BudgetExhausted(
                "no allowed model is usable with this key: every candidate returned "
                "404 (not allowed) or 402 (insufficient credit)"
            )

        last_error: Exception | None = None
        # Models that have already spun on this prompt go last, and answer with
        # reasoning off. Measured on prefix-hierarchy: runs where the edit was
        # written with reasoning on passed 2/2; runs that fell back to the same
        # model with reasoning off passed 1/6. So a runaway hands the turn to
        # the next family with its reasoning intact, and only when every family
        # has spun do we take an answer without reasoning.
        usable.sort(key=lambda m: m in self.no_reasoning)          # stable: keeps ladder order otherwise
        skipped_for_budget = False
        for model in usable:
            if model in self.unsupported:
                continue
            transient = 0          # network / 5xx style retries, at most 2
            while transient < 2:
                try:
                    cap = self.affordable_cap(model, messages, COMPLETION_CAP)
                    if cap is None:
                        log(f"{model}: even a {MIN_COMPLETION_CAP}-token reply would exceed "
                            f"the remaining budget (~${self.headroom():.3f}); skipping")
                        skipped_for_budget = True
                        break
                    reasoning = model not in self.no_reasoning
                    outgoing = messages
                    if not reasoning:
                        # This model spent a whole cap thinking earlier. Say so;
                        # the answer now has to come out in the content stream.
                        outgoing = messages + [{"role": "user", "content": NUDGE}]
                    if remaining_seconds() < timeout + 60:
                        # A call that cannot finish before the deadline is a
                        # call whose answer nobody will read. Stop cleanly.
                        raise BudgetExhausted("wall-clock budget cannot cover another completion")
                    self.calls += 1
                    trace("LLM.complete", "in", call=self.calls, model=model,
                          messages=len(outgoing), reasoning=reasoning, cap=cap,
                          prompt_chars=sum(len(str(m.get("content", ""))) for m in outgoing),
                          spent=self.spent_estimate)
                    before = (self.prompt_tokens, self.completion_tokens,
                              self.cached_tokens, self.reasoning_tokens, self.spent_estimate)
                    content = ""
                    try:
                        content = self._call_via_routes(model, outgoing, temperature,
                                                        timeout, cap, reasoning)
                    finally:
                        # In a finally, because usage is recorded before the
                        # error is raised: a runaway that spends 12k completion
                        # tokens and returns nothing is billed in full and
                        # would otherwise leave no trace at all.
                        trace("LLM.complete", "out", call=self.calls, model=model,
                              content_chars=len(content or ""),
                              prompt=self.prompt_tokens - before[0],
                              completion=self.completion_tokens - before[1],
                              cached=self.cached_tokens - before[2],
                              # Reasoning bills as completion and is never read:
                              # the single largest avoidable cost in this agent.
                              reasoning_tokens=self.reasoning_tokens - before[3],
                              usd=self.spent_estimate - before[4])
                    if content and content.strip():
                        log(f"inference ok: {model} ({len(content)} chars, ~${self.spent_estimate:.4f})")
                        return content, model
                    last_error = InferenceError("empty completion")
                except InferenceError as exc:
                    last_error = exc
                    text = str(exc)
                    lowered = text.lower()
                    if "truncated by max_tokens" in lowered or "empty completion" in lowered:
                        if reasoning:
                            self.no_reasoning.add(model)
                            log(f"{model} spent the whole cap reasoning; trying the next model "
                                f"family with reasoning on, this one answers without it from now on")
                        else:
                            self.unsupported.add(model)
                            self.blocked.add(model)
                            log(f"{model} returns no content even without reasoning; moving on")
                        break
                    if "403" in text or "access denied" in lowered:
                        # Refused by policy for this model. Retrying the same
                        # request is what got denied; move on at once.
                        self.unsupported.add(model)
                        self.blocked.add(model)
                        log(f"model refused by policy on this route: {model}")
                        break
                    if "404" in text or "not supported" in lowered or "no allowed providers" in lowered:
                        # Permanent for this key: the model is not on the
                        # gateway allowlist, or no permitted provider serves it.
                        self.unsupported.add(model)
                        self.blocked.add(model)
                        break  # try the next slug, not the same one again
                    if "402" in text or "more credits" in lowered or "insufficient" in lowered:
                        # Out of credit for this model. No amount of retrying
                        # fixes that; strike it off and move on immediately.
                        self.unsupported.add(model)
                        self.blocked.add(model)
                        log(f"model unaffordable on this key: {model}")
                        break
                    if "429" in text and "cost" in lowered:
                        raise BudgetExhausted(text) from exc
                    transient += 1
                    log(f"inference retry ({model}, attempt {transient}): {truncate(text, 200)}")
                    time.sleep(2 + 3 * (transient - 1))
        # Every candidate spun with reasoning on and none has answered yet: go
        # round once more, now without reasoning (they are all in no_reasoning).
        spun = [m for m in usable if m in self.no_reasoning and m not in self.unsupported]
        ran_dry = last_error is not None and ("empty" in str(last_error).lower()
                                              or "truncated by max_tokens" in str(last_error).lower())
        if spun and ran_dry and not self._second_pass:
            self._second_pass = True
            try:
                return self.complete(candidates, messages, temperature=temperature, timeout=timeout)
            finally:
                self._second_pass = False
        if skipped_for_budget and last_error is None:
            raise BudgetExhausted(f"remaining budget (~${self.headroom():.3f}) cannot cover "
                                  f"another completion")
        raise InferenceError(f"all models failed; last error: {last_error}")

    def affordable_cap(self, model: str, messages: list[dict], cap: int) -> int | None:
        """Largest completion cap this call can take without breaching the budget.

        The worst case for a reasoning model is a reply that uses every token
        of the cap and says nothing. Price that case before sending: shrink the
        cap to what the remaining budget covers, or refuse the call outright
        when not even MIN_COMPLETION_CAP fits. Better one small honest reply
        than a 429 from the gateway with no patch behind it.
        """
        # An unknown slug is priced like the dearest model we know of, not
        # treated as free: the budget guard exists for exactly that case.
        spec = MODELS.get(model) or ModelSpec(model, 2.0, 8.0, 128000)
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) / 3.5
        prompt_cost = prompt_tokens / 1e6 * spec.usd_per_m_in * PRICE_SAFETY
        room = self.headroom() * 0.95 - prompt_cost
        if room <= 0:
            return None
        max_tokens = int(room / (spec.usd_per_m_out * PRICE_SAFETY) * 1e6)
        if max_tokens < MIN_COMPLETION_CAP:
            trace("LLM.affordable_cap", "out", model=model, cap=None,
                  headroom=self.headroom(), reason="even the smallest reply exceeds the budget")
            return None
        if max_tokens < cap:
            log(f"{model}: shrinking completion cap {cap} -> {max_tokens} to stay in budget")
            trace("LLM.affordable_cap", "out", model=model, cap=max_tokens, asked=cap,
                  headroom=self.headroom(), reason="shrunk to fit the budget")
            return max_tokens
        trace("LLM.affordable_cap", "out", model=model, cap=cap, headroom=self.headroom())
        return cap


# ---------------------------------------------------------------------------
# Instruction parsing
# ---------------------------------------------------------------------------

# Problem taxonomy.  The class does not select a separate solver -- it selects
# the evidence we gather and the constraints we put in front of the model.
PROBLEM_PATTERNS: dict[str, tuple[str, ...]] = {
    "bounded_queries": (
        r"\bN\+1\b", r"bounded number", r"grows with", r"scale[sd]? with",
        r"bulk[_ ]?(create|update|insert)", r"per[- ]row", r"in a loop",
        r"round[- ]trip", r"query count", r"number of (SQL |)queries",
    ),
    "index_or_plan": (
        r"\bindex\b", r"\bindexes\b", r"\bEXPLAIN\b", r"\bbuffers\b",
        r"sequential scan", r"seq scan", r"full scan", r"\bslow\b",
        r"\btimeout\b", r"query plan", r"selective", r"partial index",
    ),
    "result_correctness": (
        r"wrong (count|result|value|number)", r"double[- ]count", r"incorrect",
        r"off by", r"duplicate rows", r"missing rows", r"should return",
        r"percentage", r"utilization", r"aggregat", r"\bcounts?\b",
    ),
    "authoring": (
        r"\bauthor\b", r"\bwrite\b a query", r"\bimplement\b", r"\badd\b an? annotation",
        r"annotate", r"currently (returns|raises|is) (not|un)implemented",
    ),
    "orm_layer": (
        r"\bORM\b", r"Django", r"SQLAlchemy", r"queryset", r"QuerySet",
        r"select_related", r"prefetch_related", r"ActiveRecord", r"Ecto",
        r"GORM", r"Prisma", r"query builder", r"manager method",
    ),
    "raw_sql": (
        r"\bRawSQL\b", r"raw SQL", r"\.raw\(", r"\bSELECT\b", r"\bJOIN\b",
        r"\bCTE\b", r"WITH RECURSIVE", r"window function", r"\.sql\b",
    ),
    "migration": (
        r"\bmigration\b", r"ALTER TABLE", r"schema change", r"AddIndex",
        r"RunSQL", r"alembic", r"\bDDL\b",
    ),
    "clickhouse": (
        r"ClickHouse", r"MergeTree", r"ReplacingMergeTree", r"materiali[sz]ed view",
        r"\bPREWHERE\b", r"ORDER BY key", r"\bsharding key\b", r"Distributed\(",
        r"\bpartition(ing|)\b key",
    ),
}

ENGINE_PATTERNS = {
    "clickhouse": (r"clickhouse", r"mergetree", r"prewhere", r"clickhouse_driver", r"chdb"),
    "postgresql": (r"postgres", r"postgresql", r"psycopg", r"pg_", r"\bpsql\b", r"::regclass"),
}


@dataclass
class Instruction:
    """Everything we can learn from the problem statement without an LLM."""

    text: str
    kinds: list[str] = field(default_factory=list)
    engine: str = "unknown"
    named_paths: list[str] = field(default_factory=list)
    edit_only: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    lint_paths: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    method_hint: str | None = None
    class_hint: str | None = None
    single_method: bool = False
    style_constraints: list[str] = field(default_factory=list)
    traced_hint: list[str] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)   # ranker scores, best first
    # Numeric targets the grader states outright: {"max_queries": 3, "max_read_rows": 50000,
    # "create_paths": [...]}. Absolute bounds beat relative ones when the task gives them.
    targets: dict = field(default_factory=dict)

    @property
    def primary_kind(self) -> str:
        return self.kinds[0] if self.kinds else "general"


def _fenced_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)]


def _split_shell_commands(block: str) -> list[str]:
    """Split a fenced shell block into commands, honouring backslash joins."""
    joined = re.sub(r"\\\n\s*", " ", block)
    commands = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(line)
    return commands


def _looks_like_path(token: str) -> bool:
    if not token or len(token) > 200 or " " in token:
        return False
    if token.startswith(("http://", "https://")):
        return False
    suffix = re.search(r"(\.[A-Za-z0-9]{1,5})$", token)
    if not suffix:
        return False
    # A slash makes it a path outright; without one, only a recognised source
    # extension counts, so prose like "e.g." is not mistaken for a filename.
    return "/" in token or suffix.group(1).lower() in SOURCE_SUFFIXES


def normalize_repo_path(raw: str) -> str | None:
    """Return a clean repo-relative path, or None if it escapes the repo."""
    if not raw:
        return None
    candidate = raw.strip().strip("`").replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    path = Path(candidate)
    if path.is_absolute() or ".." in path.parts or not candidate:
        return None
    return path.as_posix()


class InstructionParser:
    """Extracts everything the statement says, one rule per method.

    This started as a single function and grew a rule at a time until it was
    two hundred lines of interleaved regex, which is where every parsing bug in
    this agent has come from. Two of them were the same bug: a multi-word
    phrase that a hard wrap split in half, found once in the style clause and
    again, later, in the prohibitions -- because the rule "match phrases
    against the flattened text" lived in a comment rather than in the code.

    Here the three views of the statement are attributes with a stated purpose,
    so choosing the wrong one is a visible mistake rather than an invisible
    one, and each rule is small enough to read and test on its own:

      text   the statement exactly as written. Only for what depends on
             layout -- fenced blocks, paragraph breaks, backtick spans that
             must not cross a line.
      flat   whitespace collapsed. Every multi-word phrase matches against
             this, because a line break can fall anywhere inside one.
      prose  fenced blocks removed. For paths and identifiers, so a path
             inside `python manage.py test` is read as an invocation rather
             than as something the task wants edited.

    Rules run in a fixed order and some depend on earlier ones -- lint paths
    come out of the parsed commands, and the path filter runs last because it
    needs every path any rule found.
    """

    def __init__(self, text: str, root: Path) -> None:
        self.text = text
        self.flat = re.sub(r"\s+", " ", text)
        self.prose = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        self.lowered = text.lower()
        self.root = root
        self.parsed = Instruction(text=text)

    def parse(self) -> Instruction:
        trace("parse_instruction", "in", statement=len(self.text), root=str(self.root))
        for rule in (self._classify, self._detect_engine, self._collect_paths,
                     self._read_permissions, self._read_prohibitions, self._read_method_bound,
                     self._read_style_rules, self._read_commands, self._read_lint_paths,
                     self._collect_identifiers, self._read_numeric_targets, self._resolve_paths):
            rule()
        parsed = self.parsed
        trace("parse_instruction", "out", kinds=parsed.kinds, engine=parsed.engine,
              edit_only=parsed.edit_only, lint_paths=parsed.lint_paths,
              named=parsed.named_paths, method=parsed.method_hint, cls=parsed.class_hint,
              single_method=parsed.single_method, style=parsed.style_constraints,
              forbidden=len(parsed.forbidden), commands=parsed.commands,
              targets=parsed.targets, identifiers=parsed.identifiers[:6])
        return parsed

    # -- what kind of problem this is ------------------------------------
    def _classify(self) -> None:
        """Rank the problem shapes the statement matches.

        Selects the evidence gathered and the guidance shown; never sent to
        the model as a label, which is what it used to be.

        Against `flat` like every other phrase rule: half of PROBLEM_PATTERNS
        are two-word phrases ("bounded number", "query count", "sequential
        scan"), so the wrap that cost a style constraint and four prohibition
        clauses could silently drop a whole classification here too. Measured
        when this moved: identical on all 62 statements in the corpus, so the
        bug was latent rather than live -- but a dropped `bounded_queries`
        would cost the `measure` probe the prompt asks for.
        """
        scores: list[tuple[int, str]] = []
        for kind, patterns in PROBLEM_PATTERNS.items():
            hits = sum(1 for pattern in patterns if re.search(pattern, self.flat, re.IGNORECASE))
            if hits:
                scores.append((hits, kind))
        self.parsed.kinds = [kind for _, kind in sorted(scores, reverse=True)]

    def _detect_engine(self) -> None:
        """The engine the prose names, if it names one at all.

        Usually it does not -- a netbox statement never says "PostgreSQL". The
        live database settles it later; this is only the head start.
        """
        scores = {
            engine: sum(len(re.findall(pattern, self.lowered)) for pattern in patterns)
            for engine, patterns in ENGINE_PATTERNS.items()
        }
        best = max(scores, key=lambda key: scores[key])
        if scores[best]:
            self.parsed.engine = best

    # -- what the statement points at ------------------------------------
    def _collect_paths(self) -> None:
        """Every path the prose mentions, in the order it mentions them.

        Evidence for the ranker, never authority to edit: check_protected
        keys on an explicit permission precisely because a path can be
        mentioned by a prohibition or by a command invocation.
        """
        seen: set[str] = set()
        tokens = (re.findall(r"`([^`\n]+)`", self.prose)
                  + re.findall(r"(?<![\w`/])([\w./-]+/[\w./-]+\.\w{1,5})", self.prose))
        for token in tokens:
            token = token.strip().strip(",.;:")
            if token in seen or not _looks_like_path(token):
                continue
            seen.add(token)
            self.parsed.named_paths.append(token)

    _PERMISSION = re.compile(
        r"(?:you\s+may\s+(?:only\s+)?(?:edit|modify|change)(?:\s+only)?"
        r"|(?:edit|modify|change)\s+only"
        r"|limit\s+(?:production\s+|source\s+|)changes\s+to"
        r"|restrict\s+(?:your\s+|)(?:changes|edits)\s+to"
        r"|the\s+only\s+file\s+you\s+may\s+(?:edit|change|modify)"
        r"|confine\s+(?:your\s+|)(?:changes|edits)\s+to)",
        re.IGNORECASE,
    )

    def _read_permissions(self) -> None:
        """Files the statement actually permits us to edit.

        Scans `text`, not `flat`: the window has to end where the sentence
        does, and a 240-character window that ran into the next sentence used
        to swallow a "do not change tests/x.py" clause and read it as a
        permission.
        """
        for match in self._PERMISSION.finditer(self.text):
            window = re.split(r"\.\s|\n\s*\n",
                              self.text[match.end() : match.end() + 240], maxsplit=1)[0]
            for token in re.findall(r"([\w][\w./-]*\.\w{1,5})", window):
                if _looks_like_path(token) and token not in self.parsed.edit_only:
                    self.parsed.edit_only.append(token)

    def _read_prohibitions(self) -> None:
        """"Do not change models, fields, ..." clauses, whole.

        Against `flat`: the phrase wraps between "Do" and "not" on three of
        the six netbox samples, which cost four clauses before this moved.
        """
        for match in re.finditer(r"Do\s+not\s+(?:change|modify|edit|add|touch)\s+([^.]{0,300})\.",
                                 self.flat, re.IGNORECASE):
            self.parsed.forbidden.append(" ".join(match.group(1).split()))

    _METHOD_BOUND = re.compile(
        r"(?:change|modify|edit)\s+only\s+that\s+(?:method|function)"
        r"|only\s+that\s+(?:method|function)"
        r"|keep\s+(?:its|the)\s+signature"
        r"|(?:the\s+)?rest\s+of\s+(?:its|the)\s+file\s+unchanged"
        r"|bounded\s+to\s+one\s+method"
        r"|specifically\s+`[\w.]+\(\)`",
        re.IGNORECASE)
    _METHOD_NAME = re.compile(
        r"`(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)\(\)`"
        r"|(?:method|function)\s+`(?:(?P<cls2>[A-Za-z_]\w*)\.)?(?P<name2>[A-Za-z_]\w*)`")

    def _read_method_bound(self) -> None:
        """Whether the change is bounded to one method, and which one.

        A method hint is only trustworthy written as code -- backticked, or
        spelled with parentheses. Bare prose ("the method that assigns tags")
        names no symbol, and guessing one sends the slicer to the wrong place.
        A statement mentions many symbols, so the one being scoped is the one
        introduced as such or written with its class.
        """
        self.parsed.single_method = bool(self._METHOD_BOUND.search(self.flat))
        best_rank = -1
        for match in self._METHOD_NAME.finditer(self.text):
            name = match.group("name") or match.group("name2")
            qualifier = match.group("cls") or match.group("cls2")
            if not name:
                continue
            preceding = self.text[max(0, match.start() - 60) : match.start()].lower()
            rank = 2 if qualifier else 0
            if re.search(r"specifically|namely|the method|change only|limit .{0,40}to", preceding):
                rank += 3
            if rank > best_rank:
                best_rank = rank
                self.parsed.class_hint = qualifier
                self.parsed.method_hint = name

    # -- what the change may not contain ---------------------------------
    _FORBIDDEN_CONSTRUCTS = {
        "loops": r"loops?",
        "comprehensions": r"comprehensions?",
        "lambdas": r"lambdas?",
        "exception handling": r"exception handling|try/except",
        "context managers": r"context managers?",
        "raw SQL": r"raw SQL",
    }
    # Stated as their own sentences rather than in the "no Python ..." list,
    # so the clause scan above never reaches them. Both are graded: the first
    # is why F401 leads ERROR_HINTS, the second is what separates a
    # database-side fix from a Python-side one.
    _NO_NEW_NAMES = re.compile(
        r"use only names (?:the file|it) already imports"
        r"|only names .{0,24}already imports"
        r"|without adding (?:any )?(?:new )?imports"
        r"|do not add (?:any )?(?:new )?imports", re.IGNORECASE)
    _NO_MATERIALISE = re.compile(
        r"do not materiali[sz]e|keep .{0,40}database-backed"
        r"|must not (?:be )?(?:fetch|load|materiali[sz]e)", re.IGNORECASE)

    def _read_style_rules(self) -> None:
        """Constructs the statement rules out inside the changed code.

        Cheap to check before an edit is sent, expensive to discover from a
        grader's AST audit. Every one of these is a two-word phrase in
        hard-wrapped prose, so they match `flat`.
        """
        constraints = self.parsed.style_constraints
        if re.search(r"no Python loops|without (?:a |)loops?|plain ORM expressions",
                     self.flat, re.IGNORECASE):
            clause = re.search(r"no Python[^.]{0,200}\.", self.flat, re.IGNORECASE)
            haystack = clause.group(0) if clause else self.flat
            for label, pattern in self._FORBIDDEN_CONSTRUCTS.items():
                if re.search(pattern, haystack, re.IGNORECASE):
                    constraints.append(label)
        if self._NO_NEW_NAMES.search(self.flat):
            constraints.append("names the file does not import")
        if self._NO_MATERIALISE.search(self.flat):
            constraints.append("materialising rows in Python")

    # -- how the statement says to check the work -------------------------
    _RUNNER = re.compile(
        r"^\s*(python|python3|pytest|ruff|flake8|mypy|manage\.py|\./|npm|yarn|pnpm|go |cargo"
        r"|bundle|mvn|gradle|make|psql|clickhouse|tox|nose|rspec|phpunit)",
        re.IGNORECASE | re.MULTILINE)
    _LINTER = re.compile(r"^\s*(ruff|flake8|pylint|mypy|black|eslint|gofmt|rubocop)\b")

    def _read_commands(self) -> None:
        """The checks the statement names, fenced or inline.

        Missing these means the agent never verifies its own patch and
        reports success on something it did not run, so the inline fallback
        matters as much as the fenced one.
        """
        for block in _fenced_blocks(self.text):
            if not (self._RUNNER.match(block) or self._RUNNER.search(block)):
                continue
            lines = _split_shell_commands(block)
            lint_lines = [line for line in lines if self._LINTER.match(line)]
            run_lines = [line for line in lines if not self._LINTER.match(line)]
            # One fenced block is one script: an `export` or `cd` on line one
            # must still be in effect on line two. `set -e` so the block's exit
            # status is the first failure, not whatever the last line returned.
            if len(run_lines) == 1:
                self.parsed.commands.append(run_lines[0])
            elif run_lines:
                self.parsed.commands.append("set -e\n" + "\n".join(run_lines))
            self.parsed.commands.extend(lint_lines)   # for lint_paths; the runner skips them

        if not self.parsed.commands:
            for span in re.findall(r"`([^`]+)`", self.text, re.DOTALL):
                candidate = " ".join(span.split())     # inline spans wrap across lines
                if self._RUNNER.match(candidate) and len(candidate) > 12:
                    self.parsed.commands.append(candidate)

    def _read_lint_paths(self) -> None:
        """A lint invocation names the file the task expects to have changed --
        a strong, engine-agnostic hint, and an explicit permission."""
        for command in self.parsed.commands:
            if not self._LINTER.match(command):
                continue
            for token in command.split():
                if _looks_like_path(token) and token not in self.parsed.lint_paths:
                    self.parsed.lint_paths.append(token)

    # -- vocabulary and numbers -------------------------------------------
    _STOP_IDENTIFIERS = {"do_not", "make_sure", "the_same", "read_only", "task_toml"}

    def _collect_identifiers(self) -> None:
        """Symbols worth grepping for: they trace a symptom to its query."""
        found: list[str] = []
        found += re.findall(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b", self.prose)
        found += re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", self.prose)
        for token in re.findall(r"`([A-Za-z_][\w.]*)`", self.prose):
            if not _looks_like_path(token):
                found.append(token.split(".")[-1])
        ordered: list[str] = []
        for token in found:
            if token.lower() in self._STOP_IDENTIFIERS or len(token) < 4 or token in ordered:
                continue
            ordered.append(token)
        self.parsed.identifiers = ordered[:40]

    def _read_numeric_targets(self) -> None:
        """Bounds the grader states outright. An absolute number beats the
        relative "must not grow with input" test whenever the task gives one."""
        match = re.search(
            r"(?:at\s+most|no\s+more\s+than|a\s+maximum\s+of|not\s+exceed|≤|<=)\s*(\d+)\s+"
            r"(?:SQL\s+|database\s+)?(?:quer(?:y|ies)|statements?|round[-\s]trips?)",
            self.flat, re.IGNORECASE) \
            or re.search(r"\b(\d+)\s+(?:SQL\s+|database\s+)?quer(?:y|ies)\s+"
                         r"(?:in\s+total|total|regardless|for\s+the\s+whole)",
                         self.flat, re.IGNORECASE)
        if match:
            self.parsed.targets["max_queries"] = int(match.group(1))
        match = re.search(r"(?:read|scan)s?\s+(?:at\s+most|no\s+more\s+than|fewer\s+than|under)"
                          r"\s+([\d,]+)\s+rows", self.flat, re.IGNORECASE)
        if match:
            self.parsed.targets["max_read_rows"] = int(match.group(1).replace(",", ""))

    def _resolve_paths(self) -> None:
        """Drop mentioned paths that do not exist, but remember the ones an
        authoring task expects us to create."""
        self.parsed.named_paths = [
            path for path in self.parsed.named_paths
            if (self.root / path).exists() or (self.root / path).parent.is_dir()
        ]
        create = [path for path in self.parsed.named_paths if not (self.root / path).exists()]
        if create:
            self.parsed.targets["create_paths"] = create


def parse_instruction(text: str, root: Path) -> Instruction:
    """Everything the statement says, without an LLM. See InstructionParser."""
    return InstructionParser(text, root).parse()


# ---------------------------------------------------------------------------
# Repository index and target location
# ---------------------------------------------------------------------------

# Signals that a line participates in a query layer, weighted by specificity.
QUERY_SIGNALS: tuple[tuple[str, float], ...] = (
    # raw SQL entry points, any language
    (r"\bRawSQL\b|\.raw\(|raw_sql|text\(\s*[\"']|find_by_sql|\$queryRaw|\$executeRaw|knex\.raw\(", 3.0),
    (r"cursor\.execute|connection\.cursor|executemany|execute_batch|\.QueryRow(Context|)\(|\.Query(Context|)\(|\.Exec(Context|)\(|sql\.DB|sqlx\.", 2.0),
    # SQL keywords, one per line, so multi-line statements are counted too
    (r"\bSELECT\b", 1.2), (r"\bFROM\s+\w", 0.8), (r"\bJOIN\b", 1.0), (r"\bWHERE\b", 0.8),
    (r"\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\s+BY\b|\bPREWHERE\b", 1.0),
    (r"\bINSERT\s+INTO\b|\bUPDATE\b.+\bSET\b|\bDELETE\s+FROM\b", 2.5),
    (r"\bOVER\s*\(|\bPARTITION\s+BY\b|\bWITH\s+RECURSIVE\b|\bLATERAL\b|\bEXISTS\s*\(", 1.5),
    # Django
    (r"\.annotate\(|\.aggregate\(|Subquery\(|OuterRef\(|\bWindow\(|\bExists\(", 2.0),
    (r"\.select_related\(|\.prefetch_related\(|\.only\(|\.defer\(|\.values(_list|)\(", 1.5),
    (r"bulk_create|bulk_update|\.update\(|\.delete\(", 1.5),
    (r"session\.query|select\(|join\(|\.filter\(|\.exclude\(|\.order_by\(", 1.0),
    # SQLAlchemy
    (r"session\.(execute|scalars|scalar)\(|sa\.(select|func|text|case)\(|joinedload\(|selectinload\(|\.subquery\(", 1.5),
    # Knex / Prisma / TypeORM
    (r"knex\(|\.whereIn\(|\.whereRaw\(|\.leftJoin\(|\.innerJoin\(|\.groupBy\(|\.havingRaw\(|\.select\(|\.insert\(", 1.5),
    (r"prisma\.\w+\.(findMany|findFirst|findUnique|create|update|upsert|delete|aggregate|groupBy|count)\(", 2.0),
    (r"createQueryBuilder\(|getRepository\(|leftJoinAndSelect\(|\.getMany\(|\.getRawMany\(", 2.0),
    # Go: GORM, pgx, database/sql
    (r"db\.(Where|Preload|Joins|Select|Find|First|Model|Raw|Exec)\(|pool\.(Query|QueryRow|Exec)\(|pgx\.", 1.5),
    # Ruby ActiveRecord
    (r"\.includes\(|\.joins\(|\.left_joins\(|\.group\(|\.pluck\(|\.find_each\(|\.where\(|\.where\.not\(", 1.0),
    # Java: jOOQ / JPA
    (r"@Query\(|createQuery\(|createNativeQuery\(|JOIN FETCH|dsl\.select|\.fetch\(", 1.5),
    # ClickHouse: engines, settings, clients
    (r"MergeTree|SETTINGS\s+\w+|clickhouse|client\.(query|execute|command|query_df|insert)\(|windowFunnel|argMax|uniqExact|quantiles?", 2.5),
    # DDL / indexes
    (r"CREATE\s+(UNIQUE\s+|)INDEX|AddIndex|RemoveIndex|models\.Index\(|Index\(fields|@@index|add_index|USING\s+(gin|gist|brin|btree)", 2.5),
)


def query_density(text: str) -> float:
    """How much a file looks like it talks to a database, per line.

    Density, not volume, for the reason the ranker normalises: a 5,000-line
    view module mentioning queries throughout is not more likely to hold *this*
    query than a 67-line manager whose every line is about it. Shared by the
    call graph and the problem profiler so neither rebuilds the other's work.
    """
    raw = sum(weight * len(re.findall(pattern, text)) for pattern, weight in QUERY_SIGNALS)
    return raw / (len(text.splitlines()) ** 0.5 + 4.0)


@dataclass
class RepoFile:
    path: Path
    relative: str
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


def read_source(path: Path) -> str:
    """Read without newline translation: a CRLF file must diff as CRLF, or the
    patch rewrites every line and `git apply` rejects it against the original."""
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def write_source(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


class Repository:
    """A lazily-read, in-memory view of the application checkout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: list[str] = []
        self._cache: dict[str, str] = {}
        self._snapshots: dict[str, str | None] = {}
        self._index()

    def _index(self) -> None:
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            # Sorted: directory order is filesystem-dependent, and two validators
            # must rank and slice identically from the same checkout.
            dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS and not name.startswith("."))
            for name in sorted(filenames):
                if Path(name).suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                full = Path(dirpath) / name
                try:
                    if full.is_symlink() or full.stat().st_size > 2_000_000:
                        continue
                except OSError:
                    continue
                self.files.append(str(full.relative_to(self.root)))
                count += 1
                if count >= MAX_INDEXED_FILES:
                    log(f"index truncated at {MAX_INDEXED_FILES} files")
                    return
        log(f"indexed {len(self.files)} source files under {self.root}")
        trace("Repository", "out", files=len(self.files),
              suffixes=sorted({Path(f).suffix for f in self.files})[:8])

    def read(self, relative: str) -> str | None:
        if relative in self._cache:
            return self._cache[relative]
        path = self.root / relative
        try:
            text = read_source(path)
        except (OSError, UnicodeDecodeError):
            return None
        self._cache[relative] = text
        return text

    # -- mutation with rollback -----------------------------------------
    def snapshot(self, relative: str) -> None:
        if relative in self._snapshots:
            return
        path = self.root / relative
        try:
            self._snapshots[relative] = read_source(path) if path.exists() else None
        except (OSError, UnicodeDecodeError):
            self._snapshots[relative] = None

    def write(self, relative: str, text: str) -> None:
        """Write a file, or raise OSError naming what stopped it.

        The path comes from the model, so it is untrusted input: `a/b.py/c.py`
        where `a/b.py` is a file makes mkdir raise NotADirectoryError, and a
        path that resolves onto a directory makes the open raise
        IsADirectoryError. apply_edits catches both and reports them as edit
        errors, because a bad path is something the model can correct on the
        next round -- while an exception escaping here unwinds the whole solve
        loop and throws away a patch that may already be passing.
        """
        self.snapshot(relative)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_source(path, text)
        self._cache[relative] = text

    def revert(self, relative: str) -> None:
        """Undo one file, reporting rather than raising.

        Reverting is cleanup, and cleanup that raises turns a recoverable
        problem into a lost task: solve() reverts before every attempt and
        after every failure, and agent_main reverts in its finally so the
        verifier sees a pristine checkout. None of those callers has anything
        useful to do with an OSError, and all of them have something to lose.
        """
        if relative not in self._snapshots:
            return
        original = self._snapshots[relative]
        path = self.root / relative
        try:
            if original is None:
                path.unlink(missing_ok=True)
                self._cache.pop(relative, None)
            else:
                write_source(path, original)
                self._cache[relative] = original
        except OSError as exc:
            # Worth saying loudly: a file left modified is a file the verifier
            # will hash and grade against us.
            log(f"WARNING: could not restore {relative} ({exc}); the checkout may not be pristine")

    def revert_all(self) -> None:
        for relative in list(self._snapshots):
            self.revert(relative)

    def original(self, relative: str) -> str | None:
        if relative in self._snapshots:
            return self._snapshots[relative]
        return self.read(relative)

    def changed_files(self) -> list[str]:
        changed = []
        for relative, original in self._snapshots.items():
            current = None
            path = self.root / relative
            if path.exists():
                try:
                    current = read_source(path)
                except (OSError, UnicodeDecodeError):
                    continue
            if current != original:
                changed.append(relative)
        return sorted(changed)


STOPWORDS = frozenset("""
a an and are as at be been before but by can do does for from has have if in into is it
its may must no not of on only or should so than that the their then there these this
those to use used using was were when which while will with without you your work run
also any each every same such keep make change changed unchanged rest file files line
lines method function name names instruction task check checks finishing following
become becomes became becoming stay stays staying stayed remain remains remaining
currently instead both being handful number numbers added adding still now exact exactly
correct correctly wrong incorrect incorrectly slow slowly fast expensive cheap large small
many few several some most more less least first last new old one two three per well very
just even quite rather already again once twice however therefore because since although
whether either neither between within across through against toward towards during after
under over above below behind result results returns returned returning gives given give
takes taken take shows shown show needs needed need wants want expected expects expect
""".split())


# Words that name a kind of code artifact. A statement using one is describing
# where the problem lives, not what it looks like.
ROLE_WORDS = frozenset("""
migration index manager queryset filter filterset serializer view signal model cache
search api admin form util service repository dao schema query sql handler router
controller resource endpoint task job worker command middleware backend store
""".split())


# Test code in any of the layouts the bench uses: tests/, test/, spec/,
# __tests__/, test_x.py, x_test.go, x.test.js, x.spec.ts, conftest, fixtures.
_TEST_PATH = re.compile(
    r"(^|/)(tests?|specs?|__tests__|fixtures?)(/|$)|(^|/)test_[^/]*$|_test\.\w+$"
    r"|\.(test|spec)\.\w+$|conftest|fixture")
_SEED_PATH = re.compile(r"(^|/|_)seeds?(_|\.|/|$)|sample_data|(^|/)data/")

# Path words that carry no information about what a file is for.
_PATH_NOISE = frozenset("""
py js ts tsx jsx go rb rs java kt php sql yml yaml toml json index init main src lib app
internal pkg cmd core utils util common base
""".split())


def _path_tokens(relative: str) -> set[str]:
    """Words of a path, so `api` matches `api/` but not `capital.py`."""
    return {t for t in re.split(r"[/._\-]+", relative.lower()) if t and t not in _PATH_NOISE}


def _token_match(word: str, tokens: set[str]) -> bool:
    """Whole-token or stem-prefix match: `filter` reaches `filtersets`, `tag`
    reaches `tags`, but `api` no longer reaches `capital`."""
    stem = word.rstrip("s")
    return any(t.rstrip("s") == stem or t.startswith(stem) for t in tokens)


# Ranker weights. Tuned on rank_eval.py (56 bench tasks, symptom-only); the
# harness is the place to change them, not intuition.
MENTIONED_PATH_BONUS = 15.0      # the prose names this file in passing
ROLE_WORD_BONUS = 12.0           # statement and path share a role word (manager, filterset...)


class Ranker:
    """Rank files by how likely they hold the query the task is about.

    Three independent signals, because any one of them is gameable by file
    size: query-layer density (normalised for length), rare-word overlap with
    the problem statement, and conventional placement of query code.
    """

    def __init__(self, repo: Repository, instruction: Instruction) -> None:
        self.repo = repo
        self.instruction = instruction
        self.keywords = self._keywords()
        self.idf = self._document_frequencies()
        # Re-rank by count x IDF and drop words no file contains: a word with
        # df=0 can never match, yet it would hold one of the twelve slots the
        # hot-line pass reads. "utilization" said three times in the prose but
        # present in four files now outranks "django" said once and present in
        # nine hundred.
        absent = math.log(1.0 + max(1, len(self.repo.files)))
        present = [w for w in self.keywords if self.idf.get(w, absent) < absent]
        trace("Ranker", "in", keywords=present, dropped=[w for w in self.keywords if w not in present],
              idf={w: round(self.idf.get(w, 0.0), 2) for w in present[:8]})
        self.keywords = present                                  # variant C: original order, df=0 dropped

    def _keywords(self) -> list[str]:
        """Content words from the prose -- never from the command blocks."""
        prose = re.sub(r"```.*?```", " ", self.instruction.text, flags=re.DOTALL)
        counts: dict[str, int] = {}
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", prose):
            lowered = word.lower()
            if lowered in STOPWORDS:
                continue
            counts[lowered] = counts.get(lowered, 0) + 1
        for identifier in self.instruction.identifiers:
            lowered = identifier.lower()
            counts[lowered] = counts.get(lowered, 0) + 3
        if self.instruction.method_hint:
            counts[self.instruction.method_hint.lower()] = 12
        if self.instruction.class_hint:
            counts[self.instruction.class_hint.lower()] = 12
        self._counts = counts
        ordered = sorted(counts, key=lambda word: (-counts[word], word))
        return ordered[:30]

    def _document_frequencies(self) -> dict[str, float]:
        """Inverse document frequency, so 'tag' outweighs 'queryset'."""
        total = max(1, len(self.repo.files))
        frequency = {word: 0 for word in self.keywords}
        for relative in self.repo.files:
            text = self.repo.read(relative)
            if text is None:
                continue
            lowered = text.lower()
            for word in self.keywords:
                if word in lowered:
                    frequency[word] += 1
        return {
            word: math.log(1.0 + total / (1.0 + count))
            for word, count in frequency.items()
        }

    def score(self, relative: str) -> tuple[float, list[int]]:
        text = self.repo.read(relative)
        if text is None:
            return 0.0, []

        lines = text.splitlines()
        lowered = text.lower()
        hot: dict[int, float] = {}
        signal = 0.0

        for number, line in enumerate(lines, start=1):
            line_score = 0.0
            for pattern, weight in QUERY_SIGNALS:
                if re.search(pattern, line):
                    line_score += weight
            lowered_line = line.lower()
            for word in self.keywords[:12]:
                if word in lowered_line:
                    line_score += min(3.0, self.idf.get(word, 0.0))
            if line_score:
                hot[number] = line_score
                signal += line_score

        # Density, not volume: a 5,000-line view module mentioning queries
        # everywhere is not more likely to hold *this* query than a 60-line
        # manager whose every line is about it.
        density = signal / (len(lines) ** 0.5 + 4.0)

        # Length normalisation applies to the topical term too, or a large
        # module wins simply by containing more words.
        topical = 0.0
        for word in self.keywords:
            occurrences = lowered.count(word)
            if occurrences:
                topical += min(occurrences, 4) * self.idf.get(word, 0.0)
        topical /= len(lines) ** 0.5 + 4.0

        # A symbol *named* after the subject is far stronger evidence than a
        # passing mention in a comment or string.
        symbols = " ".join(
            match.group(2).lower()
            for match in re.finditer(r"^\s*(class|def|func|function|type|struct)\s+(\w+)",
                                     text, re.MULTILINE)
        )
        symbol_score = sum(4.0 * self.idf.get(word, 0.0) for word in self.keywords
                           if len(word) >= 4 and word in symbols)

        path_lower = relative.lower()
        path_score = 0.0
        tokens = _path_tokens(relative)
        for word in self.keywords:
            if len(word) >= 4 and _token_match(word, tokens):
                path_score += 3.0 * self.idf.get(word, 0.0)
                # A role word in both the statement and the path is where the
                # statement is telling us what KIND of file to look in.
                if word.rstrip("s") in ROLE_WORDS:
                    path_score += ROLE_WORD_BONUS
        for hint, weight in (
            ("manager", 6.0), ("queryset", 6.0), ("repositor", 6.0), ("migration", 4.0),
            ("query", 4.0), ("dao", 4.0), ("sql", 4.0), ("model", 2.0),
            ("filters", 2.0), ("store", 1.5), ("search", 1.5),
        ):
            if hint in path_lower:
                path_score += weight

        if relative in self.instruction.named_paths:
            path_score += MENTIONED_PATH_BONUS

        total = 3.0 * density + 8.0 * topical + symbol_score + path_score
        penalty = 1.0
        if _TEST_PATH.search(path_lower):
            penalty = 0.1           # a fix never lands in a test
        elif _SEED_PATH.search(path_lower):
            penalty = 0.3           # fixture data is dense with SQL but is not the query
        if "/migrations/" in path_lower and "migration" not in self.keywords:
            penalty *= 0.5
        total *= penalty

        top = sorted(hot, key=lambda number: hot[number], reverse=True)[:12]
        # Guarded rather than trusting trace() to return early: score() runs
        # once per indexed file -- 1,206 of them on netbox -- so building these
        # arguments unconditionally would cost more than the ranking does.
        # Only files that scored something are worth a line.
        if TRACE and total > 0:
            trace("Ranker.score", "out", path=relative, total=total,
                  density=3.0 * density, topical=8.0 * topical,
                  symbol=symbol_score, path_score=path_score, penalty=penalty,
                  hot_lines=len(top))
        return total, sorted(top)


class CallGraph:
    """Trace a symptom to the code that actually issues the query.

    A task names a symptom -- "the IP-address device filter is slow" -- not a
    query. The query is often several calls away:

        endpoint -> view -> service -> repository -> ORM -> SQL

    Ranking files by keyword and query-signal density finds the query only when
    the symptom and the query live in the same file. When they do not, the
    ranker points at the symptom and the fix lands in the wrong place. So follow
    the references: from the names the instruction mentions, walk outward to the
    definitions they reach, and see which of those touch a query layer.

    Deliberately approximate. It resolves names, not types, so `a.save()` and
    `b.save()` are the same symbol -- which over-connects rather than
    under-connects, and over-connecting is the safe direction for a search whose
    output is only a ranked shortlist.
    """

    _DEF = re.compile(r"^\s*(?:class|def|async def|func|function)\s+(\w+)", re.MULTILINE)
    _CALL = re.compile(r"\b([A-Za-z_]\w{2,})\s*\(")
    _ATTR = re.compile(r"\.([A-Za-z_]\w{2,})\b")

    # Same notion of "test code" as the protected-paths gate, so the graph never
    # steers the model toward a helper it would then be forbidden to edit.
    _TEST_PATH = re.compile(r"(^|/)(tests?|testing|spec|fixtures?|conftest\.py)(/|$)"
                            r"|(^|/)test_[^/]*$|_test\.[a-z]+$", re.IGNORECASE)

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.defined_in: dict[str, list[str]] = {}     # symbol -> files defining it
        self.defined_at: dict[str, list[tuple[str, int]]] = {}   # symbol -> (file, line)
        self.mentions: dict[str, set[str]] = {}        # file -> symbols it uses
        self._weights: dict[str, float] = {}           # query_weight is hot; cache it
        self._build()

    def _build(self) -> None:
        for relative in self.repo.files:
            # Production code only: a fix never belongs in a test, and test
            # modules would otherwise dominate both seeding and the walk.
            if self._TEST_PATH.search(relative):
                continue
            text = self.repo.read(relative)
            if text is None:
                continue
            for match in self._DEF.finditer(text):
                name = match.group(1)
                self.defined_in.setdefault(name, []).append(relative)
                self.defined_at.setdefault(name, []).append(
                    (relative, text.count("\n", 0, match.start()) + 1))
            used = set(self._CALL.findall(text)) | set(self._ATTR.findall(text))
            self.mentions[relative] = used

    def query_weight(self, relative: str) -> float:
        """How much this file looks like it talks to a database, per line."""
        if relative not in self._weights:
            self._weights[relative] = query_density(self.repo.read(relative) or "")
        return self._weights[relative]

    def seeds_from_keywords(self, keywords: Sequence[str], limit: int = 40) -> list[str]:
        """Symbols whose *name* echoes the task's vocabulary.

        Many instructions name no symbol at all -- "find the manager method that
        assigns tags" mentions no identifier that exists in the code. But the
        code does, e.g. a TaggableManager or TaggedItem class. Matching the task's
        content words against defined symbol names gives the walk somewhere to
        start when the prose gives us nothing.
        """
        wanted = [word for word in keywords if len(word) >= 3]
        hits: list[tuple[int, str]] = []
        for symbol, files in self.defined_in.items():
            lowered = symbol.lower()
            matched = sum(1 for word in wanted if word in lowered)
            if matched:
                weight = max((self.query_weight(f) for f in files[:3]), default=0.0)
                hits.append((matched * 100 + int(weight), symbol))
        hits.sort(key=lambda hit: (-hit[0], hit[1]))          # score, then name: deterministic
        return [symbol for _, symbol in hits[:limit]]

    def trace(self, seeds: Sequence[str], max_hops: int = 3) -> dict[str, float]:
        """Files reachable from the seed symbols, scored by query evidence.

        Score decays with distance, so a query-bearing file two calls from the
        symptom still outranks an unrelated one that merely mentions SQL.
        """
        scores: dict[str, float] = {}
        frontier = {name for name in seeds if name in self.defined_in}
        seen: set[str] = set()

        # One hop backwards first: the files that CALL a seed. A symptom named
        # after a view often has its query in the manager the view calls, but
        # just as often the reverse -- the named helper is called from the
        # file that issues the query.
        for symbol in sorted(frontier):
            owners = set(self.defined_in.get(symbol, []))
            for relative, names in self.mentions.items():
                if symbol in names and relative not in owners:
                    weight = self.query_weight(relative)
                    if weight:
                        scores[relative] = max(scores.get(relative, 0.0), weight * 0.6)

        for hop in range(max_hops):
            if not frontier:
                break
            decay = 0.6 ** hop
            reached: set[str] = set()
            for symbol in frontier:
                if symbol in seen:
                    continue
                seen.add(symbol)
                for relative in self.defined_in.get(symbol, [])[:8]:
                    weight = self.query_weight(relative)
                    if weight:
                        scores[relative] = max(scores.get(relative, 0.0), weight * decay)
                    # Follow what that file reaches, so the walk can cross from
                    # a view into the manager the view eventually calls.
                    reached |= {name for name in self.mentions.get(relative, set())
                                if name in self.defined_in and name not in seen}
            # Widening without limit turns into a full-repo scan; keep the most
            # promising names by how query-ish their defining files are.
            # Ties broken by name: `reached` is a set of strings, and set order
            # follows the per-process hash seed. Two validators must see the
            # same shortlist from the same input.
            frontier = set(sorted(reached,
                                  key=lambda n: (-max((self.query_weight(f)
                                                       for f in self.defined_in.get(n, [])[:3]),
                                                      default=0.0), n))[:40])
        return scores


def locate_targets(repo: Repository, instruction: Instruction, limit: int = 6) -> list[tuple[str, list[int]]]:
    """Rank candidate files: instruction-named ones first, then by evidence."""
    ranker = Ranker(repo, instruction)
    trace("locate_targets", "in", files=len(repo.files), keywords=ranker.keywords[:8],
          method=instruction.method_hint, explicit=bool(instruction.edit_only
                                                        or instruction.lint_paths))

    # Only a permission ("limit changes to X", or the lint command's target)
    # settles the question. A path the prose merely mentions -- "the signal in
    # `app/signals.py` must keep firing" -- is evidence for the ranker, not a
    # decision: measured, treating mentions as decisions sent the model to the
    # wrong file on 2 of 56 symptom-only cases.
    explicit = [
        path
        for path in (instruction.edit_only or instruction.lint_paths)
        if (repo.root / path).exists()
    ]
    if explicit:
        results = [(path, ranker.score(path)[1]) for path in explicit[:limit]]
        log(f"instruction names editable paths: {[path for path, _ in results]}")
        trace("locate_targets", "out", source="the instruction named them",
              candidates=[path for path, _ in results],
              hot_lines=[len(hot) for _, hot in results])
        return results

    scored: list[tuple[float, str, list[int]]] = []
    for relative in repo.files:
        value, hot = ranker.score(relative)
        if value > 0:
            scored.append((value, relative, hot))

    # Ranking alone finds the query only when it shares a file with the symptom.
    # Follow the references too, so a query two calls away is still reachable.
    traced: dict[str, float] = {}
    try:
        graph = CallGraph(repo)
        seeds = list(instruction.identifiers)
        if instruction.method_hint:
            seeds.append(instruction.method_hint)
        if instruction.class_hint:
            seeds.append(instruction.class_hint)
        resolved = [name for name in seeds if name in graph.defined_in]
        if not resolved:
            # The instruction names no symbol that exists in the code. Seed from
            # the task's vocabulary instead, matched against symbol names.
            resolved = graph.seeds_from_keywords(ranker.keywords)
            log(f"no named symbol resolved; seeding the trace from vocabulary: {resolved[:6]}")
        traced = graph.trace(resolved)
        trace("CallGraph", "in", seeds=resolved[:8], symbols=len(graph.defined_in))
        trace("CallGraph", "out", reached=len(traced),
              strongest=sorted(traced, key=lambda k: -traced[k])[:4])
    except Exception as exc:                      # tracing is a bonus, never a blocker
        log(f"call-graph tracing skipped: {exc}")

    scored.sort(key=lambda item: (-item[0], item[1]))
    log("keywords: " + ", ".join(ranker.keywords[:10]))
    if traced:
        reached = sorted(traced, key=lambda k: (-traced[k], k))[:4]
        # Deliberately NOT folded into the ranking: measured on the implicit-
        # target sample it moved the correct file from rank 2 to rank 5. Kept
        # as a hint the model can act on through need_context instead.
        instruction.traced_hint = [r for r in reached
                                   if r not in {path for _, path, _ in scored[:limit]}][:3]
        log(f"call-graph reached {len(traced)} query-bearing file(s); strongest: {reached}")
    log("top candidates: " + ", ".join(f"{path} ({value:.0f})" for value, path, _ in scored[:limit]))
    instruction.candidate_scores = [value for value, _, _ in scored[:limit]]
    top = scored[:limit]
    # The margin is what says whether rank 1 was actually decided: a 5% gap
    # over rank 2 is a coin toss dressed as a ranking.
    margin = ((top[0][0] - top[1][0]) / top[0][0]) if len(top) > 1 and top[0][0] else 1.0
    trace("locate_targets", "out", source="ranked", scored=len(scored),
          candidates=[path for _, path, _ in top],
          scores=[round(value, 1) for value, _, _ in top],
          margin_over_2nd=margin, traced_hint=instruction.traced_hint)
    return [(path, hot) for _, path, hot in top]


# ---------------------------------------------------------------------------
# Code slicing -- show the model the relevant region, not the whole file
# ---------------------------------------------------------------------------

@dataclass
class Slice:
    relative: str
    start: int          # 1-indexed, inclusive
    end: int            # 1-indexed, inclusive
    label: str

    def render(self, repo: Repository) -> str:
        text = repo.read(self.relative) or ""
        lines = text.splitlines()
        chunk = lines[self.start - 1 : self.end]
        numbered = "\n".join(f"{self.start + offset:5d}| {line}" for offset, line in enumerate(chunk))
        return f"--- {self.relative} lines {self.start}-{self.end} ({self.label}) ---\n{numbered}"


def python_definitions(text: str) -> list[tuple[str, int, int, str]]:
    """(qualified_name, start_line, end_line, kind) for every def/class."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    found: list[tuple[str, int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                found.append((name, start, child.end_lineno or child.lineno, kind))
                walk(child, f"{name}.")
    walk(tree, "")
    return found


_BLOCK_START = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|async|export|pub|def|func|function|class|"
    r"module|type|impl|interface|struct)\b|[\w<>\[\], ]+\s+\w+\s*\()",
)


def generic_blocks(text: str) -> list[tuple[str, int, int, str]]:
    """Brace-counting block finder for non-Python sources."""
    lines = text.splitlines()
    blocks: list[tuple[str, int, int, str]] = []
    for index, line in enumerate(lines):
        if not _BLOCK_START.match(line) or "{" not in line:
            continue
        depth = 0
        for end_index in range(index, min(len(lines), index + 400)):
            depth += lines[end_index].count("{") - lines[end_index].count("}")
            if depth <= 0 and end_index > index:
                name = re.sub(r"[^\w]+", " ", line).strip()[:60]
                blocks.append((name, index + 1, end_index + 1, "block"))
                break
    return blocks


def slice_around(repo: Repository, relative: str, hot_lines: Sequence[int],
                 instruction: Instruction, *, budget_lines: int = 320) -> list[Slice]:
    """Pick the enclosing definitions of the interesting lines, or a window."""
    text = repo.read(relative)
    if text is None:
        return []
    total = len(text.splitlines())
    if total <= budget_lines:
        return [Slice(relative, 1, total, "whole file")]

    if relative.endswith(".py"):
        definitions = python_definitions(text)
    else:
        definitions = generic_blocks(text)

    chosen: list[Slice] = []
    used: set[tuple[int, int]] = set()

    # A named method hint wins outright.
    if instruction.method_hint:
        for name, start, end, kind in definitions:
            if name.split(".")[-1] == instruction.method_hint and kind != "class":
                chosen.append(Slice(relative, start, end, f"{kind} {name}"))
                used.add((start, end))
                break

    for line in hot_lines:
        enclosing = [
            (name, start, end, kind)
            for name, start, end, kind in definitions
            if start <= line <= end and kind != "class"
        ]
        if enclosing:
            name, start, end, kind = min(enclosing, key=lambda item: item[2] - item[1])
        else:
            name, start, end, kind = ("context", max(1, line - 25), min(total, line + 25), "window")
        if (start, end) in used:
            continue
        used.add((start, end))
        chosen.append(Slice(relative, start, end, f"{kind} {name}"))

    if not chosen:
        return [Slice(relative, 1, min(total, budget_lines), "file head")]

    if any(piece.label.startswith(("function", "block")) for piece in chosen):
        named = [piece for piece in chosen if not piece.label.startswith("window")]
        if named:
            chosen = named

    chosen.sort(key=lambda item: item.start)
    spent = 0
    kept: list[Slice] = []
    for piece in chosen:
        span = piece.end - piece.start + 1
        if spent + span > budget_lines * 2 and kept:
            break
        kept.append(piece)
        spent += span
    return kept


def package_map(repo: Repository, relative: str, limit: int = 25) -> str:
    """One line per module in the target's package: `path: class A, def b`.

    A function under repair often receives the application's own objects as
    parameters -- a `client` whose class lives two files away. Nothing in the
    target file names that class, so the model guesses `.query()` and
    `.execute()` before reading it (measured: two of three repair rounds on
    one task). The package map costs ~400 tokens and answers that up front.
    """
    parts = Path(relative).parts
    if len(parts) < 2:
        return ""
    # The package is the nearest ancestor directory with other modules in it:
    # `app/reports/sessions.py` sits alone in reports/, so its package is
    # `app/`; `proj/pkg/querysets.py` with twenty siblings keeps
    # `proj/pkg/`, not the whole repository.
    package = str(Path(relative).parent)
    while True:
        siblings = [f for f in repo.files if f.startswith(package + "/") and f != relative
                    and not _TEST_PATH.search(f.lower()) and "/migrations/" not in f
                    and Path(f).stem != "__init__"]
        if len(siblings) >= 2 or "/" not in package:
            break
        package = str(Path(package).parent)
    depth = package.count("/") + 1
    lines: list[str] = []
    for other in sorted(siblings, key=lambda f: (f.count("/") - depth, f)):
        if not other.startswith(package + "/") or other == relative:
            continue
        text = repo.read(other)
        if text is None:
            continue
        defs = python_definitions(text) if other.endswith(".py") else generic_blocks(text)
        names = [f"{'class ' if kind == 'class' else ''}{name}" for name, _, _, kind in defs
                 if "." not in name][:6]
        if names:
            lines.append(f"  {other}: {', '.join(names)}")
        if len(lines) >= limit:
            lines.append("  ...")
            break
    return ("other modules in this package (request one with read_file if the target uses "
            "their objects):\n" + "\n".join(lines)) if lines else ""


def file_outline(repo: Repository, relative: str, limit: int = 80) -> str:
    """A compact map of a file so the model knows what else lives there."""
    text = repo.read(relative)
    if text is None:
        return ""
    if relative.endswith(".py"):
        definitions = python_definitions(text)
    else:
        definitions = generic_blocks(text)
    if not definitions:
        return ""
    rows = [f"  L{start:<5} {kind:8} {name}" for name, start, _, kind in definitions[:limit]]
    imports = ""
    if relative.endswith(".py"):
        heads = [line for line in text.splitlines()[:80]
                 if line.startswith(("import ", "from ")) or re.match(r"^\s+(import|from)\s", line)]
        if heads:
            imports = "\nimports available in this file:\n" + "\n".join(f"  {line.strip()}" for line in heads[:40])
    return f"outline of {relative}:\n" + "\n".join(rows) + imports


# ---------------------------------------------------------------------------
# Live database probe
# ---------------------------------------------------------------------------

@dataclass
class DatabaseTarget:
    engine: str                       # postgresql | clickhouse
    # Lower sorts first. Django's own settings are exact; a regex over a
    # settings file is a guess and must never outrank them.
    priority: int = 5
    dsn: str | None = None
    host: str = ""
    port: str = ""
    user: str = ""
    password: str = ""
    database: str = ""


def target_url(target: "DatabaseTarget", *, password: bool = False) -> str:
    """The target as a connection URL.

    The password is left out unless asked for: the model never connects itself
    -- it asks this agent for `sql` and `explain` and the agent runs them -- so
    a secret in the prompt buys nothing and travels to a provider.
    """
    credentials = target.user
    if credentials and password and target.password:
        credentials += f":{target.password}"
    return (f"{target.engine}://{credentials + '@' if credentials else ''}"
            f"{target.host}:{target.port}/{target.database}")


class DatabaseProbe:
    """Discover the application's database and ask it real questions.

    Credentials are read from the application's own configuration -- the same
    place the app reads them -- so this stays repo-agnostic.
    """

    def __init__(self, repo: Repository, instruction: Instruction) -> None:
        self.repo = repo
        self.instruction = instruction
        self.targets: list[DatabaseTarget] = []
        self.notes: list[str] = []
        self._discover()

    # -- discovery -------------------------------------------------------
    # A database is guaranteed to exist for every task, so "nothing found" is
    # never the right answer. Four layers, all run, cheapest first; every
    # candidate they produce is then pinged and only the reachable ones kept.
    #
    # Measured on the 50 generated bench apps: none sets a connection variable
    # in the agent's container and none ships a config file. Every one of them
    # falls back to a default written in source -- a DSN literal, a defaults
    # dict, or `envOr("X_CLICKHOUSE_HOST", "clickhouse")`. Source is therefore
    # a first-class source here, not an afterthought.
    def _discover(self) -> None:
        # (url, reachable) for every candidate the connection test tried, in the
        # order it tried them. Kept so a harness can show that the test ran and
        # what it rejected -- the log line alone reaches nobody.
        self.attempts: list[tuple[str, bool]] = []
        self._env = self._load_env_files()
        self._from_env()
        self._from_django()
        self._from_source()
        offline = bool(os.getenv("RIDGES_PROBE_NO_PING"))   # offline harnesses only
        if not self.targets and not offline:
            self._from_network()
        if offline:
            pass
        elif self.targets:
            # Stop at the first candidate that answers. Targets are already in
            # priority order -- Django's own settings before a regex over a
            # config file -- and everything downstream reads targets[0] and
            # nothing else, so verifying the rest buys no evidence and costs up
            # to the full connect timeout each. A scrape that yields four
            # candidates spent four timeouts to keep three targets no caller
            # ever looked at.
            self.targets = self._first_reachable(self.targets)
            if not self.targets:
                self._from_network()
                self.targets = self._first_reachable(self.targets)
        trace("DatabaseProbe", "out", tried=[url for url, _ in self.attempts],
              answered=[url for url, ok in self.attempts if ok],
              used=target_url(self.targets[0]) if self.targets else None)
        if self.targets:
            for target in self.targets:
                log(f"database: {target_url(target)} (SELECT 1 answered)")
        else:
            log("no database credentials discovered; proceeding on static evidence")

    # Ports that belong to something else entirely. Scraping a config file that
    # mentions "postgres" somewhere picks up every host:port pair in it, so the
    # netbox samples yield a Redis candidate at 6379 alongside the real one. The
    # ping drops those, but each costs up to 8s of connect timeout to find out.
    #
    # A denylist, deliberately, not a whitelist of 5432/8123: PostgreSQL runs on
    # 5433 in multi-instance setups and 6432 behind pgbouncer, and ClickHouse's
    # native protocol is 9000/9440 -- which _clickhouse() already handles. A
    # whitelist would refuse databases this agent can actually talk to.
    _NOT_A_DATABASE_PORT = {
        "6379", "6380",          # redis
        "11211",                 # memcached
        "27017", "27018",        # mongodb
        "9200", "9300",          # elasticsearch
        "5672", "15672",         # rabbitmq
        "2181",                  # zookeeper
        "9092",                  # kafka
        "80", "443", "3000", "8000", "8080",   # http
    }

    def _add(self, target: DatabaseTarget) -> None:
        if not target.host or not re.fullmatch(r"[\w.-]+", target.host):
            return                               # scraped junk, not a hostname
        if target.port and not target.port.isdigit():
            return                               # an env var that never expanded
        if target.port in self._NOT_A_DATABASE_PORT:
            return
        if target.user and not re.fullmatch(r"[\w.$-]+", target.user):
            return
        if target.password and not re.fullmatch(r"[^\s\"'<>{}$]+", target.password):
            return                               # a template, not a secret
        key = (target.engine, target.host, target.port, target.database, target.user)
        for existing in self.targets:
            if (existing.engine, existing.host, existing.port, existing.database, existing.user) == key:
                if target.priority < existing.priority:
                    existing.priority = target.priority
                    self.targets.sort(key=lambda item: item.priority)
                return
        if self.instruction.engine != "unknown" and target.engine != self.instruction.engine:
            target.priority += 10                # wrong engine: keep, but last
        self.targets.append(target)
        self.targets.sort(key=lambda item: item.priority)

    # Layer 1: process environment plus dotenv files -------------------------
    _ENV_FILES = (".env", ".env.local", ".env.development", ".env.test", ".env.example")

    def _load_env_files(self) -> dict[str, str]:
        """KEY=VALUE from dotenv files; the real environment wins over files."""
        merged: dict[str, str] = {}
        for name in self._ENV_FILES:
            path = self.repo.root / name
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                merged.setdefault(key, value.strip().strip('"').strip("'"))
        merged.update(os.environ)
        return merged

    def _expand(self, value: str) -> str:
        """Resolve ${VAR}, ${VAR:-default}, $VAR and env("VAR") references."""
        env = self._env

        def sub(match: re.Match) -> str:
            return env.get(match.group("name"), match.group("default") or "")

        value = re.sub(r"\$\{(?P<name>\w+)(?::-(?P<default>[^}]*))?\}", sub, value)
        value = re.sub(r"\$(?P<name>[A-Za-z_]\w*)(?P<default>)", sub, value)
        value = re.sub(r'env\(\s*["\'](?P<name>\w+)["\']\s*\)(?P<default>)', sub, value)
        return value

    _URL_KEY = re.compile(r"(DATABASE_URL|DATABASE_URI|DB_URL|DSN|CLICKHOUSE_URL|POSTGRES_URL|POSTGRESQL_URL)$")
    _PART_KEY = re.compile(r"^(?P<prefix>.*?)_?(?P<part>HOST|HOSTNAME|PORT|HTTP_PORT|USER|USERNAME|PASSWORD|PASS|DB|DATABASE|DBNAME|NAME)$")

    def _from_env(self) -> None:
        """Connection variables under any prefix: WELCOME_DATABASE_URL, PGHOST,
        POSTGRES_HOST, DRIFTWOOD_CLICKHOUSE_HOST. The prefix is the app's, so
        matching on the suffix is what generalises."""
        for key, value in self._env.items():
            if self._URL_KEY.search(key) and value:
                self._add_url(self._expand(value), priority=1)
        families: dict[str, dict[str, str]] = {}
        for key, value in self._env.items():
            match = self._PART_KEY.match(key)
            if match and value:
                families.setdefault(match.group("prefix"), {})[match.group("part")] = value
        for prefix, parts in families.items():
            upper = prefix.upper()
            if upper in ("PG",) or "POSTGRES" in upper or "PGSQL" in upper:
                engine = "postgresql"
            elif "CLICKHOUSE" in upper or upper.endswith("CH") or upper == "CH":
                engine = "clickhouse"
            elif "DB" in upper or "DATABASE" in upper or "SQL" in upper:
                engine = "postgresql"
            else:
                continue                         # SMTP_HOST, REDIS_HOST, ...
            host = parts.get("HOST") or parts.get("HOSTNAME")
            if not host:
                continue
            self._add(DatabaseTarget(
                engine=engine, priority=1, host=self._expand(host),
                port=parts.get("PORT") or parts.get("HTTP_PORT") or ("8123" if engine == "clickhouse" else "5432"),
                user=parts.get("USER") or parts.get("USERNAME") or "",
                password=parts.get("PASSWORD") or parts.get("PASS") or "",
                database=parts.get("DB") or parts.get("DATABASE") or parts.get("DBNAME") or parts.get("NAME") or ""))

    _URL = re.compile(
        r"(?P<scheme>[a-z][\w+.-]*)://(?:(?P<user>[^:@/\s]+)(?::(?P<password>[^@/\s]*))?@)?"
        r"(?P<host>[\w.-]+)(?::(?P<port>\d+))?(?:/(?P<database>[\w.-]*))?(?P<query>\?[^\s\"'`]*)?",
        re.IGNORECASE)

    def _add_url(self, url: str, priority: int) -> None:
        match = self._URL.match(url.strip())
        if not match:
            return
        scheme = match.group("scheme").lower().split("+")[0]   # postgresql+psycopg -> postgresql
        if scheme in ("postgres", "postgresql", "pgsql", "pg"):
            engine, default_port = "postgresql", "5432"
        elif scheme in ("clickhouse", "clickhouses", "ch", "chs"):
            engine, default_port = "clickhouse", "8123"
        else:
            return                               # mysql, sqlite, redis, http ...
        host = match.group("host") or "localhost"
        port = match.group("port") or default_port
        user, password = match.group("user") or "", match.group("password") or ""
        database = (match.group("database") or "").strip("/")
        # Rebuild with a scheme psql and psycopg accept; keep ?sslmode=... intact.
        auth = f"{user}:{password}@" if user else ""
        dsn = f"{engine}://{auth}{host}:{port}/{database}{match.group('query') or ''}"
        self._add(DatabaseTarget(engine=engine, priority=priority, dsn=dsn if engine == "postgresql" else None,
                                 host=host, port=port, user=user, password=password, database=database))

    # Layer 2: ask the framework ----------------------------------------------
    def _from_django(self) -> None:
        """Ask Django itself, which is exact when it works."""
        manage = self._find_manage_py()
        if not manage:
            return
        script = (
            "import json;from django.conf import settings;"
            "print('RIDGES_DB'+json.dumps({k:{kk:str(vv) for kk,vv in v.items() "
            "if kk in ('ENGINE','NAME','USER','PASSWORD','HOST','PORT')} "
            "for k,v in settings.DATABASES.items()}))"
        )
        result = run_command([app_python(self.repo.root), str(manage), "shell", "-c", script], timeout=120)
        match = re.search(r"RIDGES_DB(\{.*\})", result.stdout or "")
        if not match:
            return
        try:
            databases = json.loads(match.group(1))
        except json.JSONDecodeError:
            return
        for name, config in databases.items():
            engine_string = (config.get("ENGINE") or "").lower()
            engine = "clickhouse" if "clickhouse" in engine_string else "postgresql"
            self._add(DatabaseTarget(
                engine=engine, priority=0, host=config.get("HOST") or "localhost",
                port=str(config.get("PORT") or ("8123" if engine == "clickhouse" else "5432")),
                user=config.get("USER") or "", password=config.get("PASSWORD") or "",
                database=config.get("NAME") or "",
            ))
        self.notes.append(f"django databases: {sorted(databases)}")

    def _find_manage_py(self) -> Path | None:
        for relative in self.repo.files:
            if Path(relative).name == "manage.py" and relative.count("/") <= 2:
                return self.repo.root / relative
        return None

    # Layer 3: the application's own source and config ---------------------------
    _DSN_LITERAL = re.compile(
        r"\b(?:postgres(?:ql)?(?:\+\w+)?|pgsql|clickhouses?(?:\+\w+)?)://[^\s\"'`<>]{6,200}", re.IGNORECASE)
    # `envOr("X_HOST", "clickhouse")`, `os.environ.get("X_HOST", "clickhouse")`,
    # `ENV.fetch("X_HOST", "clickhouse")`: the default argument IS the value.
    _ENV_DEFAULT = re.compile(
        r"\(\s*[\"']\w*?(?P<part>HOST|HOSTNAME|PORT|HTTP_PORT|USER|USERNAME|PASSWORD|PASS|DB|DATABASE|DBNAME|DB_NAME|NAME)[\"']"
        r"\s*,\s*[\"'](?P<value>[^\"']*)[\"']\s*\)")
    # `process.env.X_HOST || "clickhouse"`, `process.env.X_HOST ?? "clickhouse"`
    _JS_DEFAULT = re.compile(
        r"process\.env\.\w*?(?P<part>HOST|PORT|USER|USERNAME|PASSWORD|DB|DATABASE)\s*(?:\|\||\?\?)\s*[\"'](?P<value>[^\"']*)[\"']")
    # `"host": "clickhouse"`, `HOST = 'db'`, `host: postgres`, `Host: "db"`
    _KEY_VALUE = re.compile(
        r"(?<![\w.@-])[\"']?(?P<part>host|hostname|port|http_port|user|username|password|db_?name|database|dbname|name)[\"']?"
        r"\s*[:=]\s*(?:[\"'](?P<quoted>[^\"'\n]*)[\"']|(?P<bare>[\w.-]+)(?!\s*[\(\[]))", re.IGNORECASE)
    _PART_ALIASES = {"HOSTNAME": "HOST", "HTTP_PORT": "PORT", "USERNAME": "USER", "PASS": "PASSWORD",
                     "DB": "DATABASE", "DBNAME": "DATABASE", "DB_NAME": "DATABASE", "NAME": "DATABASE"}

    def _from_source(self) -> None:
        """Defaults written in the application's code and config files.

        Two shapes. A DSN literal anywhere in source (`DEFAULT_URL =
        "postgres://app:pw@postgres:5432/app_dev"`), and a per-field default
        in the files that mention the engine: a dict, an assignment, a YAML
        key, or the fallback argument of an env lookup.
        """
        candidates: list[tuple[int, str, str]] = []           # (priority, relative, text)
        for relative in self.repo.files:
            lowered_path = relative.lower()
            text = self.repo.read(relative)
            if text is None or len(text) > 400_000:
                continue
            lowered = text.lower()
            if "postgres" not in lowered and "clickhouse" not in lowered:
                continue
            # A DSN in the module that owns the connection outranks one in a
            # test or an example file; the ping settles any remaining doubt.
            if _TEST_PATH.search(lowered_path) or "example" in lowered_path or "sample" in lowered_path:
                priority = 6
            elif re.search(r"(^|/)(db|database|conn(ection)?|client|store|settings|config)", lowered_path):
                priority = 3
            else:
                priority = 4
            candidates.append((priority, relative, text))
        candidates.sort(key=lambda item: (item[0], item[1]))

        for priority, relative, text in candidates[:80]:
            for literal in self._DSN_LITERAL.findall(text):
                if "${" in literal or "%s" in literal or "{" in literal:
                    literal = self._expand(literal)
                    if "{" in literal or "$" in literal:
                        continue             # still a template after expansion
                self._add_url(literal.rstrip(".,;)"), priority=priority)

        for priority, relative, text in candidates[:80]:
            self._scrape_fields(relative, text, priority)

    def _scrape_fields(self, relative: str, text: str, priority: int) -> None:
        # DSN literals were handled by _add_url; scrub them so `host:5432`
        # inside a URL is not read back as a host named 5432.
        text = self._DSN_LITERAL.sub(" ", text)
        lowered = text.lower()
        engine = "clickhouse" if "clickhouse" in lowered else "postgresql"
        parts: dict[str, str] = {}
        weak_name: str | None = None                 # `name` only if nothing better names the database
        for pattern in (self._ENV_DEFAULT, self._JS_DEFAULT):
            for match in pattern.finditer(text):
                raw = match.group("part").upper()
                if raw == "NAME":
                    weak_name = weak_name or match.group("value")
                    continue
                part = self._PART_ALIASES.get(raw, raw)
                parts.setdefault(part, match.group("value"))
        # Only look at plain key/value pairs when the file is about a database
        # connection; `name = "x"` is everywhere otherwise.
        if re.search(r"(^|/)(db|database|conn(ection)?|client|store|settings|config)", relative.lower()) \
                or re.search(r"DATABASES\s*=|DEFAULTS\s*=|connection|datasource", text):
            for match in self._KEY_VALUE.finditer(text):
                raw = match.group("part").upper().replace("_", "")
                value = match.group("quoted") if match.group("quoted") is not None else match.group("bare")
                if not value:
                    continue
                if raw == "NAME":
                    weak_name = weak_name or value
                    continue
                part = self._PART_ALIASES.get(raw, raw)
                if part == "DATABASE" and value.isdigit():
                    # A Redis database index, not a database name. NetBox's
                    # configuration.py carries `REDIS = {... 'DATABASE': 0}`
                    # below its DATABASES block, and because a bare `DATABASE`
                    # key outranks the weak `NAME`, the scrape produced
                    # postgresql://solver@postgres:5432/0 -- a URL that cannot
                    # connect, so the live schema was silently lost whenever
                    # asking Django directly did not work.
                    continue
                if part in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE"):
                    parts.setdefault(part, value)
        if "DATABASE" not in parts and weak_name:
            parts["DATABASE"] = weak_name
        host = parts.get("HOST")
        if not host or host.lower() in ("true", "false", "none", "null") or host.isdigit():
            return
        if parts.get("DATABASE", "").lower() in ("true", "false", "none", "null"):
            parts.pop("DATABASE")
        self._add(DatabaseTarget(
            engine=engine, priority=priority, host=self._expand(host),
            port=parts.get("PORT") or ("8123" if engine == "clickhouse" else "5432"),
            user=parts.get("USER") or "", password=parts.get("PASSWORD") or "",
            database=parts.get("DATABASE") or ""))

    # Layer 4: the container network ---------------------------------------------
    _PROBE_HOSTS = ("localhost", "127.0.0.1", "db", "database", "postgres", "postgresql", "pg",
                    "clickhouse", "ch", "clickhouse-server")
    _PROBE_CREDS = {
        "postgresql": (5432, (("postgres", "postgres"), ("postgres", ""), ("app", "app"))),
        "clickhouse": (8123, (("default", ""), ("clickhouse", "clickhouse"), ("app", "app"))),
    }

    @staticmethod
    def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
        """Connect with a bound that covers DNS too. `create_connection`'s
        timeout starts after name resolution, and an unknown host on a slow
        resolver can take ten seconds per name -- measured: the probe of ten
        conventional names hung a test run for minutes."""
        import socket
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        def attempt() -> bool:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                return False

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(attempt).result(timeout=timeout)
        except FutureTimeout:
            return False
        finally:
            pool.shutdown(wait=False)

    def _from_network(self) -> None:
        """Last resort: the sidecar is on the compose network under a
        conventional name. Knock on the conventional doors."""
        wanted = [self.instruction.engine] if self.instruction.engine in self._PROBE_CREDS \
            else list(self._PROBE_CREDS)
        deadline = time.monotonic() + 15.0          # a safety net, not a scan
        for engine in wanted:
            port, creds = self._PROBE_CREDS[engine]
            for host in self._PROBE_HOSTS:
                if time.monotonic() > deadline:
                    break
                if not self._tcp_open(host, port):
                    continue
                for user, password in creds:
                    target = DatabaseTarget(engine=engine, priority=8, host=host, port=str(port),
                                            user=user, password=password,
                                            database="postgres" if engine == "postgresql" else "default")
                    if self._ping(target):
                        if engine == "postgresql":
                            names = self.sql("SELECT datname FROM pg_database WHERE NOT datistemplate "
                                             "AND datname <> 'postgres' ORDER BY 1", target=target, timeout=10)
                            first = next((n.strip() for n in names.splitlines()
                                          if re.fullmatch(r"[\w-]+", n.strip()) and n.strip() != "datname"), "")
                            if first:
                                target.database = first
                        self._add(target)
                        break
                break                            # one reachable host per engine is enough

    def _first_reachable(self, candidates: list[DatabaseTarget]) -> list[DatabaseTarget]:
        """The first candidate that answers SELECT 1, as a one-element list."""
        for target in candidates:
            if self._verify(target):
                return [target]
            log(f"database candidate unreachable, dropped: {target_url(target)}")
        return []

    def _ping(self, target: DatabaseTarget) -> bool:
        out = self.sql("SELECT 1", target=target, timeout=8)
        return bool(re.search(r"^\s*1\s*$", out or "", re.MULTILINE))

    def _verify(self, target: DatabaseTarget) -> bool:
        """_ping, recorded. Only a target that answers SELECT 1 survives, so
        only a verified URL can ever reach the prompt."""
        ok = self._ping(target)
        self.attempts.append((target_url(target), ok))
        return ok

    # -- querying --------------------------------------------------------
    def available(self) -> bool:
        return bool(self.targets)

    def sql(self, query: str, *, target: DatabaseTarget | None = None, timeout: float = 60.0) -> str:
        """Run a read-only statement and return its text output (or an error)."""
        chosen = target or (self.targets[0] if self.targets else None)
        if chosen is None:
            trace("DatabaseProbe.sql", "out", query=query, rows=0, note="no target")
            return "[no database target discovered]"
        started = time.monotonic()
        out = (self._clickhouse(chosen, query, timeout) if chosen.engine == "clickhouse"
               else self._postgres(chosen, query, timeout))
        trace("DatabaseProbe.sql", "out", query=query, target=target_url(chosen),
              seconds=time.monotonic() - started, chars=len(out or ""),
              usable=self._usable(out))
        return out

    def _postgres(self, target: DatabaseTarget, query: str, timeout: float) -> str:
        dsn = target.dsn or (
            f"postgresql://{target.user}:{target.password}@{target.host}:{target.port}/{target.database}"
        )
        if shutil.which("psql"):
            result = run_command(["psql", dsn, "-X", "-A", "-F", " | ", "-c", query], timeout=timeout)
            if result.returncode == 0:
                return result.stdout
            error = result.stdout
        else:
            error = "[psql not installed]"
        return self._postgres_via_python(target, query, timeout) or error

    def _postgres_via_python(self, target: DatabaseTarget, query: str, timeout: float) -> str:
        script = f"""
import json, sys
try:
    import psycopg
    connect = psycopg.connect
except Exception:
    try:
        import psycopg2 as psycopg
        connect = psycopg.connect
    except Exception:
        sys.exit("[no postgres driver]")
with connect({target.dsn!r} or "dbname={target.database} user={target.user} "
             "password={target.password} host={target.host} port={target.port}") as conn:
    with conn.cursor() as cur:
        cur.execute({query!r})
        rows = cur.fetchall()
for row in rows[:200]:
    print(" | ".join("" if v is None else str(v) for v in row))
"""
        result = run_command([sys.executable, "-c", script], timeout=timeout)
        return result.stdout if result.returncode == 0 else ""

    def _clickhouse(self, target: DatabaseTarget, query: str, timeout: float) -> str:
        if shutil.which("clickhouse-client"):
            command = ["clickhouse-client", "--host", target.host, "--query", query]
            if target.user:
                command += ["--user", target.user]
            if target.password:
                command += ["--password", target.password]
            if target.database:
                command += ["--database", target.database]
            result = run_command(command, timeout=timeout)
            if result.returncode == 0:
                return result.stdout
        # HTTP interface, always present on a ClickHouse server.
        http_port = "8123" if target.port in ("9000", "9440", "") else target.port
        url = f"http://{target.host}:{http_port}/?database={target.database}"
        try:
            request = urllib.request.Request(url, data=query.encode("utf-8"), method="POST")
            if target.user:
                import base64
                token = base64.b64encode(f"{target.user}:{target.password}".encode()).decode()
                request.add_header("Authorization", f"Basic {token}")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except Exception as exc:
            return f"[clickhouse query failed: {exc}]"

    # -- structured evidence --------------------------------------------
    # Every failure path in sql() returns its complaint as text: "[psql not
    # installed]", "[no postgres driver]", "[clickhouse query failed: ...]".
    # Those are truthy, so they used to be rendered under a "Live schema
    # (columns and existing indexes)" header -- the prompt asserting it had
    # read the database when it had not.
    _QUERY_ERROR = re.compile(r"^\s*\[(?:no |psql |clickhouse |could not )", re.IGNORECASE)

    @classmethod
    def _usable(cls, output: str) -> bool:
        return bool(output and output.strip() and not cls._QUERY_ERROR.match(output))

    def schema_for(self, names: Iterable[str], limit: int = 8) -> str:
        """DDL-ish description of the tables the instruction talks about."""
        if not self.available():
            trace("schema_for", "out", chars=0, note="no verified database")
            return ""
        target = self.targets[0]
        wanted = [name for name in names if re.fullmatch(r"[A-Za-z_][\w]{2,}", name)][:limit]
        trace("schema_for", "in", asked=list(names)[:12], kept=wanted)
        if not wanted:
            return ""
        blocks: list[str] = []
        if target.engine == "postgresql":
            pattern = "|".join(re.escape(name.lower()) for name in wanted)
            columns = self.sql(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                f"WHERE table_schema='public' AND table_name ~ '{pattern}' "
                "ORDER BY table_name, ordinal_position LIMIT 400"
            )
            indexes = self.sql(
                f"SELECT tablename, indexname, indexdef FROM pg_indexes "
                f"WHERE schemaname='public' AND tablename ~ '{pattern}' ORDER BY tablename LIMIT 200"
            )
            if self._usable(columns):
                blocks.append("columns (table | column | type):\n" + truncate(columns, 4000))
            if self._usable(indexes):
                blocks.append("indexes (table | name | definition):\n" + truncate(indexes, 4000))
        else:
            for name in wanted[:4]:
                ddl = self.sql(f"SHOW CREATE TABLE {name}")
                if self._usable(ddl) and "failed" not in ddl[:40]:
                    blocks.append(truncate(ddl, 2500))
        rendered = "\n\n".join(blocks)
        trace("schema_for", "out", blocks=len(blocks), chars=len(rendered))
        return rendered

    def clickhouse_work(self, query: str) -> str:
        """Rows and bytes a ClickHouse query actually read.

        EXPLAIN alone does not say how much data was touched; system.query_log
        does, and read_rows/read_bytes are the ClickHouse equivalent of the
        buffer counts an optimisation task is graded on.
        """
        if not self.available() or self.targets[0].engine != "clickhouse":
            return ""
        marker = f"ridges_probe_{int(time.time()*1000)}"
        self.sql(f"SELECT 1 AS {marker} SETTINGS log_comment = '{marker}'")
        self.sql(f"{query} SETTINGS log_comment = '{marker}'")
        self.sql("SYSTEM FLUSH LOGS")
        stats = self.sql(
            "SELECT read_rows, read_bytes, result_rows, "
            "query_duration_ms, ProfileEvents['SelectedParts'] AS parts, "
            "ProfileEvents['SelectedMarks'] AS marks "
            "FROM system.query_log "
            f"WHERE log_comment = '{marker}' AND type = 'QueryFinish' "
            "ORDER BY event_time DESC LIMIT 1")
        first = stats.strip().splitlines()[0] if stats.strip() else ""
        # A permission or missing-table error comes back as text too; do not
        # hand the model an error message dressed up as a measurement.
        if not first or not re.match(r"^\s*\d+\s*\|", first):
            return ""
        return ("read_rows | read_bytes | result_rows | duration_ms | parts | marks\n"
                + truncate(stats, 800))

    def explain(self, query: str) -> str:
        if not self.available():
            return ""
        target = self.targets[0]
        if target.engine == "postgresql":
            return truncate(self.sql(f"EXPLAIN (ANALYZE, BUFFERS, SUMMARY OFF) {query}"), 4000)
        return truncate(self.sql(f"EXPLAIN indexes = 1 {query}"), 4000)


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def _diff_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _annotate_no_newline(diff: Iterable[str]) -> list[str]:
    """Insert git's '\\ No newline at end of file' markers where needed."""
    output: list[str] = []
    for line in diff:
        if line.startswith(("---", "+++", "@@", "diff ", "new file", "index ")):
            output.append(line if line.endswith("\n") else line + "\n")
            continue
        if line.endswith("\n"):
            output.append(line)
        else:
            output.append(line + "\n")
            output.append("\\ No newline at end of file\n")
    return output


def build_patch(repo: Repository, files: Sequence[str]) -> str:
    """A unified diff `git apply` accepts, built without a git repository."""
    parts: list[str] = []
    for relative in files:
        original = repo.original(relative)
        path = repo.root / relative
        try:
            current = read_source(path) if path.exists() else None
        except (OSError, UnicodeDecodeError) as exc:
            # exists() answered a moment ago and is not a promise: the file can
            # be unreadable, binary, or gone by the time it is opened. Losing
            # one file from the diff is bad; losing the whole run to an
            # exception raised while assembling it is worse.
            log(f"cannot read {relative} to diff it ({exc}); omitted from the patch")
            continue
        if current == original:
            continue

        if original is None:
            header = f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n"
            from_label, to_label = "/dev/null", f"b/{relative}"
            original = ""
        elif current is None:
            header = f"diff --git a/{relative} b/{relative}\ndeleted file mode 100644\n"
            from_label, to_label = f"a/{relative}", "/dev/null"
            current = ""
        else:
            header = f"diff --git a/{relative} b/{relative}\n"
            from_label, to_label = f"a/{relative}", f"b/{relative}"

        body = difflib.unified_diff(
            _diff_lines(original), _diff_lines(current),
            fromfile=from_label, tofile=to_label, n=3,
        )
        rendered = "".join(_annotate_no_newline(body))
        if rendered.strip():
            parts.append(header + rendered)
            trace("build_patch", "out", path=relative, bytes=len(header + rendered),
                  added=sum(1 for l in rendered.splitlines() if l.startswith("+")
                            and not l.startswith("+++")),
                  removed=sum(1 for l in rendered.splitlines() if l.startswith("-")
                              and not l.startswith("---")))
    patch = "".join(parts)
    trace("build_patch", "out", files=len(parts), bytes=len(patch))
    return patch


def verify_patch_applies(patch: str, repo: Repository) -> tuple[bool, str]:
    """Dry-run the patch against a pristine copy, the way the verifier will."""
    if not patch.strip():
        return False, "empty patch"
    if not shutil.which("git"):
        return True, "git unavailable; skipped apply check"

    staging = Path("/tmp/ridges-patch-check")
    touched = re.findall(r"^diff --git a/(\S+) b/\S+$", patch, re.MULTILINE)
    patch_file = Path("/tmp/ridges-candidate.diff")
    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for relative in touched:
            original = repo.original(relative)
            if original is None:
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_source(destination, original)
        patch_file.write_text(patch, encoding="utf-8")
    except OSError as exc:
        # /tmp full, read-only, or a path in the patch that cannot be staged.
        # This is a check, and a check that cannot run must not fail the run --
        # the same call it already makes when git itself is missing.
        shutil.rmtree(staging, ignore_errors=True)
        log(f"could not stage the patch for the apply check ({exc}); skipping it")
        return True, f"apply check could not run: {exc}"
    result = run_command(["git", "apply", "--check", "-v", str(patch_file)], cwd=staging, timeout=60)
    shutil.rmtree(staging, ignore_errors=True)
    trace("verify_patch_applies", "out", applies=result.returncode == 0, files=len(touched),
          detail=truncate(result.stdout, 200) if result.returncode else "")
    return result.returncode == 0, truncate(result.stdout, 1500)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


class Verifier:
    """Everything we can check locally before committing to a patch.

    Ordered cheapest-first so a syntax slip never costs a six-minute test run.
    """

    def __init__(self, repo: Repository, instruction: Instruction,
                 candidates: Sequence[str] = (), probe: "DatabaseProbe | None" = None) -> None:
        self.repo = repo
        self.instruction = instruction
        self.probe = probe
        # Files we independently judged relevant. When the instruction names no
        # editable path this is the only thing bounding the blast radius.
        self.candidates = list(candidates)

    def run_all(self, changed: Sequence[str], *, include_tests: bool = True) -> list[CheckResult]:
        trace("Verifier", "in", changed=changed, include_tests=include_tests,
              gates=[name for name, on in (("scope", True), ("protected", True), ("syntax", True),
                                           ("single-method", self.instruction.single_method),
                                           ("style", bool(self.instruction.style_constraints)),
                                           ("contract", self.instruction.single_method)) if on],
              commands=len(self.instruction.commands))
        results = [self.check_scope(changed)]
        results.append(self.check_protected(changed))
        results.append(self.check_syntax(changed))
        if self.instruction.single_method:
            results.append(self.check_single_method(changed))
        if self.instruction.style_constraints:
            results.append(self.check_style(changed))
        if self.instruction.single_method:
            results.append(self.check_verifier_contract(changed))
        if all(check.passed for check in results):
            results.extend(self.check_lint(changed))
            if include_tests:
                if not self.instruction.commands:
                    found = self.discovered_commands(changed)
                    if found:
                        log(f"instruction named no checks; running the repository's own: {found}")
                        self.instruction.commands = found
                results.extend(self.run_task_commands())
        for check in results:            # each gate's own verdict, in order
            _traced(check)
        trace("Verifier", "out",
              passed=[c.name for c in results if c.passed],
              failed=[c.name for c in results if not c.passed])
        return results

    # -- scope -----------------------------------------------------------
    # Never editable unless the instruction explicitly names them: touching any
    # of these fails source-tree conservation, whatever else is correct.
    # Two tiers, because they need different amounts of trust.
    #
    # ABSOLUTE: no instruction in this category legitimately asks for these to
    # change -- "you change the query, not the test" is the scoring rule, and
    # the verifier hashes every file it did not authorise. There is therefore
    # no exemption path at all, so no parsing mistake can open one. That
    # matters: `named_paths` collects every path-looking token in the prose
    # regardless of what its sentence says, so "Do not change tests/x.py" and
    # a `python manage.py test ...` invocation both used to land there and
    # both used to grant an exemption. Measured on the netbox samples: the
    # prefix-hierarchy statement writes its checks inline rather than fenced,
    # which put `netbox/manage.py` in named_paths and exempted it.
    #
    # CONDITIONAL: migrations and project config are legitimately edited when
    # a task asks for it (the cached-value-index sample is exactly that), but
    # only on an explicit permission -- edit_only or the lint command's target,
    # never a passing mention.
    _NEVER_EDIT_ABSOLUTE = re.compile(
        r"(^|/)(tests?|testing|spec|fixtures?|conftest\.py)(/|$)"
        r"|(^|/)test_[^/]*$|_test\.[a-z]+$"
        r"|(^|/)(setup|conftest|manage)\.py$"
        r"|\.lock$",
        re.IGNORECASE)
    _NEVER_EDIT_UNLESS_PERMITTED = re.compile(
        r"(^|/)migrations(/|$)|\.(cfg|ini)$", re.IGNORECASE)
    # Kept for callers that only ask "is this path sensitive at all".
    _NEVER_EDIT = re.compile(
        _NEVER_EDIT_ABSOLUTE.pattern + "|" + _NEVER_EDIT_UNLESS_PERMITTED.pattern,
        re.IGNORECASE)

    def allowed_paths(self) -> set[str] | None:
        if self.instruction.edit_only:
            return set(self.instruction.edit_only)
        if self.instruction.lint_paths:
            return set(self.instruction.lint_paths)
        # No named target: fall back to the files we ranked as candidates plus
        # anything the prose mentioned. Unbounded is not an option -- an edit
        # outside this set is far more likely to be a mistake than the fix.
        inferred = set(self.candidates) | set(self.instruction.named_paths)
        return inferred or None

    def check_protected(self, changed: Sequence[str]) -> CheckResult:
        # Only an explicit permission counts. A path the prose merely mentions
        # is evidence for the ranker, never authority to edit it.
        permitted = set(self.instruction.edit_only) | set(self.instruction.lint_paths)
        forbidden = [p for p in changed if self._NEVER_EDIT_ABSOLUTE.search(p)]
        unpermitted = [p for p in changed if p not in permitted
                       and self._NEVER_EDIT_UNLESS_PERMITTED.search(p)]
        if forbidden:
            return CheckResult(
                "protected paths", False,
                "tests, fixtures and project scripts are graded as untouchable however the "
                f"instruction is worded; these were modified: {forbidden}. Fix the production "
                "code instead -- a patch that edits a test scores zero for the whole problem.")
        if unpermitted:
            return CheckResult(
                "protected paths", False,
                "migrations and project config may only be changed when the instruction "
                f"explicitly permits that file; these were modified without one: {unpermitted}. "
                "Fix the production code instead.")
        return CheckResult("protected paths", True, "no test or fixture files touched")

    def check_scope(self, changed: Sequence[str]) -> CheckResult:
        allowed = self.allowed_paths()
        if allowed is None:
            return CheckResult("scope", True, f"changed: {list(changed)}")
        stray = [path for path in changed if path not in allowed]
        if stray:
            return CheckResult(
                "scope", False,
                f"the instruction permits edits only to {sorted(allowed)}, "
                f"but these files were modified: {stray}",
            )
        return CheckResult("scope", True, f"changed: {list(changed)}")

    def check_syntax(self, changed: Sequence[str]) -> CheckResult:
        problems = []
        for relative in changed:
            if not relative.endswith(".py"):
                continue
            text = self.repo.read(relative)
            if text is None:
                continue
            try:
                # parse() misses what only the compiler rejects -- a repeated
                # keyword argument reached ruff before this check caught it.
                compile(text, relative, "exec", dont_inherit=True)
            except SyntaxError as exc:
                problems.append(f"{relative}:{exc.lineno}: {exc.msg}")
            except (ValueError, RecursionError) as exc:
                problems.append(f"{relative}: {exc}")
        if problems:
            return CheckResult("syntax", False, "; ".join(problems))
        return CheckResult("syntax", True, "parsed")

    def check_single_method(self, changed: Sequence[str]) -> CheckResult:
        """Enforce 'change only that method': everything else byte-identical."""
        for relative in changed:
            if not relative.endswith(".py"):
                continue
            original = self.repo.original(relative)
            current = self.repo.read(relative)
            if original is None or current is None:
                continue
            try:
                current_defs = python_definitions(current)
            except Exception:
                continue

            original_lines = original.splitlines(keepends=True)
            current_lines = current.splitlines(keepends=True)
            matcher = difflib.SequenceMatcher(None, original_lines, current_lines, autojunk=False)
            edited = [op for op in matcher.get_opcodes() if op[0] != "equal"]
            if not edited:
                continue

            first_line = min(op[3] for op in edited) + 1
            last_line = max(op[4] for op in edited)
            enclosing = [
                (name, start, end) for name, start, end, kind in current_defs
                if kind != "class" and start <= first_line and last_line <= end
            ]
            if not enclosing:
                return CheckResult(
                    "single-method", False,
                    f"{relative}: edits at lines {first_line}-{last_line} fall outside a single "
                    "function body; the instruction bounds the change to one method, so imports "
                    "and every other line in the file must stay byte-identical",
                )
            name, start, end = min(enclosing, key=lambda item: item[2] - item[1])
            if original_lines[: start - 1] != current_lines[: start - 1]:
                return CheckResult("single-method", False,
                                   f"{relative}: source above {name} changed")
            before = next(((s0, e0) for n0, s0, e0, k0 in python_definitions(original)
                           if n0 == name and k0 != "class"), None)
            if before and original_lines[before[1]:] != current_lines[end:]:
                return CheckResult("single-method", False,
                                   f"{relative}: source below {name} changed")
        return CheckResult("single-method", True, "confined to one method")

    # Constructs the instruction rules out inside the edited region.  Cheap to
    # check here, expensive to discover from a verifier's AST audit.
    _STYLE_NODES: dict[str, tuple[type, ...]] = {
        "loops": (ast.For, ast.AsyncFor, ast.While),
        "comprehensions": (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        "lambdas": (ast.Lambda,),
        "exception handling": (ast.Try,),
        "context managers": (ast.With, ast.AsyncWith),
    }

    def check_style(self, changed: Sequence[str]) -> CheckResult:
        forbidden: tuple[type, ...] = tuple(
            node
            for label in self.instruction.style_constraints
            for node in self._STYLE_NODES.get(label, ())
        )
        problems: list[str] = []
        for relative in changed:
            if not relative.endswith(".py"):
                continue
            original = self.repo.original(relative) or ""
            current = self.repo.read(relative) or ""
            try:
                tree = ast.parse(current)
            except SyntaxError:
                continue
            edited = self._edited_line_range(original, current)
            if edited is None:
                continue
            first, last = edited
            for node in ast.walk(tree):
                line = getattr(node, "lineno", None)
                if line is None or not (first <= line <= last):
                    continue
                if forbidden and isinstance(node, forbidden):
                    problems.append(f"{relative}:{line}: {type(node).__name__}")
                if "raw SQL" in self.instruction.style_constraints and isinstance(node, ast.Name) \
                        and node.id in {"RawSQL", "raw"}:
                    problems.append(f"{relative}:{line}: raw SQL is ruled out for this change")
                if "materialising rows in Python" in self.instruction.style_constraints \
                        and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in {"list", "tuple", "set", "sorted"}:
                    problems.append(f"{relative}:{line}: {node.func.id}() pulls the rows into "
                                    "Python; the instruction requires the work stay in the database")
            if "names the file does not import" in self.instruction.style_constraints:
                problems.extend(self._undefined_names(relative, tree, first, last))
        if problems:
            return CheckResult(
                "style constraints", False,
                "the instruction forbids "
                f"{', '.join(self.instruction.style_constraints)} in the changed code, but found: "
                + "; ".join(sorted(set(problems))[:10]),
            )
        return CheckResult("style constraints", True,
                           f"none of {self.instruction.style_constraints} present")

    @staticmethod
    def _undefined_names(relative: str, tree: ast.AST, first: int, last: int) -> list[str]:
        """Names the edited region uses that this file does not have.

        "Use only names the file already imports" is stated in five of the six
        netbox samples and was checked by nothing: a model reaching for `Cast`
        or `Coalesce` that the file never imported produced a NameError at test
        time, or an added import that then failed the single-method gate. F401
        leads ERROR_HINTS for exactly this reason.

        Deliberately conservative -- it reports a name only when every binding
        site in the file has been ruled out, so a false positive would need the
        name to be genuinely absent. Attributes, keywords and locals are not
        names in this sense and are never reported.
        """
        bound: set[str] = set(dir(__builtins__) if isinstance(__builtins__, type(ast))
                              else __builtins__) | {"self", "cls", "__name__"}
        edited: list[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                if first <= node.lineno <= last or node.lineno <= first <= (node.end_lineno or 0):
                    args = node.args if not isinstance(node, ast.ClassDef) else None
                    for arg in (args.posonlyargs + args.args + args.kwonlyargs if args else []):
                        bound.add(arg.arg)
                    for extra in ((args.vararg, args.kwarg) if args else ()):
                        if extra:
                            bound.add(extra.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, (ast.comprehension,)):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            if getattr(node, "lineno", None) is not None and first <= node.lineno <= last:
                edited.append(node)

        missing: list[str] = []
        for node in edited:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
                message = (f"{relative}:{node.lineno}: `{node.id}` is not imported or defined in "
                           "this file, and the instruction allows only names it already has")
                if message not in missing:
                    missing.append(message)
        return missing

    @staticmethod
    def _edited_line_range(original: str, current: str) -> tuple[int, int] | None:
        matcher = difflib.SequenceMatcher(
            None, original.splitlines(keepends=True), current.splitlines(keepends=True), autojunk=False
        )
        edited = [op for op in matcher.get_opcodes() if op[0] != "equal"]
        if not edited:
            return None
        return min(op[3] for op in edited) + 1, max(max(op[4] for op in edited), 1)

    # The verifier audits the edited method structurally, and any violation is a
    # zero for the whole problem however correct the SQL is. These mirror the
    # checks in the sample verify.py and are applied whenever the change is
    # bounded to one method -- not only when the prose happens to mention them.
    _DANGEROUS_NAMES = {"__import__", "breakpoint", "compile", "eval", "exec",
                        "getattr", "globals", "locals", "open", "setattr", "vars"}
    _FORBIDDEN_NODES = (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete,
                        ast.Global, ast.Lambda, ast.Match, ast.Nonlocal, ast.While,
                        ast.Yield, ast.YieldFrom)
    _MAX_METHOD_BYTES = 5000
    _MAX_METHOD_NODES = 400

    def _method_violations(self, method: ast.FunctionDef) -> list[str]:
        """Forbidden constructs in a method body, as position-independent keys.

        Keys omit line numbers so the same construct in the original and the
        edited method compares equal even after lines shift.
        """
        found: list[str] = []
        for node in ast.walk(ast.Module(body=method.body, type_ignores=[])):
            if isinstance(node, ast.FunctionDef) and node is not method:
                found.append(f"nested function definition `{node.name}`")
            elif isinstance(node, self._FORBIDDEN_NODES):
                found.append(f"{type(node).__name__} is not allowed")
            elif isinstance(node, ast.ImportFrom):
                found.append(f"import inside the method: from {node.module} import "
                             f"{', '.join(a.name for a in node.names)}")
            elif isinstance(node, ast.Import):
                found.append(f"import inside the method: import "
                             f"{', '.join(a.name for a in node.names)}")
            elif isinstance(node, ast.Name) and (node.id in self._DANGEROUS_NAMES or "__" in node.id):
                found.append(f"forbidden name {node.id}")
            elif isinstance(node, ast.Attribute) and "__" in node.attr:
                found.append(f"forbidden attribute {node.attr}")
        return found

    @staticmethod
    def _method_named(tree: ast.AST, name: str) -> ast.FunctionDef | None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def check_verifier_contract(self, changed: Sequence[str]) -> CheckResult:
        problems: list[str] = []
        for relative in changed:
            if not relative.endswith(".py"):
                continue
            current = self.repo.read(relative) or ""
            original = self.repo.original(relative) or ""
            try:
                tree = ast.parse(current)
                original_tree = ast.parse(original)
            except SyntaxError:
                continue
            edited = self._edited_line_range(original, current)
            if edited is None:
                continue
            first, last = edited

            enclosing = [n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.lineno <= first and (n.end_lineno or n.lineno) >= last]
            if not enclosing:
                continue
            method = min(enclosing, key=lambda n: (n.end_lineno or 0) - n.lineno)

            body = "\n".join(current.splitlines()[method.lineno - 1 : method.end_lineno])
            if len(body.encode()) > self._MAX_METHOD_BYTES:
                problems.append(f"method is {len(body.encode())} bytes (limit {self._MAX_METHOD_BYTES})")
            nodes = list(ast.walk(ast.Module(body=method.body, type_ignores=[])))
            if len(nodes) > self._MAX_METHOD_NODES:
                problems.append(f"method has {len(nodes)} AST nodes (limit {self._MAX_METHOD_NODES})")

            # Only constructs the EDIT introduced count. The grader accommodates
            # what was already there (one task's verify.py skips the method's
            # first statement precisely because it is a pre-existing local
            # import); flagging pre-existing code sent a correct one-token fix
            # into three repair rounds and shipped a worse patch.
            before = self._method_named(original_tree, method.name)
            inherited = set(self._method_violations(before)) if before else set()
            for violation in self._method_violations(method):
                if violation not in inherited:
                    problems.append(f"{relative}: {violation}"
                                    + (" -- use only names the file already imports"
                                       if violation.startswith("import inside") else ""))
        if problems:
            return CheckResult(
                "verifier contract", False,
                "the grader rejects the patch outright for these, however correct the query is: "
                + "; ".join(sorted(set(problems))[:8]))
        return CheckResult("verifier contract", True, "method body within the graded limits")

    def check_lint(self, changed: Sequence[str]) -> list[CheckResult]:
        python_files = [path for path in changed if path.endswith(".py")]
        if not python_files or not shutil.which("ruff"):
            return []
        result = run_command(["ruff", "check", "--no-cache", *python_files], timeout=120)
        return [CheckResult(
            "ruff", result.returncode == 0, truncate(result.stdout, 2000),
        )]

    # -- measuring database work -----------------------------------------
    # Optimisation is graded on the work the query performs, and the hidden
    # tests measure query count with CaptureQueriesContext at two selection
    # sizes. Passing a test suite says nothing about that, so measure it here:
    # a fix that still scales with input is a zero we can detect ourselves.
    _DJANGO_PROBE = """
import json
from django.db import connection
from django.test.utils import CaptureQueriesContext
{setup}
counts = {{}}
for _n in ({small}, {large}):
    N = _n
    with CaptureQueriesContext(connection) as _ctx:
        {call}
    counts[_n] = len(_ctx)
print("RIDGES_QC" + json.dumps(counts))
print("RIDGES_QS" + json.dumps([q["sql"] for q in _ctx.captured_queries[:6]]))
"""

    def measure_query_scaling(self, probe: dict) -> CheckResult | None:
        """Run the model's probe at two sizes and compare query counts."""
        setup = (probe.get("setup") or "").strip()
        call = (probe.get("call") or "").strip()
        if not call:
            return None
        manage = next((p for p in self.repo.files
                       if Path(p).name == "manage.py" and p.count("/") <= 2), None)
        if not manage:
            return None
        try:
            small = int(probe.get("small") or 1)
            large = int(probe.get("large") or 10)
        except (TypeError, ValueError):
            small, large = 1, 10

        script = self._DJANGO_PROBE.format(
            setup="\n".join(f"{line}" for line in setup.splitlines()),
            call="\n        ".join(call.splitlines()),
            small=small, large=large)
        result = run_command([app_python(self.repo.root), manage, "shell", "-c", script],
                             timeout=min(300.0, max(60.0, remaining_seconds() - 200)))
        match = re.search(r"RIDGES_QC(\{.*\})", result.stdout or "")
        if not match:
            return CheckResult("query scaling", True,
                               f"probe did not report a count; treating as unmeasured. "
                               f"{truncate(result.stdout, 600)}")
        try:
            counts = {int(k): int(v) for k, v in json.loads(match.group(1)).items()}
        except (ValueError, json.JSONDecodeError):
            return CheckResult("query scaling", True, "probe output unreadable")

        trace("measure_query_scaling", "out", counts=counts,
              limit=self.instruction.targets.get("max_queries"))
        low, high = counts.get(small), counts.get(large)
        if low is None or high is None:
            return CheckResult("query scaling", True, f"incomplete measurement: {counts}")
        limit = self.instruction.targets.get("max_queries")
        failure = None
        if limit and high > limit:
            failure = (f"the instruction allows at most {limit} queries; measured {low} at "
                       f"N={small} and {high} at N={large}.")
        # Bounded means the count does not grow with the selection. Allow one
        # extra statement for a larger IN list or an added round trip.
        elif high > low + 1:
            failure = (f"query count still grows with input: {low} queries at N={small}, "
                       f"{high} at N={large}. The work must be bounded -- fold the per-item "
                       f"statements into one set-based query.")
        if failure:
            return CheckResult("query scaling", False, failure + self._explain_captured(result.stdout or ""))
        return CheckResult("query scaling", True,
                           f"bounded: {low} queries at N={small}, {high} at N={large}"
                           + (f" (limit {limit})" if limit else ""))

    def _explain_captured(self, stdout: str) -> str:
        """The plan of the statements the probe captured, so a scaling failure
        arrives with the evidence the model would otherwise have to ask for."""
        match = re.search(r"RIDGES_QS(\[.*\])", stdout)
        if not match or self.probe is None or not self.probe.available():
            return ""
        try:
            statements = json.loads(match.group(1))
        except json.JSONDecodeError:
            return ""
        shown: list[str] = []
        for sql in statements[:3]:
            if is_read_only_sql(sql):
                shown.append(f"$ EXPLAIN {truncate(sql, 300)}\n{truncate(self.probe.explain(sql), 1500)}")
        return ("\n\nStatements issued at N=large, with their plans:\n" + "\n\n".join(shown)) if shown else ""

    # -- discovering tests when the instruction names none ---------------
    def discovered_commands(self, changed: Sequence[str]) -> list[str]:
        """Find the repository's own tests for the code we just changed.

        An instruction may name its checks in a format we cannot parse, in
        another language, or not at all. The repository always knows how to test
        itself, so derive the command from the project layout instead of from
        prose. Narrow to the package containing the edit: running everything is
        usually too slow for the agent's time budget.
        """
        root = self.repo.root
        for relative in changed:
            path = Path(relative)
            parts = path.parts

            # Django: <...>/<app>/<module>.py next to a manage.py
            manage = next((p for p in self.repo.files
                           if Path(p).name == "manage.py" and p.count("/") <= 2), None)
            if manage and path.suffix == ".py":
                app = next((parts[i] for i in range(len(parts) - 2, -1, -1)
                            if (root / Path(*parts[: i + 1]) / "tests").is_dir()
                            or (root / Path(*parts[: i + 1]) / "tests.py").is_file()), None)
                if app:
                    return [f"{app_python(root)} {manage} test {app} --keepdb --noinput"]

            # pytest: the nearest tests/ directory above the changed file
            if path.suffix == ".py":
                for i in range(len(parts) - 1, 0, -1):
                    candidate = root / Path(*parts[:i]) / "tests"
                    if candidate.is_dir():
                        return [f"{app_python(root)} -m pytest {candidate.relative_to(root)} -x -q"]

            # Go / Rust / Ruby, by the package or crate holding the change
            if path.suffix == ".go":
                return [f"go test ./{path.parent.as_posix()}/..."]
            if path.suffix == ".rs":
                return ["cargo test"]
            if path.suffix == ".rb":
                return ["bundle exec rspec"]
            if path.suffix in (".ex", ".exs") and (root / "mix.exs").is_file():
                return ["mix test"]
            # JavaScript / TypeScript: whatever the project's own test script is
            if path.suffix in (".js", ".ts", ".tsx", ".jsx") and (root / "package.json").is_file():
                try:
                    scripts = json.loads((root / "package.json").read_text()).get("scripts", {})
                except (OSError, json.JSONDecodeError):
                    scripts = {}
                if "test" in scripts:
                    return ["npm test --silent"]
                return ["node --test"]
            if path.suffix in (".java", ".kt"):
                if (root / "pom.xml").is_file():
                    return ["mvn -q test"]
                if (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
                    return ["gradle test -q"]
            if path.suffix == ".php" and (root / "vendor/bin/phpunit").exists():
                return ["vendor/bin/phpunit"]
            if path.suffix == ".cs":
                return ["dotnet test"]
        return []

    # -- the task's own commands ----------------------------------------
    def selected_commands(self) -> list[str]:
        commands = []
        for command in self.instruction.commands:
            if command.startswith("ruff "):
                continue  # already covered, and cheaper, in check_lint
            commands.append(command)
        return commands

    def run_task_commands(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for command in self.selected_commands():
            if remaining_seconds() < 180:
                results.append(CheckResult(f"$ {command}", True, "skipped: out of time"))
                continue
            log(f"running task check: {command}")
            result = run_command(command, timeout=min(600.0, max(60.0, remaining_seconds() - 120)))
            passed = result.returncode == 0
            results.append(CheckResult(f"$ {command}", passed, truncate(result.stdout, 6000, head_ratio=0.25)))
            if not passed:
                break  # the first failure is the one worth reporting
        return results


def _traced(check: CheckResult) -> CheckResult:
    """A gate's verdict as it is produced, with the detail it would have to
    explain to the model. Aggregated pass/fail hides which rule fired."""
    trace("gate", "out", name=check.name, passed=check.passed,
          detail="" if check.passed else check.detail)
    return check


def summarise(checks: Sequence[CheckResult]) -> str:
    lines = []
    for check in checks:
        lines.append(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}")
        if not check.passed and check.detail:
            lines.append(truncate(check.detail, 5000, head_ratio=0.3))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

# The four elements a prompt is built from (promptingguide.ai/introduction/elements)
# map onto this agent's turns as:
#
#   instruction       DIRECTIVE, below -- the procedure, stated once, up front
#   context           what this agent found: verified database URL, ranked shortlist
#   input data        the task statement verbatim, the source slices, the live schema
#   output indicator  EDIT_PROTOCOL, last -- the exact shape of the reply
#
# Three rules from that guide shape the wording throughout, and each is easy to
# undo by accident:
#
#   * Instructions go first and lead with a command verb. The task statement is
#     input data, not the instruction: it says what is wrong with the
#     application, never what to do with this prompt.
#   * State what to do, not what to avoid. Prohibitions are reported to measure
#     against ("DO NOT ASK FOR INTERESTS" backfires); the desired behaviour and
#     its fallback are what actually steer the reply.
#   * Replace vague limits with exact ones -- "keep it short" becomes a
#     sentence count -- because a limit the model cannot measure is not a limit.

SYSTEM_PROMPT = """\
You are a database query engineer. You fix, author, and optimise the queries a \
real application issues against PostgreSQL or ClickHouse, working inside the \
application's own repository: raw SQL, ORM code, or query-builder code.

How you work:

* Edit production query code, and write the fix so it holds for data you have \
never seen. Correctness is graded on a hidden dataset, so implement the general \
rule the task states, in terms of the columns and relations it names.
* Reduce the database work the query performs -- statements issued, rows and \
buffers touched, index usage. That measurement is the grade, so remove work the \
query genuinely does not need.
* Keep everything outside the blast radius the instruction sets byte-identical, \
imports included, and build the fix from names already in scope.
* Make the smallest change that fixes the underlying cause.

Reasoning you should apply, by symptom:

* Work that grows with input size -- a statement per element, per row, or per \
iteration -- becomes one set-based statement: a single bulk insert/update, one \
`IN`/`ANY` predicate, a join, or a CTE. Compute the set difference in the \
database, and keep any signal/callback contract firing exactly once with the \
same payload.
* A slow or unselective plan usually means the predicate the application \
actually issues is not the one the index serves. Match index column order and \
partiality to the real predicate, including equality columns first.
* Wrong aggregates over a hierarchy or a many-to-many usually mean fan-out: \
rows multiplied by a join. Fix it with DISTINCT on the counted key, a \
subquery/lateral, or a nested-set/recursive descendant predicate, keeping the \
correction in the database rather than in application code.
* Percentages and ratios should be computed in the database with explicit \
numeric casting and a zero-denominator guard.
* On ClickHouse, favour the primary key order and PREWHERE, prefer set-based \
expressions over per-row subqueries, and remember that JOIN semantics and \
nullability differ from PostgreSQL. An `explain` request there also returns \
read_rows, read_bytes, selected parts and marks from system.query_log -- that \
is the number an optimisation is graded on, so check it fell. Graders measure a \
report's statements in system.query_log and count only statements that name no \
`system.` table, so build any series with numbers(N) or arrayJoin(range(...)).

You answer only with a single JSON object, described in the user message."""

# The instruction element: first in the user turn, one command verb per step.
# The statement that follows it is input data -- it describes a defect in an
# application and knows nothing about this protocol -- so the procedure has to
# be stated here or it is not stated at all. Before this existed the only
# directive in the prompt was EDIT_PROTOCOL, which the model reached some
# fifteen thousand characters after the material it governs.
DIRECTIVE = """\
# Instructions

Diagnose the database defect described in the task statement below, then reply \
with one JSON object in the format given at the end of this message.

Work in this order:

1. Read the task statement. It is the authority on what to change, which files \
you may edit, and which checks to run.
2. Locate the code that issues the query in question, using the source \
provided below.
3. Name the database-level cause in at most two sentences.
4. Write the smallest edit that fixes that cause.
5. Reply with the JSON object.

When the statement does not name the file to edit and the provided source does \
not settle which file issues the query, reply with the `need_context` object \
and request what would settle it. Request context whenever you are unsure \
rather than editing a file you have not read."""

EDIT_PROTOCOL = """\
Reply with ONE JSON object and nothing else. Begin the reply with `{` and end \
it with `}`. Two shapes are allowed.

To gather more evidence before deciding (the agent tells you how many rounds remain; \
an unnamed target allows more than a named one, and each failed attempt grants another):

{"action": "need_context",
 "why": "<one sentence>",
 "requests": [
   {"kind": "read_file", "path": "<repo-relative path>", "start": 1, "end": 200},
   {"kind": "grep", "pattern": "<python regex>", "path_filter": "<optional substring>"},
   {"kind": "sql", "query": "<read-only statement to run against the live database>"},
   {"kind": "explain", "query": "<SELECT ... to EXPLAIN on the live database>"},
   {"kind": "read_file", "path": "<repo-relative path>", "symbol": "<def or class name: returns that definition whole>"},
   {"kind": "schema", "tables": ["<table name>", "..."]},
   {"kind": "callers", "symbol": "<function or class name: who defines and who references it>"},
   {"kind": "count", "setup": "<Django shell setup>", "call": "<one line using N>", "small": 1, "large": 10}
 ]}
`count` measures the query count at two sizes BEFORE you edit -- use it on \
bounded-work tasks so you know the number you are trying to change.

To make the change:

{"action": "edit",
 "diagnosis": "<the database-level cause, one or two sentences>",
 "verify": ["<any shell command the instruction says to run before finishing, copied verbatim; [] if it names none>"],
 "constraints": {"editable_files": ["<repo-relative paths the instruction allows you to change>"],
                 "bounded_to_method": "<Class.method the instruction restricts the change to, or null>"},
 "edits": [
   {"path": "<repo-relative path>",
    "search": "<exact contiguous text from the current file, unique within it>",
    "replace": "<replacement text>"}
 ]}

A complete `edit` reply, to copy the shape of:

{"action": "edit",
 "diagnosis": "share_pct divides two integer columns, so the fraction is truncated before Round() runs.",
 "verify": ["python manage.py test shop.tests.test_reports --keepdb --noinput"],
 "constraints": {"editable_files": ["shop/reports/querysets.py"],
                 "bounded_to_method": "OrderQuerySet.annotate_share"},
 "edits": [
   {"path": "shop/reports/querysets.py",
    "search": "        return self.annotate(\\n            share_pct=Round(F('paid') * 100 / F('total'), 2),",
    "replace": "        return self.annotate(\\n            share_pct=Round(F('paid') * 100.0 / F('total'), 2),"}
 ]}

Rules for edits:
* `search` must reproduce the existing file byte for byte, including \
indentation. Include 2 to 5 surrounding lines, enough to appear exactly once in \
the file.
* Give the smallest `search`/`replace` pair that expresses the change -- one \
edit per distinct change, each covering the lines that change plus that much \
context.
* Change only what the fix requires. Leave docstrings, comments, formatting, \
blank lines and import order exactly as they are unless the task asks for them \
to change -- an unnecessary edit is a way to fail a scope check, never a way to \
pass one.
* Keep `diagnosis` to at most 2 sentences and the whole reply under 2000 \
characters unless the edit itself is longer. Reason as far as naming the cause \
and writing the edit; deliberation past that point is billed and is not read.
* `constraints` is read only when the instruction named no file and no method: \
state what it DOES allow, exactly as written. The agent enforces it against your \
own edits, so claim only what the instruction grants.
* After a failed attempt you may request context again -- the failure output \
usually points at something worth reading before the next edit.
* `verify` matters: those commands are run against the live database and their \
output comes back to you if they fail. Copy every check the instruction names, \
wherever it states them -- fenced block, inline text, or prose.
* When the task is about work that must not grow with input size, add a probe \
so the agent can measure it before submitting:
    "measure": {"setup": "<imports and fixture creation, Django shell>",
                "call": "<one line exercising the change, using N as the size>",
                "small": 1, "large": 10}
  Use `N` as the selection size in `call`. The agent runs it at both sizes and \
tells you the query counts; if they grow with N the fix is not bounded.
* To create a new file instead, use {"path": ..., "new_file": true, \
"content": "<full text>"}.
* Escape newlines properly -- the whole reply must parse as JSON."""


class PromptBuilder:
    """Assembles the first user turn: instruction, findings, code, schema.

    One method per section, and a section that has nothing to say returns "".
    That shape is the point: what reaches the model is now enumerable, and the
    reason each section exists can sit on the method that produces it.

    The governing rule is that the instruction speaks for itself. It is printed
    whole and first, and nothing below it restates a rule it already states --
    measured on the six netbox samples, the old fact list restated nine things
    the prose said, one of them less accurately than the prose said it. What
    remains is what the statement cannot contain: what this agent found by
    reading the repository and connecting to the database, and how to drive
    machinery the statement knows nothing about.
    """

    def __init__(self, repo: Repository, instruction: Instruction,
                 candidates: Sequence[tuple[str, list[int]]], probe: DatabaseProbe) -> None:
        self.repo = repo
        self.instruction = instruction
        self.candidates = candidates
        self.probe = probe
        # Whether the statement pointed at a file. It decides how much of each
        # candidate is worth showing and whether the shortlist needs explaining.
        self.named_target = bool(instruction.edit_only or instruction.lint_paths)

    # Characters per token, the divisor affordable_cap() prices calls with.
    CHARS_PER_TOKEN = 3.5

    # Between elements. Costs three tokens and removes the one ambiguity a
    # heading cannot: whether a `#` line belongs to the task statement or to
    # this agent.
    SEPARATOR = "###"

    def build(self) -> str:
        trace("PromptBuilder", "in", candidates=[path for path, _ in self.candidates],
              named_target=self.named_target, database=self.probe.available())
        named = ("directive", "task statement", "what this agent found",
                 "relevant source", "live schema")
        sections = (DIRECTIVE, self._task_instruction(), self._agent_findings(),
                    self._relevant_source(), self._live_schema())
        # Sections are separated by a rule as well as a heading. A statement
        # can contain any markdown it likes, fenced blocks and headings
        # included, so a heading alone does not reliably mark where the
        # statement stops and this agent's own findings start.
        prompt = f"\n\n{self.SEPARATOR}\n\n".join(s for s in sections if s)
        # Section by section, because "the prompt is too long" is never
        # actionable until you know which part of it is long.
        for label, section in zip(named, sections):
            if section:
                trace("PromptBuilder", "out", section=label, chars=len(section),
                      tokens=int(len(section) / self.CHARS_PER_TOKEN),
                      share=len(section) / max(1, len(prompt)))
        trace("PromptBuilder", "out", section="TOTAL", chars=len(prompt),
              tokens=int(len(prompt) / self.CHARS_PER_TOKEN))
        return prompt

    def _task_instruction(self) -> str:
        """The statement, verbatim and unabridged. The model's authority.

        Input data, not the instruction element: it describes a defect in an
        application and says nothing about this prompt or the reply it wants.
        The heading says so, because a statement that opens with its own `#
        Repair ...` title otherwise reads as the top of the document.
        """
        return f"# Task statement (the authority on what to change)\n\n{self.instruction.text.strip()}"

    def _agent_findings(self) -> str:
        """Only what the statement cannot know.

        Everything else -- the editable file, the bounded method, the
        forbidden constructs -- is parsed to ENFORCE it, not to retell it.
        """
        facts: list[str] = []
        if self.probe.available():
            # Only a target that answered SELECT 1 survives discovery, so this
            # URL is a connection this agent made, not a scraped guess.
            facts.append(f"live database (this agent connected to it and it answered "
                         f"SELECT 1): {target_url(self.probe.targets[0])}")
        if not (self.instruction.edit_only or self.instruction.lint_paths
                or self.instruction.named_paths):
            # The one case where the statement is genuinely silent: it names no
            # file, so the ranked shortlist below is the only guidance there is.
            facts.append("the instruction names no file to edit: the candidates below are this "
                         "agent's ranking, not the task's -- identify which one actually issues "
                         "the query, and fill `constraints` with what the instruction does permit")
        if ("bounded_queries" in self.instruction.kinds
                or self.instruction.targets.get("max_queries")):
            facts.append("supply a `measure` probe with your edit: this agent runs it at two "
                         "selection sizes and checks the query count before submitting, so a fix "
                         "that still scales with input is caught here rather than by the grader")
        if self.instruction.targets.get("create_paths"):
            facts.append(f"these paths named by the instruction do not exist yet: "
                         f"{self.instruction.targets['create_paths']} -- create them with a "
                         f"new_file edit")
        if not facts:
            return ""
        return ("# What this agent found (not stated in the instruction)\n\n"
                + "\n".join(f"- {fact}" for fact in facts))

    def _budget_for(self, position: int, length: int) -> int:
        """How many lines of this candidate are worth sending.

        Context is the dominant cost of a run. When the statement names the
        file, spend the budget there. When it does not, rank 1 is only a guess,
        so budget by size instead: query code usually lives in compact
        managers and querysets, and showing a short file whole costs less than
        a slice of a long one.
        """
        if self.named_target:
            return 320
        if length <= 140:
            return length + 10           # small enough to show entirely
        return 160 if position == 0 else 40

    def _relevant_source(self) -> str:
        """The code itself, with an outline and the target's neighbours."""
        blocks: list[str] = []
        for position, (relative, hot) in enumerate(self.candidates):
            length = len((self.repo.read(relative) or "").splitlines())
            budget = self._budget_for(position, length)
            outline = file_outline(self.repo, relative, limit=80 if self.named_target else 20)
            if outline:
                blocks.append(outline)
                trace("file_outline", "out", rank=position + 1, path=relative, chars=len(outline))
            if position == 0:
                neighbours = package_map(self.repo, relative)
                if neighbours:
                    blocks.append(neighbours)
                    trace("package_map", "out", path=relative, chars=len(neighbours))
            pieces = slice_around(self.repo, relative, hot or [], self.instruction,
                                  budget_lines=budget)
            kept = pieces if budget > 40 else pieces[:1]
            for piece in kept:
                blocks.append(piece.render(self.repo))
            trace("slice_around", "out", rank=position + 1, path=relative, file_lines=length,
                  budget=budget, slices=[f"{s.start}-{s.end} {s.label}" for s in kept],
                  shown=sum(s.end - s.start + 1 for s in kept),
                  chars=sum(len(b) for b in blocks[-len(kept):]) if kept else 0)
        return f"{self._source_header()}\n\n" + "\n\n".join(blocks)

    def _source_header(self) -> str:
        header = "# Relevant source"
        if self.instruction.traced_hint:
            header += ("\n\nReference tracing from the task's vocabulary also reached these files, "
                       "not shown below; request one with need_context if the candidates above "
                       f"do not contain the query: {self.instruction.traced_hint}")
        if not self.named_target and len(self.candidates) > 1:
            header += (
                "\n\nThe instruction does not name a file. These are the strongest candidates, "
                "best first; identify which one actually issues the query in question -- request "
                "more of it with need_context if the excerpt is not enough."
            )
        return header

    def _live_schema(self) -> str:
        """Columns and existing indexes, read from the database itself.

        schema_for returns "" when the lookup failed, so a connection error is
        never dressed up under this header.
        """
        schema = self.probe.schema_for(
            _table_candidates(self.repo, self.instruction, self.candidates))
        return ("# Live schema (columns and existing indexes)\n\n" + schema) if schema else ""


def render_evidence(
    repo: Repository,
    instruction: Instruction,
    candidates: Sequence[tuple[str, list[int]]],
    probe: DatabaseProbe,
) -> str:
    """The first user turn. See PromptBuilder."""
    return PromptBuilder(repo, instruction, candidates, probe).build()


def _table_candidates(repo: Repository, instruction: Instruction,
                      candidates: Sequence[tuple[str, list[int]]]) -> list[str]:
    """Table-ish names to look up: from the instruction and from the code."""
    names: list[str] = []
    for relative, _ in candidates:
        text = repo.read(relative) or ""
        names.extend(re.findall(r"\bFROM\s+([A-Za-z_][\w.]*)", text, re.IGNORECASE))
        names.extend(re.findall(r"\bJOIN\s+([A-Za-z_][\w.]*)", text, re.IGNORECASE))
        names.extend(re.findall(r"\b(?:db_table|table_name)\s*=\s*['\"]([\w.]+)['\"]", text))
    names.extend(instruction.identifiers)
    ordered: list[str] = []
    for name in names:
        clean = name.split(".")[-1].strip('"')
        if clean and clean not in ordered:
            ordered.append(clean)
    return ordered[:12]


def extract_json(content: str) -> dict | None:
    """Pull the first JSON object out of a model reply, fences or not."""
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
    candidates.extend(fenced)
    candidates.append(content)
    for text in candidates:
        text = text.strip()
        start = text.find("{")
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start : index + 1])
                        trace("extract_json", "out", ok=True, keys=sorted(payload)
                              if isinstance(payload, dict) else None, reply_chars=len(content))
                        return payload
                    except json.JSONDecodeError:
                        break
    trace("extract_json", "out", ok=False, reply_chars=len(content),
          head=content[:120])
    return None


_READ_ONLY_START = re.compile(
    r"^\s*(?:\(\s*)*(SELECT|WITH|EXPLAIN|SHOW|DESCRIBE|DESC|TABLE|VALUES)\b", re.IGNORECASE)
_MUTATING = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE|CREATE|ALTER|DROP|TRUNCATE|RENAME|GRANT|REVOKE"
    r"|COPY|VACUUM|ANALYZE\s+\w|REINDEX|CLUSTER|LOCK|SET\s+(?!TRANSACTION)|RESET|DO|CALL|EXECUTE"
    r"|OPTIMIZE|ATTACH|DETACH|KILL|SYSTEM|INTO\s+OUTFILE|pg_terminate|pg_cancel|lo_)\b",
    re.IGNORECASE)


def is_read_only_sql(query: str) -> bool:
    """Only statements that cannot change data or schema may run during investigation.

    An allowlist on the leading keyword, plus a denylist for anything mutating
    smuggled inside (a CTE with a data-modifying statement, SELECT ... INTO,
    a semicolon-separated second statement). The database is disposable, but a
    write here would corrupt the agent's own subsequent test run.
    """
    body = re.sub(r"--[^\n]*|/\*.*?\*/", " ", query, flags=re.DOTALL).strip()
    if not body or ";" in body.rstrip(";"):
        return False
    if not _READ_ONLY_START.match(body):
        return False
    # SHOW / DESCRIBE never mutate, and "SHOW CREATE TABLE" would otherwise trip
    # the CREATE check below.
    if re.match(r"^\s*(SHOW|DESCRIBE|DESC)\b", body, re.IGNORECASE):
        return True
    return not _MUTATING.search(body)


def _request_key(kind: str, request: dict) -> str:
    fields = {k: v for k, v in request.items() if k != "kind"}
    return kind + ":" + json.dumps(fields, sort_keys=True, default=str)


def fulfil_requests(repo: Repository, probe: DatabaseProbe, requests: Sequence[dict],
                    graph: "CallGraph | None" = None, verifier: "Verifier | None" = None,
                    served: "set[str] | None" = None) -> str:
    """Answer the model's context requests deterministically and cheaply.

    `served` remembers what earlier rounds already returned. Asking for the
    same thing twice yields "already shown" instead of a second copy: the
    round is still spent, so a model that loops runs out of rounds rather than
    running up the bill.
    """
    trace("fulfil_requests", "in", asked=len(requests),
          kinds=[(r.get("kind") or "?") for r in requests[:6]])
    blocks: list[str] = []
    for request in requests[:6]:
        kind = (request.get("kind") or "").lower()
        key = _request_key(kind, request)
        if served is not None:
            if key in served:
                blocks.append(f"{kind}: already shown in an earlier round -- it has not changed; "
                              "request something new or reply with the edit")
                continue
            served.add(key)
        try:
            if kind == "callers":
                # The call graph, on request: who defines and who references a
                # symbol. Used this way it expands what the model can see without
                # perturbing the ranking -- the integration that measured well.
                symbol = (request.get("symbol") or "").strip()
                if not symbol or graph is None:
                    blocks.append(f"callers: {'no symbol given' if not symbol else 'unavailable'}")
                    continue
                defined = graph.defined_in.get(symbol, [])
                where = [f"{f}:{line}" for f, line in graph.defined_at.get(symbol, [])[:6]]
                referenced = sorted(f for f, names in graph.mentions.items()
                                    if symbol in names and f not in defined)
                blocks.append(f"callers {symbol}:\n  defined at: {where or 'nowhere found'}"
                              f"\n  referenced by {len(referenced)} file(s): {referenced[:20]}")
                continue
            if kind == "count":
                # Query count at two sizes, BEFORE editing -- the baseline number
                # the model should be trying to move.
                if verifier is None:
                    blocks.append("count: unavailable"); continue
                measured = verifier.measure_query_scaling(request)
                blocks.append("count: " + (measured.detail if measured else
                              "needs a Django project and a `call` using N"))
                continue
            if kind == "schema":
                tables = request.get("tables") or request.get("table") or []
                if isinstance(tables, str):
                    tables = [tables]
                detail = probe.schema_for(tables, limit=8) if probe and tables else ""
                blocks.append("schema " + ", ".join(map(str, tables)) + ":\n"
                              + (detail or "[no database, or no such tables]"))
                continue
            if kind == "read_file":
                relative = normalize_repo_path(request.get("path") or "") or ""
                text = repo.read(relative) if relative else None
                if text is None:
                    blocks.append(f"read_file {relative}: not found")
                    continue
                lines = text.splitlines()
                symbol = (request.get("symbol") or "").strip()
                if symbol:
                    # The definition, whole, without the model guessing line numbers.
                    defs = python_definitions(text) if relative.endswith(".py") else generic_blocks(text)
                    hit = next(((st, en) for name, st, en, kind_ in defs
                                if kind_ != "class" and name.split(".")[-1] == symbol), None) \
                        or next(((st, en) for name, st, en, kind_ in defs
                                 if re.search(rf"\b{re.escape(symbol)}\b", name)), None)
                    if hit is None:
                        blocks.append(f"read_file {relative}: no definition named {symbol!r}; "
                                      f"definitions here: {[d[0] for d in defs][:30]}")
                        continue
                    request = dict(request, start=max(1, hit[0] - 3), end=min(len(lines), hit[1] + 3))
                start = max(1, int(request.get("start") or 1))
                end = min(len(lines), int(request.get("end") or min(len(lines), start + 200)))
                body = "\n".join(f"{number:5d}| {lines[number - 1]}" for number in range(start, end + 1))
                blocks.append(f"read_file {relative} lines {start}-{end}:\n{truncate(body, 12000)}")
            elif kind == "grep":
                pattern = request.get("pattern") or ""
                path_filter = request.get("path_filter") or ""
                compiled = re.compile(pattern)
                hits: list[str] = []
                for relative in repo.files:
                    if path_filter and path_filter not in relative:
                        continue
                    text = repo.read(relative)
                    if text is None or not compiled.search(text):
                        continue
                    for number, line in enumerate(text.splitlines(), start=1):
                        if compiled.search(line):
                            hits.append(f"{relative}:{number}: {line.strip()[:200]}")
                            if len(hits) >= 60:
                                break
                    if len(hits) >= 60:
                        break
                blocks.append(f"grep {pattern!r}:\n" + ("\n".join(hits) or "no matches"))
            elif kind in ("sql", "explain"):
                query = (request.get("query") or "").strip().rstrip(";")
                if not is_read_only_sql(query):
                    blocks.append(f"{kind}: refused -- investigation queries must be read-only "
                                  "(SELECT / WITH ... SELECT / EXPLAIN / SHOW / DESCRIBE)")
                    continue
                if kind == "explain":
                    output = probe.explain(query)
                    work = probe.clickhouse_work(query)
                    if work:
                        output += "\n\ndata actually read (system.query_log):\n" + work
                else:
                    output = truncate(probe.sql(query), 4000)
                blocks.append(f"{kind} {truncate(query, 400)}:\n{output or '[no output]'}")
            else:
                blocks.append(f"unsupported request kind: {kind!r}")
        except Exception as exc:
            blocks.append(f"{kind} request failed: {exc}")
        trace("fulfil_requests", "out", kind=kind, chars=len(blocks[-1]) if blocks else 0)
    answer = "\n\n".join(blocks) if blocks else "no context returned"
    # This is appended to the message stack and re-sent on every later call
    # until _compact_old_context shrinks it, so its size is a running cost.
    trace("fulfil_requests", "out", blocks=len(blocks), chars=len(answer))
    return answer


def apply_edits(repo: Repository, edits: Sequence[dict]) -> tuple[list[str], list[str]]:
    """Apply search/replace edits. Returns (changed_files, errors)."""
    trace("apply_edits", "in", edits=len(edits),
          paths=[e.get("path") for e in edits],
          bytes=sum(len(str(e.get("replace", "") or e.get("content", ""))) for e in edits))
    changed: list[str] = []
    errors: list[str] = []
    for edit in edits:
        raw_path = edit.get("path") or ""
        if not raw_path:
            errors.append("an edit is missing its 'path'")
            continue
        relative = normalize_repo_path(raw_path)
        if relative is None:
            errors.append(f"refusing to edit path outside the repository: {raw_path}")
            continue

        if edit.get("new_file"):
            content = edit.get("content")
            if not isinstance(content, str):
                errors.append(f"{relative}: new_file edit has no 'content' string")
                continue
            try:
                repo.write(relative, content)
            except OSError as exc:
                errors.append(f"{relative}: could not be created ({exc}). Check the "
                              "directory part of the path -- a component of it may be a file.")
                continue
            changed.append(relative)
            continue

        text = repo.read(relative)
        if text is None:
            errors.append(f"{relative}: file not found")
            continue
        search = edit.get("search")
        replace = edit.get("replace")
        if not isinstance(search, str) or not isinstance(replace, str):
            errors.append(f"{relative}: edit needs both 'search' and 'replace' strings")
            continue

        occurrences = text.count(search)
        if occurrences == 0:
            relaxed = _relaxed_find(text, search)
            if relaxed is None:
                errors.append(
                    f"{relative}: the 'search' text was not found. It must reproduce the current "
                    "file byte for byte, including indentation."
                )
                continue
            start, end = relaxed
            updated = text[:start] + replace + text[end:]
        elif occurrences > 1:
            errors.append(f"{relative}: the 'search' text appears {occurrences} times; make it unique")
            continue
        else:
            updated = text.replace(search, replace, 1)

        if updated == text:
            errors.append(f"{relative}: the edit is a no-op")
            continue
        try:
            repo.write(relative, updated)
        except OSError as exc:
            errors.append(f"{relative}: could not be written ({exc})")
            continue
        if relative not in changed:
            changed.append(relative)
    trace("apply_edits", "out", changed=changed, errors=errors)
    return changed, errors


def _relaxed_find(text: str, search: str) -> tuple[int, int] | None:
    """Locate `search` ignoring trailing whitespace differences per line."""
    def normalise(value: str) -> list[str]:
        return [line.rstrip() for line in value.splitlines()]

    haystack = text.splitlines(keepends=True)
    needle = normalise(search)
    if not needle:
        return None
    flat = [line.rstrip() for line in haystack]
    for index in range(len(flat) - len(needle) + 1):
        if flat[index : index + len(needle)] == needle:
            start = sum(len(line) for line in haystack[:index])
            end = start + sum(len(line) for line in haystack[index : index + len(needle)])
            # keep any trailing newline outside the replaced span
            replaced = "".join(haystack[index : index + len(needle)])
            if replaced.endswith("\n") and not search.endswith("\n"):
                end -= 1
            return start, end
    return None


# ---------------------------------------------------------------------------
# Solve loop
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    patch: str
    checks: list[CheckResult]
    diagnosis: str

    @property
    def score(self) -> tuple[int, int, int]:
        # A candidate the tests actually exercised outranks one that merely
        # passed static gates, whatever the raw counts say.
        passed = sum(1 for check in self.checks if check.passed)
        failed = sum(1 for check in self.checks if not check.passed)
        return (int(self.verified), -failed, passed)

    @property
    def verified(self) -> bool:
        """Did anything actually exercise the change against the database?"""
        return any(c.name.startswith("$ ") and c.passed and "skipped" not in c.detail
                   for c in self.checks)

    @property
    def clean(self) -> bool:
        # all([]) is True, so an empty check list would otherwise read as
        # success. A patch nothing executed is not a verified patch.
        return (bool(self.patch.strip())
                and all(check.passed for check in self.checks)
                and self.verified)


class Solver:
    def __init__(self, repo: Repository, instruction: Instruction, probe: DatabaseProbe, llm: LLM) -> None:
        self.repo = repo
        self.instruction = instruction
        self.probe = probe
        self.llm = llm
        self.verifier = Verifier(repo, instruction, candidates=[], probe=probe)
        self.best: Candidate | None = None
        self._graph_cache: CallGraph | None = None
        self._served: set[str] = set()          # context requests already answered
        # Run counters. Nothing in the agent reads these; they are what
        # routing_lab.telemetry_record() needs when routing is being measured
        # locally, and they cost nothing to keep.
        self.attempts = 0
        self.first_attempt_clean: bool | None = None
        self.final_clean = False
        self.failures_seen = 0
        self.slips_seen = 0
        self.stalls_seen = 0
        self.stop_reason = ""

    @property
    def _graph(self) -> CallGraph:
        if self._graph_cache is None:
            self._graph_cache = CallGraph(self.repo)
        return self._graph_cache

    def tier_for(self, failures: int) -> list[str]:
        """Escalate only when a *verified* failure justifies the extra spend."""
        if FORCE_MODEL:
            trace("Solver.tier_for", "out", tier="forced", models=[FORCE_MODEL])
            return [FORCE_MODEL]
        index = min(failures, len(LADDER) - 1)
        capped = self.llm.spent() > COST_TARGET_USD and index > 1
        if capped:
            index = 1
        trace("Solver.tier_for", "out", failures=failures, tier=index, models=LADDER[index],
              spent=self.llm.spent(), held_back_by_cost=capped)
        return LADDER[index]

    def solve(self, candidates: Sequence[tuple[str, list[int]]]) -> str:
        self.verifier.candidates = [path for path, _ in candidates]
        evidence = render_evidence(self.repo, self.instruction, candidates, self.probe)
        log(f"evidence bundle: {len(evidence)} characters")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
             f"{evidence}\n\n{PromptBuilder.SEPARATOR}\n\n# Response format\n\n{EDIT_PROTOCOL}"},
        ]
        # Investigation budget scales with how much the instruction gives us. A
        # named file needs at most one look around; an unnamed one may need to
        # trace symptom -> query across several files. Each verified failure
        # buys one more round, because the failure output is exactly when the
        # model needs to look at something it had not considered.
        named_target = bool(self.instruction.edit_only or self.instruction.lint_paths)
        context_budget = 1 if named_target else 3
        context_rounds = 0
        failures = 0
        rounds = 0
        stalls = 0
        # Enough iterations for the largest legitimate path: every context
        # round, every repair, and a protocol slip or two.
        slips = 0
        unverified_retries = 0
        stalled = 0                 # byte-identical edits in a row: the same model will not change its mind
        last_edit_sig = None
        max_rounds = MAX_REPAIR_ROUNDS + MAX_CONTEXT_ROUNDS + MAX_GATE_SLIPS + 2

        while failures < MAX_REPAIR_ROUNDS and rounds < max_rounds:
            rounds += 1
            if remaining_seconds() < 200:
                log("stopping: wall-clock budget nearly spent")
                break
            try:
                content, model = self.llm.complete(self.tier_for(failures + stalled), self._trim(messages))
            except BudgetExhausted as exc:
                log(f"stopping: {exc}")
                break
            except InferenceError as exc:
                # Permanent only if every model we tried was refused outright.
                permanent = bool(self.llm.blocked) and self.llm.blocked >= self.llm.unsupported
                if permanent or stalls >= 2 or remaining_seconds() < 420:
                    log(f"inference failed, giving up: {exc}")
                    break
                stalls += 1
                pause = 30 * stalls
                log(f"inference failed (looks transient), retrying the roster in {pause}s: "
                    f"{truncate(str(exc), 200)}")
                time.sleep(pause)
                self.llm.unsupported -= self.llm.blocked   # re-probe the transient ones only
                continue

            trace("Solver.round", "", round=rounds, model=model,
                  action=(extract_json(content) or {}).get("action", "edit"),
                  failures=failures, slips=slips, context_rounds=context_rounds,
                  stack_chars=sum(len(str(m.get("content", ""))) for m in messages),
                  seconds_left=remaining_seconds())
            payload = extract_json(content)
            if payload is None:
                messages.append({"role": "assistant", "content": truncate(content, 2000)})
                messages.append({"role": "user", "content":
                                 "That reply did not contain a parseable JSON object. "
                                 "Reply with exactly one JSON object in the documented shape."})
                continue

            action = (payload.get("action") or "").lower()

            if action == "need_context":
                if context_rounds >= min(context_budget, MAX_CONTEXT_ROUNDS):
                    messages.append({"role": "assistant", "content": json.dumps(payload)[:2000]})
                    messages.append({"role": "user", "content":
                                     "No investigation rounds remain. Reply with the 'edit' "
                                     "object using what you already have."})
                    continue
                context_rounds += 1
                requests = payload.get("requests") or []
                log(f"model requested context ({context_rounds}/{context_budget}): "
                    f"{[r.get('kind') for r in requests]}")
                answers = fulfil_requests(self.repo, self.probe, requests,
                                          graph=self._graph, verifier=self.verifier,
                                          served=self._served)
                remaining = min(context_budget, MAX_CONTEXT_ROUNDS) - context_rounds
                self._compact_old_context(messages)
                messages.append({"role": "assistant", "content": json.dumps(payload)})
                messages.append({"role": "user", "content":
                                 f"# Requested context\n\n{truncate(answers, 30000)}\n\n"
                                 + (f"You may request context {remaining} more time(s), "
                                    "or reply with the 'edit' object."
                                    if remaining else "Now reply with the 'edit' object.")})
                continue

            edits = payload.get("edits") or []
            if not edits:
                messages.append({"role": "assistant", "content": truncate(content, 1500)})
                messages.append({"role": "user", "content":
                                 "No edits were provided. Reply with an 'edit' object containing "
                                 "at least one search/replace edit."})
                continue

            diagnosis = str(payload.get("diagnosis") or "")
            edit_sig = hashlib.sha1(json.dumps(edits, sort_keys=True, default=str).encode()).hexdigest()
            repeated = edit_sig == last_edit_sig
            last_edit_sig = edit_sig
            self._adopt_constraints(payload.get("constraints"))
            if not self.instruction.commands:
                supplied = [c for c in (payload.get("verify") or [])
                            if isinstance(c, str) and 8 < len(c) < 400]
                if supplied:
                    log(f"no checks parsed from the instruction; using the {len(supplied)} "
                        f"the model extracted: {supplied}")
                    self.instruction.commands = supplied
            log(f"attempt {failures + 1} via {model}: {truncate(diagnosis, 300)}")
            self.attempts += 1

            self.repo.revert_all()
            changed, errors = apply_edits(self.repo, edits)

            if errors and not changed:
                messages.append({"role": "assistant", "content": json.dumps(payload)[:4000]})
                messages.append({"role": "user", "content":
                                 "None of the edits applied:\n" + "\n".join(errors) +
                                 "\n\nRe-read the source shown above and reply with corrected edits."})
                continue

            include_tests = remaining_seconds() > 400
            checks = self.verifier.run_all(changed, include_tests=include_tests)

            # Optimisation is graded on database work. If the model supplied a
            # probe, measure the scaling it claims to have fixed.
            if include_tests and all(c.passed for c in checks):
                measured = self.verifier.measure_query_scaling(payload.get("measure") or {})
                if measured is not None:
                    checks.append(measured)
                    log(f"query scaling: {measured.detail}")
            patch = build_patch(self.repo, changed)
            applies, apply_detail = verify_patch_applies(patch, self.repo)
            checks.append(CheckResult("patch applies cleanly", applies, apply_detail))

            candidate = Candidate(patch=patch, checks=checks, diagnosis=diagnosis)
            kept = self.best is None or candidate.score >= self.best.score
            trace("Candidate", "out", attempt=self.attempts, bytes=len(patch),
                  score=candidate.score, verified=candidate.verified, clean=candidate.clean,
                  kept_as_best=kept,
                  previous_best=self.best.score if self.best else None)
            if kept:   # ties: the later, better-informed attempt
                self.best = candidate
            log("checks:\n" + summarise(checks))

            if self.first_attempt_clean is None:
                self.first_attempt_clean = candidate.clean
            if candidate.clean:
                log("all checks passed (including the task's own tests)")
                self.final_clean = True
                return patch

            # Two kinds of not-clean, and they deserve different responses.
            #   L2, a gate slip: scope, syntax, bounded-method, contract, or the
            #       patch did not apply. No test ran; the model broke a rule of
            #       the protocol, not its diagnosis. Same model, tell it the rule,
            #       do not spend a repair round or escalate.
            #   L3, a verified failure: the tests or the measurement said the
            #       fix is wrong. That is real evidence -- switch model family
            #       and grant another investigation round.
            runtime_failed = [c for c in checks if not c.passed and c.name.startswith("$ ")
                              or (not c.passed and c.name == "query scaling")]
            unverified = not candidate.verified and all(c.passed for c in checks)
            if unverified:
                log("WARNING: the change passed every static check but nothing ran it "
                    "against the database -- no usable test command was found")

            self._compact_old_context(messages)
            feedback = self._failure_message(errors, checks)
            messages.append({"role": "assistant", "content": json.dumps(payload)[:4000]})
            self.repo.revert_all()

            if runtime_failed:
                failures += 1                       # L3
                self.failures_seen = failures
                context_budget += 1                 # the failure is worth investigating
                messages.append({"role": "user", "content": feedback})
                continue

            if unverified:
                # Ask once for a way to verify. If none exists, the patch that
                # passed every static gate is the answer -- a second and third
                # request cost calls and change nothing.
                unverified_retries += 1
                if unverified_retries > 1:
                    log("no verification available after one request; adopting the "
                        "statically clean patch as best effort")
                    return patch
                feedback += ("\n\nNo test command was available to exercise this change. Supply "
                             "the checks the instruction names in `verify`, or name the test "
                             "module for the code you changed, so the patch can be verified. "
                             "If the task truly names none, reply with the same edit and an empty verify.")
                messages.append({"role": "user", "content": feedback})
                continue

            slips += 1                              # L2
            self.slips_seen = slips
            if slips > MAX_GATE_SLIPS:
                log(f"giving up after {slips} protocol slips without a verified attempt")
                break
            if repeated:
                # At temperature 0 the same model given the same feedback
                # returns the same bytes. Measured: four identical invalid
                # edits in a row, twice. Say so, and let the next family try.
                stalled += 1
                self.stalls_seen = stalled
                log(f"identical edit repeated; handing the retry to the next model family (stall {stalled})")
                feedback += ("\n\nYour reply was byte-identical to the previous one, which failed this "
                             "same check. Do not resend it: change the construct that the check names.")
            feedback += ("\n\nThis was a rule violation, not a wrong diagnosis: the tests "
                         "did not run. Keep your analysis; fix only what the failing check "
                         "names.")
            messages.append({"role": "user", "content": feedback})

        return self.best.patch if self.best else ""

    def _adopt_constraints(self, supplied) -> None:
        """Take the model's reading of the edit boundary when ours found none.

        Deterministic parsing wins whenever it produces anything -- it is exact
        and identical across validators. This runs only when the instruction
        named no file, no lint path, no method and no path in prose, which is
        the case where the scope gate would otherwise fall back to the whole
        candidate list.
        """
        ins = self.instruction
        if not isinstance(supplied, dict):
            return
        static_found = bool(ins.edit_only or ins.lint_paths or ins.named_paths or ins.single_method)
        if static_found:
            return

        files = supplied.get("editable_files") or []
        accepted = []
        for raw in files if isinstance(files, list) else []:
            relative = normalize_repo_path(str(raw))
            if relative and (self.repo.root / relative).is_file():
                accepted.append(relative)
        if accepted:
            ins.edit_only = accepted[:4]
            self.verifier.instruction = ins
            log(f"scope adopted from the model's reading of the instruction: {ins.edit_only}")
            trace("adopt_constraints", "out", editable=ins.edit_only, source="the model")

        bound = supplied.get("bounded_to_method")
        if isinstance(bound, str) and bound.strip():
            name = bound.strip().split(".")[-1].rstrip("()")
            # Corroborate: the instruction must actually mention this symbol.
            # Otherwise a hallucinated bound would gate the model's own fix.
            if re.search(rf"\b{re.escape(name)}\b", ins.text):
                ins.single_method = True
                ins.method_hint = name
                if "." in bound:
                    ins.class_hint = bound.strip().split(".")[0]
                log(f"method bound adopted from the model, corroborated by the instruction: {bound}")
            else:
                log(f"ignored model-supplied method bound {bound!r}: not mentioned in the instruction")

    def _failure_message(self, errors: Sequence[str], checks: Sequence[CheckResult]) -> str:
        parts = ["The change was applied but did not pass verification."]
        if errors:
            parts.append("Edit problems:\n" + "\n".join(errors))
        failures = [check for check in checks if not check.passed]
        for check in failures:
            parts.append(f"## {check.name}\n{truncate(check.detail, 6000, head_ratio=0.3)}")
        hints = error_hints("\n".join(check.detail for check in failures))
        if hints:
            parts.append("## What this output means\n" + "\n".join(f"- {h}" for h in hints))
        parts.append(
            "Diagnose what this output says about the database work being done, then reply with "
            "a corrected 'edit' object. The edits replace the ORIGINAL file contents shown "
            "earlier -- your previous attempt has been reverted. Do not weaken or edit tests, "
            "and do not special-case fixture values."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _compact_old_context(messages: list[dict]) -> None:
        """Shrink investigation payloads the model has already read.

        A context round can return 30k characters. Left in place, that is
        re-sent on every later call: one such round cost $0.0775 on a real task
        because four payloads rode along on the next three calls. The model's
        own reply already carries what it learned, so keep a stub.
        """
        marker = "# Requested context"
        saved = 0
        for message in messages[:-1]:            # never touch the newest turn
            content = message.get("content") or ""
            if message.get("role") == "user" and content.startswith(marker) and len(content) > 1200:
                head = content[:600].rstrip()
                message["content"] = (f"{head}\n\n[... {len(content) - 600:,} characters of "
                                      "already-consumed context trimmed; request again if needed]")
                saved += len(content) - len(message["content"])
        if saved:
            trace("compact_old_context", "out", chars_reclaimed=saved,
                  stack_now=sum(len(str(m.get("content", ""))) for m in messages))

    @staticmethod
    def _trim(messages: list[dict], keep: int = 6) -> list[dict]:
        """Keep the system prompt and opening brief, then the recent exchange."""
        if len(messages) <= keep + 2:
            return messages
        trimmed = messages[:2] + messages[-keep:]
        # Prompt tokens are charged per call, so what this drops is not saved
        # once -- it is saved on every remaining call in the loop.
        trace("Solver._trim", "out", turns=f"{len(messages)}->{len(trimmed)}",
              chars=sum(len(str(m.get("content", ""))) for m in trimmed),
              dropped=sum(len(str(m.get("content", ""))) for m in messages)
              - sum(len(str(m.get("content", ""))) for m in trimmed))
        return trimmed


# Translations of failure output the models keep misreading. Each is a fact
# about SQL or the ORM, not about any task; they only add text to the
# feedback, so a wrong match costs nothing but a sentence.
ERROR_HINTS: tuple[tuple[str, str], ...] = (
    (r"F401 .*imported but unused",
     "F401: your change stopped using a name the file imports. Where the instruction says to "
     "keep imports unchanged, removing the import is not an option -- the method must keep "
     "using that name (the placeholder you replaced did)."),
    (r"more than one row returned by a subquery used as an expression",
     "A correlated subquery used as an annotation must return exactly one row. Aggregate the "
     "whole correlated set: no GROUP BY on a column that varies per matched row (in the ORM, "
     "`.values()` before `.annotate(Count)` groups by that column). Group on a constant or "
     "write the COUNT in SQL."),
    (r"null value in column .* violates not-null constraint|AssertionError: None != 0",
     "An aggregate subquery yields NULL when nothing matches. Rows with no matches must get 0: "
     "COALESCE(..., 0) around the count, using only what the file already imports."),
    (r"invalid-syntax: Duplicate keyword argument|keyword argument repeated",
     "The same keyword was passed twice in one call. To put two conditions on one field, "
     "use separate Q objects joined with & or |, e.g. Q(field__isnull=True) & Q(field=OuterRef(...)), "
     "or compare with OuterRef/F -- never repeat the keyword."),
    (r"Cannot resolve keyword '(\w+)' into field",
     "That field name does not exist on the model; check the model's actual field names in the "
     "evidence before guessing another."),
    (r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
     "That object is one of the application's own classes, not a library client. Find its "
     "definition (the package map above, `callers`, or grep for `class <Name>`) and call a method "
     "it actually defines, with the statement format it expects."),
    (r"NotSupportedError|not supported by this database backend",
     "The expression is not available on this database engine; use the engine's native "
     "operators or a RawSQL fallback."),
)


def error_hints(text: str) -> list[str]:
    hints = []
    for pattern, hint in ERROR_HINTS:
        if re.search(pattern, text) and hint not in hints:
            hints.append(hint)
    return hints


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class MissingInstruction(RuntimeError):
    """No problem statement was supplied -- a bad input, not an agent fault."""


HARNESS_INSTRUCTION = Path("/installed-agent/instruction.md")


def _instruction_text(payload: dict, root: Path,
                      harness_copy: Path = HARNESS_INSTRUCTION) -> str:
    for key in ("problem_statement", "instruction", "problem", "task", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # File fallbacks, for local and manual runs. The harness-written copy comes
    # first: it can only be this task. A file of the same name inside the
    # application repository might be the project's own documentation.
    for location in (harness_copy, root / "instruction.md"):
        if location.is_file():
            try:
                return location.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    raise MissingInstruction("no problem statement supplied")


def agent_main(input: dict) -> str:
    """Return a unified diff that solves the task described in `input`."""
    repo: Repository | None = None
    # Most detailed mode wins: the per-line tracer already reports every call
    # and return, so installing both would print each of them twice.
    if TRACE_VARS:
        install_variable_tracer()
    elif TRACE_ALL:
        install_call_tracer()
    try:
        root = workdir()
        text = _instruction_text(input or {}, root)
        log(f"model={FORCE_MODEL or 'ladder'} workdir={root} timeout={os.getenv('AGENT_TIMEOUT', '?')}s "
            f"budget=${os.getenv('RIDGES_MAX_COST_USD', DEFAULT_MAX_COST_USD)}")

        trace("agent_main", "in", root=str(root), statement=len(text),
              timeout=os.getenv("AGENT_TIMEOUT", "unset"),
              budget=os.getenv("RIDGES_MAX_COST_USD", str(DEFAULT_MAX_COST_USD)),
              seconds_available=remaining_seconds())
        instruction = parse_instruction(text, root)
        log(f"kind={instruction.primary_kind} engine={instruction.engine} "
            f"single_method={instruction.single_method} commands={len(instruction.commands)}")

        repo = Repository(root)
        targets = locate_targets(repo, instruction)
        if not targets:
            log("no candidate files found; falling back to instruction-named paths")
            targets = [(path, []) for path in instruction.named_paths]

        probe = DatabaseProbe(repo, instruction)
        if instruction.engine == "unknown" and probe.available():
            instruction.engine = probe.targets[0].engine
            log(f"engine resolved from the live database: {instruction.engine}")
        llm = LLM()
        if not FORCE_MODEL:
            llm.discover_models()
        solver = Solver(repo, instruction, probe, llm)
        patch = solver.solve(targets)

        if not patch.strip():
            log("no patch produced")
            trace("agent_main", "out", patch_bytes=0, calls=llm.calls, usd=llm.spent(),
                  seconds_left=remaining_seconds(), outcome="no patch")
            return ""

        log(f"returning patch: {len(patch)} bytes, "
            f"{len(re.findall(r'^diff --git', patch, re.MULTILINE))} file(s)")
        log(llm.report())
        if TRACE:
            log(llm.ledger())
        trace("agent_main", "out", patch_bytes=len(patch),
              files=len(re.findall(r"^diff --git", patch, re.MULTILINE)),
              calls=llm.calls, usd=llm.spent(), attempts=solver.attempts,
              seconds_left=remaining_seconds(),
              outcome="verified" if solver.final_clean else "best effort")
        return patch
    except MissingInstruction as exc:
        # Nothing to solve. Report it in one line rather than a stack trace, and
        # still return a string: a raised exception is scored as an agent crash.
        log(f"no work to do: {exc}")
        return ""
    except Exception:
        log("agent failed:\n" + traceback.format_exc())
        return ""
    finally:
        # The verifier hashes every file it did not authorise us to change, and
        # it grades the patch, not our container. Leave the checkout pristine.
        if repo is not None:
            try:
                repo.revert_all()
            except Exception:
                pass


if __name__ == "__main__":
    statement = sys.stdin.read() if not sys.stdin.isatty() else ""
    print(agent_main({"problem_statement": statement}))
