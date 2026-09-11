"""Ridges DB-query agent — general big-repo edition.

Entry point: agent_main({"problem_statement": ...}) -> unified diff

Stages:
  0. Scope    — reads the named file, method and check commands out of the
                instruction itself; no model call
  1. Locator  — finds the target file(s) with search tools, only when the
                instruction names none
  2. Planner  — reads the target and forms a plan, only when the locator ran
  3. Driver   — edits, runs checks, submits; submit re-runs the named checks
                and refuses edits outside the named files

Standard library only. No git and no network: the application folder is
snapshotted into memory at start, the patch is a difflib diff against that
snapshot in the format `git apply` reads, and the folder is restored after.
"""
from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import urllib.error
import urllib.request


# =========================================================================
# Constants
# =========================================================================

FALLBACK_BASE_URL = "https://openrouter.ai/api/v1"

DRIVER_MODEL = os.getenv("RIDGES_AGENT_MODEL", "openai/gpt-5.6-luna")
RELIEF_MODEL = os.getenv("RIDGES_RELIEF_MODEL", "deepseek/deepseek-v4-pro-0813")
LOCATOR_MODEL = os.getenv("RIDGES_LOCATOR_MODEL", "openai/gpt-5.6-luna")
PLANNER_MODEL = os.getenv("RIDGES_PLAN_MODEL", "openai/gpt-5.6-luna")

HIGH_REASONING = (os.getenv("RIDGES_HIGH_REASONING") or "0").strip().lower() not in (
    "0", "no", "off", "false", "")

# cache-read price per token, context window
SEAT_CACHE_TERMS = {
    "xiaomi/mimo-v2.5-pro": (0.0050e-6, 262_144),
    "xiaomi/mimo-v2.5": (0.0050e-6, 262_144),
    "minimax/minimax-m2.5": (0.0500e-6, 204_800),
    "minimax/minimax-m3": (0.0750e-6, 204_800),
    "deepseek/deepseek-v4-pro": (0.0036e-6, 1_048_576),
    "deepseek/deepseek-v4-pro-0813": (0.0220e-6, 1_048_576),
    "openai/gpt-5.6-luna": (0.0200e-6, 400_000),
    "qwen/qwen3.8-2.4t-a95b": (0.2500e-6, 1_000_000),
    "@preset/qwen38-24t-lowthink": (0.2500e-6, 1_000_000),
    "openai/gpt-5.6-terra": (0.2000e-6, 400_000),
    "google/gemini-3.7-flash": (0.0375e-6, 1_048_576),
    "deepseek/deepseek-v4-flash-0731": (0.0280e-6, 1_048_576),
    "tencent/hy3": (0.0330e-6, 262_144),
}
UNKNOWN_CACHE_TERMS = (0.1000e-6, 131_072)

# fresh-in, out
MODEL_PRICING = {
    "qwen/qwen3.8-2.4t-a95b": (2.000e-6, 6.000e-6),
    "@preset/qwen38-24t-lowthink": (2.000e-6, 6.000e-6),
    "xiaomi/mimo-v2.5-pro": (0.600e-6, 1.201e-6),
    "xiaomi/mimo-v2.5": (0.140e-6, 0.280e-6),
    "minimax/minimax-m2.5": (0.150e-6, 0.900e-6),
    "minimax/minimax-m3": (0.375e-6, 1.500e-6),
    "deepseek/deepseek-v4-pro-0813": (0.660e-6, 1.980e-6),
    "openai/gpt-5.6-luna": (0.200e-6, 1.200e-6),
    "openai/gpt-5.6-terra": (2.000e-6, 12.000e-6),
    "google/gemini-3.7-flash": (0.375e-6, 1.875e-6),
    "deepseek/deepseek-v4-flash-0731": (0.440e-6, 1.320e-6),
    "tencent/hy3": (0.132e-6, 0.528e-6),
}
UNKNOWN_TOKEN_PRICE = (1.0e-6, 4.0e-6)

# budget
DEFAULT_COST_LIMIT_USD = 0.29
DEFAULT_WALL_SEC = 1500.0
COST_SHARE = 0.88
WALL_SHARE = 0.90
WALL_RESERVE_SEC = 45.0
HARD_TAIL_RESERVE_SEC = 30.0     # for the harness's own git apply after we return
ANCHOR_FILE = "/installed-agent/instruction.md"   # written when the harness clock starts
COST_SYNC_TURNS = 5
USAGE_PATH = "/api/v1/usage"

# loop
TURN_CEILING = 200
BLANK_REPLY_CEILING = 3
TOOL_FAULT_CEILING = 12
SUBMIT_REFUSAL_CEILING = 3
IDENTICAL_REPLY_CEILING = 3
EDIT_PRESSES_MAX = 3
FIRST_EDIT_DEADLINE_TURN = 8
WRAPUP_TURN = 80
WRAPUP_CLOCK_SEC = 240.0

# stages
LOCATOR_TURN_CAP = 25
LOCATOR_READ_BUDGET = 80_000
LOCATOR_NOTE_CHARS = 1_500
LOCATOR_SPEND_SHARE = 0.20
LOCATOR_CLOCK_SHARE = 0.25       # of the run's clock, measured from the start

PLANNER_TURN_CAP = 30
PLANNER_READ_BUDGET = 100_000
PLAN_NOTE_CHARS = 4_000
PLANNER_SPEND_SHARE = 0.25
PLANNER_CLOCK_SHARE = 0.45

# output caps
READ_OUTPUT_CAP = 24_000
SHELL_OUTPUT_CAP = 8_000
SEARCH_OUTPUT_CAP = 8_000
SHELL_REPORT_CAP = 120
REPLY_TOKEN_CEILING = 8_000

# transcript
TRANSCRIPT_SPEND_SHARE = 0.35
TURNS_PLANNED = 50
CHARS_PER_TOKEN = 3.5
TRANSCRIPT_FLOOR_CHARS = 40_000

# seat
TEMPERATURE = 0.0
SEED = 4242
SEAT_CALL_TIMEOUT_SEC = 90.0
SEAT_RETRY_GROWTH = 1.5          # each retry waits longer, not shorter
SEAT_RETRY_FLOOR_SEC = 20.0
SEAT_REFUSED_WAIT_SEC = 20.0
REQUEST_ATTEMPTS = 3
CALL_CLOCK_SHARE = 0.34
ABANDONED_REPLY_TOKENS = REPLY_TOKEN_CEILING // 4   # charged for a call we gave up on

# shell
SHELL_BUDGET_CEILING_SEC = 180.0
BACKGROUND_JOBS_MAX = 2
BACKGROUND_POLL_WAIT_SEC = 75.0
BACKGROUND_POLL_CEILING_SEC = 150.0
POLLS_PER_JOB_ADVISE = 8
CHILD_MEMORY_BYTES = 3 * 1024 * 1024 * 1024   # per child, address space; Node and Go start under it
CHILD_CAP_FLOOR_BYTES = 1024 * 1024 * 1024    # never squeeze a child below this
MEMORY_HEADROOM_BYTES = 768 * 1024 * 1024     # the agent's own footprint plus slack
SMALL_CONTAINER_BYTES = 4 * 1024 * 1024 * 1024  # under this, one job at a time
MEMORY_STOP_SHARE = 0.45      # of the container limit: no second job past this
MEMORY_REFUSE_SHARE = 0.75    # no new background job past this
CHILD_OOM_SCORE_ADJ = 800     # the OOM killer takes a child before the agent
CHILD_NICE = 5

# git
GIT_TIMED_OUT = 124

# repeat reads
REPEAT_READ_CEILING = 2

# history / network fences
HISTORY_GIT = re.compile(
    r"\bgit\s+(?:-[^\s]+\s+)*(commit|stash|checkout|switch|restore|reset|clean|"
    r"revert|rebase|merge|cherry-pick|push)\b"
)
NETWORK_COMMAND = re.compile(
    r"(?:^|[|&;]|\$\(|`)\s*(?:sudo\s+)?"
    r"(curl|wget|nc|ncat|telnet|ssh|scp|rsync|ftp|"
    r"git\s+(?:fetch|pull|clone|remote|ls-remote|submodule))(?![\w-])"
)

# tests
TEST_PATH = re.compile(
    r"(^|/)conftest\.py$|(^|/)tests?(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
NOQA_DIRECTIVE = re.compile(r"#\s*(?:(?:ruff|flake8)\s*:\s*)?noqa\b", re.I)

# statement parsing
QUOTED_RE = re.compile(r"[`'\"]([A-Za-z_][A-Za-z0-9_.]{2,})[`'\"]")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
COMMON_WORDS = frozenset(
    """the this that with from when what which should would could have been
    test tests file files line lines code error errors return returns value
    values method function class module import python true false none self
    argument arguments result results object objects string strings expected
    also any each every same such keep make change changed unchanged rest
    currently instead both being handful number numbers added adding still
    exact exactly correct correctly wrong incorrect incorrectly slow slowly
    fast expensive cheap large small many few several some most more less
    least first last new old one two three per well very just even quite
    rather already again once twice however therefore because since although
    whether either neither between within across through against toward
    towards during after under over above below behind""".split()
)


# =========================================================================
# Small helpers
# =========================================================================

def say(message: str) -> None:
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


def one_line(value: object) -> str:
    return " ".join(str(value or "").split())


def foreign(text: object) -> str:
    body = "" if text is None else str(text)
    return "%dB" % len(body.encode("utf-8", "replace")) if body else "empty"


def num_env(name: str, default: float) -> float:
    try:
        value = float((os.getenv(name) or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value == value and value not in (
        float("inf"), float("-inf")) else default


def flag(name: str, default: str = "1") -> bool:
    return (os.getenv(name) or default).strip().lower() not in (
        "0", "no", "off", "false", "")


def clip(text: str, cap: int, label: str = "output") -> str:
    if len(text) <= cap:
        return text
    note_template = "\n... [%d characters of %s elided] ...\n"
    keep = cap
    for _ in range(4):
        room = max(0, cap - len(note_template % (len(text) - keep, label)))
        if room == keep:
            break
        keep = room
    if keep <= 0:
        return (note_template % (len(text), label))[:cap]
    note = note_template % (len(text) - keep, label)
    return text[: keep // 2] + note + text[len(text) - (keep - keep // 2):]


def reply_fingerprint(message: dict) -> str:
    calls = (message or {}).get("tool_calls") if isinstance(message, dict) else None
    parts = []
    for call in calls if isinstance(calls, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            function = {}
        parts.append("%s(%s)" % (function.get("name") or "",
                                 function.get("arguments") or ""))
    content = message.get("content") if isinstance(message, dict) else ""
    said = "\n".join(parts) if parts else str(content or "")
    return hashlib.sha256(said.encode("utf-8", "replace")).hexdigest()[:8]


def reasoning_tokens(usage: dict) -> int:
    details = (usage or {}).get("completion_tokens_details")
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -1
    return int(value)


# =========================================================================
# Allowance
# =========================================================================

class Spent(Exception):
    pass


class Allowance:
    """The run's clock and money. Durations are measured on the monotonic
    clock, which is what the harness times us on; the wall clock can be
    stepped under us and is only used to compare file mtimes."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.wall = max(60.0, num_env("AGENT_TIMEOUT", DEFAULT_WALL_SEC))
        self.ceiling_usd = num_env("RIDGES_MAX_COST_USD", DEFAULT_COST_LIMIT_USD)
        self.soft_usd = self.ceiling_usd * COST_SHARE
        self.deadline = self.started + max(30.0, self.wall * WALL_SHARE - WALL_RESERVE_SEC)
        # After this the harness may already have given up on us.
        self.hard_deadline = self.started + max(45.0, self.wall - HARD_TAIL_RESERVE_SEC)
        self.spent = 0.0
        self.calls = 0
        self.billed = 0
        self.edits = 0
        self.synced_usd = None    # last total the proxy reported
        self.sync_failed = False

    def shift(self, seconds: float) -> None:
        """The harness started its clock `seconds` before we started ours."""
        if seconds > 0:
            self.started -= seconds
            self.deadline -= seconds
            self.hard_deadline -= seconds

    def clock_left(self) -> float:
        return self.deadline - time.monotonic()

    def hard_left(self) -> float:
        return self.hard_deadline - time.monotonic()

    def run_length(self) -> float:
        """Seconds between the start and the soft deadline."""
        return self.deadline - self.started

    def money_left(self) -> float:
        return self.soft_usd - self.spent

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def sync(self, proxy: str = None, timeout: float = 5.0) -> bool:
        """Adopt the proxy's total when it is above ours. The proxy is what
        refuses the next call once the budget is gone, so its number wins
        upward; a lower one may just be lagging."""
        base = (proxy if proxy is not None
                else (os.getenv("SANDBOX_PROXY_URL") or "")).strip().rstrip("/")
        if not base or self.sync_failed:
            return False
        try:
            with urllib.request.urlopen(base + USAGE_PATH, timeout=timeout) as response:
                report = json.loads(response.read().decode("utf-8", "replace"))
            total = report.get("total_cost_usd") if isinstance(report, dict) else None
            if isinstance(total, bool) or not isinstance(total, (int, float)):
                raise ValueError("no total_cost_usd in the usage reply")
        except Exception as error:
            self.sync_failed = True
            say("[COST] usage endpoint unavailable: %s: %s"
                % (type(error).__name__, str(error)[:120]))
            return False
        self.synced_usd = float(total)
        if total > self.spent:
            say("[COST] proxy=$%.4f local=$%.4f; adopting the proxy's total"
                % (total, self.spent))
            self.spent = float(total)
            return True
        return False

    def halt_reason(self) -> str:
        if self.clock_left() <= 0:
            return "wall clock"
        if self.money_left() <= 0:
            return "budget"
        return ""

    def charge(self, model: str, usage: dict) -> float:
        self.calls += 1
        quoted = usage.get("cost")
        if (isinstance(quoted, (int, float)) and not isinstance(quoted, bool)
                and quoted >= 0):
            self.billed += 1
            self.spent += float(quoted)
            return float(quoted)
        cost = self.estimate(model, usage)
        self.spent += cost
        return cost

    @staticmethod
    def estimate(model: str, usage: dict) -> float:
        """List-price estimate from token counts, for when the endpoint
        does not quote a cost."""
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
        fresh = max(0, prompt - cached)
        in_price, out_price = MODEL_PRICING.get(model, UNKNOWN_TOKEN_PRICE)
        cache_price = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)[0]
        return fresh * in_price + cached * cache_price + completion * out_price


def transcript_cap_chars(model: str, ceiling_usd: float) -> int:
    cache_price, window = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)
    affordable = (ceiling_usd * TRANSCRIPT_SPEND_SHARE) / (TURNS_PLANNED * cache_price)
    tokens = min(affordable, window * 0.6)
    return int(max(TRANSCRIPT_FLOOR_CHARS, tokens * CHARS_PER_TOKEN))


def rounds_left(allowance: Allowance, model: str, transcript_chars: int,
                history: list) -> int:
    """How many more rounds the money can pay for."""
    cache_price, _ = SEAT_CACHE_TERMS.get(model, UNKNOWN_CACHE_TERMS)
    in_price, out_price = MODEL_PRICING.get(model, UNKNOWN_TOKEN_PRICE)
    money = allowance.money_left()
    if money <= 0:
        return 0
    if len(history) >= 3:
        recent = history[-5:]
        avg_growth = sum(h["growth"] for h in recent) / len(recent)
        avg_reply = sum(h["reply_tokens"] for h in recent) / len(recent)
    else:
        avg_growth = 4000.0
        avg_reply = 1200.0
    # Each round re-reads the transcript at the cache price, pays the fresh
    # price once for what the last round added, and pays for its reply.
    a = cache_price * avg_growth / (2.0 * CHARS_PER_TOKEN)
    b = (cache_price * transcript_chars / CHARS_PER_TOKEN
         + in_price * avg_growth / CHARS_PER_TOKEN + out_price * avg_reply)
    if a <= 0:
        return int(money / b) if b > 0 else TURN_CEILING
    disc = b * b + 4 * a * money
    r = (-b + disc ** 0.5) / (2 * a)
    return max(1, min(TURN_CEILING, int(r)))


# =========================================================================
# Beacon
# =========================================================================

class Beacon:
    def __init__(self, slug: str) -> None:
        self.slug = slug.upper()
        self.calls = 0
        self.usd = 0.0

    def reached(self, step: int, spent: float, clock: float) -> None:
        say("[%s] reached step=%d spent=$%.4f clock=%.0fs"
            % (self.slug, step, spent, clock))

    def skipped(self, reason: str) -> None:
        say("[%s] skipped: %s" % (self.slug, reason))

    def fired(self, detail: str) -> None:
        say("[%s] fired: %s" % (self.slug, detail[:400]))

    def bill(self) -> None:
        say("[%s] cost calls=%d usd=%.4f" % (self.slug, self.calls, self.usd))


# =========================================================================
# Tree — a snapshot of the application folder. No git, no network.
# =========================================================================
#
# The task container holds the application at /app with its .git removed,
# and some task images do not ship a git binary at all. Everything below
# therefore works from the files on disk: the tree is read into memory at
# start, the patch is a difflib diff of snapshot-vs-disk written in the
# format `git apply` reads, and restore writes the snapshot back.

SNAPSHOT_SKIP_DIRS = frozenset((
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", ".tox", ".cache"))
SNAPSHOT_KEEP_BYTES = 2 * 1024 * 1024        # per file, kept in memory
SNAPSHOT_TOTAL_BYTES = 128 * 1024 * 1024     # across the whole snapshot
SNAPSHOT_FILE_CEILING = 200_000
SNAPSHOT_HASH_CEILING = 64 * 1024 * 1024      # bigger files are not hashed
PRISTINE_BUDGET_SEC = 90.0                    # copying the tree at start
PRISTINE_BYTES_CEILING = 2 * 1024 * 1024 * 1024
RACY_WINDOW_NS = 2_000_000_000     # mtimes this close to the snapshot are re-hashed
LINE_SPLIT = re.compile(r"(?<=\n)")
NO_NEWLINE = "\\ No newline at end of file\n"
HUNK_RANGE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def git(args: list, cwd: str, timeout: float = 60.0) -> tuple:
    """Only ever used for `git apply --check`, and only when git exists."""
    try:
        done = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True,
            timeout=max(2.0, timeout), errors="replace",
        )
    except subprocess.TimeoutExpired:
        return GIT_TIMED_OUT, "git timed out"
    except Exception as error:
        return 1, "%s: %s" % (type(error).__name__, error)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


class ToolFault(Exception):
    pass


def parse_python(source: str):
    """ast.parse over text read with surrogateescape: the bytes go back
    exactly, so a coding declaration or a stray byte is the parser's call
    and not a UnicodeEncodeError."""
    return ast.parse(source.encode("utf-8", "surrogateescape"))


def match_line_endings(text: str, old: str, new: str) -> tuple:
    """`old` and `new` in the line-ending convention `text` uses. A file
    whose first line break is CRLF gets CRLF spans; anything else LF."""
    crlf = text.find("\r\n")
    lf = text.find("\n")
    uses_crlf = crlf >= 0 and crlf + 1 == lf
    out = []
    for span in (old, new):
        span = span.replace("\r\n", "\n")
        if uses_crlf:
            span = span.replace("\n", "\r\n")
        out.append(span)
    return out[0], out[1]


def split_lines(text: str) -> list:
    """Split on \\n only, keeping the newline on each line. git counts lines
    this way; str.splitlines also breaks on \\r, \\f and \\x1c, which would
    misnumber hunks in files that contain them."""
    if not text:
        return []
    return [part for part in LINE_SPLIT.split(text) if part]


def looks_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def file_mode(st_mode: int) -> str:
    return "100755" if st_mode & 0o111 else "100644"


DIFF_CONTEXT = 3
DIFF_LINES_CEILING = 20_000   # differing lines beyond which the file is replaced whole


def hunk_range(count: int, start: int = 1) -> str:
    if count == 0:
        return "%d,0" % max(0, start - 1)
    if count == 1:
        return "%d" % start
    return "%d,%d" % (start, count)


def shift_hunk_header(line: str, offset: int) -> str:
    match = HUNK_RANGE.match(line)
    if not match or not offset:
        return line
    old_start, old_count, new_start, new_count = match.groups()
    old = "%d" % (int(old_start) + offset) + ("," + old_count if old_count is not None else "")
    new = "%d" % (int(new_start) + offset) + ("," + new_count if new_count is not None else "")
    return "@@ -%s +%s @@" % (old, new) + line[match.end():]


def diff_lines(a: list, b: list, src: str, dst: str, whole: bool = False) -> list:
    """The unified diff of two line lists, in the shape difflib emits.

    The lines the two share at the head and the tail are left out of the
    comparison (their last few kept as context), so a small edit to a long
    file costs a few lines of work whatever the file's size. A middle that
    is still too large to compare in bounded time is replaced whole: one
    hunk that removes every old line and adds every new one, which git
    applies just the same."""
    limit = min(len(a), len(b))
    head = 0
    while head < limit and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < limit - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    keep_head = max(0, head - DIFF_CONTEXT)
    keep_tail = max(0, tail - DIFF_CONTEXT)
    mid_a = a[keep_head:len(a) - keep_tail]
    mid_b = b[keep_head:len(b) - keep_tail]
    if whole or len(mid_a) + len(mid_b) > DIFF_LINES_CEILING:
        say("[PATCH] whole-file hunk for %s: %d differing lines%s"
            % (dst if dst != "/dev/null" else src, len(mid_a) + len(mid_b),
               " (asked for)" if whole else ""))
        out = ["--- %s\n" % src, "+++ %s\n" % dst,
               "@@ -%s +%s @@\n" % (hunk_range(len(a)), hunk_range(len(b)))]
        out += ["-" + line for line in a]
        out += ["+" + line for line in b]
        return out
    out = []
    for line in difflib.unified_diff(mid_a, mid_b, src, dst, n=DIFF_CONTEXT):
        if keep_head and line.startswith("@@ "):
            line = shift_hunk_header(line, keep_head)
        out.append(line)
    return out


def file_diff(rel: str, kind: str, before: bytes, after: bytes,
              mode_before: str, mode_after: str, whole: bool = False) -> str:
    """One file's section of a patch, in the shape `git apply` reads."""
    a = split_lines(before.decode("utf-8", "surrogateescape"))
    b = split_lines(after.decode("utf-8", "surrogateescape"))
    head = ["diff --git a/%s b/%s\n" % (rel, rel)]
    if kind == "added":
        head.append("new file mode %s\n" % (mode_after or "100644"))
        src, dst = "/dev/null", "b/" + rel
    elif kind == "deleted":
        head.append("deleted file mode %s\n" % (mode_before or "100644"))
        src, dst = "a/" + rel, "/dev/null"
    else:
        if mode_before and mode_after and mode_before != mode_after:
            head.append("old mode %s\nnew mode %s\n" % (mode_before, mode_after))
        src, dst = "a/" + rel, "b/" + rel
    body = []
    for line in diff_lines(a, b, src, dst, whole):
        # header lines carry their own newline; a body line does too unless
        # it is the file's last line and the file has no trailing newline.
        body.append(line if line.endswith("\n") else line + "\n" + NO_NEWLINE)
    if not body and kind == "modified" and len(head) == 1:
        return ""
    return "".join(head + body)


def patch_dry_run(patch: str, original) -> bool:
    """Would `git apply` accept this patch against the snapshot? A strict,
    offset-free check of every context and removed line. `original(rel)`
    returns the snapshot bytes for a path, or None when they are not held."""
    for part in split_by_file(patch):
        rel, new_file, deleted = "", False, False
        for line in part.split("\n"):
            if line.startswith("--- "):
                if line[4:].strip() == "/dev/null":
                    new_file = True
                elif line.startswith("--- a/"):
                    rel = rel or line[6:].rstrip("\t")
            elif line.startswith("+++ "):
                if line[4:].strip() == "/dev/null":
                    deleted = True
                elif line.startswith("+++ b/"):
                    rel = rel or line[6:].rstrip("\t")
            elif line.startswith("@@ "):
                break
        if not rel:
            return False
        before = b"" if new_file else original(rel)
        if before is None:
            return None
        src = [row.rstrip("\n") for row in
               split_lines(before.decode("utf-8", "surrogateescape"))]
        pos = 0
        in_hunk = False
        for line in part.split("\n"):
            head = HUNK_RANGE.match(line)
            if head:
                start = int(head.group(1))
                count = int(head.group(2)) if head.group(2) is not None else 1
                pos = max(0, start - 1) if count else start
                in_hunk = True
                continue
            if not in_hunk or not line:
                continue
            tag, text = line[0], line[1:]
            if tag in (" ", "-"):
                if pos >= len(src) or src[pos] != text:
                    return False
                pos += 1
            elif tag in ("+", "\\"):
                continue
            else:
                in_hunk = False
        if deleted and pos != len(src):
            return False
    return True


class Tree:
    """The application folder at `root`, read into memory at start."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        self.base = "snapshot"
        self.index: dict = {}       # rel -> (size, mtime_ns, st_mode)
        self.digests: dict = {}     # rel -> sha256 of the file at start
        self.originals: dict = {}   # rel -> bytes at start, when small enough
        self.unkept: set = set()    # rel paths hashed but not held in memory
        self.pristine_dir = None
        self.taken_ns = 0
        self.dropped: list = []     # why the last patch is incomplete, if it is
        self._snapshot()

    # ---- walking -----------------------------------------------------

    def _walk(self):
        """(rel, full) for every regular file, skipping noise directories
        and symlinks. Never follows links out of the tree."""
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in SNAPSHOT_SKIP_DIRS
                and not os.path.islink(os.path.join(dirpath, d)))
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                try:
                    if os.path.islink(full) or not os.path.isfile(full):
                        continue
                except OSError:
                    continue
                count += 1
                if count > SNAPSHOT_FILE_CEILING:
                    say("[TREE] more than %d files; the rest are not tracked"
                        % SNAPSHOT_FILE_CEILING)
                    return
                yield os.path.relpath(full, self.root).replace(os.sep, "/"), full

    def _snapshot(self) -> None:
        began = time.monotonic()
        kept = 0
        unhashed = 0
        for rel, full in self._walk():
            try:
                st = os.stat(full)
                if st.st_size > SNAPSHOT_HASH_CEILING:
                    # Too big to hash in the time this run has; size and
                    # mtime are all that tells a change on it.
                    self.index[rel] = (st.st_size, st.st_mtime_ns, st.st_mode)
                    self.digests[rel] = ""
                    self.unkept.add(rel)
                    unhashed += 1
                    continue
                if (st.st_size > SNAPSHOT_KEEP_BYTES
                        or kept + st.st_size > SNAPSHOT_TOTAL_BYTES):
                    # Hash streaming; never pull a big file into memory.
                    self.index[rel] = (st.st_size, st.st_mtime_ns, st.st_mode)
                    self.digests[rel] = self._digest(full)
                    self.unkept.add(rel)
                    continue
                with open(full, "rb") as fh:
                    data = fh.read()
            except OSError as error:
                say("[TREE] unreadable, not tracked: %s (%s)" % (rel, error))
                continue
            self.index[rel] = (st.st_size, st.st_mtime_ns, st.st_mode)
            self.digests[rel] = hashlib.sha256(data).hexdigest()
            self.originals[rel] = data
            kept += len(data)
        self.taken_ns = time.time_ns()
        if unhashed:
            say("[TREE] %d file(s) over %dMB tracked by size and mtime only"
                % (unhashed, SNAPSHOT_HASH_CEILING // (1024 * 1024)))
        say("[TREE] snapshot of %s in %.1fs: %d files, %d held (%dB), "
            "%d hashed only"
            % (self.root, time.monotonic() - began, len(self.index),
               len(self.originals), kept, len(self.unkept)))

    def files(self) -> list:
        return sorted(self.index)

    def current_files(self) -> list:
        return [rel for rel, _ in self._walk()]

    # ---- paths and IO -------------------------------------------------

    def absolute(self, path: str) -> str:
        root = os.path.normpath(self.root)
        joined = os.path.normpath(os.path.join(root, path))
        if joined != root and not joined.startswith(root + os.sep):
            raise ToolFault("path escapes the repository: %s" % path)
        return joined

    def relative(self, path: str) -> str:
        return os.path.relpath(self.absolute(path), self.root).replace(os.sep, "/")

    def read_bytes(self, path: str) -> bytes:
        full = self.absolute(path)
        if not os.path.isfile(full):
            raise ToolFault("no such file: %s" % path)
        with open(full, "rb") as fh:
            return fh.read()

    def read(self, path: str) -> str:
        """The file as text, byte-faithful: bytes that are not UTF-8 come
        back as surrogates and line endings are left alone, so text that
        goes through `edit` and back writes the same bytes it read."""
        return self.read_bytes(path).decode("utf-8", "surrogateescape")

    def write_bytes(self, path: str, data: bytes) -> None:
        full = self.absolute(path)
        rel = self.relative(path)
        if rel in self.unkept and rel not in self.originals:
            # About to overwrite a file we only hashed: keep its bytes now
            # so the patch and the restore still have the original.
            try:
                with open(full, "rb") as fh:
                    self.originals[rel] = fh.read()
                self.unkept.discard(rel)
            except OSError:
                pass
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(data)

    def write(self, path: str, text: str) -> None:
        self.write_bytes(path, text.encode("utf-8", "surrogateescape"))

    # ---- change detection ---------------------------------------------

    @staticmethod
    def _digest(full: str) -> str:
        digest = hashlib.sha256()
        with open(full, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def changed_paths(self) -> list:
        """[(rel, kind)] with kind one of modified / added / deleted."""
        out = []
        seen = set()
        for rel, full in self._walk():
            seen.add(rel)
            known = self.index.get(rel)
            if known is None:
                out.append((rel, "added"))
                continue
            try:
                st = os.stat(full)
                # Same size and mtime means unchanged, unless the mtime is
                # so close to the snapshot that a rewrite could share it
                # (filesystem clocks tick coarsely). Those are re-hashed.
                if (st.st_size == known[0] and st.st_mtime_ns == known[1]
                        and st.st_mtime_ns < self.taken_ns - RACY_WINDOW_NS):
                    continue
                known_digest = self.digests.get(rel, "")
                if not known_digest:
                    # Never hashed: size or mtime moving is the only signal.
                    if st.st_size != known[0] or st.st_mtime_ns != known[1]:
                        out.append((rel, "modified"))
                    continue
                if self._digest(full) != known_digest:
                    out.append((rel, "modified"))
            except OSError:
                continue
        for rel in self.index:
            if rel not in seen:
                out.append((rel, "deleted"))
        return sorted(out)

    def has_changes(self, budget: float = 10.0):
        try:
            return bool(self.changed_paths())
        except OSError:
            return None

    def original(self, rel: str):
        """Bytes of `rel` as it was at start, or None if not held."""
        data = self.originals.get(rel)
        if data is not None:
            return data
        if rel in self.unkept and self.pristine_dir:
            try:
                with open(os.path.join(self.pristine_dir, rel), "rb") as fh:
                    return fh.read()
            except OSError:
                return None
        return None

    def _current(self, rel: str) -> bytes:
        with open(self.absolute(rel), "rb") as fh:
            return fh.read()

    # ---- the patch ------------------------------------------------------

    def diff(self, budget: float = 60.0, whole=()) -> str:
        """The patch. Anything a section could not be built for is listed
        in self.dropped, so the caller can say the patch is incomplete.
        Paths in `whole` are written as whole-file hunks."""
        deadline = time.monotonic() + max(1.0, budget)
        pieces, skipped = [], []
        whole = set(whole or ())
        for rel, kind in self.changed_paths():
            if time.monotonic() > deadline:
                say("[TREE] diff ran out of time at %s" % rel)
                skipped.append("%s and later (out of time)" % rel)
                break
            before = b"" if kind == "added" else self.original(rel)
            if before is None:
                skipped.append("%s (original not held)" % rel)
                continue
            try:
                after = b"" if kind == "deleted" else self._current(rel)
            except OSError as error:
                skipped.append("%s (%s)" % (rel, error))
                continue
            if looks_binary(before) or looks_binary(after):
                skipped.append("%s (binary)" % rel)
                continue
            mode_before = (file_mode(self.index[rel][2])
                           if rel in self.index else None)
            try:
                mode_after = (None if kind == "deleted"
                              else file_mode(os.stat(self.absolute(rel)).st_mode))
            except OSError:
                mode_after = mode_before
            pieces.append(file_diff(rel, kind, before, after,
                                    mode_before, mode_after, rel in whole))
        for note in skipped:
            say("[PATCH] INCOMPLETE: left out %s" % note)
        self.dropped = list(skipped)
        return "".join(pieces)

    def fingerprint(self, changed: list = None) -> dict:
        """{rel: sha256 or None} of every changed path as it is on disk now;
        None for a deleted path, '?' for one that could not be read."""
        out = {}
        for rel, kind in (self.changed_paths() if changed is None else changed):
            if kind == "deleted":
                out[rel] = None
                continue
            try:
                out[rel] = self._digest(self.absolute(rel))
            except (OSError, ToolFault):
                out[rel] = "?"
        return out

    def round_trip(self, patch: str, expected: dict, budget: float = 20.0):
        """Apply the patch to a copy of the files as they were at start and
        compare the result with `expected`, the fingerprint of what the run
        left on disk. A patch that applies is not yet a patch that rebuilds
        the tree the model tested; this is the proof of that.

        Returns the rels that differ, [] when every file is reproduced, and
        None when git could not answer. Reads originals from memory or the
        pristine copy, never from the working tree, so it can run before
        or after restore()."""
        if not patch.strip() or not expected:
            return []
        if not shutil.which("git"):
            say("[PATCH] round trip skipped: no git")
            return []
        deadline = time.monotonic() + max(1.0, budget)
        try:
            base = tempfile.mkdtemp(prefix="roundtrip-")
        except OSError as error:
            say("[PATCH] round trip failed: %s" % error)
            return None
        try:
            rels = set(expected)
            for part in split_by_file(patch):
                rels.add(patch_section_path(part))
            for rel in sorted(r for r in rels if r):
                data = self.original(rel)
                if data is None:
                    continue            # added in this run: nothing to seed
                target = os.path.join(base, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(data)
                if rel in self.index:
                    os.chmod(target, stat.S_IMODE(self.index[rel][2]))
            fd, path = tempfile.mkstemp(prefix="ridges-patch-", suffix=".diff", dir=base)
            with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as fh:
                fh.write(patch)
            code, out = git(["apply", path], base, max(1.0, deadline - time.monotonic()))
            if code != 0:
                say("[PATCH] round trip could not apply: %s" % out.strip()[:200])
                return None
            differs = []
            for rel, digest in sorted(expected.items()):
                target = os.path.join(base, rel)
                got = self._digest(target) if os.path.isfile(target) else None
                if got != digest:
                    differs.append(rel)
            return differs
        except (OSError, ToolFault) as error:
            say("[PATCH] round trip failed: %s" % error)
            return None
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def _check(self, patch: str, budget: float):
        """True / False / None(unknown). Prefers `git apply --check`, which
        needs no repository, and falls back to a Python dry run when the
        image has no git."""
        if shutil.which("git"):
            try:
                fd, path = tempfile.mkstemp(prefix="ridges-patch-", suffix=".diff")
            except OSError as error:
                say("[PATCH] could not be written out: %s" % error)
                return None
            try:
                with os.fdopen(fd, "w", encoding="utf-8",
                               errors="surrogateescape") as fh:
                    fh.write(patch)
                code, out = git(["apply", "--check", path], self.root, budget)
            except OSError as error:
                say("[PATCH] could not be filled: %s" % error)
                return None
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            if code == GIT_TIMED_OUT:
                return None
            if code != 0:
                say("[PATCH] git apply --check: %s" % out.strip()[:200])
            return code == 0
        try:
            return patch_dry_run(patch, self.original)
        except Exception as error:
            say("[PATCH] dry run failed: %s: %s" % (type(error).__name__, error))
            return None

    def applies(self, patch: str, budget: float = 30.0):
        if not patch.strip():
            say("[PATCH] empty: the run finished without changing a line")
            return False
        answer = self._check(patch, budget)
        if answer is True:
            say("[PATCH] applies cleanly")
        elif answer is False:
            say("[PATCH] will not apply")
        else:
            say("[PATCH] could not be checked")
        return answer

    def applies_quietly(self, patch: str, budget: float):
        if not patch.strip():
            return False
        return self._check(patch, budget)

    def salvage(self, patch: str, budget: float = 45.0) -> str:
        deadline = time.monotonic() + max(1.0, budget)
        parts = split_by_file(patch)
        if len(parts) < 2:
            say("[PATCH] nothing to salvage: %d section(s)" % len(parts))
            return ""
        kept, dropped, unread = [], 0, 0
        for part in parts:
            left = deadline - time.monotonic()
            answer = self.applies_quietly(part, left) if left > 0 else None
            if answer is None:
                unread = len(parts) - len(kept) - dropped
                say("[PATCH] salvage left %d unread" % unread)
                break
            if answer:
                kept.append(part)
            else:
                dropped += 1
        if not kept:
            say("[PATCH] salvage kept nothing of %d" % len(parts))
            return ""
        joined = "".join(kept)
        whole = self.applies_quietly(joined, max(1.0, deadline - time.monotonic()))
        if whole is None:
            say("[PATCH] salvage could not re-check %d" % len(kept))
            return ""
        if not whole:
            say("[PATCH] salvage kept %d that will not apply together" % len(kept))
            return ""
        say("[PATCH] salvaged %d of %d, dropped %d, unread %d"
            % (len(kept), len(parts), dropped, unread))
        lost = [patch_section_path(p) for p in parts if p not in kept]
        note = "salvage dropped %d section(s): %s" % (
            len(lost), ", ".join(p for p in lost if p)[:300])
        say("[PATCH] INCOMPLETE: %s" % note)
        self.dropped.append(note)
        return joined

    # ---- restore ----------------------------------------------------------

    def put_back(self, rel: str, kind: str) -> str:
        """Return one path to its state at start. '' on success, else why not."""
        try:
            full = self.absolute(rel)
        except ToolFault as fault:
            return str(fault)
        if kind == "added":
            try:
                os.remove(full)
            except OSError as error:
                return str(error)
            return ""
        data = self.original(rel)
        if data is None:
            return "original not held"
        try:
            os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
            with open(full, "wb") as fh:
                fh.write(data)
            os.chmod(full, stat.S_IMODE(self.index[rel][2]))
        except OSError as error:
            return str(error)
        return ""

    def restore(self, budget: float = 60.0) -> None:
        deadline = time.monotonic() + max(1.0, budget)
        put_back, removed, failed = 0, 0, []
        for rel, kind in self.changed_paths():
            if time.monotonic() > deadline:
                failed.append("%s (out of time)" % rel)
                break
            why = self.put_back(rel, kind)
            if why:
                failed.append("%s (%s)" % (rel, why))
            elif kind == "added":
                removed += 1
            else:
                put_back += 1
        for note in failed:
            say("[TREE] could not restore %s" % note)
        say("[TREE] restored: %d file(s) put back, %d removed"
            % (put_back, removed))

    def revert_outside(self, allowed: list, budget: float = 30.0) -> list:
        """Undo every change to a path the instruction did not allow. The
        verifier hashes those files, so a stray edit there scores zero."""
        if not allowed:
            return []
        deadline = time.monotonic() + max(1.0, budget)
        undone = []
        for rel, kind in self.changed_paths():
            if rel in allowed:
                continue
            if time.monotonic() > deadline:
                say("[TREE] out of time reverting; %s left as is" % rel)
                break
            why = self.put_back(rel, kind)
            if why:
                say("[TREE] could not revert %s (%s)" % (rel, why))
            else:
                undone.append(rel)
        if undone:
            say("[TREE] reverted outside the allowed files: %s"
                % ", ".join(undone))
        return undone

    # ---- a separate pristine copy, for running the baseline suite ----------

    def make_pristine(self, budget: float = PRISTINE_BUDGET_SEC):
        """A copy of the tree as it was at start, or None. Built once.

        Only the files the snapshot could not hold in memory need to be
        in it, so only those are copied. A copy that cannot finish inside
        the budget, or that the disk cannot take, is thrown away whole: a
        partial copy would be trusted as an original and is worse than
        none."""
        if self.pristine_dir and os.path.isdir(self.pristine_dir):
            return self.pristine_dir
        wanted = [rel for rel in self.unkept if rel in self.index]
        if not wanted:
            say("[TREE] pristine copy not needed: every original is in memory")
            return None
        need = sum(self.index[rel][0] for rel in wanted)
        if need > PRISTINE_BYTES_CEILING:
            say("[TREE] no pristine copy: %dMB of unheld files is over the "
                "ceiling" % (need // (1024 * 1024)))
            return None
        began = time.monotonic()
        deadline = began + max(1.0, budget)
        try:
            base = tempfile.mkdtemp(prefix="start")
            free = shutil.disk_usage(base).free
        except OSError as error:
            say("[TREE] no pristine copy: %s" % error)
            return None
        if free < need * 2:
            say("[TREE] no pristine copy: %dMB free, %dMB needed"
                % (free // (1024 * 1024), need // (1024 * 1024)))
            shutil.rmtree(base, ignore_errors=True)
            return None
        where = os.path.join(base, "tree")
        copied = 0
        why = ""
        for rel in sorted(wanted):
            if time.monotonic() > deadline:
                why = "out of time after %d file(s)" % copied
                break
            target = os.path.join(where, rel)
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(self.absolute(rel), target)
                copied += 1
            except (OSError, ToolFault) as error:
                why = "%s: %s" % (rel, error)
                break
        if why:
            say("[TREE] no pristine copy: %s" % why)
            shutil.rmtree(base, ignore_errors=True)
            return None
        self.pristine_dir = where
        say("[TREE] pristine copy of %d unheld file(s) at %s in %.1fs"
            % (copied, where, time.monotonic() - began))
        return where


# =========================================================================
# search_files, workdir, package roots, inside
# =========================================================================

PROJECT_MARKERS = ("manage.py", "pyproject.toml", "setup.py", "requirements.txt",
                   "package.json", "go.mod", "Gemfile", "pom.xml",
                   "build.gradle", "composer.json", "Cargo.toml", "mix.exs",
                   "Makefile", "src")

SEARCH_SKIP_DIRS = tuple(sorted(SNAPSHOT_SKIP_DIRS)) + (
    "site-packages", "dist", "build", "target", "vendor", "coverage")


def _looks_like_project(path: str) -> bool:
    try:
        return any(os.path.exists(os.path.join(path, m)) for m in PROJECT_MARKERS)
    except OSError:
        return False


def workdir() -> str:
    override = os.getenv("RIDGES_WORKDIR")
    if override and os.path.isdir(override):
        return os.path.realpath(override)
    for candidate in ("/app", "/repo", "/workspace"):
        if os.path.isdir(candidate) and _looks_like_project(candidate):
            return os.path.realpath(candidate)
    cwd = os.path.realpath(os.getcwd())
    if cwd not in ("/", "/installed-agent") and _looks_like_project(cwd):
        return cwd
    return cwd


def search_files(root: str, pattern: str, mode: str = "content",
                 include: str = None, context: int = 0,
                 timeout: float = 15.0, where: str = None) -> str:
    """grep -r over the tree on disk. Paths come back relative to root."""
    flags = {"files": "-rlE", "count": "-rcE"}.get(mode, "-rEn")
    args = ["grep", flags, "--binary-files=without-match", "-I"]
    if context and mode == "content":
        args.append("-C%d" % max(0, min(20, context)))
    if include:
        args.append("--include=" + include)
    for skip in SEARCH_SKIP_DIRS:
        args.append("--exclude-dir=" + skip)
    target = "."
    if where and where.strip(" ./"):
        target = "./" + where.strip().strip("/")
    args += ["--", pattern, target]
    try:
        done = subprocess.run(args, cwd=root, capture_output=True, text=True,
                              timeout=timeout, errors="replace")
    except (subprocess.TimeoutExpired, OSError):
        return ""
    rows = []
    for line in (done.stdout or "").splitlines():
        if line.startswith("./"):
            line = line[2:]
        if mode == "count" and line.endswith(":0"):
            continue
        rows.append(line)
    return "\n".join(rows) + ("\n" if rows else "")


def inside(path: str, root: str) -> bool:
    try:
        path, root = os.path.realpath(path), os.path.realpath(root)
        if (os.path.splitdrive(path)[0].lower()
                != os.path.splitdrive(root)[0].lower()):
            return False
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError, TypeError):
        return True


# =========================================================================
# Diff parsing
# =========================================================================

def patch_section_path(section: str) -> str:
    """The b/ path a `diff --git` section is about, or ''."""
    first = (section or "").split("\n", 1)[0]
    match = re.match(r"diff --git a/(.*) b/(.*)$", first)
    return match.group(2) if match else ""


def split_by_file(patch: str) -> list:
    out, current = [], []
    for line in (patch or "").splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                out.append("".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        out.append("".join(current))
    return out


def path_tail(name: str, root: str = "") -> str:
    text = str(name or "").replace("\\", "/")
    base = str(root or "").replace("\\", "/").rstrip("/")
    if base and text.startswith(base + "/"):
        text = text[len(base) + 1:]
    while text.startswith("./"):
        text = text[2:]
    return text


CANDIDATE_BUDGET_SEC = 12.0
CANDIDATE_TERMS_MAX = 16


def candidate_paths(root: str, statement: str, limit: int = 30,
                    budget: float = CANDIDATE_BUDGET_SEC) -> list:
    """grep -r over the statement's rare terms, quoted terms first, inside a
    fixed time budget so a large tree cannot stall the start of the run.
    Returns up to `limit` paths."""
    quoted = {m.lower() for m in QUOTED_RE.findall(statement)}
    terms = {w.lower() for w in WORD_RE.findall(statement)} - COMMON_WORDS
    terms |= quoted
    terms = {t for t in terms if len(t) > 3}
    if not terms:
        return []
    ordered = sorted(terms, key=lambda t: (t not in quoted, -len(t), t))
    deadline = time.monotonic() + max(1.0, budget)
    scores = {}
    for term in ordered[:CANDIDATE_TERMS_MAX]:
        left = deadline - time.monotonic()
        if left <= 0:
            say("[SEED] out of time after %d term(s)" % len(scores))
            break
        out = search_files(root, term, mode="files",
                           timeout=max(0.5, min(3.0, left)))
        hits = [p for p in out.splitlines() if p]
        if not hits or len(hits) > 60:
            continue
        weight = (1.0 / len(hits)) * (3.0 if term in quoted else 1.0)
        for path in hits:
            penalty = 0.25 if ("test" in path.lower()
                               or path.startswith("docs/")) else 1.0
            scores[path] = scores.get(path, 0.0) + weight * penalty
    return [p for p, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:limit]


# =========================================================================
# Scope — what the statement itself says about where the change goes
# =========================================================================
#
# Most instructions name the file, often the method, and always the checks.
# Reading those out deterministically costs nothing and removes the search
# the model would otherwise do across the whole tree. Nothing here is tied
# to one repository: the regexes match phrasing, and every path is verified
# on disk before it is trusted.

SCOPE_FILE_RES = (
    re.compile(
        r"(?:limit(?:ed)?\s+(?:production\s+)?changes?\s+to|"
        r"you\s+may\s+(?:only\s+)?(?:edit|change|modify)(?:\s+only)?|"
        r"(?:edit|change|modify)\s+only|"
        r"only\s+(?:edit|change|modify)|"
        r"confine\s+(?:your\s+)?changes?\s+to|"
        r"production\s+changes?\s+(?:are\s+)?limited\s+to)"
        r"\s*[:\s]*`([^`\n]+)`", re.I),
    re.compile(r"`([^`\s]+\.[A-Za-z0-9]{1,6})`\s*,?\s*(?:specifically|and\s+only)\b",
               re.I),
)
SCOPE_METHOD_RE = re.compile(r"\bspecifically\s+`([^`\n]+)`", re.I)
SCOPE_NOT_SYMBOL = (".py", ".js", ".ts", ".go", ".rb", ".php", ".sql", ".md",
                    ".json", ".yaml", ".yml", ".toml")
CHECK_RUNNERS = frozenset((
    "python", "python3", "pytest", "ruff", "flake8", "mypy", "black",
    "node", "npm", "npx", "pnpm", "yarn", "deno", "jest", "vitest", "eslint", "tsc",
    "go", "gofmt", "ruby", "bundle", "rails", "rspec", "rake",
    "cargo", "mvn", "gradle", "php", "composer", "mix", "make",
    "manage.py", "poetry", "uv", "dotnet"))
ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
FENCED_BLOCK = re.compile(r"```(?:bash|sh|shell|console|zsh)?[ \t]*\r?\n(.*?)```", re.S)
INLINE_SPAN = re.compile(r"`([^`]{4,400})`")
CD_LINE = re.compile(r"^cd\s+(\S+)\s*$")
SCOPE_EXCERPT_CHARS = 14_000
CHECK_COMMANDS_MAX = 4
REGION_LINES_MAX = 250
IMPORT_LINES_MAX = 60

# One definition-line pattern per language; %s is the escaped name.
LANG_DEF_RES = {
    "js": (r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+%s\s*\(|"
           r"^\s*(?:export\s+)?(?:const|let|var)\s+%s\s*=|"
           r"^\s*(?:static\s+)?(?:async\s+)?%s\s*\([^;{}]*\)\s*\{|"
           r"^\s*%s\s*:\s*(?:async\s*)?(?:function\b|\()"),
    "go": r"^func\s+(?:\([^)]*\)\s*)?%s\s*[\(\[]",
    "rb": r"^\s*def\s+(?:self\.)?%s\b",
    "php": r"^\s*(?:(?:public|private|protected|static|final|abstract)\s+)*function\s+%s\s*\(",
}
LANG_OF_EXT = {".js": "js", ".mjs": "js", ".cjs": "js", ".jsx": "js", ".ts": "js",
               ".tsx": "js", ".go": "go", ".rb": "rb", ".php": "php"}
IMPORT_LINE = re.compile(
    r"^\s*(?:import\b|from\b|require\b|package\b|use\b|"
    r"(?:const|let|var)\s+.*=\s*(?:await\s+)?(?:require|import)\s*\(|"
    r"#|//|/\*|\*|\)|\"|'|$)")


SCOPE_TOP_DENY = frozenset((
    "docs", "doc", "fixtures", "examples", "example", "tests", "test",
    "scripts", "build", "dist"))


def _resolve_named(rel: str, root: str) -> str:
    """The path as it exists under root, or '' when it does not.

    A path written from inside the app's package dir, e.g. `ipam/x.py`
    for netbox/ipam/x.py, is looked for one level down. When more than one
    top-level dir holds it, the answer is '' rather than a guess: a wrong
    scope would gate every edit to the real target."""
    if os.path.isfile(os.path.join(root, rel)):
        return rel
    try:
        tops = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))
                      and not d.startswith("."))
    except OSError:
        return ""
    hits = [top + "/" + rel for top in tops
            if os.path.isfile(os.path.join(root, top, rel))]
    if len(hits) > 1:
        hits = [h for h in hits if h.split("/", 1)[0] not in SCOPE_TOP_DENY] or hits
    if len(hits) > 1:
        marked = [h for h in hits
                  if _looks_like_project(os.path.join(root, h.split("/", 1)[0]))]
        hits = marked or hits
    if len(hits) == 1:
        return hits[0]
    if hits:
        say("[SCOPE] `%s` is under %d top-level dirs (%s); not guessing"
            % (rel, len(hits), ", ".join(h.split("/", 1)[0] for h in hits)))
    return ""


def named_files(statement: str, root: str) -> tuple:
    """(files that exist, files named but not found), in statement order."""
    found, missing = [], []
    for pattern in SCOPE_FILE_RES:
        for hit in pattern.findall(statement or ""):
            rel = path_tail(path_tail(hit.strip(), root), "/app").strip("/")
            if not rel or "/" not in rel and "." not in rel:
                continue
            if rel in found or rel in missing or TEST_PATH.search(rel):
                continue
            resolved = _resolve_named(rel, root)
            if resolved:
                if resolved not in found:
                    found.append(resolved)
            else:
                missing.append(rel)
    return found, missing


def named_symbol(statement: str) -> tuple:
    """(symbol, owner, name) after the word `specifically`, or empties."""
    match = SCOPE_METHOD_RE.search(statement or "")
    if not match:
        return "", "", ""
    symbol = re.sub(r"\(\s*\)$", "", match.group(1).strip()).strip()
    if (not re.match(r"^[A-Za-z_][\w.:]*$", symbol)
            or symbol.lower().endswith(SCOPE_NOT_SYMBOL)):
        return "", "", ""
    if "." in symbol:
        owner, name = symbol.rsplit(".", 1)
    elif "::" in symbol:
        owner, name = symbol.rsplit("::", 1)
    else:
        owner, name = "", symbol
    return symbol, owner, name


def _python_region(source: str, owner: str, name: str) -> tuple:
    try:
        module = parse_python(source)
    except (SyntaxError, ValueError):
        return None, None, "whole"
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef)
    matches = []
    if owner:
        owner_name = owner.split(".")[-1].split("::")[-1]
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef) and node.name == owner_name:
                matches += [c for c in node.body
                            if isinstance(c, kinds) and c.name == name]
    else:
        matches = [n for n in module.body
                   if isinstance(n, kinds) and n.name == name]
    if not matches:
        matches = [n for n in ast.walk(module)
                   if isinstance(n, kinds) and n.name == name]
    if len(matches) != 1:
        return None, None, "whole"
    node = matches[0]
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, getattr(node, "end_lineno", node.lineno), "ast"


def _brace_end(lines: list, start: int) -> int:
    """Last line (1-based) of a brace-delimited block starting at `start`."""
    depth = 0
    opened = False
    last = min(len(lines), start + REGION_LINES_MAX - 1)
    for number in range(start, last + 1):
        text = lines[number - 1]
        for char in text:
            if char == "{":
                depth += 1
                opened = True
            elif char == "}":
                depth -= 1
        if opened and depth <= 0:
            return number
    return last


def _ruby_end(lines: list, start: int) -> int:
    indent = len(lines[start - 1]) - len(lines[start - 1].lstrip())
    last = min(len(lines), start + REGION_LINES_MAX - 1)
    for number in range(start + 1, last + 1):
        text = lines[number - 1]
        if not text.strip():
            continue
        lead = len(text) - len(text.lstrip())
        if lead <= indent and text.strip().startswith("end"):
            return number
    return last


def _regex_region(source: str, path: str, owner: str, name: str) -> tuple:
    lang = LANG_OF_EXT.get(os.path.splitext(path)[1].lower())
    if not lang:
        return None, None, "whole"
    escaped = re.escape(name)
    pattern = re.compile(LANG_DEF_RES[lang].replace("%s", escaped))
    lines = source.split("\n")
    hits = [n for n, text in enumerate(lines, 1) if pattern.match(text)]
    if len(hits) > 1 and owner:
        narrowed = [n for n in hits if owner.split("::")[-1] in lines[n - 1]]
        hits = narrowed or hits
    if len(hits) != 1:
        return None, None, "whole"
    start = hits[0]
    end = _ruby_end(lines, start) if lang == "rb" else _brace_end(lines, start)
    return start, end, "regex"


def resolve_region(source: str, path: str, owner: str, name: str) -> tuple:
    """(start, end, kind): 1-based inclusive lines of the named definition,
    kind 'ast' for Python, 'regex' for other languages, 'whole' when the
    name is missing, ambiguous, or the language is unknown."""
    if not name or not source:
        return None, None, "whole"
    if path.endswith(".py"):
        return _python_region(source, owner, name)
    return _regex_region(source, path, owner, name)


def import_block(source: str, path: str) -> str:
    if path.endswith(".py"):
        try:
            module = parse_python(source)
            found = []
            for item in module.body:
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    text = ast.get_source_segment(source, item)
                    if text:
                        found.append(text)
            return "\n".join(found)
        except (SyntaxError, ValueError):
            pass
    kept = []
    for text in source.split("\n")[:IMPORT_LINES_MAX]:
        if IMPORT_LINE.match(text):
            kept.append(text)
        elif kept:
            break
    return "\n".join(kept).strip()


def scope_excerpt(tree: "Tree", scope: dict) -> str:
    parts = []
    for index, rel in enumerate(scope.get("files") or []):
        try:
            source = tree.read(rel)
        except ToolFault:
            continue
        lines = source.split("\n")
        total = len(lines)
        if index == 0 and scope.get("region"):
            start, end = scope["region"]
            parts.append(
                "FILE %s (%d lines)\nTARGET %s, lines %d-%d\n\nIMPORTS\n%s\n\n"
                "DEFINITION (verbatim, current code)\n%s"
                % (rel, total, scope.get("symbol") or "", start, end,
                   import_block(source, rel) or "(none)",
                   "\n".join(lines[start - 1:end])))
        elif index == 0:
            parts.append("FILE %s (%d lines), whole file\n\n%s" % (rel, total, source))
        else:
            outline = ""
            if rel.endswith(".py"):
                try:
                    outline = "\n".join(outline_source(source))
                except SyntaxError:
                    outline = ""
            parts.append("FILE %s (%d lines)\n\nIMPORTS\n%s\n\nOUTLINE\n%s"
                         % (rel, total, import_block(source, rel) or "(none)",
                            outline or "(read it for detail)"))
    return clip("\n\n".join(parts), SCOPE_EXCERPT_CHARS, "scope excerpt")


def _check_line(line: str) -> str:
    """The command if the line runs a known checker, else ''."""
    line = line.strip()
    if line.startswith("$ "):
        line = line[2:].strip()
    if not line or line.startswith("#"):
        return ""
    try:
        argv = shlex.split(line)
    except ValueError:
        argv = line.split()
    index = 0
    while index < len(argv) and ENV_ASSIGN.match(argv[index]):
        index += 1
    rest = argv[index:]
    if not rest or os.path.basename(rest[0]) not in CHECK_RUNNERS:
        return ""
    return line


def parse_checks(statement: str) -> list:
    """The commands the statement asks the run to execute, at most four."""
    found = []
    text = statement or ""
    for block in FENCED_BLOCK.findall(text):
        joined = re.sub(r"\\\r?\n\s*", " ", block)
        prefix = ""
        for raw in joined.splitlines():
            stripped = raw.strip()
            if stripped.startswith("$ "):
                stripped = stripped[2:].strip()
            moved = CD_LINE.match(stripped)
            if moved:
                prefix = "cd %s && " % moved.group(1)
                continue
            command = _check_line(stripped)
            if command and prefix + command not in found:
                found.append(prefix + command)
    for span in INLINE_SPAN.findall(text):
        span = " ".join(span.split())
        tokens = span.split()
        # An inline mention needs a target to be a command rather than a
        # tool name: `ruff check --no-cache` alone is advice, not a check.
        if len(tokens) < 3 or not any("/" in t or "." in t for t in tokens[1:]):
            continue
        command = _check_line(span)
        if command and command not in found:
            found.append(command)
    return found[:CHECK_COMMANDS_MAX]


def parse_scope(statement: str, tree: "Tree") -> dict:
    """Everything the statement says about where the change goes and how
    it is checked. Never raises."""
    scope = {"files": [], "missing": [], "symbol": "", "owner": "", "name": "",
             "region": None, "kind": "", "excerpt": "", "checks": []}
    try:
        scope["files"], scope["missing"] = named_files(statement, tree.root)
        scope["symbol"], scope["owner"], scope["name"] = named_symbol(statement)
        scope["checks"] = parse_checks(statement)
        if scope["files"]:
            try:
                source = tree.read(scope["files"][0])
            except ToolFault:
                source = ""
            start, end, kind = resolve_region(source, scope["files"][0],
                                              scope["owner"], scope["name"])
            scope["kind"] = kind
            if start:
                scope["region"] = (start, end)
            scope["excerpt"] = scope_excerpt(tree, scope)
    except Exception as error:
        traceback.print_exc()
        say("[SCOPE] parse failed: %s: %s" % (type(error).__name__, error))
    say("[SCOPE] files=%s symbol=%s kind=%s region=%s checks=%d missing=%s"
        % (",".join(scope["files"]) or "-", scope["symbol"] or "-",
           scope["kind"] or "-", scope["region"] or "-", len(scope["checks"]),
           ",".join(scope["missing"]) or "-"))
    return scope


def repo_sketch(files: list) -> str:
    if not files:
        return "No files found under the working directory."
    tops, kinds = {}, {}
    for path in files:
        top = path.split("/", 1)[0]
        tops[top] = tops.get(top, 0) + 1
        ext = os.path.splitext(path)[1] or "(none)"
        kinds[ext] = kinds.get(ext, 0) + 1
    return ("%d files.\nTop level: %s\nExtensions: %s"
            % (len(files),
               ", ".join("%s (%d)" % (k, v) for k, v in
                         sorted(tops.items(), key=lambda kv: -kv[1])[:12]),
               ", ".join("%s (%d)" % (k, v) for k, v in
                         sorted(kinds.items(), key=lambda kv: -kv[1])[:8])))


# =========================================================================
# Source analysis helpers
# =========================================================================

def outline_source(source: str) -> list:
    rows = []

    def walk(node, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = ("class" if isinstance(child, ast.ClassDef)
                        else "async def" if isinstance(child, ast.AsyncFunctionDef)
                        else "def")
                rows.append((child.lineno, getattr(child, "end_lineno", child.lineno),
                             depth, "%s %s" % (kind, child.name)))
                walk(child, depth + 1)
            else:
                walk(child, depth)
    walk(parse_python(source), 0)
    rows.sort()
    return ["%5d-%-5d %s%s" % (start, end, "  " * depth, name)
            for start, end, depth, name in rows]


def visible_definitions(source: str) -> list:
    try:
        tree = parse_python(source)
    except SyntaxError:
        return []
    found = []

    def walk(node, prefix, inside):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async" if isinstance(child, ast.AsyncFunctionDef) else "def"
                if not inside:
                    found.append("%s %s%s" % (kind, prefix, child.name))
                walk(child, prefix + child.name + ".", True)
            elif isinstance(child, ast.ClassDef):
                if not inside:
                    found.append("class %s%s" % (prefix, child.name))
                walk(child, prefix + child.name + ".", inside)
            else:
                walk(child, prefix, inside)
    walk(tree, "", False)
    return sorted(found)

# =========================================================================
# Shell + ShellPool
# =========================================================================

STILL_RUNNING = "[still running]"


def bounded_output_file(path: str, cap: int, label: str = "output") -> str:
    """At most cap bytes, keeping both ends. Seeks past the middle."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    with open(path, "rb") as fh:
        if size <= cap:
            return fh.read().decode("utf-8", "replace")
        half = max(1, cap // 2)
        head = fh.read(half)
        fh.seek(max(0, size - half))
        tail = fh.read(half)
    note = "\n... [%d characters of %s elided] ...\n" % (
        max(0, size - len(head) - len(tail)), label)
    room = max(0, cap - len(note.encode("utf-8")))
    left, right = room // 2, room - room // 2
    data = head[:left] + note.encode("utf-8") + (tail[-right:] if right else b"")
    return data.decode("utf-8", "replace")


# ---- the container's own limits ----------------------------------------
#
# The task gives the container a fixed memory limit and CPU count. A child
# that outgrows the per-process cap is killed on its own; a set of children
# that outgrows the container is killed by the OOM killer, which may take
# the agent itself, and then no patch comes back at all.

CGROUP_MEMORY = (
    ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
    ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
     "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
CGROUP_CPU = ("/sys/fs/cgroup/cpu.max",
              "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        return fh.read().strip()


def memory_limit_bytes() -> int:
    """The container's memory limit, or 0 when there is none to be found."""
    for _, limit_path in CGROUP_MEMORY:
        try:
            raw = _read_text(limit_path)
            if raw == "max":
                return 0
            limit = int(raw)
            if 0 < limit < 1 << 50:
                return limit
        except (OSError, ValueError):
            continue
    return 0


def memory_used_bytes() -> int:
    for used_path, _ in CGROUP_MEMORY:
        try:
            return int(_read_text(used_path))
        except (OSError, ValueError):
            continue
    return 0


def memory_share() -> float:
    """Container memory in use as a fraction of its limit; -1 when unknown."""
    limit = memory_limit_bytes()
    if limit <= 0:
        return -1.0
    used = memory_used_bytes()
    return used / float(limit) if used > 0 else -1.0


def cpu_quota() -> int:
    """CPUs the container may use, floored at 1."""
    try:
        quota, period = _read_text(CGROUP_CPU[0]).split()[:2]
        if quota != "max":
            return max(1, int(round(int(quota) / float(period))))
    except (OSError, ValueError, IndexError):
        try:
            quota, period = int(_read_text(CGROUP_CPU[1])), int(_read_text(CGROUP_CPU[2]))
            if quota > 0 and period > 0:
                return max(1, int(round(quota / float(period))))
        except (OSError, ValueError):
            pass
    return max(1, os.cpu_count() or 1)


def child_cap(limit: int, jobs: int) -> int:
    """Address-space cap for one child: the container's limit less the
    agent's headroom, shared between the jobs allowed at once, kept between
    a floor a real test run needs and the fixed per-process cap."""
    if limit <= 0:
        return CHILD_MEMORY_BYTES
    share = (limit - MEMORY_HEADROOM_BYTES) // max(1, jobs)
    return int(min(CHILD_MEMORY_BYTES, max(CHILD_CAP_FLOOR_BYTES, share)))


def child_hook(cap: int):
    """What every child does between fork and exec: lower its priority,
    volunteer for the OOM killer ahead of the agent, and cap its address
    space. Each step is best-effort."""
    def apply() -> None:
        try:
            os.nice(CHILD_NICE)
        except OSError:
            pass
        try:
            with open("/proc/self/oom_score_adj", "w") as fh:
                fh.write(str(CHILD_OOM_SCORE_ADJ))
        except OSError:
            pass
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            wanted = cap
            if hard != resource.RLIM_INFINITY:
                wanted = min(wanted, hard)
            resource.setrlimit(resource.RLIMIT_AS, (wanted, hard))
        except Exception:
            pass
    return apply


class Shell:
    counter = 0

    def __init__(self, command: str, cwd: str, cap: int = CHILD_MEMORY_BYTES,
                 cpus: int = 2) -> None:
        Shell.counter += 1
        self.name = "job%d" % Shell.counter
        self.command = command
        self.started = time.monotonic()
        self.sink = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", errors="replace", suffix=".out",
            delete=False)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CI"] = "1"
        jobs = str(max(1, int(cpus)))
        for name, value in (("GOMAXPROCS", jobs), ("CARGO_BUILD_JOBS", jobs),
                            ("MAKEOPTS", "-j" + jobs), ("MAKEFLAGS", "-j" + jobs),
                            ("npm_config_jobs", jobs), ("BUNDLE_JOBS", jobs)):
            env.setdefault(name, value)
        try:
            self.process = subprocess.Popen(
                ["bash", "-lc", command], cwd=cwd, env=env,
                stdout=self.sink, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=child_hook(cap))
        except Exception:
            self.sink.close()
            try:
                os.unlink(self.sink.name)
            except OSError:
                pass
            raise

    def _text(self) -> str:
        try:
            self.sink.flush()
            return bounded_output_file(self.sink.name, READ_OUTPUT_CAP)
        except OSError:
            return ""

    def wait(self, timeout: float) -> tuple:
        try:
            self.process.wait(timeout=max(1.0, timeout))
            return True, self._text()
        except subprocess.TimeoutExpired:
            return False, self._text()

    def finished(self) -> bool:
        return self.process.poll() is not None

    def drain(self) -> str:
        if self.process.poll() is None:
            return self._text() + "\n" + STILL_RUNNING
        return self._text()

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    self.process.kill()
                except Exception:
                    pass
        try:
            self.sink.close()
            os.unlink(self.sink.name)
        except OSError:
            pass


class ShellPool:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.jobs = {}
        self.limit = memory_limit_bytes()
        self.cpus = cpu_quota()
        self.jobs_allowed = (1 if 0 < self.limit < SMALL_CONTAINER_BYTES
                             else BACKGROUND_JOBS_MAX)
        self.child_cap = child_cap(self.limit, self.jobs_allowed)

    def describe(self) -> str:
        limit = ("%dMB" % (self.limit // (1024 * 1024))) if self.limit else "unknown"
        return ("memory %s, cpus %d, child cap %dMB, %d job(s) at a time"
                % (limit, self.cpus, self.child_cap // (1024 * 1024), self.jobs_allowed))

    def start(self, command: str, background: bool = False) -> Shell:
        running = [j for j in self.jobs.values() if not j.finished()]
        share = memory_share()
        if background and share > MEMORY_REFUSE_SHARE:
            raise ToolFault(
                "memory is at %d%% of the container's limit; a new background "
                "job would risk the whole run. Collect or stop a running job "
                "with bash_poll(job, stop=true), or run the command in the "
                "foreground once memory is back." % (share * 100))
        allowed = 1 if share > MEMORY_STOP_SHARE else self.jobs_allowed
        if len(running) >= allowed:
            raise ToolFault(
                "%d command(s) still running (%s)%s; collect one with "
                "bash_poll, or end one with bash_poll(job, stop=true), "
                "before starting another"
                % (len(running), ", ".join(j.name for j in running),
                   " and memory is at %d%% of the limit" % (share * 100)
                   if share > MEMORY_STOP_SHARE else ""))
        job = Shell(command, self.cwd, cap=self.child_cap, cpus=self.cpus)
        self.jobs[job.name] = job
        return job

    def get(self, name: str) -> Shell:
        job = self.jobs.get(name)
        if job is None:
            raise ToolFault("no background job named %s" % name)
        return job

    def close(self) -> None:
        for job in list(self.jobs.values()):
            try:
                self._report(job, job.drain())
                job.stop()
            except Exception:
                pass
        self.jobs.clear()

    @staticmethod
    def _report(job: Shell, out: str) -> None:
        tail = ""
        for line in reversed((out or "").splitlines()):
            if line.strip() and line.strip() != STILL_RUNNING:
                tail = line.strip()
                break
        say("[SHELL] %.1fs %dc :: %s :: %s"
            % (time.monotonic() - job.started, len(out or ""),
               " ".join(job.command.split())[:SHELL_REPORT_CAP],
               tail[:SHELL_REPORT_CAP]))


# =========================================================================
# Tool schemas — the surface the driver sees
# =========================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file, optionally a line range. Prefer a range once you "
                "know where you are looking. A read that does not fit stops "
                "early and the header says where to start again."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Path relative to the repository root."},
                    "start": {"type": "integer",
                              "description": "First line, 1-based."},
                    "count": {"type": "integer",
                              "description": "How many lines to return."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "Extended-regex search across the repository. Use mode=files "
                "first to see where the matches are, then read narrowly."),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string",
                             "description": "Directory to search under. Defaults to the whole repository."},
                    "mode": {"type": "string",
                             "enum": ["content", "files", "count"],
                             "description": "content returns matching lines; files returns paths only; count returns per-file totals."},
                    "include": {"type": "string",
                                "description": "Only search paths matching this glob, e.g. *.py"},
                    "context": {"type": "integer",
                                "description": "Lines of surrounding code to return with each match, up to 20."},
                    "head_limit": {"type": "integer",
                                   "description": "Stop after this many matching lines."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "List files whose path matches a glob, e.g. src/**/*.py",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "outline",
            "description": (
                "Index one Python file: every class and function in it with "
                "the lines it spans, and nothing of what they say. Call it "
                "before reading a long file, then read the range it names."),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Replace an exact span of text in a file. Set replace_all when "
                "the same span occurs several places and every one of them "
                "needs the same change."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string",
                            "description": "Exact text to find, including indentation."},
                    "new": {"type": "string",
                            "description": "Replacement text."},
                    "replace_all": {"type": "boolean",
                                    "description": "Replace every occurrence instead of requiring a unique one."},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Write a whole file, creating it or overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the repository root. Set background "
                "for anything slow, such as a test suite, and collect it "
                "later with bash_poll instead of waiting."),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "background": {"type": "boolean"},
                    "timeout": {"type": "integer",
                                "description": "Seconds to wait when not backgrounded."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash_poll",
            "description": ("Collect output from a background command started "
                            "with bash. Blocks until it finishes or the wait "
                            "runs out; a test that takes two minutes needs one "
                            "poll with wait=150, not six short ones."),
            "parameters": {
                "type": "object",
                "properties": {
                    "job": {"type": "string"},
                    "wait": {"type": "integer",
                             "description": "Seconds to block for, up to 150. Default 75."},
                    "stop": {"type": "boolean",
                             "description": "Kill the job instead of waiting; its output so far is returned and its slot freed."},
                },
                "required": ["job"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restore_file",
            "description": ("Put one file back exactly as it was when the run "
                            "started, or remove a file this run created. Use "
                            "it to undo an edit outside the files the "
                            "instruction allows."),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Finish the run. Call this only once the change is complete, "
                "the checks the instruction names have been run, and your "
                "reasoning is against the final code."),
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]


# The three stages see different subsets.
def locator_tools() -> list:
    names = {"read_file", "search_text", "find_files", "outline"}
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]


def planner_tools() -> list:
    names = {"read_file", "outline", "search_text"}
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]


# =========================================================================
# Kit — the dispatcher
# =========================================================================

class Finished(Exception):
    """The model called submit and the checks accepted."""


class Kit:
    """Tool dispatch. A read-only call repeated with identical arguments is
    refused rather than served."""

    def __init__(self, tree: Tree, pool: ShellPool, allowance: Allowance,
                 warden=None, label: str = "", scope: dict = None) -> None:
        self.tree = tree
        self.pool = pool
        self.allowance = allowance
        self.warden = warden
        self.label = label
        self.scope = scope or {}
        self.allowed = list(self.scope.get("files") or [])
        self.polls: dict = {}
        self.seen: dict = {}
        self.last_refusal = ""      # "empty" | "warden" after a refused submit
        self.fence = Beacon("fence")
        self.bg = Beacon("bgshell")
        self.statement = getattr(warden, "statement", "") if warden else ""

    # ---- dispatch --------------------------------------------------

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, "do_" + name, None)
        if handler is None:
            raise ToolFault("no tool named %s" % name)
        return handler(args)

    def guard_repeat(self, key: str) -> None:
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] > REPEAT_READ_CEILING:
            raise ToolFault(
                "this exact call was already answered %d times and nothing "
                "has changed since. Scroll up and use the earlier result."
                % (self.seen[key] - 1))

    def note_read(self, what: str) -> None:
        say("[READ]%s %s" % (" " + self.label if self.label else "", what))

    def scope_gate(self, path: str) -> None:
        """Refuse a write outside the files the instruction allows."""
        if not self.allowed:
            return
        rel = self.tree.relative(path)
        if rel in self.allowed:
            return
        raise ToolFault(
            "the instruction limits production changes to %s; %s is outside "
            "that scope, and anything outside it is reverted at hand-in. Make "
            "the change inside the named file%s."
            % (", ".join(self.allowed), rel, "s" if len(self.allowed) > 1 else ""))

    # ---- read_file -------------------------------------------------

    def do_read_file(self, args: dict) -> str:
        path = str(args.get("path") or "")
        start = args.get("start")
        count = args.get("count")
        self.guard_repeat("read:%s:%s:%s" % (path, start, count))
        text = self.tree.read(path)
        lines = text.splitlines()
        first = max(1, int(start or 1))
        wanted = int(count) if count and int(count) > 0 else 0
        asked = min(len(lines), first + wanted - 1) if wanted else len(lines)
        rows, used, last = [], 0, first - 1
        for n in range(first, asked + 1):
            row = "%6d\t%s" % (n, lines[n - 1])
            if rows and used + len(row) + 1 > READ_OUTPUT_CAP:
                break
            if not rows:
                row = clip(row, READ_OUTPUT_CAP, "line")
            rows.append(row)
            used += len(row) + 1
            last = n
        if not rows:
            served = ("%s is empty" % path if not lines
                      else "%s has %d line(s); start=%d is past the end"
                      % (path, len(lines), first))
            self.note_read("read_file %s:%d- of %d -> %dc"
                           % (path, first, len(lines), len(served)))
            return served
        while True:
            head = "%s lines %d-%d of %d" % (path, first, last, len(lines))
            if last < asked:
                head += " -- pass start=%d to read on" % (last + 1)
            if len(head) + 1 + used <= READ_OUTPUT_CAP:
                break
            if len(rows) > 1:
                used -= len(rows.pop()) + 1
                last -= 1
                continue
            rows[0] = clip(rows[0], max(0, READ_OUTPUT_CAP - len(head) - 1), "line")
            used = len(rows[0]) + 1
            break
        served = head + "\n" + "\n".join(rows)
        self.note_read("read_file %s:%d-%d of %d -> %dc"
                       % (path, first, last, len(lines), len(served)))
        return served

    # ---- outline ---------------------------------------------------

    def do_outline(self, args: dict) -> str:
        path = str(args.get("path") or "")
        self.guard_repeat("outline:%s" % path)
        text = self.tree.read(path)
        try:
            rows = outline_source(text)
        except SyntaxError as bad:
            raise ToolFault("%s does not parse as Python around line %s, so "
                            "it has no index; read it instead"
                            % (path, bad.lineno))
        if not rows:
            raise ToolFault("%s defines nothing, so an index of it would be "
                            "empty; read it instead" % path)
        served = clip("%s, %d lines, %d definitions\n"
                      % (path, len(text.splitlines()), len(rows))
                      + "\n".join(rows), SEARCH_OUTPUT_CAP, "outline")
        self.note_read("outline %s %d defs -> %dc" % (path, len(rows), len(served)))
        return served

    # ---- search_text -----------------------------------------------

    def do_search_text(self, args: dict) -> str:
        pattern = str(args.get("pattern") or "")
        where = str(args.get("path") or ".")
        mode = str(args.get("mode") or "content")
        include = args.get("include")
        context = int(args.get("context") or 0)
        head_limit = int(args.get("head_limit") or 250)
        self.guard_repeat("grep:%s:%s:%s:%s" % (pattern, where, mode, include))
        if where.strip(" ./"):
            if not os.path.isdir(self.tree.absolute(where)):
                raise ToolFault("no such directory: %s" % where)
            where = self.tree.relative(where)
        out = search_files(self.tree.root, pattern, mode=mode,
                           include=include, context=context, timeout=30.0,
                           where=where)
        if not out.strip():
            self.note_read("search_text %r %s -> no matches" % (pattern, mode))
            return "no matches for %r under %s" % (pattern, where)
        rows = out.splitlines()
        if len(rows) > head_limit:
            out = "\n".join(rows[:head_limit]) + (
                "\n... %d more matching lines; narrow the pattern or the path\n"
                % (len(rows) - head_limit))
        served = clip(out, SEARCH_OUTPUT_CAP, "matches")
        self.note_read("search_text %r %s -> %dc" % (pattern, mode, len(served)))
        return served

    # ---- find_files ------------------------------------------------

    def do_find_files(self, args: dict) -> str:
        pattern = str(args.get("pattern") or "*")
        self.guard_repeat("glob:%s" % pattern)
        files = self.tree.current_files()
        hits = [p for p in files if fnmatch.fnmatch(p, pattern)]
        if not hits:
            loose = pattern if pattern.startswith("*") else "*" + pattern
            hits = [p for p in files if fnmatch.fnmatch(p, loose)]
        if not hits:
            return "no file matches %s" % pattern
        return clip("\n".join(hits[:400]), SEARCH_OUTPUT_CAP, "paths")

    # ---- edit ------------------------------------------------------

    def do_edit(self, args: dict) -> str:
        path = str(args.get("path") or "")
        old = str(args.get("old") or "")
        new = str(args.get("new") or "")
        every = bool(args.get("replace_all"))
        if not old:
            raise ToolFault("old must not be empty; use create_file to write "
                            "a whole file")
        self.scope_gate(path)
        text = self.tree.read(path)
        # The file keeps its own line endings; the model's text is matched
        # in that convention so a CRLF file never turns into an LF one.
        old, new = match_line_endings(text, old, new)
        hits = text.count(old)
        if hits == 0:
            raise ToolFault("that exact text is not in %s; read the file "
                            "again and copy it verbatim" % path)
        if hits > 1 and not every:
            raise ToolFault("that text occurs %d times in %s. Either extend "
                            "it until it is unique, or pass replace_all=true "
                            "if all %d should change the same way."
                            % (hits, path, hits))
        updated = text.replace(old, new) if every else text.replace(old, new, 1)
        if updated == text:
            raise ToolFault("the edit is a no-op")
        self.tree.write(path, updated)
        self.allowance.edits += 1
        note = self._compile_check(path)
        return ("edited %s (%d occurrence%s)%s"
                % (path, hits if every else 1, "" if hits == 1 else "s", note))

    def do_create_file(self, args: dict) -> str:
        path = str(args.get("path") or "")
        self.scope_gate(path)
        self.tree.write(path, str(args.get("content") or ""))
        self.allowance.edits += 1
        return "wrote %s%s" % (path, self._compile_check(path))

    def do_restore_file(self, args: dict) -> str:
        rel = self.tree.relative(str(args.get("path") or ""))
        changed = dict(self.tree.changed_paths())
        if rel not in changed:
            return "%s is unchanged since the run started" % rel
        why = self.tree.put_back(rel, changed[rel])
        if why:
            raise ToolFault("could not restore %s: %s" % (rel, why))
        return "%s is back exactly as it was at the start" % rel

    def _compile_check(self, path: str) -> str:
        if not path.endswith(".py"):
            return ""
        try:
            compile(self.tree.read_bytes(path), path, "exec", dont_inherit=True)
        except SyntaxError as bad:
            return ("\n\nWARNING: the file no longer parses: %s:%s: %s"
                    % (path, bad.lineno, bad.msg))
        except (ValueError, RecursionError) as error:
            return "\n\nWARNING: %s: %s" % (type(error).__name__, error)
        return ""

    # ---- bash ------------------------------------------------------

    def do_bash(self, args: dict) -> str:
        command = str(args.get("command") or "")
        if not command.strip():
            raise ToolFault("command must not be empty")
        outward = NETWORK_COMMAND.search(command)
        if outward:
            self.fence.fired("refused %r" % outward.group(1))
            raise ToolFault(
                "%s is not available here. Everything this task needs is "
                "already in the tree. Work from the code in front of you."
                % outward.group(1))
        blocked = HISTORY_GIT.search(command)
        if blocked:
            raise ToolFault(
                "git %s is not available here. Your changes are collected "
                "from the working tree as it stands, so moving or discarding "
                "them loses the work. Read-only git is fine."
                % blocked.group(1))
        want_bg = bool(args.get("background"))
        asked = float(args.get("timeout") or 120)
        room = self.allowance.clock_left() - WALL_RESERVE_SEC
        if room < 5.0:
            raise ToolFault("not enough of the run left to wait on a command; "
                            "make the change you already have evidence for, "
                            "or submit")
        budget = max(5.0, min(asked, SHELL_BUDGET_CEILING_SEC, room))
        if want_bg:
            job = self.pool.start(command, background=True)
            self.bg.fired("started %s: %s" % (job.name, command[:120]))
            return ("started in the background as %s; collect it with "
                    "bash_poll" % job.name)
        job = self.pool.start(command)
        done, out = job.wait(budget)
        if done:
            self.pool.jobs.pop(job.name, None)
            ShellPool._report(job, out)
            return clip(out, SHELL_OUTPUT_CAP, "shell output") or "(no output)"
        self.bg.fired("kept %s alive past %.0fs: %s"
                      % (job.name, budget, command[:120]))
        return (clip(out, SHELL_OUTPUT_CAP, "partial output")
                + "\n\n[still running after %.0fs, moved to the background "
                  "as %s; keep working and collect it later with bash_poll]"
                % (budget, job.name))

    def do_bash_poll(self, args: dict) -> str:
        job = self.pool.get(str(args.get("job") or ""))
        if args.get("stop"):
            out = job.drain()
            was_running = not job.finished()
            job.stop()
            self.pool.jobs.pop(job.name, None)
            ShellPool._report(job, out)
            body = clip(out.replace(STILL_RUNNING, "").rstrip(),
                        SHELL_OUTPUT_CAP, "shell output") or "(no output)"
            return body + ("\n\n[%s stopped; its slot is free]" % job.name
                           if was_running else "\n\n[%s had already finished]" % job.name)
        room = self.allowance.clock_left() - WALL_RESERVE_SEC
        try:
            asked = float(args.get("wait") or BACKGROUND_POLL_WAIT_SEC)
        except (TypeError, ValueError):
            asked = BACKGROUND_POLL_WAIT_SEC
        window = max(1.0, min(asked, BACKGROUND_POLL_CEILING_SEC, room))
        if not job.finished() and room > 1.0:
            job.wait(window)
        out = job.drain()
        self.polls[job.name] = self.polls.get(job.name, 0) + 1
        if not out.endswith(STILL_RUNNING):
            self.pool.jobs.pop(job.name, None)
            ShellPool._report(job, out)
            return clip(out, SHELL_OUTPUT_CAP, "shell output") or "(no output)"
        note = ("\n[%s has run for %.0fs; poll again, or pass wait=%d to "
                "block longer]" % (job.name, time.monotonic() - job.started,
                                   int(BACKGROUND_POLL_CEILING_SEC)))
        if self.polls[job.name] >= POLLS_PER_JOB_ADVISE:
            note += ("\n[polled %d times; if its result no longer matters, "
                     "continue with the evidence you have]" % self.polls[job.name])
        return clip(out, SHELL_OUTPUT_CAP, "partial output") + note

    # ---- submit ----------------------------------------------------

    def do_submit(self, args: dict) -> str:
        self.last_refusal = ""
        changed = None
        try:
            changed = self.tree.has_changes(10.0)
        except BaseException:
            changed = None
        if not changed:
            # An empty patch is rejected outright by the harness, so there
            # is never a point at which handing one in beats another try.
            self.last_refusal = "empty"
            why = ("nothing in the repository has changed" if changed is False
                   else "the change check could not read the tree")
            return ("Not handed in: %s, so this would hand in an empty "
                    "answer. Re-read the requested behaviour, open the "
                    "definition the statement names, and make the narrowest "
                    "real edit before submitting again." % why)
        faults = self.warden.verdict() if self.warden else []
        if faults:
            self.last_refusal = "warden"
            return ("Not handed in. The task states conditions this run can "
                    "check itself, and they are not met yet:\n\n"
                    + "\n\n".join(faults[:3])
                    + "\n\nThere is budget left. Fix this and call submit again.")
        raise Finished(str(args.get("summary") or ""))

# =========================================================================
# Seat — the model request layer
# =========================================================================

class SeatRefused(Exception):
    """The endpoint will not serve this model at all."""


class SeatTimedOut(SeatRefused):
    """The seat did not answer inside the time this call was given."""


class SeatUnreachable(SeatRefused):
    """No base answered at all: connection or parse failures only. That is
    the network, not the model, so the cure is waiting, not benching."""


def base_urls() -> list:
    out = []
    injected = (os.getenv("OPENROUTER_BASE_URL") or "").strip().rstrip("/")
    if injected:
        out.append(injected)
    proxy = (os.getenv("SANDBOX_PROXY_URL") or "").strip().rstrip("/")
    if proxy:
        for suffix in ("/api/v1", "/v1"):
            if proxy + suffix not in out:
                out.append(proxy + suffix)
    if not out:
        out.append(FALLBACK_BASE_URL)
    return out


def recorded_calls(calls: list, turn: int = 0) -> list:
    """The reply's tool calls in the shape the next request must echo:
    arguments as a JSON object string, and every call with an id. A call
    that came without one gets a stable made-up id, since a null id is
    refused by the endpoint on the next turn."""
    kept = []
    for index, call in enumerate(calls):
        function = dict(call.get("function") or {})
        written = function.get("arguments")
        try:
            if isinstance(written, dict):
                written = json.dumps(written)
            if not isinstance(written, str) or not written.strip():
                raise ValueError("nothing was written")
            if not isinstance(json.loads(written), dict):
                raise ValueError("not an object")
        except Exception:
            written = "{}"
        function["arguments"] = written
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = "call_%d_%d" % (turn, index + 1)
        kept.append({"id": call_id, "type": "function", "function": function})
    return kept


class Seat:
    roster: list = []
    patient: bool = True
    include_usage: bool = True   # ask the endpoint to quote each call's cost

    def __init__(self, allowance: Allowance, models: list = None,
                 patient: bool = True) -> None:
        self.allowance = allowance
        self.patient = patient
        if models:
            self.models = list(models)
        else:
            self.models = [DRIVER_MODEL]
            if RELIEF_MODEL and RELIEF_MODEL != DRIVER_MODEL:
                self.models.append(RELIEF_MODEL)
        self.roster = list(self.models)
        self.timeouts: dict = {}
        self.bases = base_urls()
        self.key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("RIDGES_OPENROUTER_API_KEY")
            or os.getenv("AI_PROXY_KEY")
            or ""
        )

    def current(self) -> str:
        return self.models[0]

    def retire(self, model: str) -> bool:
        if model in self.models and len(self.models) > 1:
            self.models.remove(model)
            if model in self.roster:
                self.roster.remove(model)
            say("[SEAT] retired %s, now on %s" % (model, self.models[0]))
            return True
        return False

    def ask(self, messages: list, tools: list = None) -> dict:
        budget = self._budget()
        while True:
            model = self.current()
            try:
                return self._attempt(model, messages, tools, budget)
            except SeatRefused as refusal:
                say("[SEAT] %s refused: %s" % (model, str(refusal)[:200]))
                if isinstance(refusal, SeatTimedOut):
                    # The abandoned calls were charged by estimate; the
                    # proxy knows what they really cost.
                    self.allowance.sync()
                if isinstance(refusal, SeatUnreachable):
                    if not self.patient:
                        raise
                    wait = min(SEAT_REFUSED_WAIT_SEC,
                               self.allowance.clock_left() - WALL_RESERVE_SEC - 60.0)
                    if wait <= 0:
                        raise Spent("endpoint unreachable: %s" % refusal)
                    say("[SEAT] endpoint unreachable; asking again in %.0fs" % wait)
                    time.sleep(wait)
                    budget = self._budget()
                    continue
                if (isinstance(refusal, SeatTimedOut)
                        and self.timeouts.get(model, 0) >= 2
                        and self.retire(model)):
                    budget = self._budget()
                    continue
                if len(self.models) > 1:
                    self.models.remove(model)
                    say("[SEAT] benched %s, now on %s"
                        % (model, self.models[0]))
                    if isinstance(refusal, SeatTimedOut):
                        budget = self._budget()
                    continue
                if isinstance(refusal, SeatTimedOut):
                    raise Spent("every seat went quiet: %s" % refusal)
                if not self.patient:
                    raise
                wait = min(SEAT_REFUSED_WAIT_SEC,
                           self.allowance.clock_left() - WALL_RESERVE_SEC - 60.0)
                if wait <= 0:
                    raise Spent("no seat will serve this run")
                say("[SEAT] every seat refused; asking again in %.0fs" % wait)
                time.sleep(wait)
                self.models = list(self.roster)
                budget = self._budget()

    def _budget(self):
        share = self.allowance.clock_left() * CALL_CLOCK_SHARE
        return share, time.monotonic(), self.allowance.clock_left()

    def _attempt(self, model: str, messages: list, tools: list,
                 budget=None) -> dict:
        body = {
            "model": model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": REPLY_TOKEN_CEILING,
        }
        if HIGH_REASONING:
            body["reasoning"] = {"effort": "high", "exclude": True}
        if Seat.include_usage:
            body["usage"] = {"include": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key

        backoff = 3.0
        sweep: list = []
        timed_out = 0
        walled = 0
        unreachable = 0
        asked = 0
        attempts = REQUEST_ATTEMPTS if self.patient else 1
        ceiling = SEAT_CALL_TIMEOUT_SEC if self.patient else 60.0
        share, began, started_with = budget if budget else self._budget()
        # What the endpoint bills for a request we walk away from: the
        # prompt it read plus part of a reply. Estimated, then corrected
        # from the proxy's own total.
        prompt_chars = len(payload)

        def spent_share() -> float:
            return max(time.monotonic() - began,
                       started_with - self.allowance.clock_left())

        for attempt in range(attempts):
            if self.allowance.clock_left() <= 5:
                raise Spent("clock ran out mid-request")
            walled = 0
            quiet = 0
            sweep = []
            # A slow endpoint gets more time on the retry, never less: a
            # reply that needs 100s cannot arrive in 45.
            granted = max(SEAT_RETRY_FLOOR_SEC,
                          ceiling * (SEAT_RETRY_GROWTH ** timed_out))
            for base in list(self.bases):
                room = min(share - spent_share(),
                           self.allowance.clock_left() - 10)
                if room < SEAT_RETRY_FLOOR_SEC:
                    break
                asked += 1
                try:
                    request = urllib.request.Request(
                        base + "/chat/completions", data=payload,
                        headers=headers)
                    with urllib.request.urlopen(
                            request, timeout=min(granted, room)) as response:
                        parsed = json.loads(
                            response.read().decode("utf-8", "replace"))
                    return self._book(model, parsed)
                except urllib.error.HTTPError as error:
                    try:
                        detail = error.read()[:400].decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                    sweep.append("%s %s" % (error.code, foreign(detail)))
                    if error.code == 404 and len(self.bases) > 1:
                        # Not a route here; stop paying for it on every call.
                        self.bases.remove(base)
                        say("[SEAT] %s is not served; dropped" % base)
                        continue
                    if error.code == 403:
                        walled += 1
                        continue
                    if (error.code == 400 and Seat.include_usage
                            and "usage" in detail.lower()):
                        Seat.include_usage = False
                        say("[SEAT] endpoint rejects usage.include; "
                            "estimating cost from now on")
                        return self._attempt(model, messages, tools, budget)
                    if error.code in (400, 402, 413):
                        raise Spent("endpoint refused the request: "
                                    + sweep[-1])
                    if error.code == 429 and (
                            "budget" in detail.lower()
                            or "cost" in detail.lower()):
                        raise Spent("allowance exhausted upstream")
                    if error.code == 429:
                        walled += 1
                        continue
                except Exception as error:
                    reason = getattr(error, "reason", None)
                    if isinstance(error, TimeoutError) or isinstance(
                            reason, TimeoutError):
                        timed_out += 1
                        quiet += 1
                        sweep.append("timed out after %.0fs"
                                     % min(granted, room))
                        charged = self.allowance.charge(model, {
                            "prompt_tokens": int(prompt_chars / CHARS_PER_TOKEN),
                            "completion_tokens": ABANDONED_REPLY_TOKENS})
                        say("[SEAT] charged ~$%.4f for the request that timed "
                            "out; total=$%.4f" % (charged, self.allowance.spent))
                    else:
                        unreachable += 1
                        sweep.append("%s: %s"
                                     % (type(error).__name__, foreign(error)))
            if asked and walled and not quiet:
                # Nothing but walls in this sweep.
                if attempt + 1 >= 2:
                    raise SeatRefused(" | ".join(sweep))
            if attempt + 1 >= attempts or spent_share() >= share:
                break
            nap = min(backoff, max(0.0, self.allowance.clock_left() - 5),
                      max(0.0, share - spent_share()))
            if nap <= 0:
                break
            time.sleep(nap)
            backoff = min(backoff * 2, 15.0)
        said = " | ".join(sweep) or "no base was asked"
        if timed_out:
            self.timeouts[model] = self.timeouts.get(model, 0) + 1
            raise SeatTimedOut("%d timeout(s): %s" % (timed_out, said))
        if asked and walled and not quiet:
            raise SeatRefused(said)
        if asked and unreachable and not walled:
            raise SeatUnreachable(said)
        raise Spent("no reply after %d attempts: %s" % (attempt + 1, said))

    def _book(self, model: str, parsed: dict) -> dict:
        choices = parsed.get("choices") or []
        if not choices:
            raise Spent("reply carried no choices")
        message = choices[0].get("message") or {}
        usage = parsed.get("usage") or {}
        cost = self.allowance.charge(model, usage)
        message["_usage"] = usage
        message["_model"] = model
        served = one_line(parsed.get("provider") or parsed.get("served_by"))
        answered = one_line(parsed.get("model"))
        aside = ""
        if served:
            aside += " via=%s" % served[:40]
        if answered and answered != model:
            aside += " answered=%s" % answered[:60]
        thought = reasoning_tokens(usage)
        if thought >= 0:
            aside += " thought=%d" % thought
        if "cost" in usage:
            aside += " est=$%.4f" % self.allowance.estimate(model, usage)
        say("[SEAT] %s call=%d $%.4f total=$%.4f left=%.0fs said=%s%s"
            % (model, self.allowance.calls, cost, self.allowance.spent,
               self.allowance.clock_left(), reply_fingerprint(message), aside))
        finish = choices[0].get("finish_reason") or ""
        message["_finish"] = finish
        if finish not in ("stop", "tool_calls", ""):
            say("[SEAT] %s reply ended on %s after %d token(s)"
                % (model, finish if finish in
                   ("stop", "length", "tool_calls", "content_filter", "error")
                   else "other",
                   int(usage.get("completion_tokens") or 0)))
        return message


# =========================================================================
# Warden — the task's own conditions, checked before hand-in
# =========================================================================

WARDEN_REFUSALS_MAX = 3
WARDEN_RELEASE_SEC = 150.0
CHECK_SEC = 360.0            # one named check; the database sidecar has a quarter CPU
CHECK_GATE_SEC = 480.0       # all of them together
CHECK_MIN_ROOM_SEC = 60.0
CHECK_OUTPUT_CHARS = 2000


class Warden:
    """What the instruction itself requires: only the named files change,
    no test files touched, no definitions dropped, no suppressions added,
    and the check commands it names exit clean."""

    def __init__(self, tree: Tree, pool: ShellPool, allowance: Allowance,
                 statement: str = "", scope: dict = None) -> None:
        self.tree = tree
        self.pool = pool
        self.allowance = allowance
        self.statement = statement
        self.scope = scope or {}
        self.allowed = list(self.scope.get("files") or [])
        self.checks = list(self.scope.get("checks") or [])[:CHECK_COMMANDS_MAX]
        self.beacon = Beacon("warden")
        self.refusals = 0
        self.stood_down = False
        self.passed_key = None

    def tree_key(self, changed: list = None) -> tuple:
        """Identifies the exact set of changes on disk."""
        key = []
        for rel, kind in (self.tree.changed_paths() if changed is None else changed):
            digest = ""
            if kind != "deleted":
                try:
                    digest = Tree._digest(self.tree.absolute(rel))
                except (OSError, ToolFault):
                    digest = "?"
            key.append((rel, kind, digest))
        return tuple(key)

    def scope_faults(self, changed: list = None) -> list:
        if not self.allowed:
            return []
        extras = [p for p in self.changed_paths(changed) if p not in self.allowed]
        if not extras:
            return []
        return ["The instruction limits changes to %s. This run also changed "
                "%s. Put %s back exactly as at the start (restore_file does "
                "that in one call) and keep the fix inside the named file%s."
                % (", ".join(self.allowed), ", ".join(extras[:6]),
                   "it" if len(extras) == 1 else "them",
                   "s" if len(self.allowed) > 1 else "")]

    def check_faults(self, changed: list = None) -> list:
        """Run the commands the instruction names; the first red one is
        the fault. A check that cannot finish in the time left is skipped,
        never counted against the run, and never recorded as passed."""
        if not self.checks:
            return []
        room = self.allowance.clock_left() - WALL_RESERVE_SEC
        if room < CHECK_MIN_ROOM_SEC:
            self.beacon.skipped("too little of the run left for the named checks")
            return []
        key = self.tree_key(changed)
        ran_all = True
        if self.passed_key is not None and key == self.passed_key:
            self.beacon.skipped("checks already passed on this exact tree")
            return []
        # The model's own background runs are stale once it submits, and
        # they hold the pool's slots.
        for job in list(self.pool.jobs.values()):
            if not job.finished():
                self.beacon.fired("stopping %s before the named checks" % job.name)
            try:
                ShellPool._report(job, job.drain())
                job.stop()
            except Exception:
                pass
        self.pool.jobs.clear()
        deadline = time.monotonic() + min(CHECK_GATE_SEC, room)
        for command in self.checks:
            left = deadline - time.monotonic()
            if left < 10.0:
                self.beacon.skipped("out of time before: %s" % command[:100])
                ran_all = False
                break
            try:
                job = self.pool.start(command)
            except (ToolFault, OSError) as error:
                self.beacon.skipped("could not start %s: %s" % (command[:80], error))
                ran_all = False
                continue
            granted = min(CHECK_SEC, left)
            done, out = job.wait(granted)
            self.pool.jobs.pop(job.name, None)
            if not done:
                job.stop()
                self.beacon.skipped("timed out after %.0fs: %s" % (granted, command[:100]))
                ran_all = False
                continue
            code = job.process.returncode
            ShellPool._report(job, out)
            job.stop()
            if code == 0:
                self.beacon.fired("clean: %s" % command[:100])
                continue
            self.beacon.fired("red (exit %s): %s" % (code, command[:100]))
            return ["The instruction names this check and it does not pass "
                    "(exit %s):\n  %s\n\n%s"
                    % (code, command,
                       clip(out or "(no output)", CHECK_OUTPUT_CHARS, "check output"))]
        self.passed_key = key if ran_all else None
        return []

    def changed_paths(self, changed: list = None) -> list:
        if changed is None:
            changed = self.tree.changed_paths()
        return [rel for rel, _ in changed]

    def original(self, path: str):
        data = self.tree.original(path)
        return None if data is None else data.decode("utf-8", "replace")

    def change_faults(self, changed: list = None) -> list:
        faults = []
        for path in self.changed_paths(changed):
            if TEST_PATH.search(path):
                faults.append(
                    "This run edited %s. The task is to change the code under "
                    "repair, not the tests that check it. Put it back exactly "
                    "as it was and make the source satisfy the test instead."
                    % path)
                continue
            faults.extend(self.file_faults(path))
        return faults

    def file_faults(self, path: str) -> list:
        before = self.original(path)
        if before is None:
            return []
        if not path.endswith((".py", ".rb", ".go", ".js", ".ts", ".php")):
            return []
        carried = visible_definitions(before) if path.endswith(".py") else []
        try:
            after = self.tree.read(path)
        except ToolFault:
            if not carried:
                return []
            return ["This run deleted %s, and callers of %s lose it with the "
                    "file. Keep the file and the names in it."
                    % (path, carried[0])]
        out = []
        if carried:
            import collections
            gone = sorted((collections.Counter(carried)
                           - collections.Counter(visible_definitions(after))).elements())
            if gone:
                out.append(
                    "These definitions were in %s when the run started and "
                    "are not there now: %s. A refactor may not drop a name "
                    "callers can use." % (path, ", ".join(gone[:6])))
        added = (len(NOQA_DIRECTIVE.findall(after))
                 - len(NOQA_DIRECTIVE.findall(before)))
        if added > 0:
            out.append(
                "This run added %d suppression comment(s) to %s. Silencing "
                "the check is the one answer the task rules out."
                % (added, path))
        return out

    def verdict(self) -> list:
        if self.refusals >= WARDEN_REFUSALS_MAX:
            self.stood_down = True
            self.beacon.skipped("already sent the run back %d times"
                                % self.refusals)
            return []
        if self.allowance.clock_left() < WARDEN_RELEASE_SEC:
            self.stood_down = True
            self.beacon.skipped("too little of the run left to act on a refusal")
            return []
        # One walk of the tree serves all three gates.
        changed = self.tree.changed_paths()
        faults = (self.scope_faults(changed) or self.change_faults(changed)
                  or self.check_faults(changed))
        if faults:
            self.refusals += 1
            self.beacon.fired("refused hand-in #%d: %s"
                              % (self.refusals, faults[0][:120]))
        return faults

# =========================================================================
# Shared loop helpers
# =========================================================================

def extract_tool_call(call: dict) -> tuple:
    """Returns (name, args_dict, args_error). Never raises. Some endpoints
    hand the arguments over already parsed; both shapes are accepted."""
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    written = function.get("arguments")
    try:
        args = written if isinstance(written, dict) else json.loads(written or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments were not an object")
    except Exception as error:
        return name, None, "could not read the arguments: %s" % error
    return name, args, ""


def dispatch(kit: Kit, name: str, args: dict) -> tuple:
    """Run one tool. Returns (result_text, faulted_bool)."""
    try:
        result = kit.run(name, args)
        return str(result), False
    except Finished:
        raise
    except ToolFault as fault:
        return "error: %s" % fault, True
    except BaseException as error:
        traceback.print_exc()
        return "error: %s: %s" % (type(error).__name__, error), True


def shrink_transcript(messages: list, cap: int, beacon: Beacon) -> bool:
    """Blank the oldest bulky tool results so the transcript fits."""
    total = sum(len(str(m.get("content") or "")) for m in messages)
    if total <= cap:
        return False
    beacon.fired("transcript %dB over cap %dB" % (total, cap))
    freed = 0
    for message in messages[2: max(2, len(messages) - 12)]:
        if message.get("role") != "tool":
            continue
        body = str(message.get("content") or "")
        if len(body) <= 400:
            continue
        message["content"] = ("[%d characters of earlier tool output dropped "
                              "to fit the context]" % len(body))
        freed += len(body)
        total -= len(body)
        if total <= cap * 0.7:
            break
    if not freed:
        beacon.skipped("nothing bulky enough to drop")
        return False
    say("[TRIM] freed %dB, transcript now ~%dB" % (freed, total))
    return True


# =========================================================================
# Stage 1: the Locator
# =========================================================================

LOCATOR_BRIEF = """You are locating, not editing. You can read the code in front of you but you cannot change it. A separate planner will take your target list and decide how to solve the task; the driver after that will make the edit. Your job is to hand them the right file.

The instruction is below. Its goal is the thing to change; its constraints are the bounds on what may change. Neither is a filename.

HOW TO FIND THE TARGET

The code sits in the working directory and holds tens of thousands of files. You cannot read them all. For every distinctive term the instruction uses -- an entity, an operation, a symptom, a parameter name -- search for it:

  search_text(pattern="discount_code", mode="files")
  search_text(pattern="order_total", mode="files")

The target is the file that mentions several of these terms. Intersect the results in your own reasoning; do not read a file because it sounds plausible. Read it because a search put it in front of you.

When a search returns more than 60 files, the pattern is too common. Add a second term, or narrow with include="*.py" or path="src/".

Once you have two or three candidate files, use outline() before reading: it tells you which lines hold which function, so a read costs 40 lines instead of 400. Then read only the function the instruction is about.

WHAT TO RECORD

Call set_targets with the files you believe are the target. Each row needs:
  path       -- the file's path, relative to the working directory
  symbol     -- the class.method or function the change will land in, if you can tell
  lines      -- the line range of that symbol
  why        -- one sentence: what this code does that the task names
  confidence -- high / medium / low

Keep it short. Three high-confidence rows beat ten guesses. The planner reads this list, not your conversation.

Call done_locating as soon as you have a hypothesis worth handing over. You are not required to spend the whole budget; a wrong list handed early is recoverable, a right list handed too late is not."""


LOCATOR_TOOLS = locator_tools() + [
    {
        "type": "function",
        "function": {
            "name": "set_targets",
            "description": ("Record the target files, replacing any earlier "
                            "list. Call this as soon as you have a hypothesis "
                            "worth handing to the planner, and again whenever "
                            "it improves. This list IS your output."),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "lines": {"type": "string"},
                                "symbol": {"type": "string"},
                                "why": {"type": "string"},
                                "confidence": {"type": "string",
                                               "enum": ["high", "medium", "low"]},
                            },
                            "required": ["path", "why"],
                        },
                    },
                    "note": {"type": "string"},
                },
                "required": ["targets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done_locating",
            "description": "Stop locating and hand the target list to the planner.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def normalise_targets(raw, root: str) -> list:
    out = []
    seen = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().lstrip("./")
        if not path or path in seen:
            continue
        if not os.path.isfile(os.path.join(root, path)):
            continue
        seen.add(path)
        out.append({
            "path": path,
            "symbol": str(item.get("symbol") or "")[:120],
            "lines": str(item.get("lines") or "")[:40],
            "why": str(item.get("why") or "")[:300],
            "confidence": str(item.get("confidence") or "medium")[:10],
        })
    return out[:8]


def locator_opening(statement: str, tree: "Tree", seed_paths: list,
                    missing: list = None) -> str:
    blocks = [
        "Task instruction:\n\n" + statement.strip(),
        "\nThe code at a glance:\n" + repo_sketch(tree.files()),
    ]
    if missing:
        blocks.append(
            "\nThe instruction names %s, but nothing is at that path. Find "
            "where that file lives now; a search for its base name is the "
            "quickest start." % ", ".join("`%s`" % m for m in missing))
    if seed_paths:
        blocks.append(
            "\nFiles whose contents overlap the rare terms in the instruction, "
            "most overlap first. This is a text-matching starting point, not "
            "an answer:\n" + "\n".join("  " + p for p in seed_paths))
    blocks.append(
        "\nFind the file(s) the change belongs in. Use search_text to look "
        "for the instruction's own terms, then outline and read narrowly. "
        "When you have a hypothesis worth handing to the planner, call "
        "set_targets, then done_locating.")
    return "\n".join(blocks)


def run_locator(statement: str, tree: Tree, pool: ShellPool,
                allowance: Allowance, beacon: Beacon,
                missing: list = None) -> dict:
    """Returns {"targets": [...], "note": "..."}. Never raises."""
    beacon.reached(0, allowance.spent, allowance.clock_left())
    spent_at_entry = allowance.spent
    calls_at_entry = allowance.calls
    ceiling = spent_at_entry + allowance.soft_usd * LOCATOR_SPEND_SHARE

    kit = Kit(tree, pool, allowance, label="LOCATE")
    seat = Seat(allowance, models=[LOCATOR_MODEL], patient=False)

    seed = candidate_paths(tree.root, statement, limit=30)
    beacon.fired("seed: %d path(s)" % len(seed))

    messages = [
        {"role": "system", "content": LOCATOR_BRIEF},
        {"role": "user", "content": locator_opening(statement, tree, seed,
                                                    missing)},
    ]

    targets: list = []
    note = ""
    read = 0
    step = 0
    stop = "turns"

    try:
        while step < LOCATOR_TURN_CAP:
            step += 1
            if read >= LOCATOR_READ_BUDGET:
                stop = "budget"
                break
            if allowance.spent >= ceiling or allowance.money_left() <= 0:
                stop = "spend"
                break
            if (allowance.clock_left() < 120 or allowance.elapsed()
                    > allowance.run_length() * LOCATOR_CLOCK_SHARE):
                stop = "clock"
                break

            reply = seat.ask(messages, LOCATOR_TOOLS)
            calls = reply.get("tool_calls") or []
            entry = {"role": "assistant",
                     "content": str(reply.get("content") or "")}
            if calls:
                entry["tool_calls"] = recorded_calls(calls)
            messages.append(entry)
            if not calls:
                stop = "silent"
                break

            finished = False
            for call in calls:
                name, args, arg_err = extract_tool_call(call)
                if arg_err:
                    result = arg_err
                elif name == "done_locating":
                    finished = True
                    result = "locating ended"
                elif name == "set_targets":
                    targets = normalise_targets(args.get("targets") or [],
                                                tree.root)
                    note = str(args.get("note") or "")[:LOCATOR_NOTE_CHARS]
                    result = "recorded %d target(s)" % len(targets)
                else:
                    result, _ = dispatch(kit, name, args)
                served = clip(str(result), READ_OUTPUT_CAP)
                read += len(served)
                messages.append({"role": "tool",
                                 "tool_call_id": call.get("id"),
                                 "content": served})
            if finished:
                stop = "done"
                break
            messages.append({"role": "user", "content":
                "Reading budget: %d used, %d left."
                % (read, max(0, LOCATOR_READ_BUDGET - read))})
    except Exception as error:
        stop = "error"
        say("[LOCATE] gave up: %s: %s"
            % (type(error).__name__, str(error)[:200]))

    # Fallback: if the model produced nothing, use the seed.
    if not targets and seed:
        targets = [{"path": p, "why": "ranked by term overlap",
                    "confidence": "low"} for p in seed[:5]]
        stop += "+seed"

    allowance.sync()
    beacon.calls = allowance.calls - calls_at_entry
    beacon.usd = allowance.spent - spent_at_entry
    beacon.fired("stopped=%s steps=%d read=%dc targets=%d note=%dB"
                 % (stop, step, read, len(targets), len(note)))
    if targets:
        say("[LOCATE] targets: " + ", ".join(t["path"] for t in targets[:6]))
    if note:
        say("[LOCATE] note %s" % note.replace("\n", " | ")[:500])
    beacon.bill()
    return {"targets": targets, "note": note}


# =========================================================================
# Stage 2: the Planner
# =========================================================================

PLANNER_BRIEF = """You are planning, not editing. You can read the code in front of you but you cannot change it. A driver will make the edit from your plan.

The locator seat already found the target file(s). They are below. Trust the locator enough to start there; verify enough that you do not plan against the wrong code.

YOUR JOB

Read the target file(s) and the code around them. Then answer, concretely:

  1. What the current code does -- one or two sentences, in terms of the columns, the relations, and the operation the instruction names.
  2. Which function or method the change belongs in, with its line range.
  3. What the code must do instead -- the specific condition, ordering, join, aggregate, or guard that changes. Not "fix the bug"; the new shape.
  4. Which constraints from the instruction apply to the fix: the files it permits, the method it bounds, the names the file already imports, the constructs it forbids, the row/semantics it requires on ties, NULLs, empty groups, or boundaries.
  5. One concrete check the driver can run to prove the change is correct.

Record it with set_plan. The plan MAY BE READ AT ANY MOMENT by the driver, so keep it worth reading from the first call onward. Call done_planning when it is good enough.

A good plan names a file and a line range and says what must become true. It does not restate the instruction and it does not narrate your reading."""


PLANNER_TOOLS = planner_tools() + [
    {
        "type": "function",
        "function": {
            "name": "set_plan",
            "description": ("Record the current best plan, replacing any "
                            "earlier one. Call this as soon as you have "
                            "something worth handing over, and again whenever "
                            "it improves."),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done_planning",
            "description": ("Stop planning and hand the note over. Call this "
                            "as soon as the plan is good enough."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def planner_opening(statement: str, targets: list) -> str:
    if not targets:
        return ("Task instruction:\n\n" + statement.strip()
                + "\n\nThe locator found no target file. Read the instruction, "
                  "search for its terms, and plan against whatever the search "
                  "turns up.")
    rows = []
    for t in targets:
        row = "- %s" % t["path"]
        if t.get("symbol"):
            row += " :: %s" % t["symbol"]
        if t.get("lines"):
            row += " (lines %s)" % t["lines"]
        if t.get("confidence"):
            row += " [%s]" % t["confidence"]
        row += "\n    %s" % t.get("why", "")
        rows.append(row)
    return ("Task instruction:\n\n" + statement.strip()
            + "\n\nThe locator seat found these target(s):\n\n"
            + "\n".join(rows)
            + "\n\nRead the target(s), verify the locator's reading, then "
              "write the plan with set_plan and call done_planning.")


def run_planner(statement: str, tree: Tree, pool: ShellPool,
                allowance: Allowance, targets: list,
                beacon: Beacon) -> str:
    """Returns the plan note. Never raises."""
    beacon.reached(0, allowance.spent, allowance.clock_left())
    spent_at_entry = allowance.spent
    calls_at_entry = allowance.calls
    ceiling = spent_at_entry + allowance.soft_usd * PLANNER_SPEND_SHARE

    kit = Kit(tree, pool, allowance, label="PLAN")
    seat = Seat(allowance, models=[PLANNER_MODEL], patient=False)

    messages = [
        {"role": "system", "content": PLANNER_BRIEF},
        {"role": "user", "content": planner_opening(statement, targets)},
    ]

    note = ""
    read = 0
    step = 0
    stop = "turns"

    try:
        while step < PLANNER_TURN_CAP:
            step += 1
            if read >= PLANNER_READ_BUDGET:
                stop = "budget"
                break
            if allowance.spent >= ceiling or allowance.money_left() <= 0:
                stop = "spend"
                break
            if (allowance.clock_left() < 120 or allowance.elapsed()
                    > allowance.run_length() * PLANNER_CLOCK_SHARE):
                stop = "clock"
                break

            reply = seat.ask(messages, PLANNER_TOOLS)
            calls = reply.get("tool_calls") or []
            entry = {"role": "assistant",
                     "content": str(reply.get("content") or "")}
            if calls:
                entry["tool_calls"] = recorded_calls(calls)
            messages.append(entry)
            if not calls:
                stop = "silent"
                break

            finished = False
            for call in calls:
                name, args, arg_err = extract_tool_call(call)
                if arg_err:
                    result = arg_err
                elif name == "done_planning":
                    finished = True
                    result = "planning ended"
                elif name == "set_plan":
                    note = str(args.get("text") or "")[:PLAN_NOTE_CHARS]
                    result = "plan recorded, %d characters" % len(note)
                else:
                    result, _ = dispatch(kit, name, args)
                served = clip(str(result), READ_OUTPUT_CAP)
                read += len(served)
                messages.append({"role": "tool",
                                 "tool_call_id": call.get("id"),
                                 "content": served})
            if finished:
                stop = "done"
                break
            messages.append({"role": "user", "content":
                "Reading budget: %d used, %d left."
                % (read, max(0, PLANNER_READ_BUDGET - read))})
    except Exception as error:
        stop = "error"
        say("[PLAN] gave up: %s: %s"
            % (type(error).__name__, str(error)[:200]))

    allowance.sync()
    beacon.calls = allowance.calls - calls_at_entry
    beacon.usd = allowance.spent - spent_at_entry
    beacon.fired("stopped=%s steps=%d read=%dc note=%dB"
                 % (stop, step, read, len(note)))
    if note:
        say("[PLAN] note %s" % note.replace("\n", " | ")[:PLAN_NOTE_CHARS])
    beacon.bill()
    return note.strip()


# =========================================================================
# Stage 3: the Driver
# =========================================================================

DRIVER_BRIEF = """You are changing how a real application fetches data from its database. You have shell access, file tools, and the ability to run the checks the instruction names. When you are done, the working tree is the answer: your changes are read straight off it, so leave the change in place and call submit.

Work inside the repository as it is. Do not add dependencies and do not rewrite unrelated code. When the task limits the change to one file or one method, everything else in that file stays byte-for-byte as it was, including imports: use only names the file already imports. Fix the cause in the code the application actually runs; a private snippet that never sits on that path changes nothing.

CRITICAL -- spend turns carefully. Every reply costs one exchange with the model, and exchanges are the scarcest thing you have. Put every tool call that does not depend on another one into the SAME reply. Reading four files is four calls in one reply, not four replies. Searching for three patterns is three calls in one reply. Only wait for a result when the next thing you do genuinely depends on it.

Do not sit idle while a slow command runs. Start a long check with background=true, keep reading code, and collect it with bash_poll when you need the answer. Run only the checks the instruction names; a whole-project build or test run is neither asked for nor affordable.

HOW TO WORK

1. Read the target's current code before you change it. Use read_file with a range, and outline first if the file is long.

2. Decide the change: what does one output row stand for, what does each JOIN do to the row count, what the code must do instead. The plan names this; verify it against the code before trusting it.

3. Make the narrowest edit the instruction requires. Change only the body of the function the problem is about. Never change a function's signature, parameter defaults, decorators, docstring, imports, class attributes or module-level lines.

4. Do not introduce numeric constants, index constants such as result[0], or new literals; unpack results into named variables and reuse the values the code already has.

5. Run the checks the instruction names, exactly as written, and read their output.

6. Read your own diff before submitting. Ask whether it changes anything the problem did not ask for.

BEFORE YOU SUBMIT

Run the checks the instruction names. Then read your own diff and confirm it stays inside any scope the instruction sets. Call submit with a one-line summary of what you changed. submit runs the instruction's named checks itself and refuses a failing one with its output, so a clean submit is the proof."""


def driver_opening(statement: str, tree: "Tree", located: dict,
                   plan_note: str, scope: dict = None) -> str:
    scope = scope or {}
    blocks = [
        "Task:\n\n" + statement.strip(),
        "\nThe code at a glance:\n" + repo_sketch(tree.files()),
    ]
    files = scope.get("files") or []
    if files:
        blocks.append(
            "\nThe instruction limits changes to: %s. An edit anywhere else "
            "is refused, and anything outside these files is reverted at "
            "hand-in." % ", ".join(files))
        if scope.get("excerpt"):
            blocks.append(
                "\nCurrent code of the named scope, verbatim, so you can copy "
                "spans from it straight into edit. This is the code as it "
                "stands now, not a proposed answer:\n\n" + scope["excerpt"])
    targets = located.get("targets") or []
    if targets and not files:
        rows = []
        for t in targets:
            row = "  - %s" % t["path"]
            if t.get("symbol"):
                row += " :: %s" % t["symbol"]
            if t.get("lines"):
                row += " (lines %s)" % t["lines"]
            rows.append(row)
        blocks.append(
            "\nA locating pass read the instruction and searched the code. "
            "These are the files it believes the change belongs in. Verify "
            "them with read_file before trusting:\n" + "\n".join(rows))
    locate_note = located.get("note") or ""
    if locate_note:
        blocks.append("\nLocator note:\n" + locate_note)
    if plan_note:
        blocks.append(
            "\nA planning pass read the targets and left this plan. It did "
            "not run anything and may be wrong -- check it against the file "
            "before you act on it.\n\n" + plan_note)
    checks = scope.get("checks") or []
    if checks:
        blocks.append(
            "\nAt submit these commands are run and must exit 0; a failure "
            "sends the run back with the output:\n"
            + "\n".join("  " + c for c in checks))
    blocks.append(
        "\nWork the plan: read the target's current code, make the narrowest "
        "edit the instruction requires, run the checks the instruction names, "
        "and submit.")
    return "\n".join(blocks)


def edit_press(turn: int, pressed: int) -> str:
    return (
        "No edit yet. Narrow down and change something now; an imperfect fix "
        "in the tree beats a perfect one you never wrote. Reading more before "
        "the first edit cannot help: nothing you have learned so far is in "
        "the answer until it is in the file. Do not submit while the tree is "
        "unchanged.")


def wrap_up() -> str:
    return (
        "You are near the end of the run. Finish the change you are on, "
        "re-run the checks the instruction names, and call submit.")


def drive(statement: str, tree: Tree, pool: ShellPool, allowance: Allowance,
          located: dict, plan_note: str, warden: Warden,
          scope: dict = None) -> None:
    seat = Seat(allowance)
    kit = Kit(tree, pool, allowance, warden=warden, scope=scope)

    messages = [
        {"role": "system", "content": DRIVER_BRIEF},
        {"role": "user", "content": driver_opening(statement, tree,
                                                   located, plan_note, scope)},
    ]
    cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
    say("[LOOP] transcript cap set for %s" % seat.current())

    turn = 1
    blanks = 0
    tool_faults = 0
    refusals = 0
    pressed = 0
    wrapped_up = False
    last_fingerprint = ""
    identical = 0
    history: list = []

    while True:
        # --- exits, unconditional, at the top of every turn ---
        if allowance.clock_left() <= 0:
            say("[LOOP] wall clock at turn %d" % turn)
            return
        if allowance.money_left() <= 0:
            say("[LOOP] budget at turn %d" % turn)
            return
        if blanks >= BLANK_REPLY_CEILING:
            say("[LOOP] %d blank replies" % blanks)
            return
        if tool_faults >= TOOL_FAULT_CEILING:
            say("[LOOP] %d tool faults" % tool_faults)
            return
        if identical >= IDENTICAL_REPLY_CEILING:
            say("[LOOP] %d identical replies" % identical)
            return
        if refusals >= SUBMIT_REFUSAL_CEILING:
            say("[LOOP] %d submit refusals" % refusals)
            return

        # --- the proxy's total is the one that counts ---
        if turn % COST_SYNC_TURNS == 0:
            allowance.sync()

        # --- dynamic round ceiling ---
        transcript_chars = sum(len(str(m.get("content") or "")) for m in messages)
        budget_rounds = rounds_left(allowance, seat.current(),
                                    transcript_chars, history)
        if turn > TURN_CEILING:
            say("[LOOP] hard turn ceiling at %d" % TURN_CEILING)
            return

        # --- nudges, once each ---
        if (allowance.edits == 0 and turn > FIRST_EDIT_DEADLINE_TURN
                and pressed < EDIT_PRESSES_MAX):
            pressed += 1
            say("[LOOP] %d turns without an edit; pressing (#%d)"
                % (turn - 1, pressed))
            messages.append({"role": "user",
                             "content": edit_press(turn, pressed)})
        if (not wrapped_up and (turn >= WRAPUP_TURN
                                or budget_rounds <= 2
                                or allowance.clock_left() < WRAPUP_CLOCK_SEC
                                or allowance.money_left() < allowance.soft_usd * 0.15)):
            wrapped_up = True
            say("[LOOP] wrapping up at turn %d: ~%d round(s) in $%.4f, %.0fs left"
                % (turn, budget_rounds, allowance.money_left(),
                   allowance.clock_left()))
            messages.append({"role": "user", "content": wrap_up()})

        if transcript_chars > cap:
            shrink_transcript(messages, cap, Beacon("trim"))

        # --- one model turn ---
        answering = seat.current()
        reply = seat.ask(messages, TOOL_SCHEMAS)
        if seat.current() != answering:
            cap = transcript_cap_chars(seat.current(), allowance.ceiling_usd)
            say("[LOOP] seat changed to %s" % seat.current())

        calls = reply.get("tool_calls") or []
        text = str(reply.get("content") or "")

        # A reply that only polls a background job is legitimately the same
        # as the last one; it neither counts as a repeat nor clears one.
        poll_only = bool(calls) and all(
            extract_tool_call(c)[0] == "bash_poll" for c in calls)
        fingerprint = reply_fingerprint(reply)
        if poll_only:
            pass
        elif fingerprint == last_fingerprint:
            identical += 1
        else:
            identical = 0
            last_fingerprint = fingerprint

        entry = {"role": "assistant", "content": text}
        recorded = recorded_calls(calls, turn) if calls else []
        if calls:
            entry["tool_calls"] = recorded
        messages.append(entry)

        if not calls:
            blanks += 1
            if not text.strip() and blanks >= BLANK_REPLY_CEILING:
                if seat.retire(seat.current()):
                    say("[LOOP] %d blank replies; changed seats" % blanks)
                    blanks = 0
                    cap = transcript_cap_chars(seat.current(),
                                               allowance.ceiling_usd)
                    turn += 1
                    continue
                say("[LOOP] %d blank replies and no seat left" % blanks)
                return
            messages.append({"role": "user", "content":
                "That reply carried no tool call. Take the next concrete "
                "step, or call submit if the change is complete."})
            turn += 1
            continue

        blanks = 0
        say("[LOOP] turn %d: %d tool call(s)" % (turn, len(calls)))

        turn_growth = len(text)
        for call, kept in zip(calls, recorded):
            name, args, arg_err = extract_tool_call(call)
            if arg_err:
                result = arg_err
            else:
                try:
                    result = kit.run(name, args)
                except Finished as done:
                    say("[LOOP] submit at turn %d: %s"
                        % (turn, str(done)[:200]))
                    raise
                except ToolFault as fault:
                    tool_faults += 1
                    result = "error: %s" % fault
                except BaseException as error:
                    tool_faults += 1
                    traceback.print_exc()
                    result = "error: %s: %s" % (type(error).__name__, error)
                else:
                    # The Warden counts its own refusals and stands down
                    # after its share; only an empty hand-in counts here.
                    if name == "submit" and kit.last_refusal == "empty":
                        refusals += 1
            body = clip(str(result), READ_OUTPUT_CAP)
            turn_growth += len(body)
            messages.append({"role": "tool", "tool_call_id": kept["id"],
                             "content": body})

        usage = reply.get("_usage") or {}
        history.append({
            "growth": turn_growth,
            "reply_tokens": int(usage.get("completion_tokens") or 1200),
        })
        turn += 1

# =========================================================================
# Part 5: agent_main
# =========================================================================

def _instruction_text(payload: dict, root: str) -> str:
    """Where the instruction comes from, in order of preference."""
    for key in ("problem_statement", "instruction", "problem", "task", "prompt"):
        value = (payload or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value
    for location in ("/installed-agent/instruction.md",
                     os.path.join(root, "instruction.md"),
                     "/app/instruction.md"):
        if os.path.isfile(location):
            try:
                with open(location, "r", encoding="utf-8",
                          errors="replace") as fh:
                    return fh.read()
            except OSError:
                continue
    return ""


def harness_lead(wall: float, anchor: str = ANCHOR_FILE) -> float:
    """Seconds the harness's clock has been running longer than ours. The
    harness writes the instruction file the moment its timer starts, then
    commits a git baseline of the whole tree before we run; that file's age
    is the lead. An absurd age (a stale file, a clock ahead of the file's)
    counts as none."""
    try:
        age = time.time() - os.stat(anchor).st_mtime
    except OSError:
        return 0.0
    return age if 0.0 < age < wall / 2 else 0.0


def tail_budget(allowance: Allowance, share: float, cap: float,
                floor: float = 3.0) -> float:
    """Part of what is left before the harness deadline, for one tail step."""
    return max(floor, min(cap, allowance.hard_left() * share))


def agent_main(input: dict) -> str:
    """Every path out of this function returns a patch built from the tree.

    Even a crash returns whatever the model managed to write before it. The
    only way to return "" is if the tree is genuinely unchanged.
    """
    allowance = Allowance()
    lead = harness_lead(allowance.wall)
    if lead:
        allowance.shift(lead)
        say("[RUN] harness clock started ~%.0fs before this one" % lead)
    root = workdir()
    statement = _instruction_text(input or {}, root)

    say("[RUN] workdir=%s budget=$%.3f clock=%.0fs (hard %.0fs)"
        % (root, allowance.ceiling_usd, allowance.clock_left(),
           allowance.hard_left()))

    if not statement.strip():
        say("[RUN] no instruction found; nothing to do")
        return ""

    # Nothing has been edited yet, so a failure here loses no work; it is
    # still reported rather than raised so the log says what happened.
    try:
        tree = Tree(root)
        pool = ShellPool(root)
    except BaseException as error:
        traceback.print_exc()
        say("[RUN] snapshot failed: %s: %s" % (type(error).__name__, error))
        return ""
    say("[RUN] container: %s" % pool.describe())

    # A copy of the tree as it starts, before anything can touch it, so
    # restore and revert never depend on what fits in memory.
    try:
        tree.make_pristine()
    except BaseException:
        traceback.print_exc()

    # --- Stage 0: what the instruction itself says ---
    scope = parse_scope(statement, tree)

    # --- Stage 1: locate the target, unless the instruction named it ---
    located = {"targets": [], "note": ""}
    if scope["files"]:
        region = scope.get("region")
        located["targets"] = [{
            "path": path,
            "symbol": scope["symbol"] if index == 0 else "",
            "lines": "%d-%d" % region if (index == 0 and region) else "",
            "why": "named by the instruction",
            "confidence": "high",
        } for index, path in enumerate(scope["files"])]
        say("[LOCATOR] skipped: the instruction names %s" % ", ".join(scope["files"]))
    else:
        try:
            located = run_locator(statement, tree, pool, allowance,
                                  Beacon("locator"), missing=scope["missing"])
        except BaseException as error:
            traceback.print_exc()
            say("[RUN] locator crashed: %s: %s" % (type(error).__name__, error))

    # --- Stage 2: plan against the target, unless the code is already in hand ---
    plan_note = ""
    if scope["files"]:
        say("[PLANNER] skipped: the driver opens with the named code")
    else:
        try:
            plan_note = run_planner(statement, tree, pool, allowance,
                                    located.get("targets", []),
                                    Beacon("planner"))
        except BaseException as error:
            traceback.print_exc()
            say("[RUN] planner crashed: %s: %s" % (type(error).__name__, error))

    # --- Stage 3: driver loop ---
    warden = Warden(tree, pool, allowance, statement, scope)

    try:
        drive(statement, tree, pool, allowance, located, plan_note, warden, scope)
    except Finished as done:
        say("[RUN] submitted: %s" % str(done)[:200])
    except Spent as stop:
        say("[RUN] stopped: %s" % stop)
    except BaseException as error:
        traceback.print_exc()
        say("[RUN] crashed: %s: %s" % (type(error).__name__, error))

    # --- clean up the shell pool ---
    try:
        pool.close()
    except BaseException:
        pass

    # Every step below takes a share of what is left before the harness
    # gives up on us, so the sum stays inside it whatever the wall clock is.
    say("[RUN] tail: %.0fs before the harness deadline" % allowance.hard_left())

    # --- nothing outside the named files may travel ---
    try:
        tree.revert_outside(scope["files"], tail_budget(allowance, 0.25, 30.0))
    except BaseException:
        traceback.print_exc()

    # --- what the run left on disk, so the patch can be held to it ---
    edited = {}
    try:
        edited = tree.fingerprint()
    except BaseException:
        traceback.print_exc()

    # --- the patch is ALWAYS built from the tree ---
    patch = ""
    try:
        patch = tree.diff(tail_budget(allowance, 0.5, 60.0))
    except BaseException:
        traceback.print_exc()

    # --- a patch that applies must also rebuild the edited tree ---
    try:
        differs = tree.round_trip(patch, edited, tail_budget(allowance, 0.3, 20.0))
        if differs:
            say("[PATCH] round trip differs for %s; rebuilding as whole-file hunks"
                % ", ".join(differs[:6]))
            patch = tree.diff(tail_budget(allowance, 0.5, 60.0), whole=differs)
            differs = tree.round_trip(patch, edited, tail_budget(allowance, 0.3, 20.0))
            if differs:
                tree.dropped.append("round trip differs for %s" % ", ".join(differs[:6]))
        if differs == []:
            say("[PATCH] round trip: %d file(s) reproduced" % len(edited))
    except BaseException:
        traceback.print_exc()

    # --- restore the tree for the next run ---
    try:
        tree.restore(tail_budget(allowance, 0.6, 60.0))
    except BaseException:
        pass

    # --- validate the patch before returning ---
    usable = None
    try:
        usable = tree.applies(patch, tail_budget(allowance, 0.5, 30.0))
    except BaseException:
        pass

    if usable is False and patch.strip():
        try:
            rescued = tree.salvage(patch, tail_budget(allowance, 0.9, 45.0))
        except BaseException as error:
            say("[PATCH] salvage failed: %s" % type(error).__name__)
            rescued = ""
        if rescued:
            patch, usable = rescued, True

    try:
        patch.encode("utf-8")
    except UnicodeEncodeError:
        say("[PATCH] carries bytes that are not UTF-8; replacing them")
        patch = patch.encode("utf-8", "surrogateescape").decode("utf-8", "replace")

    status = {True: "yes", False: "no"}.get(usable, "unknown")
    if tree.dropped:
        say("[PATCH] INCOMPLETE: %d part(s) of the change are not in the "
            "patch: %s" % (len(tree.dropped), "; ".join(tree.dropped)[:400]))
        if usable:
            status = "partial"
    say("[RUN] done in %.0fs, $%.4f over %d calls (%d quoted by the endpoint), "
        "%d edits, patch %dB, usable=%s"
        % (allowance.elapsed(), allowance.spent, allowance.calls,
           allowance.billed, allowance.edits, len(patch), status))
    return patch