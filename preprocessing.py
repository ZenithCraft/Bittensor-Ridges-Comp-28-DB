#!/usr/bin/env python3
"""Evaluate the agent's pre-processing on its own, with no inference.

Everything `agent_main` does before the first model call is deterministic:
resolve the repository, parse the instruction, index the source, rank the
candidate files, trace the call graph, discover and connect to the database,
slice the code and render the prompt. That half decides what the model is able
to answer with at all -- a prompt that does not contain the method to change
cannot be rescued by a better model -- and it costs nothing to run, so it can
be measured on every task as often as you like.

    ./preprocessing.py pg-netbox-prefix-hierarchy-annotations-001
    ./preprocessing.py ./fast-tasks/pg-netbox-bulk-tag-assignment-001
    ./preprocessing.py --all --no-db
    ./preprocessing.py <task> --prompt             # the exact bytes the model receives
    ./preprocessing.py --all --json now.json
    ./preprocessing.py --all --baseline before.json
    ./preprocessing.py --repo /path/to/app --instruction ./instruction.md

Four things are graded here that the other harnesses do not cover:

  * containment -- is the code that actually has to change inside the excerpt?
    rank_eval.py answers "which file ranked first"; this answers "were the
    lines the fix must touch actually sent". A file at rank 1 whose relevant
    method was sliced away is a prompt nobody can answer from.
  * enforcement -- every rule the agent gates on, and whether the instruction
    states it. A gate that rejects a patch for a rule the model was never told
    is a trap: the model cannot avoid it and the failure looks like stupidity.
  * economics -- what the first call costs before any reply, split by section,
    so it is clear which part of the prompt is worth attacking.
  * completeness -- which constraints the prose states and the parser missed.

Containment is read back out of the rendered prompt rather than recomputed:
the slice headers say exactly which lines were sent, so this measures the
artifact itself and cannot drift from render_evidence's budgeting.

The prompt reported here is the whole first user turn plus the system message,
assembled exactly as Solver.solve assembles it -- not just the evidence bundle.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent
import rank_eval
import selftest

BENCH = selftest.BENCH
G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"

# `--- path lines 12-88 (function Foo.bar) ---`, written by Slice.render().
SLICE_HEADER = re.compile(
    r"^--- (?P<path>\S+) lines (?P<start>\d+)-(?P<end>\d+) \((?P<label>[^)]*)\) ---$", re.M)

# Characters per token: the same divisor affordable_cap() prices calls with, so
# a prompt that looks affordable here looks affordable to the agent too.
CHARS_PER_TOKEN = 3.5

# Discovery falls back to knocking on conventional hostnames, and that pass is
# bounded by a 15-second deadline in DatabaseProbe._from_network. With no
# database reachable it spends all of it, which dominates a whole-corpus run.
SLOW_DISCOVERY_SECONDS = 5.0

# Rules the instructions state in prose, each with the constraint label the
# parser is expected to produce for it. A rule the prose states and the parser
# misses is a rule nothing enforces, so the grader finds the violation instead
# of the agent -- which is how "use only names the file already imports" went
# unchecked while leading ERROR_HINTS.
STATED_RULES: tuple[tuple[str, str, str], ...] = (
    ("names the file does not import",
     r"use only names (?:the file|it) already imports|only names .{0,24}already imports"
     r"|without adding (?:any )?(?:new )?imports",
     "'use only names the file already imports'"),
    ("materialising rows in Python",
     r"[Dd]o not materiali[sz]e|keep .{0,40}database-backed",
     "'do not materialize ... in Python'"),
)

# How each parsed constraint is worded in the prose, so the enforcement audit
# looks for the sentence rather than for its own label.
RULE_EVIDENCE: dict[str, str] = {
    "names the file does not import": r"already imports|adding .{0,12}imports",
    "materialising rows in Python": r"materiali[sz]e|database-backed",
}


# ---------------------------------------------------------------------------
# Task resolution and staging
# ---------------------------------------------------------------------------

def resolve_task(argument: str, tasks_dir: Path) -> Path:
    """A task given as a path, or as a name under the task directory.

    Both are natural to type -- `pg-netbox-...` from the bench, or
    `./fast-tasks/pg-netbox-...` for a local copy -- and a bare name that does
    not resolve is far more often a typo than a task that does not exist, so
    say which one was probably meant.
    """
    direct = Path(argument)
    if (direct / "instruction.md").is_file():
        return direct.resolve()
    named = tasks_dir / argument
    if (named / "instruction.md").is_file():
        return named.resolve()

    known: set[str] = set()
    for parent in (tasks_dir, direct.parent, Path.cwd()):
        if parent.is_dir():
            known |= {p.name for p in parent.iterdir() if (p / "instruction.md").is_file()}
    close = difflib.get_close_matches(direct.name, sorted(known), n=3, cutoff=0.6)
    hint = (f"\n  did you mean: {', '.join(close)}" if close else
            f"\n  {len(known)} task(s) found, e.g. {', '.join(sorted(known)[:3])}")
    raise SystemExit(f"no task at {argument!r}, and no instruction.md there or under "
                     f"{tasks_dir}{hint}")


def task_root(task: Path, checkout: Path | None, staging: Path) -> Path | None:
    """The application checkout for a task, staged if it needs staging."""
    if task.name in selftest.EXPECTED:
        if checkout is None:
            return None
        return selftest.stage(checkout, task, staging / task.name)
    generated = task / "environment" / "app"
    return generated if generated.is_dir() else None


def ground_truth(task: Path, root: Path, parsed: agent.Instruction) -> tuple[str | None, str | None]:
    """(file, symbol) the reference solution actually changes, where known.

    For the six netbox samples selftest pins both. For the generated tasks the
    solution script names the file; the symbol comes from the prose the same
    way rank_eval reads it.
    """
    if task.name in selftest.EXPECTED:
        return selftest.EXPECTED[task.name], selftest.EXPECTED_METHOD.get(task.name)
    return (rank_eval.target_from_solution(task, root),
            rank_eval.target_symbol((task / "instruction.md").read_text(), parsed))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def analyse(root: Path, text: str, *, truth_file: str | None = None,
            truth_symbol: str | None = None, probe_db: bool = True) -> dict:
    """Run every deterministic stage and return what each produced."""
    timings: dict[str, float] = {}

    started = perf_counter()
    parsed = agent.parse_instruction(text, root)
    timings["parse"] = perf_counter() - started

    started = perf_counter()
    repo = agent.Repository(root)
    timings["index"] = perf_counter() - started

    started = perf_counter()
    candidates = agent.locate_targets(repo, parsed)
    timings["locate"] = perf_counter() - started

    started = perf_counter()
    if not probe_db:
        os.environ["RIDGES_PROBE_NO_PING"] = "1"      # discovery without the pings
    probe = agent.DatabaseProbe(repo, parsed)
    # agent_main resolves the engine from the live database before rendering.
    # Skipping it here reported `engine: unknown` on prompts the agent sends as
    # postgresql -- a harness that lies about the artifact it is measuring.
    engine_from_prose = parsed.engine
    if parsed.engine == "unknown" and probe.available():
        parsed.engine = probe.targets[0].engine
    timings["database"] = perf_counter() - started

    started = perf_counter()
    evidence = agent.render_evidence(repo, parsed, candidates, probe)
    # The whole first call, exactly as Solver.solve assembles it.
    user_turn = (f"{evidence}\n\n{agent.PromptBuilder.SEPARATOR}\n\n"
                 f"# Response format\n\n{agent.EDIT_PROTOCOL}")
    prompt = f"{agent.SYSTEM_PROMPT}\n\n{user_turn}"
    timings["render"] = perf_counter() - started

    ranked = [path for path, _ in candidates]
    spans: dict[str, list[tuple[int, int, str]]] = {}
    for match in SLICE_HEADER.finditer(evidence):
        spans.setdefault(match.group("path"), []).append(
            (int(match.group("start")), int(match.group("end")), match.group("label")))

    lead = agent.MODELS.get(agent.LADDER[0][0])
    tokens = int(len(prompt) / CHARS_PER_TOKEN)
    result = {
        "root": str(root),
        "timings": timings,
        "total_seconds": sum(timings.values()),
        "instruction": {
            "kinds": parsed.kinds,
            "engine": parsed.engine,
            "engine_from_prose": engine_from_prose,
            "edit_only": parsed.edit_only,
            "lint_paths": parsed.lint_paths,
            "named_paths": parsed.named_paths,
            "forbidden": parsed.forbidden,
            "commands": parsed.commands,
            "identifiers": parsed.identifiers[:12],
            "method_hint": parsed.method_hint,
            "class_hint": parsed.class_hint,
            "single_method": parsed.single_method,
            "style_constraints": parsed.style_constraints,
            "targets": parsed.targets,
            "traced_hint": parsed.traced_hint,
        },
        "index": {"files": len(repo.files)},
        "ranking": {
            "candidates": ranked,
            "scores": [round(s, 1) for s in parsed.candidate_scores],
            "explicit": bool(parsed.edit_only or parsed.lint_paths),
        },
        "database": {
            # target_url is what the prompt itself prints, so what is shown
            # here is what the model would read, character for character.
            "found": [agent.target_url(t) for t in probe.targets],
            "engine": probe.targets[0].engine if probe.targets else None,
            "verified": probe_db,
            # (url, answered) per candidate the SELECT 1 test actually tried.
            # Candidates rejected by the port filter never appear: they are
            # dropped in _add before any connection is attempted.
            "attempts": list(getattr(probe, "attempts", [])),
        },
        "slices": {path: [{"start": s, "end": e, "label": label} for s, e, label in items]
                   for path, items in spans.items()},
        "shown_lines": sum(e - s + 1 for items in spans.values() for s, e, _ in items),
        "prompt": {
            "chars": len(prompt),
            "tokens": tokens,
            "usd": round(tokens / 1e6 * lead.usd_per_m_in, 5) if lead else None,
            "model": agent.LADDER[0][0],
            "composition": composition(prompt, evidence, user_turn),
        },
        "enforcement": enforcement_audit(parsed, text),
        "structure": prompt_structure(prompt, evidence, user_turn, text),
        "gaps": parse_gaps(parsed, text),
    }
    result["containment"] = containment(repo, spans, ranked, truth_file, truth_symbol)
    result["_prompt_text"] = prompt                  # dropped before any JSON dump
    return result


def composition(prompt: str, evidence: str, user_turn: str) -> dict[str, int]:
    """Tokens per section, so it is clear which part is worth attacking.

    Sections map onto the four elements a prompt is built from: `instructions`
    is the instruction, `task statement` and `relevant source` and `live schema`
    are input data and context, `reply format` is the output indicator. The
    statement is the model's authority and the code is the material; the rest is
    the agent talking about itself, and is where a prompt gets long without
    getting clearer.

    `instructions` and `reply format` are fixed cost -- identical on every task
    and every retry -- so they are the two rows to judge against what they buy,
    not against their size.
    """
    def size(text: str) -> int:
        return int(len(text) / CHARS_PER_TOKEN)

    parts: dict[str, int] = {"system prompt": size(agent.SYSTEM_PROMPT),
                             "reply format": size(agent.EDIT_PROTOCOL)}
    # Split on the headers render_evidence emits, not on every "# " line: the
    # statement is markdown and carries headings of its own, which otherwise
    # get counted as prompt sections and leave "task statement" at 5 tokens.
    # "Instructions" is matched first and anchored to the top of the prompt for
    # the same reason -- a statement is free to open with one.
    starts = [(m.start(), m.group(1)) for m in
              re.finditer(r"^# (Instructions|Task statement[^\n]*|What this agent found[^\n]*"
                          r"|Relevant source|Live schema[^\n]*)$", evidence, re.M)]
    for index, (offset, header) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(evidence)
        parts[header.lower().split(" (")[0]] = size(evidence[offset:end])
    return parts


def prompt_structure(prompt: str, evidence: str, user_turn: str,
                     statement: str) -> list[tuple[str, bool, str]]:
    """Whether the assembled prompt still obeys the rules it was designed to.

    Composition says how big each element is; this says whether the elements
    are the right ones, in the right order, and still separable. Both matter
    and neither implies the other -- a prompt can be perfectly proportioned and
    still open with input data instead of an instruction, which is what this
    one did before the elements were named.

    Per task rather than once, because three of these depend on the statement:
    a statement carrying its own `#` headings, or one whose text the renderer
    truncates, breaks them without changing a line of agent.py.
    """
    checks: list[tuple[str, bool, str]] = []
    order = ["# Instructions", "# Task statement", "# Relevant source", "# Response format"]
    found = [(h, user_turn.find(h)) for h in order]
    present = [(h, i) for h, i in found if i >= 0]

    checks.append(("instruction element leads the prompt",
                   evidence.startswith("# Instructions"),
                   f"starts with {evidence[:14]!r}"))
    positions = [i for _, i in present]
    checks.append(("elements in reading order",
                   positions == sorted(positions),
                   " -> ".join(h.lstrip('# ') for h, _ in present)))
    # The statement is quoted, never paraphrased. This is the property the whole
    # design rests on: the prompt stopped restating the instruction's own rules
    # on the grounds that the prose says them better, which is only true while
    # the prose is actually there, whole.
    checks.append(("task statement carried verbatim",
                   statement.strip() in prompt,
                   f"{len(statement.strip()):,} chars quoted"))
    rules = evidence.count(f"\n\n{agent.PromptBuilder.SEPARATOR}\n\n")
    checks.append(("elements separated by a rule",
                   rules >= len(present) - 2,
                   f"{rules} separator(s) for {len(present)} element(s)"))
    blocks = re.findall(r'^\{"action".*?^ \]\}', agent.EDIT_PROTOCOL, re.S | re.M)
    bad = []
    for block in blocks:
        try:
            json.loads(block)
        except json.JSONDecodeError as error:
            bad.append(str(error))
    checks.append(("reply-format examples parse as JSON", not bad and bool(blocks),
                   f"{len(blocks) - len(bad)}/{len(blocks)} parse"
                   + (f"; first error: {bad[0]}" if bad else "")))
    return checks


def enforcement_audit(parsed: agent.Instruction, text: str) -> list[dict]:
    """Every rule a gate applies, and whether the instruction states it.

    The prompt no longer restates the instruction's own rules -- the prose is
    printed in full and says them better. That only holds while the parser and
    the prose agree: a gate enforcing something the instruction never states
    rejects a patch for a rule the model could not have known, which reads as
    the model being stupid and is really the agent being unfair.
    """
    flat = re.sub(r"\s+", " ", text)
    rules: list[dict] = []

    def add(rule: str, gate: str, active: bool, evidence: str) -> None:
        if active:
            rules.append({"rule": rule, "gate": gate,
                          "stated": bool(re.search(evidence, flat, re.IGNORECASE))})

    for path in parsed.edit_only:
        add(f"edit only {path}", "check_scope", True,
            rf"{re.escape(path)}")
    add(f"only {parsed.class_hint or ''}{'.' if parsed.class_hint else ''}{parsed.method_hint} "
        f"may change", "check_single_method", parsed.single_method and bool(parsed.method_hint),
        rf"{re.escape(parsed.method_hint or '!')}")
    for construct in parsed.style_constraints:
        evidence = RULE_EVIDENCE.get(
            construct, construct.replace(" ", r"\s+") + "|" + construct.rstrip("s").replace(" ", r"\s+"))
        add(f"no {construct} in the edited code", "check_style", True, evidence)
    if parsed.targets.get("max_queries"):
        add(f"at most {parsed.targets['max_queries']} queries", "measure_query_scaling", True,
            r"\bat most\b|\bno more than\b|\bbounded\b")
    # Two tiers with different trust: the absolute one admits no exemption at
    # all, the conditional one needs an explicit permission. Reported apart
    # because only the second can be unlocked by anything the prose says.
    add("tests, fixtures and project scripts are never editable", "check_protected", True,
        r"tests|fixtures")
    add("migrations and project config need an explicit permission", "check_protected",
        bool(parsed.edit_only or parsed.lint_paths)
        and any(agent.Checker._NEVER_EDIT_UNLESS_PERMITTED.search(p)
                for p in parsed.edit_only + parsed.lint_paths),
        r"migration|\.cfg|\.ini")
    return rules


def parse_gaps(parsed: agent.Instruction, text: str) -> list[str]:
    """What the parser did not find, and what the agent therefore falls back on."""
    gaps: list[str] = []
    if not parsed.commands:
        gaps.append("no check command: the patch can only be verified statically")
    if not (parsed.edit_only or parsed.lint_paths or parsed.named_paths):
        gaps.append("no named file: scope falls back to the ranked candidates")
    if not parsed.method_hint:
        gaps.append("no named method: the slicer picks by hot line instead")
    if parsed.engine == "unknown":
        gaps.append("engine unresolved: neither the prose nor a live database settled it")
    flat = re.sub(r"\s+", " ", text)
    for label, pattern, quoted in STATED_RULES:
        if re.search(pattern, flat, re.IGNORECASE) and label not in parsed.style_constraints:
            gaps.append(f"{quoted} is stated but not parsed: no gate checks it")
    return gaps


def containment(repo: agent.Repository, spans: dict[str, list[tuple[int, int, str]]],
                ranked: list[str], truth_file: str | None, truth_symbol: str | None) -> dict:
    """Did the prompt carry the lines the fix has to touch?

    Four outcomes, and they are not the same failure. `absent` means ranking
    sent the model to the wrong file. `missed` means ranking was right and the
    slicer cut the wrong region -- a much cheaper thing to fix, and invisible
    to a rank-only metric. `partial` means the method was truncated, which the
    model can sometimes still work from via need_context. `unknown` means this
    task has no recorded ground truth, not that it passed.
    """
    out: dict = {"file": truth_file, "symbol": truth_symbol,
                 "rank": (ranked.index(truth_file) + 1) if truth_file in ranked else None,
                 "verdict": "unknown", "symbol_lines": None, "shown_lines": None}
    if not truth_file:
        return out
    if truth_file not in spans:
        # A file can be in the prompt whole without a slice header only when it
        # was short enough to show entirely, which Slice.render still labels.
        out["verdict"] = "absent"
        return out
    shown = spans[truth_file]
    out["shown_lines"] = [[s, e] for s, e, _ in shown]
    if not truth_symbol:
        out["verdict"] = "file-only"          # right file, no symbol to check against
        return out
    span = rank_eval.symbol_range(repo, truth_file, truth_symbol)
    if span is None:
        out["verdict"] = "file-only"
        return out
    start, end = span
    out["symbol_lines"] = [start, end]
    if any(s <= start and end <= e for s, e, _ in shown):
        out["verdict"] = "full"
    elif any(s <= end and start <= e for s, e, _ in shown):
        out["verdict"] = "partial"
    else:
        out["verdict"] = "missed"
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

VERDICT_COLOUR = {"full": G, "partial": Y, "missed": R, "absent": R,
                  "file-only": B, "unknown": D}


def paint(verdict: str) -> str:
    return f"{VERDICT_COLOUR.get(verdict, D)}{verdict}{X}"


def dash(value) -> str:
    return f"{value}" if value else f"{D}-{X}"


def report(name: str, result: dict, *, show_prompt: bool) -> None:
    ins, rank = result["instruction"], result["ranking"]
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"{D}{result['root']}{X}")

    print(f"\n{B}1. instruction parsed{X}   {result['timings']['parse'] * 1000:.0f}ms")
    print(f"   {D}the prompt sends instruction.md verbatim; these drive the gates, the{X}")
    print(f"   {D}ranker and the slicer -- they are no longer restated to the model{X}")
    print(f"   edit_only      {dash(ins['edit_only'])}")
    print(f"   lint_paths     {dash(ins['lint_paths'])}")
    print(f"   named_paths    {dash(ins['named_paths'])}   {D}ranking evidence only{X}")
    method = f"{ins['class_hint']}.{ins['method_hint']}" if ins["class_hint"] else ins["method_hint"]
    print(f"   method         {dash(method)}"
          + (f"   {Y}single-method{X}" if ins["single_method"] else ""))
    print(f"   style          {dash(ins['style_constraints'])}")
    print(f"   forbidden      {len(ins['forbidden'])} clause(s)")
    for clause in ins["forbidden"][:3]:
        print(f"     {D}x{X} {clause[:96]}")
    print(f"   targets        {dash(ins['targets'])}")
    print(f"   commands       {len(ins['commands'])}")
    for command in ins["commands"][:4]:
        flat = command.replace(chr(10), " ; ")
        print(f"     {D}${X} {flat[:110]}" + (f"{D}...{X}" if len(flat) > 110 else ""))
    print(f"   kinds          {dash(', '.join(ins['kinds']))}   {D}selects evidence, not sent{X}")
    source = "the live database" if result["database"]["verified"] else "unverified discovery"
    print(f"   engine         {ins['engine']}"
          + (f"   {D}(from {source}; the prose said {ins['engine_from_prose']}){X}"
             if ins["engine"] != ins["engine_from_prose"] else f"   {D}(from the prose){X}"))
    print(f"   identifiers    {dash(', '.join(ins['identifiers'][:8]))}")

    print(f"\n{B}2. index{X}                {result['timings']['index'] * 1000:.0f}ms"
          f"   {result['index']['files']} source files")

    print(f"\n{B}3. ranking{X}              {result['timings']['locate'] * 1000:.0f}ms"
          + (f"   {D}(instruction named the path; no ranking needed){X}" if rank["explicit"] else ""))
    contained = result["containment"]
    for position, path in enumerate(rank["candidates"], start=1):
        score = rank["scores"][position - 1] if position <= len(rank["scores"]) else None
        marker = f" {G}<- the file the solution changes{X}" if path == contained["file"] else ""
        print(f"   {position}. {path}" + (f"  {D}{score}{X}" if score is not None else "") + marker)
    if ins["traced_hint"]:
        print(f"   {D}call graph also reached (offered, not ranked): {ins['traced_hint']}{X}")

    seconds = result["timings"]["database"]
    database = result["database"]
    print(f"\n{B}4. database{X}             {seconds * 1000:.0f}ms"
          + ("" if database["verified"]
             else f"   {Y}--no-db: nothing was connected to, so the candidates below are "
                  f"guesses the agent would not use{X}"))
    if database["attempts"]:
        # The connection test, candidate by candidate. Only a url that answered
        # SELECT 1 can reach the prompt, so this is the whole gate.
        for url, answered in database["attempts"]:
            print(f"   [{G}connected{X}] {url}" if answered
                  else f"   [{R}no answer{X}] {url}   {D}dropped{X}")
    elif database["verified"]:
        print(f"   {D}no candidate survived discovery to be tried{X}")
    for found in database["found"][:3] or [f"{D}nothing reached the prompt{X}"]:
        print(f"   {D}->{X} {found}")
    if database["verified"] and seconds > SLOW_DISCOVERY_SECONDS:
        # Each SELECT 1 can burn its 8s timeout, and a discovery that ends with
        # nothing then knocks on conventional hostnames under a 15s deadline.
        # Both are waiting, not work, and they dominate a whole-corpus run.
        print(f"   {D}{seconds:.0f}s of that is connect timeouts and the _from_network "
              f"fallback; --no-db skips both{X}")

    print(f"\n{B}5. slices{X}               {result['timings']['render'] * 1000:.0f}ms"
          f"   {result['shown_lines']} lines shown across {len(result['slices'])} file(s)")
    for path, items in result["slices"].items():
        for piece in items[:4]:
            print(f"   {path}:{piece['start']}-{piece['end']}  {D}{piece['label']}{X}")
        if len(items) > 4:
            print(f"   {D}... {len(items) - 4} more slice(s) of {path}{X}")

    print(f"\n{B}6. containment{X}")
    if contained["file"]:
        where = f"rank {contained['rank']}" if contained["rank"] else f"{R}not in the candidates{X}"
        print(f"   solution changes {contained['file']}"
              + (f" {contained['symbol']}()" if contained["symbol"] else "") + f"   ({where})")
        if contained["symbol_lines"]:
            print(f"   symbol at lines  {contained['symbol_lines'][0]}-{contained['symbol_lines'][1]}"
                  f"   shown: {contained['shown_lines']}")
        print(f"   verdict          {paint(contained['verdict'])}")
    else:
        print(f"   {D}no recorded ground truth for this task{X}")

    print(f"\n{B}7. enforcement{X}   {D}a gate rejecting a rule the instruction never states "
          f"is a trap{X}")
    for rule in result["enforcement"]:
        mark = f"{G}stated{X}" if rule["stated"] else f"{R}NOT STATED{X}"
        print(f"   [{mark}] {rule['rule']:<58} {D}{rule['gate']}{X}")

    prompt = result["prompt"]
    print(f"\n{B}8. first call{X}           {prompt['chars']:,} chars"
          f"   ~{prompt['tokens']:,} tokens   ~${prompt['usd']:.5f} on {prompt['model']}")
    for section, size in sorted(prompt["composition"].items(), key=lambda item: -item[1]):
        share = size / max(1, prompt["tokens"])
        print(f"   {size:>6,} tok  {share:>4.0%}  {section}")
    fixed = sum(prompt["composition"].get(k, 0)
                for k in ("system prompt", "instructions", "reply format"))
    print(f"   {D}system prompt, instructions and reply format are identical on every task "
          f"and every retry: {fixed:,} tok ({fixed / max(1, prompt['tokens']):.0%}) of fixed cost{X}")

    print(f"\n{B}9. prompt structure{X}   {D}the four elements, in the order the model reads "
          f"them{X}")
    for check, ok, detail in result["structure"]:
        mark = f"{G}ok{X}" if ok else f"{R}NO{X}"
        print(f"   [{mark}] {check:<44} {D}{detail}{X}")

    if result["gaps"]:
        print(f"\n{B}10. gaps{X}")
        for gap in result["gaps"]:
            print(f"   {Y}-{X} {gap}")

    print(f"\n{D}total pre-processing {result['total_seconds']:.2f}s   (no inference, no cost){X}")

    if show_prompt:
        print(f"\n{'=' * 78}\nPROMPT AS SENT (system message, then the first user turn)\n{'=' * 78}")
        print(result["_prompt_text"])


ROW = f"{'task':<46}{'files':>6}{'rank':>6}{'contained':>11}{'lines':>7}{'tokens':>9}{'$':>9}{'secs':>7}"


def row(name: str, r: dict) -> str:
    c = r["containment"]
    rank = str(c["rank"]) if c["rank"] else "-"
    colour = VERDICT_COLOUR.get(c["verdict"], D)
    return (f"{name[:45]:<46}{r['index']['files']:>6}{rank:>6}"
            f"{colour}{c['verdict']:>11}{X}{r['shown_lines']:>7}"
            f"{r['prompt']['tokens']:>9,}{r['prompt']['usd']:>9.5f}"
            f"{r['total_seconds']:>7.2f}")


def summary(results: dict[str, dict]) -> None:
    print("-" * 100)
    verdicts: dict[str, int] = {}
    for r in results.values():
        verdicts[r["containment"]["verdict"]] = verdicts.get(r["containment"]["verdict"], 0) + 1
    graded = sum(count for verdict, count in verdicts.items() if verdict != "unknown")
    tokens = [r["prompt"]["tokens"] for r in results.values()]
    spend = [r["prompt"]["usd"] for r in results.values()]
    seconds = [r["total_seconds"] for r in results.values()]
    ranked_first = sum(1 for r in results.values() if r["containment"]["rank"] == 1)

    print("containment   " + "  ".join(f"{paint(v)} {n}" for v, n in sorted(verdicts.items())))
    if graded:
        print(f"              {verdicts.get('full', 0)}/{graded} graded tasks sent the whole "
              f"method   ({ranked_first}/{graded} also ranked its file first)")
    print(f"first call    mean {sum(tokens) // len(tokens):,} tokens, max {max(tokens):,}"
          f"   mean ${sum(spend) / len(spend):.5f}, max ${max(spend):.5f}")
    print(f"runtime       mean {sum(seconds) / len(seconds):.2f}s, max {max(seconds):.2f}s"
          f"   {D}(spent before the first call on every task){X}")

    # A structural break is silent in the row view -- the prompt still renders,
    # still costs what it cost, and still contains the code. It only shows as a
    # worse reply, several minutes and one inference bill later.
    broken = [(name, check, detail) for name, r in results.items()
              for check, ok, detail in r["structure"] if not ok]
    if broken:
        print(f"{R}prompt structure broken{X}   {len(broken)} check(s) across "
              f"{len({n for n, _, _ in broken})} task(s)")
        for name, check, detail in broken[:6]:
            print(f"   {name[:44]:<46}{check}   {D}{detail}{X}")
    else:
        print(f"prompt        {G}structure intact on every task{X}   "
              f"{D}instruction first, elements in order, statement verbatim{X}")

    unstated = [(name, rule["rule"]) for name, r in results.items()
                for rule in r["enforcement"] if not rule["stated"]]
    if unstated:
        print(f"{R}enforced but not stated{X}   {len(unstated)} rule(s) across "
              f"{len({n for n, _ in unstated})} task(s)")
        for name, rule in unstated[:6]:
            print(f"   {name[:44]:<46}{rule}")

    gaps: dict[str, int] = {}
    for r in results.values():
        for gap in r["gaps"]:
            gaps[gap] = gaps.get(gap, 0) + 1
    if gaps:
        print("gaps")
        for gap, count in sorted(gaps.items(), key=lambda item: -item[1]):
            print(f"   {count:>3} tasks  {gap}")


def compare(results: dict[str, dict], baseline_path: Path) -> None:
    """What a change to the pre-processing moved, task by task."""
    baseline = json.loads(baseline_path.read_text())
    order = {"full": 3, "partial": 2, "file-only": 2, "missed": 1, "absent": 0, "unknown": -1}
    moved, tokens_delta = [], 0
    for name, now in results.items():
        was = baseline.get(name)
        if not was:
            continue
        # Tolerate dumps written before the prompt/evidence rename.
        before_tokens = (was.get("prompt") or was.get("evidence") or {}).get("tokens", 0)
        tokens_delta += now["prompt"]["tokens"] - before_tokens
        before, after = was["containment"]["verdict"], now["containment"]["verdict"]
        rank_before, rank_after = was["containment"]["rank"], now["containment"]["rank"]
        if before != after or rank_before != rank_after:
            moved.append((name, before, after, rank_before, rank_after,
                          order.get(after, -1) - order.get(before, -1)))
    print(f"\n{'=' * 100}\nAGAINST {baseline_path.name}\n{'=' * 100}")
    if not moved:
        print(f"{D}containment and rank unchanged on every shared task{X}")
    for name, before, after, rank_before, rank_after, direction in sorted(moved, key=lambda m: m[5]):
        arrow = f"{R}worse{X}" if direction < 0 else f"{G}better{X}" if direction > 0 else f"{Y}moved{X}"
        print(f"  {arrow:<16}{name[:44]:<46}{before} -> {after}"
              f"   rank {rank_before} -> {rank_after}")
    shared = sum(1 for name in results if name in baseline)
    if shared:
        print(f"\nfirst call {tokens_delta:+,} tokens across {shared} shared task(s)"
              f"   {D}({tokens_delta / shared:+,.0f} per task){X}")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", nargs="?", help="task name, or a path to a task directory")
    parser.add_argument("--all", action="store_true", help="every task in the task directory")
    parser.add_argument("--tasks-dir", help=f"where named tasks live (default: {BENCH})")
    parser.add_argument("--repo", help="an arbitrary repository instead of a task")
    parser.add_argument("--instruction", help="problem statement file (with --repo)")
    parser.add_argument("--prompt", "--evidence", dest="prompt", action="store_true",
                        help="print the prompt itself, exactly as the model receives it")
    parser.add_argument("--no-db", action="store_true",
                        help="skip the connection test (fast, but the candidates are unverified)")
    parser.add_argument("--json", help="dump the full result here")
    parser.add_argument("--baseline", help="compare against an earlier --json dump")
    parser.add_argument("--checkout", help="pinned checkout for the netbox samples")
    parser.add_argument("--verbose", action="store_true", help="leave the agent's own log on")
    parser.add_argument("--trace", action="store_true",
                        help="per-module input/output lines from the agent")
    parser.add_argument("--trace-all", action="store_true",
                        help="every function call, argument, local and state change")
    parser.add_argument("--trace-vars", action="store_true",
                        help="every variable change, statement by statement, with the value "
                             "before and after. The loudest mode; use on one task")
    args = parser.parse_args()

    if args.trace or args.trace_all or args.trace_vars:
        agent.TRACE = True
        # The per-line tracer reports every call and return itself, so the two
        # are alternatives rather than layers.
        if args.trace_vars:
            agent.install_variable_tracer()
        elif args.trace_all:
            agent.install_call_tracer()
    elif not args.verbose:
        agent.log = lambda message: None

    results: dict[str, dict] = {}
    if args.repo:
        if not args.instruction:
            parser.error("--repo needs --instruction")
        root = Path(args.repo).resolve()
        result = analyse(root, Path(args.instruction).read_text(), probe_db=not args.no_db)
        report(root.name, result, show_prompt=args.prompt)
        results = {root.name: result}
    elif args.all or args.task:
        checkout = selftest.default_checkout(args.checkout)
        tasks_dir = Path(args.tasks_dir).resolve() if args.tasks_dir else BENCH
        if args.task:
            wanted = [resolve_task(args.task, tasks_dir)]
        else:
            wanted = sorted(p for p in tasks_dir.iterdir() if (p / "instruction.md").is_file())
            if not wanted:
                parser.error(f"no tasks under {tasks_dir}")
        if args.all and not args.no_db:
            print(f"{Y}note{X} each task with no reachable database spends ~15s in the "
                  f"discovery fallback; add --no-db to skip the connection test", file=sys.stderr)
        if args.all:
            # Print each row as it lands. Accumulating and rendering at the end
            # meant a run killed part-way produced nothing at all, not even the
            # tasks that had already finished.
            print(f"\n{'=' * 100}\nPRE-PROCESSING\n{'=' * 100}\n{ROW}\n" + "-" * 100)
        with tempfile.TemporaryDirectory() as staging:
            for task in wanted:
                root = task_root(task, checkout, Path(staging))
                if root is None:
                    print(f"{D}skip {task.name}: needs a pinned checkout{X}", file=sys.stderr)
                    continue
                text = (task / "instruction.md").read_text()
                parsed = agent.parse_instruction(text, root)
                truth_file, truth_symbol = ground_truth(task, root, parsed)
                result = analyse(root, text, truth_file=truth_file, truth_symbol=truth_symbol,
                                 probe_db=not args.no_db)
                results[task.name] = result
                if args.all:
                    print(row(task.name, result), flush=True)
                else:
                    report(task.name, result, show_prompt=args.prompt)
        if args.all and results:
            summary(results)
    else:
        parser.error("give a task name, --all, or --repo with --instruction")

    if args.baseline:
        compare(results, Path(args.baseline))
    if args.json:
        dump = {name: {k: v for k, v in r.items() if not k.startswith("_")}
                for name, r in results.items()}
        Path(args.json).write_text(json.dumps(dump, indent=2, default=str))
        print(f"\n{D}wrote {args.json}{X}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
