#!/usr/bin/env python3
"""Ranking-accuracy harness for agent.locate_targets -- offline, no inference.

Every bench task names its target file, so the agent normally bypasses the
ranker. This harness removes that information and asks: with only the symptom
description, does the ranker put the right file in front of the model?

    python3 rank_eval.py <pinned-netbox-checkout> [--json out.json] [--baseline prev.json]
    python3 rank_eval.py <netbox> --seed-check       # same result under two hash seeds?

Cases come from the bench itself: the 6 netbox samples plus the 50 generated
`-test` tasks (Python, JavaScript, Go, Ruby; PostgreSQL and ClickHouse). Two
variants per task:

    noscope   the instruction minus the paragraphs that name the file and the
              check commands -- everything else the task says is kept
    opening   the first paragraph only -- the hardest, symptom-only reading

Metrics: top-1 / top-3 / top-6 hit rate, mean reciprocal rank, and whether the
target file's hot lines fall inside the target symbol (slice quality). Top-3 is
the number that matters: ranks 4-6 reach the model as 40-line stubs.
"""
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agent, selftest

BENCH = Path("/home/ajh/Documents/ridges-bench/db-engineering")
NETBOX = set(selftest.EXPECTED)
CODE_SUFFIXES = (".py", ".js", ".ts", ".go", ".rb", ".sql")

SCOPE_PARAGRAPH = re.compile(
    r"Limit production changes|Work in `/app`|Run these checks|Run `|before finishing",
    re.IGNORECASE)


def paragraphs(text: str) -> list[str]:
    body = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def variant_texts(text: str) -> dict[str, str]:
    parts = paragraphs(text)
    heading = parts[0] if parts and parts[0].startswith("#") else ""
    prose = [p for p in parts if not p.startswith("#")]
    noscope = [p for p in prose if not SCOPE_PARAGRAPH.search(p)]
    return {
        "noscope": "\n\n".join([heading] + noscope).strip(),
        "opening": "\n\n".join([heading] + prose[:1]).strip(),
        # The prose mentions the file in passing, without a scope sentence. Not
        # a permission, so the ranker must still earn rank 1 -- but a mention
        # is strong evidence and losing it would be a regression.
        "mentioned": "\n\n".join([heading] + prose[:1] + ["The code in question lives in `{target}`."]).strip(),
    }


def target_from_solution(task: Path, root: Path) -> str | None:
    script = task / "solution" / "solve.sh"
    if not script.is_file():
        return None
    for token in re.findall(r"['\"]([\w./-]+\.\w{1,5})['\"]", script.read_text()):
        if token.endswith(CODE_SUFFIXES) and (root / token).is_file():
            return token
    return None


def target_symbol(text: str, parsed: agent.Instruction) -> str | None:
    if parsed.method_hint:
        return parsed.method_hint
    match = re.search(r"specifically `([\w.]+)`", text)
    return match.group(1).split(".")[-1] if match else None


def symbol_range(repo: agent.Repository, relative: str, symbol: str) -> tuple[int, int] | None:
    text = repo.read(relative) or ""
    defs = agent.python_definitions(text) if relative.endswith(".py") else agent.generic_blocks(text)
    for name, start, end, kind in defs:
        if kind != "class" and (name.split(".")[-1] == symbol or re.search(rf"\b{re.escape(symbol)}\b", name)):
            return start, end
    # Go / Ruby / JS without a brace on the def line: a window around the def
    for number, line in enumerate(text.splitlines(), start=1):
        if re.search(rf"\b(func|def|function|const|async function)\b.*\b{re.escape(symbol)}\b", line):
            return number, number + 40
    return None


def build_cases(pristine: Path, staging: Path, only: str | None) -> list[dict]:
    cases = []
    for task in sorted(p for p in BENCH.iterdir() if p.is_dir()):
        if task.name in NETBOX:
            if only == "test":
                continue
            root = selftest.stage(pristine, task, staging / task.name)
        elif task.name.endswith("-test"):
            if only == "netbox":
                continue
            root = task / "environment" / "app"
        else:
            continue
        text = (task / "instruction.md").read_text()
        full = agent.parse_instruction(text, root)
        target = next((p for p in full.edit_only + full.lint_paths if (root / p).is_file()), None) \
            or target_from_solution(task, root)
        if target is None:
            print(f"skip {task.name}: no target found", file=sys.stderr)
            continue
        variants = {k: v.replace("{target}", target) for k, v in variant_texts(text).items()}
        cases.append({"task": task.name, "root": root, "target": target,
                      "symbol": target_symbol(text, full), "variants": variants})
    return cases


def evaluate(cases: list[dict], verbose: bool) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not verbose:
        agent.log = lambda message: None
    for case in cases:
        repo = agent.Repository(case["root"])
        for variant, text in case["variants"].items():
            ins = agent.parse_instruction(text, case["root"])
            ins.edit_only, ins.lint_paths = [], []             # force the unnamed situation
            ranked = agent.locate_targets(repo, ins, limit=10)
            paths = [p for p, _ in ranked]
            rank = paths.index(case["target"]) + 1 if case["target"] in paths else None
            hot_ok = None
            if rank and case["symbol"]:
                span = symbol_range(repo, case["target"], case["symbol"])
                hot = dict(ranked)[case["target"]]
                hot_ok = bool(span and any(span[0] <= h <= span[1] for h in hot)) if hot else False
            results[f"{case['task']}::{variant}"] = {
                "rank": rank, "hot_in_symbol": hot_ok, "top": paths[:6],
                "named_paths": ins.named_paths, "target": case["target"]}
    return results


def summarise(results: dict[str, dict]) -> None:
    for variant in ("noscope", "opening", "mentioned"):
        rows = {k: v for k, v in results.items() if k.endswith("::" + variant)}
        n = len(rows) or 1
        ranks = [v["rank"] for v in rows.values()]
        hits = {k: sum(1 for r in ranks if r and r <= k) for k in (1, 3, 6)}
        mrr = sum(1 / r for r in ranks if r) / n
        hot = [v["hot_in_symbol"] for v in rows.values() if v["hot_in_symbol"] is not None]
        hot_rate = (sum(hot) / len(hot)) if hot else float("nan")
        print(f"{variant:8} n={n:3}  top1={hits[1]/n:.2f}  top3={hits[3]/n:.2f}  "
              f"top6={hits[6]/n:.2f}  mrr={mrr:.2f}  hot_in_symbol={hot_rate:.2f} "
              f"(of {len(hot)} ranked)")


def compare(results: dict[str, dict], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text())
    better, worse = [], []
    for key, now in results.items():
        before = baseline.get(key)
        if not before:
            continue
        a, b = before["rank"] or 99, now["rank"] or 99
        if b < a:
            better.append(f"  + {key}: {before['rank']} -> {now['rank']}")
        elif b > a:
            worse.append(f"  - {key}: {before['rank']} -> {now['rank']}")
    print(f"vs baseline: {len(better)} improved, {len(worse)} regressed")
    for line in better + worse:
        print(line)


def seed_check(argv: list[str]) -> int:
    outputs = []
    for seed in ("1", "2"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            out = handle.name
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run([sys.executable, __file__, *argv, "--json", out, "--quiet"],
                       env=env, check=True)
        outputs.append(json.loads(Path(out).read_text()))
    diffs = [k for k in outputs[0] if outputs[0][k]["top"] != outputs[1].get(k, {}).get("top")]
    print(f"seed check: {len(diffs)} case(s) differ between PYTHONHASHSEED=1 and 2")
    for key in diffs:
        print(f"  {key}\n    seed1: {outputs[0][key]['top']}\n    seed2: {outputs[1][key]['top']}")
    return 1 if diffs else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pristine", nargs="?",
                        help="pinned checkout (default: $RIDGES_NETBOX or the local cache)")
    parser.add_argument("--only", choices=["netbox", "test"])
    parser.add_argument("--json", help="dump per-case results here")
    parser.add_argument("--baseline", help="compare against a previous --json dump")
    parser.add_argument("--seed-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="metrics only, no per-case rows")
    args = parser.parse_args()
    if args.seed_check:
        extra = ([str(args.pristine)] if args.pristine else []) \
            + (["--only", args.only] if args.only else [])
        return seed_check(extra)

    with tempfile.TemporaryDirectory() as tmp:
        pristine = selftest.default_checkout(args.pristine)
        if pristine is None and args.only != "test":
            parser.error(f"no pinned checkout: pass one, set $RIDGES_NETBOX, or place it at "
                         f"{selftest.CHECKOUT_CACHE} (or use --only test)")
        cases = build_cases(pristine, Path(tmp), args.only)
        results = evaluate(cases, args.verbose)
    if not args.quiet:
        for key, row in results.items():
            flag = "" if row["rank"] and row["rank"] <= 3 else "  <-- miss"
            print(f"{key:70} rank={row['rank']!s:5} hot={row['hot_in_symbol']!s:5} "
                  f"top3={row['top'][:3]}{flag}")
    summarise(results)
    if args.baseline:
        compare(results, Path(args.baseline))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
