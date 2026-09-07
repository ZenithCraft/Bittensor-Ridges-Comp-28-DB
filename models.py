#!/usr/bin/env python3
"""Catalogue OpenRouter models and rank them for this agent.

Pulls the full model list, extracts capability/pricing/benchmark data, marks
which are on the Ridges allowlist and which this key can actually reach, and
projects per-task cost from the token profile measured in real runs.

    ./models.py                 ranked shortlist (Ridges allowlist first)
    ./models.py --all           every model, ranked by coding index
    ./models.py --reachable     only models this key can currently call
    ./models.py --probe         additionally live-test reachability
    ./models.py --json out.json dump the full dataset
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import urllib.request
from pathlib import Path

ENV_MINER = Path.home() / ".ridges/.env.miner"
RIDGES_SOURCE = Path("/home/ajh/Documents/ridges/inference_gateway/providers/openrouter.py")

# Token profile measured on pg-netbox-contact-group-counts-001 (real runs):
# ~8.5k prompt, and completion varying by how much the model "thinks".
PROMPT_TOKENS = 8600
COMPLETION_LEAN = 3700       # observed with a terse model
COMPLETION_VERBOSE = 20500   # observed with a reasoning-heavy model

G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"


def key() -> str:
    if not ENV_MINER.exists():
        return ""
    for line in ENV_MINER.read_text().splitlines():
        if line.startswith("RIDGES_OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def get(url: str, auth: bool = False, timeout: int = 30):
    headers = {"Authorization": f"Bearer {key()}"} if auth else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ridges_allowlist() -> set[str]:
    """The competition's allowed slugs, read from the gateway source."""
    if not RIDGES_SOURCE.exists():
        return set()
    return set(re.findall(r'openrouter_name="([^"]+)"', RIDGES_SOURCE.read_text()))


def catalogue() -> list[dict]:
    allow = ridges_allowlist()
    rows = []
    for m in get("https://openrouter.ai/api/v1/models")["data"]:
        pricing = m.get("pricing") or {}
        bench = ((m.get("benchmarks") or {}).get("artificial_analysis") or {})
        params = set(m.get("supported_parameters") or [])
        try:
            prompt_usd = float(pricing.get("prompt") or 0) * 1e6
            completion_usd = float(pricing.get("completion") or 0) * 1e6
        except (TypeError, ValueError):
            prompt_usd = completion_usd = 0.0
        rows.append({
            "id": m["id"],
            "name": m.get("name", ""),
            "context": m.get("context_length") or 0,
            "usd_per_m_in": prompt_usd,
            "usd_per_m_out": completion_usd,
            "cached_in": float(pricing.get("input_cache_read") or 0) * 1e6,
            "tools": "tools" in params,
            "structured": "structured_outputs" in params,
            "reasoning": "reasoning" in params or "include_reasoning" in params,
            "intelligence": bench.get("intelligence_index"),
            "coding": bench.get("coding_index"),
            "agentic": bench.get("agentic_index"),
            "ridges": m["id"] in allow,
            "modality": (m.get("architecture") or {}).get("modality", ""),
        })
    return rows


def cost_per_task(row: dict, completion: int) -> float:
    return (PROMPT_TOKENS / 1e6) * row["usd_per_m_in"] + (completion / 1e6) * row["usd_per_m_out"]


def probe(rows: list[dict]) -> None:
    """Live-test which models this key can actually reach."""
    def test(row):
        body = json.dumps({"model": row["id"],
                           "messages": [{"role": "user", "content": "hi"}],
                           "max_tokens": 1}).encode()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                row["reachable"] = "choices" in json.loads(response.read())
        except Exception as exc:
            row["reachable"] = False
            row["blocked"] = "no allowed providers" if "404" in str(exc) else str(exc)[:40]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(test, rows))


def render(rows: list[dict], title: str) -> None:
    print(f"\n{title}\n" + "=" * 118)
    print(f"{'model':<34}{'ctx':>8}{'$/Min':>8}{'$/Mout':>8}{'code':>6}{'intel':>6}{'agent':>6}"
          f"{'tool':>5}{'lean $':>9}{'verbose $':>10}{'':>3}")
    print("-" * 118)
    for r in rows:
        flags = ""
        if r.get("ridges"):
            flags += f"{G}R{X}"
        if r.get("reachable") is True:
            flags += f"{B}✓{X}"
        elif r.get("reachable") is False:
            flags += f"{R}✗{X}"
        num = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) else "-"
        print(f"{r['id'][:32]:<34}{r['context']//1000:>7}k{r['usd_per_m_in']:>8.2f}"
              f"{r['usd_per_m_out']:>8.2f}{num(r['coding']):>6}{num(r['intelligence']):>6}"
              f"{num(r['agentic']):>6}{'yes' if r['tools'] else '-':>5}"
              f"{cost_per_task(r, COMPLETION_LEAN):>9.4f}{cost_per_task(r, COMPLETION_VERBOSE):>10.4f}"
              f"  {flags}")
    print("-" * 118)
    print(f"{D}lean $ = {PROMPT_TOKENS} prompt + {COMPLETION_LEAN} completion tokens; "
          f"verbose $ = + {COMPLETION_VERBOSE} completion (both measured in real runs){X}")
    print(f"{D}flags: {G}R{X}{D} = on the Ridges allowlist, {B}✓{X}{D} reachable with your key, "
          f"{R}✗{X}{D} blocked{X}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reachable", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--json")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    rows = catalogue()
    print(f"fetched {len(rows)} models from OpenRouter; "
          f"{sum(r['ridges'] for r in rows)} are on the Ridges allowlist")

    if args.probe or args.reachable:
        targets = rows if args.all else [r for r in rows if r["ridges"] or (r["coding"] or 0) >= 30]
        print(f"probing {len(targets)} models for reachability with your key...")
        probe(targets)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")

    if args.reachable:
        rows = [r for r in rows if r.get("reachable")]

    ridges = sorted([r for r in rows if r["ridges"]], key=lambda r: -(r["coding"] or 0))
    if ridges:
        render(ridges, "RIDGES ALLOWLIST — the only models valid in competition")

    if args.all or args.reachable:
        rest = sorted([r for r in rows if not r["ridges"] and r["coding"] is not None],
                      key=lambda r: -(r["coding"] or 0))[:args.limit]
        render(rest, f"TOP {len(rest)} BY CODING INDEX (not competition-valid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
