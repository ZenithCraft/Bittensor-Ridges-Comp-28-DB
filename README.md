# Ridges DB-engineering miner agent

A single-file miner agent (`agent.py`) for the Subnet 62 database query
engineering round. Entry point, per `ridges_harbor/ridges_miner_runtime.py`:

```python
def agent_main(input: dict) -> str   # {"problem_statement": "<markdown>"} -> unified diff
```

## What the environment actually is

These facts came out of the Ridges repo and the six sample tasks, and several
of them contradict assumptions that are easy to make:

| | |
|---|---|
| Inference | Transparent MITM proxy over `openrouter.ai`, keyed by `OPENROUTER_API_KEY`. Older runtimes expose `SANDBOX_PROXY_URL` + `/api/inference`; `run-local` sets `RIDGES_INFERENCE_*`. `agent.py` supports all three. |
| Models | OpenRouter slugs from the gateway allowlist: `qwen/qwen3-coder-next`, `z-ai/glm-4.7`, `z-ai/glm-5`, `moonshotai/kimi-k2.5`, `minimax/minimax-m2.5`, `qwen/qwen3.5-397b-a17b`, `deepseek/deepseek-r1-0528`. **No OpenAI or Anthropic models are available.** |
| Budget | `RIDGES_MAX_COST_USD`; live spend at `http://sandbox-proxy:80/api/v1/usage`. |
| Repo | At the task workdir (`/app`). **`.git` is deleted** in every sample image, so patches are built with `difflib`, not `git diff`. |
| Database | Live and reachable from the agent container. Credentials are in the app's own config. |
| Verifier | Runs in a **separate, pristine container**. It copies out only `/logs/agent/patch.diff`, `git apply`s it, and re-runs the tests. |

That last row is the leverage: the agent can run the test suite, `EXPLAIN`, and
schema queries in its own container without any of it reaching the verifier —
as long as the emitted diff is minimal. Sample verifiers hash every file they
did not authorise, ban imports and constructs inside the target method, and cap
its AST size, so a diff that strays anywhere else scores zero.

## Design

Deterministic work first; inference only where judgement is required.

1. **Parse the instruction** (no LLM). Problem kind, engine, editable paths,
   `Class.method()` hint, the fenced check commands, and style rules such as
   "no Python loops, comprehensions, lambdas".
2. **Locate the target.** If the instruction names a file, use it. Otherwise
   rank every file by three independent signals — query-layer density
   normalised for file length, IDF-weighted overlap with the problem prose, and
   symbol-name/path priors. Length normalisation matters: without it a
   5,000-line view module outranks the 67-line manager that holds the bug.
3. **Gather evidence.** Discover the DSN from the app's own config, then pull
   columns and existing indexes for the tables in play.
4. **One strong call** with a tight context: the target method, the file
   outline with its imports, the live schema, and the explicit constraints.
   The model may spend one round asking for more — `read_file`, `grep`, `sql`,
   `explain` — answered deterministically and cheaply.
5. **Verify locally, cheapest first**: scope → syntax → single-method
   confinement → forbidden constructs → ruff → the task's own commands. A
   failure feeds the real output back and retries; the model tier escalates
   only on a *verified* failure, never on a protocol slip or a context round.
6. **Emit a minimal diff** and dry-run `git apply --check` against a pristine
   copy the way the verifier will. The working tree is always reverted.

Typical cost is one or two calls on the cheapest coder model.

## Notes on the original plan

Kept: single-file agent, rules-based classification, model routing with
escalation, token-frugal context, retry-with-feedback.

Changed, with reasons:

- **Model roster.** `gpt-luna-5.6`, `claude-4-sonnet`, `claude-3.5-sonnet` are
  not on the allowlist and Chutes is not the production transport.
- **Eight strategy classes collapsed to one loop.** Six of the eight were
  `pass`. What differs between a bad index, a fan-out aggregate, and an N+1 is
  the *evidence* and the *constraints*, not the control flow — so the taxonomy
  now steers those instead of forking into near-duplicate solvers.
- **`quick_validate` requiring `bulk_create` in the output** was exactly the
  hardcoding the rules warn against; it would also reject a correct fix that
  used a different bulk primitive. Replaced with real verification.
- **`git diff --no-index` into `<file>.fixed`** would have failed on two counts:
  no `.git`, and a scratch file left in the tree that the source manifest hashes.
- **Regex method extraction** replaced with AST for Python and brace matching
  otherwise; regex mis-slices any method containing a nested `def`.
- **Hardcoded `/app/netbox` paths and tag-specific heuristics** replaced with a
  repo-agnostic ranker — the samples come from one repo, the competition will not.
- **Added: live database probing, constraint enforcement, patch dry-run.**

## Tests

```bash
python3 -m unittest test_units test_e2e test_routing   # 132 tests, no inference
python3 selftest.py                      # locate/slice/patch each of the 6 samples
./qualify.py                             # layered pre-upload readiness report
```

Measuring one part at a time, all free of inference:

```bash
./preprocessing.py <task>                # everything before the first model call
./preprocessing.py --all --json now.json # ...across the bench, for diffing
python3 rank_eval.py --quiet             # ranking accuracy under three prose variants
python3 probe_eval.py                    # database discovery on every app
python3 consistency_report.py            # pass rate and cost across repeated live runs
```

The pinned checkout is netbox at revision
`00791344e68213bde942218283dce03cc3941c30`; the harnesses apply each task's own
`baseline_mutation.py` to reproduce its starting state. They find it at
`~/.cache/ridges-db-agent/netbox` or `$RIDGES_NETBOX`, or take a path
positionally.

Against the real graded environment:

```bash
ridges miner run-local --task-path ./db-engineering/<task> --agent-path <this-dir>/agent.py
```
