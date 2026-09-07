"""Deterministic problem profiling, model-capability matching and initial routing.

NOT part of the uploaded agent. This subsystem was built, measured, and then
taken back out of `agent.py` on 2026-09-07 because it was not being applied:
`ROUTING_MODE` defaulted to `baseline`, so the profile and the model decision
were computed and logged on every task and then discarded. Two measurements
argued against switching it on:

  * a four-task A/B gave the router 3 of 4 against the baseline's 4 of 4;
  * across all 56 local trial tasks the capability selector disagrees with the
    baseline first model on 56 of 56, which would discard the largest sample
    in hand (35 tasks, 33 first-attempt solves, $0.030 mean) on the strength
    of priors written by hand rather than measured.

Production logs are not visible to a miner during or after an evaluation, so
leaving it running in the uploaded agent could not have produced the data
needed to settle the question either. It lives here instead, importable by the
local harnesses, so the work survives and can be reinstated when competition
results give a reason.

To reinstate: import `profile_task` and `select_initial_route` in
`agent_main`, restore `Solver.route` plus the `failures == 0` branch of
`Solver.tier_for`, and add back the `context_bonus` line in `Solver.solve`.

    python3 profile_eval.py <pinned-netbox-checkout>
"""

import os
import re
from dataclasses import dataclass
from typing import Sequence

import agent
from agent import (DatabaseProbe, Instruction, LADDER, LLM, MAX_GATE_SLIPS, MAX_REPAIR_ROUNDS,
                   MODELS, ModelSpec, Repository, Solver, log, query_density)

#
# Answers four questions before the first model call, from data the parser,
# ranker, tracer and probe already produced:
#   1. what kind of problem is this?          -> TaskProfile.task_types
#   2. where is the complexity?               -> four risk dimensions
#   3. how risky is a weak first attempt?     -> overall_risk, tier
#   4. what is the cheapest safe first route? -> select_initial_route()
#
# It chooses the FIRST rung only. The verifier decides whether that rung was
# enough, and the existing ladder (L2 slips, L3 verified failures, stalls,
# reasoning runaway) decides what happens next, unchanged.
#
# Uncertainty raises risk. A false-EASY costs a failed attempt, a test run and
# a repair round; a false-HARD costs a few cents. Under unanimity scoring the
# first is far more expensive, so every unknown pushes upward.
#
# Nothing here runs during a graded run: `agent.py` does not import this module.

ROUTING_MODE = (os.getenv("RIDGES_DB_AGENT_ROUTING") or "baseline").strip().lower()

# Each dimension is scored 0..10 from additive signals; weights are the points
# each signal contributes. Tuned on the 56 bench tasks (profile_eval.py), not
# on intuition -- change them there, with the harness open.
RISK_WEIGHTS: dict[str, float] = {
    # structural: how much code surface the change touches or must be found in
    "unnamed_target": 2.0, "many_candidates": 1.0, "cross_module": 2.0,
    "large_target_file": 1.0, "huge_target_file": 1.0, "large_repo": 1.0, "huge_repo": 1.0,
    "multiple_target_files": 2.0, "migration": 1.0,
    # semantic: how hard the requested behaviour is to pin down
    "ambiguous_ranking": 2.0, "authoring": 2.0, "no_task_kind": 3.0, "no_method_named": 1.0,
    "few_identifiers": 1.0, "long_instruction": 1.0, "semantic_terms": 1.0,   # per term, capped
    # query / database: how much database semantics the fix depends on
    "bounded_queries": 3.0, "index_or_plan": 2.0, "clickhouse": 2.0, "raw_sql": 1.0,
    "orm_layer": 1.0, "numeric_target": 2.0, "query_terms": 1.0,             # per term, capped
    "query_dense_target": 1.0,
    # correctness: how sharply the hidden tests will judge edge cases
    "single_method": 1.0, "style_constraints": 1.0, "many_forbidden": 1.0,
    "result_correctness": 2.0, "edge_case_terms": 1.0,                      # per term, capped
    "hidden_data_wording": 2.0,
}
RISK_CAP_PER_DIMENSION = 10.0
TERM_CAP = 3.0                       # a family of terms contributes at most this much
UNCERTAINTY_WEIGHT = 0.5             # overall = max(dimensions) + weight x uncertainty
RISK_THRESHOLDS: dict[str, float] = {"LOW": 3.0, "MEDIUM": 6.0, "HIGH": 9.0}   # VERY_HIGH above

# Vocabulary. Each term is one signal among several; none decides a tier alone.
_SEMANTIC_TERMS = (r"\btie[sd]?\b", r"\blatest\b", r"\bcurrent state\b", r"\bgrain\b",
                   r"\bdedup", r"\bsession", r"\bcohort", r"\bwindow\b", r"\brecursive\b",
                   r"\bhierarch", r"\bancestor", r"\bdescendant")
_QUERY_TERMS = (r"\bjoin\b", r"\baggregat", r"\bannotat", r"\bsubquer", r"\bwindow function",
                r"\bpartition", r"\bargMax\b", r"\bLIMIT BY\b", r"\bmaterializ", r"\bindex(es)?\b",
                r"\btransaction", r"\bpaginat", r"\bkeyset\b", r"\bN\+1\b", r"\bround[- ]trips?\b",
                r"\bquery count\b", r"\bread_rows\b", r"\bgranule", r"\bprun")
_EDGE_TERMS = (r"\bNULL\b", r"\bempty (groups?|days?|weeks?|result)", r"\bzero[- ]fill", r"\bduplicate",
               r"\binclusive\b", r"\bboundar", r"\bedge cases?\b", r"\bexactly\b", r"\border(ed|ing)? by\b",
               r"\bdistinct\b", r"\bregression")
_HIDDEN_DATA = (r"data (the task|you) (do(es)? )?not (show|see)", r"\bhidden\b", r"not enough to match",
                r"generalis|generaliz", r"unseen")

TASK_TYPE_RULES: dict[str, str] = {
    # label -> the signal(s) that assert it; multi-label by design
    "query_optimization": "bounded_queries|index_or_plan|numeric_target",
    "orm_change": "orm_layer",
    "raw_sql_change": "raw_sql",
    "clickhouse": "clickhouse",
    "schema_migration": "migration",
    "authoring": "authoring",
    "correctness_sensitive": "result_correctness|hidden_data_wording|edge_case_terms",
    "constraint_heavy": "single_method|style_constraints|many_forbidden",
    "cross_module": "cross_module|unnamed_target",
}


@dataclass
class InitialRoute:
    tier: str
    models: list[str]                # first rung of the ladder for this task
    reasoning: str                   # reasoning effort sent with every call
    context_bonus: int               # extra investigation rounds before the first edit
    rationale: str
    decision: "ModelDecision | None" = None   # set when capability matching chose the model


# Which rung to start from, per tier. Models come from the existing roster.
#
# The asymmetry here is deliberate and follows the evidence:
#
#  * Routing a HARD task to a stronger start is supported. On the hardest
#    measured task deepseek spun at the edit turn in 4 of 5 runs while
#    qwen3.8 with reasoning solved it; and the HIGH tier's live mean cost
#    ($0.104) is four times the LOW and MEDIUM tiers', so the tier does
#    isolate the expensive tail. The one A/B data point for the VERY_HIGH
#    route (qwen first) halved that task's cost, $0.031 -> $0.015.
#
#  * Routing an EASY task to a cheaper model is NOT supported. No measurement
#    exists of minimax-m3 as the first attempt on these tasks, and two of the
#    seven LOW-tier tasks with live data needed two and three attempts. Under
#    unanimity scoring a wrong cheap start costs far more than it saves, so
#    LOW starts on the same rung as MEDIUM until an A/B says otherwise.
#    Set RIDGES_DB_AGENT_CHEAP_LOW=1 to measure the cheap variant.
#
# Reasoning effort stays "low" everywhere: higher efforts were never measured,
# and the runaway that costs ~$0.04 a call grows with effort rather than
# shrinking. Model choice and reasoning choice remain separate settings.
_CHEAP_LOW = (os.getenv("RIDGES_DB_AGENT_CHEAP_LOW") or "").strip().lower() in ("1", "true", "yes")

ROUTING_POLICY: dict[str, dict] = {
    "LOW":       {"models": (["minimax/minimax-m3", "deepseek/deepseek-v4-pro-0813", "qwen/qwen3.8-27b"]
                             if _CHEAP_LOW else list(LADDER[0])),
                  "reasoning": "low", "context_bonus": 0},
    "MEDIUM":    {"models": list(LADDER[0]), "reasoning": "low", "context_bonus": 0},
    # HIGH started with an extra investigation round; the first A/B (n=1 per
    # task) showed that round costing $0.04 and $0.02 on the two HIGH tasks
    # without changing the outcome, so it is off until a larger run earns it.
    "HIGH":      {"models": list(LADDER[0]), "reasoning": "low", "context_bonus": 0},
    "VERY_HIGH": {"models": ["qwen/qwen3.8-27b", "deepseek/deepseek-v4-pro-0813", "minimax/minimax-m3"],
                  "reasoning": "low", "context_bonus": 1},
}


@dataclass
class TaskProfile:
    task_types: list[str]
    repo_files: int
    relevant_files: int
    target_files: int
    target_symbols: int
    target_lines: int
    dependency_depth: int            # traced files the ranker did not already show
    constraint_count: int
    test_signal: int                 # check commands the instruction names
    structural_risk: float
    semantic_risk: float
    query_risk: float
    correctness_risk: float
    uncertainty: float
    overall_risk: float
    tier: str
    reasons: list[str]
    uncertainty_reasons: list[str]

    def explain(self, route: "InitialRoute", applied: bool) -> str:
        lines = ["Task risk profile",
                 f"  types: {', '.join(self.task_types) or 'unknown'}",
                 f"  structural {self.structural_risk:.1f} | semantic {self.semantic_risk:.1f} | "
                 f"query {self.query_risk:.1f} | correctness {self.correctness_risk:.1f} | "
                 f"uncertainty {self.uncertainty:.1f}",
                 f"  overall {self.overall_risk:.1f} -> {self.tier}",
                 "  reasons: " + ("; ".join(self.reasons) or "none"),
                 "  uncertainty: " + ("; ".join(self.uncertainty_reasons) or "none"),
                 f"  initial route ({'applied' if applied else 'logged only, baseline routing in effect'}): "
                 f"{route.models[0]} first, reasoning {route.reasoning}, +{route.context_bonus} context round(s)"]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("reasons", "uncertainty_reasons")}


def _tier_for(risk: float) -> str:
    for tier, ceiling in RISK_THRESHOLDS.items():
        if risk <= ceiling:
            return tier
    return "VERY_HIGH"


def profile_task(instruction: Instruction, repo: Repository,
                 candidates: Sequence[tuple[str, list[int]]], probe: "DatabaseProbe | None") -> TaskProfile:
    """Score the task on four risk dimensions from existing deterministic data."""
    W = RISK_WEIGHTS
    text = instruction.text
    lowered = text.lower()
    reasons: list[str] = []
    unc_reasons: list[str] = []
    fired: set[str] = set()

    def hit(signal: str, reason: str) -> float:
        fired.add(signal)
        reasons.append(reason)
        return W[signal]

    def count_terms(patterns: Sequence[str]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))

    named = bool(instruction.edit_only or instruction.lint_paths)
    target_files = instruction.edit_only or instruction.lint_paths or [p for p, _ in candidates[:1]]
    target_lines = sum(len((repo.read(p) or "").splitlines()) for p in target_files)
    kinds = set(instruction.kinds)

    # -- structural --------------------------------------------------------
    structural = 0.0
    if not named:
        structural += hit("unnamed_target", "instruction names no file; the target is ranked, not given")
        if len(candidates) >= 4:
            structural += hit("many_candidates", f"{len(candidates)} candidate files reach the model")
    if instruction.traced_hint:
        structural += hit("cross_module", f"call graph reaches {len(instruction.traced_hint)} file(s) outside the shortlist")
    if target_lines > 400:
        structural += hit("large_target_file", f"target spans {target_lines} lines")
    if target_lines > 1500:
        structural += hit("huge_target_file", "target file over 1500 lines")
    if len(repo.files) > 1000:
        structural += hit("large_repo", f"{len(repo.files)} indexed files")
    if len(repo.files) > 5000:
        structural += hit("huge_repo", "repository over 5000 files")
    if len(target_files) > 1:
        structural += hit("multiple_target_files", f"{len(target_files)} files may change")
    if "migration" in kinds:
        structural += hit("migration", "schema or migration change")

    # -- semantic ----------------------------------------------------------
    semantic = 0.0
    scores = instruction.candidate_scores
    if not named and len(scores) >= 2 and scores[1] > 0 and scores[0] / scores[1] < 1.3:
        semantic += hit("ambiguous_ranking", f"top two candidates score within 30% ({scores[0]:.0f} vs {scores[1]:.0f})")
    if "authoring" in kinds:
        semantic += hit("authoring", "the query must be written, not corrected")
    if not kinds:
        semantic += hit("no_task_kind", "no known task shape matched the instruction")
    if named and not instruction.method_hint and not instruction.lint_paths:
        semantic += hit("no_method_named", "file named but no method named")
    if len(instruction.identifiers) < 3:
        semantic += hit("few_identifiers", "fewer than three identifiers to search for")
    if len(text) > 2500:
        semantic += hit("long_instruction", f"instruction is {len(text)} characters")
    n = count_terms(_SEMANTIC_TERMS)
    if n:
        semantic += min(TERM_CAP, n * W["semantic_terms"]); fired.add("semantic_terms")
        reasons.append(f"{n} semantics term(s): ties, latest, hierarchy, sessions, windows")

    # -- query / database ------------------------------------------------
    query = 0.0
    if "bounded_queries" in kinds:
        query += hit("bounded_queries", "work must not grow with input (bounded queries)")
    if "index_or_plan" in kinds:
        query += hit("index_or_plan", "plan or index behaviour is graded")
    if "clickhouse" in kinds or instruction.engine == "clickhouse":
        query += hit("clickhouse", "ClickHouse: graded on statements and rows read")
    if "raw_sql" in kinds:
        query += hit("raw_sql", "raw SQL in play")
    if "orm_layer" in kinds:
        query += hit("orm_layer", "ORM or query-builder layer")
    if instruction.targets.get("max_queries") or instruction.targets.get("max_read_rows"):
        query += hit("numeric_target", f"explicit numeric target {instruction.targets}")
    n = count_terms(_QUERY_TERMS)
    if n:
        query += min(TERM_CAP, n * W["query_terms"]); fired.add("query_terms")
        reasons.append(f"{n} query term(s): joins, aggregation, windows, indexes, pagination")
    # Reuses the ranker's own density signal; no second call graph is built.
    if any(query_density(repo.read(p) or "") > 1.0 for p in target_files):
        query += hit("query_dense_target", "target file is dense with query code")

    # -- correctness -----------------------------------------------------
    correctness = 0.0
    if instruction.single_method:
        correctness += hit("single_method", "change bounded to one method, file otherwise byte-identical")
    if instruction.style_constraints:
        correctness += hit("style_constraints", f"forbidden constructs: {', '.join(instruction.style_constraints)}")
    if len(instruction.forbidden) >= 2:
        correctness += hit("many_forbidden", f"{len(instruction.forbidden)} do-not-change clauses")
    if "result_correctness" in kinds:
        correctness += hit("result_correctness", "wrong results are the symptom")
    n = count_terms(_EDGE_TERMS)
    if n:
        correctness += min(TERM_CAP, n * W["edge_case_terms"]); fired.add("edge_case_terms")
        reasons.append(f"{n} edge-case term(s): NULL, empty groups, duplicates, boundaries, order")
    if count_terms(_HIDDEN_DATA):
        correctness += hit("hidden_data_wording", "instruction warns that grading uses unseen data")

    # -- uncertainty -----------------------------------------------------
    uncertainty = 0.0
    if not instruction.commands:
        uncertainty += 2.0; unc_reasons.append("no check command named: the first edit cannot be verified")
    if not named:
        uncertainty += 1.0; unc_reasons.append("target must be inferred")
    if not kinds:
        uncertainty += 2.0; unc_reasons.append("task shape unknown")
    if instruction.engine == "unknown" and not (probe and probe.available()):
        uncertainty += 2.0; unc_reasons.append("engine unknown and no live database")
    if len(candidates) < 1:
        uncertainty += 2.0; unc_reasons.append("no candidate file at all")

    dims = [min(RISK_CAP_PER_DIMENSION, d) for d in (structural, semantic, query, correctness)]
    overall = min(12.0, max(dims) + UNCERTAINTY_WEIGHT * uncertainty)
    task_types = [label for label, signals in TASK_TYPE_RULES.items()
                  if any(sig in fired for sig in signals.split("|"))]
    if not task_types:
        task_types = ["local_code_change" if named else "unknown"]

    return TaskProfile(
        task_types=task_types, repo_files=len(repo.files), relevant_files=len(candidates),
        target_files=len(target_files), target_symbols=int(bool(instruction.method_hint)),
        target_lines=target_lines, dependency_depth=len(instruction.traced_hint),
        constraint_count=len(instruction.forbidden) + len(instruction.style_constraints) + int(instruction.single_method),
        test_signal=len(instruction.commands),
        structural_risk=dims[0], semantic_risk=dims[1], query_risk=dims[2], correctness_risk=dims[3],
        uncertainty=uncertainty, overall_risk=overall, tier=_tier_for(overall),
        reasons=reasons, uncertainty_reasons=unc_reasons)


# ---------------------------------------------------------------------------
# Model capability profile and initial model selection
# ---------------------------------------------------------------------------
#
# The problem profiler above says what kind of problem this is. This section
# says what each model is good at, and matches the two. It chooses the FIRST
# rung only -- LADDER still owns recovery, the verifier still owns truth.
#
# The scores below are ROUTING PRIORS, not success probabilities. Nothing here
# is calibrated against contest outcomes: the official problems are hidden and
# no run of them exists. Two kinds of evidence went in, and they are labelled
# per model in `source_notes`:
#
#   measured   this agent's own local runs on the public bench (56 tasks).
#              Reliability especially: deepseek-v4-pro spent its whole
#              completion cap on reasoning and returned nothing at the edit
#              turn in 4 of 5 runs on the hardest task, while qwen3.8 with
#              reasoning solved it. That is a real reliability signal.
#   published  price and context from the provider; capability rank from
#              public coding/agentic indices at the time of writing.
#
# Treat every number as an ordering, not a measurement. Two models 0.02 apart
# are indistinguishable; 0.2 apart is a real difference.


@dataclass(frozen=True)
class ModelCapabilityProfile:
    """A static, deterministic prior for one model. No network, no lookups."""

    model_name: str
    role: str                       # why this model is in the pool at all
    coding: float                   # writing correct code in an unfamiliar repo
    debugging: float                # reading a failure and repairing from it
    sql: float                      # SQL and engine semantics: joins, NULLs, ties
    repository: float               # holding a large slice of a repo in mind
    reasoning: float                # multi-step derivation before answering
    agentic: float                  # following a strict tool/response protocol
    reliability: float              # answers at all, in the required shape
    confidence: float               # how much the row above should be trusted
    source_notes: str

    @property
    def spec(self) -> ModelSpec | None:
        return MODELS.get(self.model_name)

    @property
    def context_length(self) -> int:
        spec = self.spec
        return spec.context if spec else 0

    @property
    def blended_price(self) -> float:
        """One number for cost comparison: this agent's runs are roughly two
        parts prompt to one part completion, so weight them that way."""
        spec = self.spec
        if spec is None:
            return 4.0
        return (2.0 * spec.usd_per_m_in + spec.usd_per_m_out) / 3.0


# The pool is exactly the models already in MODELS -- this feature adds none.
# Six roles plus one cheap fallback; every row says why it is here.
MODEL_CAPABILITIES: dict[str, ModelCapabilityProfile] = {
    "deepseek/deepseek-v4-pro-0813": ModelCapabilityProfile(
        model_name="deepseek/deepseek-v4-pro-0813", role="strong coding workhorse",
        coding=0.90, debugging=0.85, sql=0.82, repository=0.88, reasoning=0.88,
        agentic=0.85, reliability=0.72, confidence=0.85,
        source_notes="measured: solved the large majority of local trial tasks on the first "
                     "attempt, at the lowest cost per solved task among the capable models. "
                     "Spends its whole completion cap on reasoning at the edit turn in about "
                     "half of Django-shaped runs (~$0.04 each); the inference loop already "
                     "hands those to the next family, so reliability is docked once only."),
    "qwen/qwen3.8-27b": ModelCapabilityProfile(
        model_name="qwen/qwen3.8-27b", role="strong alternative family, database semantics",
        coding=0.84, debugging=0.90, sql=0.90, repository=0.80, reasoning=0.86,
        agentic=0.84, reliability=0.86, confidence=0.65,
        source_notes="measured, but on a biased sample: nearly every observation is a recovery "
                     "turn, where it had the failure output in front of it. It repeatedly "
                     "solved the hardest local trial task, and several ClickHouse repairs, "
                     "after the first model produced nothing -- so its edge is credited to "
                     "SQL semantics and debugging, and confidence is held low because it has "
                     "seldom gone first."),
    "moonshotai/kimi-k2.6": ModelCapabilityProfile(
        model_name="moonshotai/kimi-k2.6", role="frontier reliability, expensive",
        coding=0.84, debugging=0.82, sql=0.80, repository=0.80, reasoning=0.82,
        agentic=0.72, reliability=0.80, confidence=0.55,
        source_notes="published index plus light local use at ladder tier 2. The dearest output "
                     "price in the pool, so it earns its place only on hard tasks."),
    "moonshotai/kimi-k2.7-code": ModelCapabilityProfile(
        model_name="moonshotai/kimi-k2.7-code", role="coding specialist",
        coding=0.85, debugging=0.80, sql=0.74, repository=0.76, reasoning=0.76,
        agentic=0.72, reliability=0.78, confidence=0.50,
        source_notes="published: the coding-tuned sibling of the row above, and cheaper. Barely "
                     "exercised locally, so confidence is low and the selector discounts it."),
    "minimax/minimax-m3": ModelCapabilityProfile(
        model_name="minimax/minimax-m3", role="cheap capable, very large context",
        coding=0.74, debugging=0.72, sql=0.70, repository=0.80, reasoning=0.74,
        agentic=0.74, reliability=0.82, confidence=0.60,
        source_notes="measured: answered correctly on its first attempt after two stronger "
                     "families had produced nothing. Cheapest output price among the capable "
                     "models, with a very large context, so it suits big repositories cheaply."),
    "qwen/qwen3.6-35b-a3b": ModelCapabilityProfile(
        model_name="qwen/qwen3.6-35b-a3b", role="cheap fallback",
        coding=0.55, debugging=0.52, sql=0.50, repository=0.52, reasoning=0.52,
        agentic=0.55, reliability=0.70, confidence=0.45,
        source_notes="published: well below the pool on coding. Kept as a last resort for when "
                     "the remaining budget cannot cover anything better."),
    "google/gemma-4-31b-it": ModelCapabilityProfile(
        model_name="google/gemma-4-31b-it", role="cheapest fallback",
        coding=0.52, debugging=0.48, sql=0.45, repository=0.48, reasoning=0.48,
        agentic=0.50, reliability=0.68, confidence=0.45,
        source_notes="published: cheapest in the pool. Produced a usable patch once, after five "
                     "stronger families had each produced nothing. The floor of the ladder, "
                     "never a first choice."),
}

# Routing configuration. Every coefficient is here, not scattered in the code.
MODEL_SELECTION_ENABLED = True           # capability matching inside the deterministic arm
MODEL_SELECTION_COST_WEIGHT = 0.05       # tie-break only; never part of the score itself
MODEL_SELECTION_RELIABILITY_WEIGHT = 0.30
MODEL_SELECTION_CONTEXT_WEIGHT = 0.20    # penalty when the task crowds the context window
MODEL_SELECTION_UNCERTAINTY_PENALTY = 0.04   # per uncertainty point, against weak models
MODEL_SELECTION_CONFIDENCE_WEIGHT = 0.10     # a low-confidence prior is pulled toward the mean
MODEL_SELECTION_MARGIN = 0.05            # below this, two models count as equal and cost decides
CONTEXT_SAFETY = 3.0                     # the conversation grows over repair rounds
PRIOR_MEAN = 0.72                        # what an unknown model is assumed to be worth

# A weak model must not be handed a task whose risk says it will fail. These
# floors are the conservative half of the policy: they cost a few cents and
# they protect the score, which is what the competition actually ranks.
CAPABILITY_FLOOR_BY_TIER: dict[str, float] = {
    "LOW": 0.0, "MEDIUM": 0.62, "HIGH": 0.74, "VERY_HIGH": 0.80}

# How much each capability dimension matters, per risk dimension of the task.
# Read a row as: "when THIS risk is high, these model abilities matter".
DIMENSION_WEIGHTS: dict[str, dict[str, float]] = {
    "base":        {"coding": 0.35, "debugging": 0.15, "sql": 0.20, "repository": 0.10,
                    "reasoning": 0.10, "agentic": 0.10},
    "query":       {"sql": 0.45, "reasoning": 0.25, "coding": 0.20, "debugging": 0.10},
    "structural":  {"repository": 0.40, "coding": 0.25, "reasoning": 0.20, "agentic": 0.15},
    "semantic":    {"reasoning": 0.40, "sql": 0.25, "coding": 0.20, "debugging": 0.15},
    "correctness": {"debugging": 0.30, "sql": 0.30, "reasoning": 0.25, "coding": 0.15},
}


@dataclass
class ModelDecision:
    """Why one model was chosen. Recorded in telemetry, never sent to a model."""

    model: str
    reasoning_level: str
    score: float
    capability_score: float
    cost_penalty: float
    reliability_score: float
    context_penalty: float
    confidence: float
    rationale: str
    ranked: list[tuple[str, float]]      # every candidate, best first

    def explain(self) -> str:
        runners = ", ".join(f"{m.split('/')[-1]} {v:+.2f}" for m, v in self.ranked[:4])
        return (f"initial model: {self.model} (score {self.score:.2f} = capability "
                f"{self.capability_score:.2f} + reliability {self.reliability_score:.2f} "
                f"- cost {self.cost_penalty:.2f} - context {self.context_penalty:.2f})\n"
                f"  because: {self.rationale}\n  ranked: {runners}")


def _risk_mix(profile: TaskProfile) -> dict[str, float]:
    """Blend the dimension weight tables in proportion to the task's own risk.

    A task whose risk is all query gets the query row; one with query and
    correctness both high gets a blend. The base row is always present so no
    task is scored on a single ability.
    """
    risks = {"query": profile.query_risk, "structural": profile.structural_risk,
             "semantic": profile.semantic_risk, "correctness": profile.correctness_risk}
    total = sum(risks.values())
    shares = {"base": 1.0} if total <= 0 else {
        "base": 1.0, **{name: 2.0 * value / total for name, value in risks.items() if value > 0}}
    mix: dict[str, float] = {}
    for row, share in shares.items():
        for ability, weight in DIMENSION_WEIGHTS[row].items():
            mix[ability] = mix.get(ability, 0.0) + weight * share
    scale = sum(mix.values()) or 1.0
    return {ability: weight / scale for ability, weight in mix.items()}


def _estimated_context_tokens(profile: TaskProfile) -> int:
    """What this task will ask a context window to hold.

    Built from the profile's own sizing, not the repository total: the evidence
    bundle is the target file's slices plus the instruction and the protocol,
    and it grows as repair rounds accumulate.
    """
    evidence_chars = 6000 + profile.target_lines * 45 + profile.relevant_files * 400
    return int(evidence_chars / 3.5 * CONTEXT_SAFETY)


def select_initial_model(profile: TaskProfile,
                         available: Sequence[str] | None = None,
                         headroom: float | None = None) -> ModelDecision | None:
    """Rank the pool for this task and return the best first attempt.

    Deterministic and local: same profile plus same registry gives the same
    decision, with no network call and no model call. Returns None when nothing
    can be scored, which sends the caller back to the tier policy.
    """
    names = [m for m in (available or MODEL_CAPABILITIES) if m in MODEL_CAPABILITIES and m in MODELS]
    if not names:
        return None

    mix = _risk_mix(profile)
    needed = _estimated_context_tokens(profile)
    floor = CAPABILITY_FLOOR_BY_TIER.get(profile.tier, 0.0)
    reasons: list[str] = []
    scored: list[tuple[float, dict]] = []

    for name in names:
        cap = MODEL_CAPABILITIES[name]
        raw = sum(getattr(cap, ability) * weight for ability, weight in mix.items())
        # A low-confidence row is pulled toward the pool mean, so a guess never
        # outranks a measurement by more than the evidence supports.
        capability = raw + (PRIOR_MEAN - raw) * (1.0 - cap.confidence) * MODEL_SELECTION_CONFIDENCE_WEIGHT

        # A reliability claim is only as good as the evidence behind it, so it
        # is damped by the same confidence as the capability row.
        reliability = (cap.reliability - 0.75) * MODEL_SELECTION_RELIABILITY_WEIGHT * cap.confidence
        # Cost is NOT in the score. Between two similar models its spread is
        # wider than any capability difference, so scoring it would let price
        # decide correctness -- the inversion the competition punishes. It
        # breaks ties inside MODEL_SELECTION_MARGIN and nowhere else.
        cost_penalty = cap.blended_price * MODEL_SELECTION_COST_WEIGHT
        context_penalty = 0.0
        if cap.context_length and needed > cap.context_length:
            context_penalty = 10.0                      # cannot hold the task at all
        elif cap.context_length and needed > 0.6 * cap.context_length:
            context_penalty = MODEL_SELECTION_CONTEXT_WEIGHT
        # Uncertainty is where a weak model turns a cheap start into a wasted
        # one, so it is charged against capability shortfall, not flat.
        mismatch = (profile.uncertainty * MODEL_SELECTION_UNCERTAINTY_PENALTY
                    * max(0.0, PRIOR_MEAN + 0.10 - capability) * 10.0)
        if capability < floor:
            mismatch += 10.0                            # risk floor: not eligible to go first

        total = capability + reliability - context_penalty - mismatch
        scored.append((total, {"model": name, "capability": capability, "reliability": reliability,
                               "cost": cost_penalty, "context": context_penalty,
                               "mismatch": mismatch, "confidence": cap.confidence}))

    # Deterministic order: score first, then name, so ties never depend on dict order.
    scored.sort(key=lambda item: (-item[0], item[1]["model"]))
    best_score, best = scored[0]

    # Within the indifference margin, capability is a coin toss -- spend the
    # tie on cost. Above it, capability wins and cost does not get a vote.
    close = [row for score, row in scored if best_score - score <= MODEL_SELECTION_MARGIN]
    if len(close) > 1:
        cheaper = min(close, key=lambda row: (MODEL_CAPABILITIES[row["model"]].blended_price, row["model"]))
        if cheaper["model"] != best["model"]:
            reasons.append(f"within {MODEL_SELECTION_MARGIN:.2f} of the top score, so the cheaper "
                           f"of {[row['model'].split('/')[-1] for row in close]} wins")
            best = cheaper
            best_score = next(score for score, row in scored if row is cheaper)

    dominant = max(("query", profile.query_risk), ("correctness", profile.correctness_risk),
                   ("structural", profile.structural_risk), ("semantic", profile.semantic_risk),
                   key=lambda item: item[1])
    if dominant[1] > 0:
        reasons.append(f"{dominant[0]} risk {dominant[1]:.1f} dominates, so "
                       f"{max(mix, key=mix.get)} capability is weighted highest")
    if floor:
        blocked = [row["model"].split("/")[-1] for _, row in scored if row["mismatch"] >= 10.0]
        if blocked:
            reasons.append(f"tier {profile.tier} floors capability at {floor:.2f}, excluding {blocked}")
    if profile.uncertainty:
        reasons.append(f"uncertainty {profile.uncertainty:.1f} penalises weaker models")
    if best["context"]:
        reasons.append(f"needs ~{needed:,} tokens of context")
    if headroom is not None and headroom < 0.05:
        reasons.append(f"budget headroom ${headroom:.3f}")

    # Everything after the first pick is the ladder's business; the rest of the
    # ranking is passed only so the transport can fall through a 404 or a 403.
    eligible = [row["model"] for _, row in scored if row["mismatch"] < 10.0] or [best["model"]]
    rest = [m for m in eligible if m != best["model"]]
    # A different family first among the fallbacks: a second opinion from the
    # same architecture tends to repeat the first one's mistake. Ordering only
    # -- reliability already decided who goes first.
    family = best["model"].split("/")[0]
    rest.sort(key=lambda m: (m.split("/")[0] == family, eligible.index(m)))
    order = [best["model"]] + rest
    return ModelDecision(
        model=best["model"], reasoning_level="low", score=best_score,
        capability_score=best["capability"], cost_penalty=best["cost"],
        reliability_score=best["reliability"], context_penalty=best["context"],
        confidence=best["confidence"], rationale="; ".join(reasons) or "highest capability for this task",
        ranked=[(row["model"], round(score, 3)) for score, row in scored])


def select_initial_route(profile: TaskProfile, llm: "LLM | None" = None) -> InitialRoute:
    """The first rung of the ladder for this task.

    Capability matching picks the model when it can; the tier policy is the
    fallback, and the ladder is unchanged either way.
    """
    policy = ROUTING_POLICY[profile.tier]
    models = [m for m in policy["models"] if m in MODELS] or list(LADDER[0])
    reasoning = policy["reasoning"]
    rationale = f"{profile.tier}: overall risk {profile.overall_risk:.1f}"
    decision = None

    if MODEL_SELECTION_ENABLED:
        try:
            available = [m for m in (llm.discovered if llm and llm.discovered else MODELS)
                         if m not in (llm.unsupported if llm else set())]
            decision = select_initial_model(profile, available,
                                            headroom=llm.headroom() if llm else None)
        except Exception as exc:                 # a routing bug must not cost the task
            log(f"model capability selection skipped ({exc.__class__.__name__}: {exc})")
            decision = None
        if decision is not None:
            # Keep the policy's own order behind the chosen model, so a 404 or a
            # 403 on the first slug still falls through to a sensible second.
            models = [decision.model] + [m for m in models + [n for n, _ in decision.ranked]
                                         if m != decision.model]
            reasoning = decision.reasoning_level
            rationale = decision.rationale

    return InitialRoute(tier=profile.tier, models=models, reasoning=reasoning,
                        context_bonus=int(policy["context_bonus"]),
                        rationale=rationale, decision=decision)


def telemetry_record(profile: TaskProfile, route: InitialRoute, solver: "Solver", llm: "LLM",
                     patch: str, runtime_s: float) -> dict:
    """One JSON line per task so routing can be tuned from real outcomes."""
    if solver.final_clean:
        failure = ""
    elif solver.attempts == 0:
        failure = solver.stop_reason or "no_attempt"
    elif solver.failures_seen >= MAX_REPAIR_ROUNDS:
        failure = "verified_failures_exhausted"
    elif solver.slips_seen > MAX_GATE_SLIPS:
        failure = "gate_slips_exhausted"
    else:
        failure = "unverified_or_stopped"
    return {
        "routing_mode": ROUTING_MODE, "tier": profile.tier, "overall_risk": round(profile.overall_risk, 2),
        "structural": profile.structural_risk, "semantic": profile.semantic_risk,
        "query": profile.query_risk, "correctness": profile.correctness_risk,
        "uncertainty": profile.uncertainty, "task_types": profile.task_types,
        "route_model": route.models[0], "route_reasoning": route.reasoning,
        "route_applied": ROUTING_MODE == "deterministic" and not FORCE_MODEL,
        # The routing prior, kept apart from the observed outcome below: these
        # are capability estimates, never success probabilities.
        "selection_used": route.decision is not None,
        "selection_score": round(route.decision.score, 3) if route.decision else None,
        "capability_score": round(route.decision.capability_score, 3) if route.decision else None,
        "cost_penalty": round(route.decision.cost_penalty, 3) if route.decision else None,
        "reliability_score": round(route.decision.reliability_score, 3) if route.decision else None,
        "selection_confidence": route.decision.confidence if route.decision else None,
        "recovery_used": solver.attempts > 1 or solver.failures_seen > 0 or solver.stalls_seen > 0,
        "attempts": solver.attempts, "first_attempt_success": solver.first_attempt_clean,
        "final_success": solver.final_clean, "patch_returned": bool(patch.strip()),
        "verified_failures": solver.failures_seen, "gate_slips": solver.slips_seen, "stalls": solver.stalls_seen,
        "calls": getattr(llm, "calls", 0), "prompt_tokens": getattr(llm, "prompt_tokens", 0),
        "cached_tokens": getattr(llm, "cached_tokens", 0),
        "completion_tokens": getattr(llm, "completion_tokens", 0),
        "reasoning_tokens": getattr(llm, "reasoning_tokens", 0),
        "cost_usd": round(llm.spent(), 5), "runtime_s": round(runtime_s, 1),
        "runaways": len(getattr(llm, "no_reasoning", ())), "failure": failure,
    }


