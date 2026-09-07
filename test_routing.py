"""Tests for routing_lab: the profiler and selector held out of the agent.

They cover a subsystem no graded run executes, kept so the work stays correct
and can be reinstated from a known-good state.

    python3 test_routing.py
"""
import json, os, sys, tempfile, textwrap, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import agent
import routing_lab as routing


def make_repo(files: dict[str, str]) -> agent.Repository:
    root = Path(tempfile.mkdtemp())
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return agent.Repository(root)


def a_profile(**kw) -> routing.TaskProfile:
    """A neutral TaskProfile; each test raises only the risk it is about."""
    base = dict(task_types=[], repo_files=500, relevant_files=1, target_files=1, target_symbols=1,
                target_lines=200, dependency_depth=0, constraint_count=0, test_signal=1,
                structural_risk=0.0, semantic_risk=0.0, query_risk=0.0, correctness_risk=0.0,
                uncertainty=0.0, overall_risk=0.0, tier="LOW", reasons=[], uncertainty_reasons=[])
    base.update(kw)
    return routing.TaskProfile(**base)


class TestModelCapabilityRegistry(unittest.TestCase):
    """The registry is data, and the routing engine must not know model names."""

    def test_every_capability_row_matches_a_real_model(self):
        for name, cap in routing.MODEL_CAPABILITIES.items():
            self.assertIn(name, agent.MODELS, f"{name} has capabilities but no ModelSpec")
            self.assertEqual(cap.model_name, name)
            self.assertGreater(cap.context_length, 0)
            self.assertGreater(cap.blended_price, 0)

    def test_pool_is_small_and_every_row_states_its_role_and_evidence(self):
        self.assertLessEqual(len(routing.MODEL_CAPABILITIES), 7)
        self.assertGreaterEqual(len(routing.MODEL_CAPABILITIES), 5)
        for cap in routing.MODEL_CAPABILITIES.values():
            self.assertTrue(cap.role.strip(), cap.model_name)
            self.assertTrue(cap.source_notes.strip(), cap.model_name)
            self.assertTrue(cap.source_notes.startswith(("measured", "published")),
                            f"{cap.model_name}: source_notes must say which kind of evidence it is")

    def test_scores_are_coarse_priors_not_false_precision(self):
        fields = ("coding", "debugging", "sql", "repository", "reasoning", "agentic",
                  "reliability", "confidence")
        for cap in routing.MODEL_CAPABILITIES.values():
            for field_name in fields:
                value = getattr(cap, field_name)
                self.assertTrue(0.0 <= value <= 1.0, f"{cap.model_name}.{field_name}={value}")
                self.assertEqual(round(value, 2), value,
                                 f"{cap.model_name}.{field_name} claims more precision than exists")

    def test_routing_engine_contains_no_model_names(self):
        # The algorithm must read capabilities, never `if model == "x"`.
        import inspect
        for function in (routing.select_initial_model, routing._risk_mix, routing._estimated_context_tokens):
            body = inspect.getsource(function)
            for name in agent.MODELS:
                self.assertNotIn(name, body, f"{function.__name__} hardcodes {name}")


class TestInitialModelSelection(unittest.TestCase):
    """Mechanisms, not particular models: the priors are meant to be retuned."""

    def cap(self, name): return routing.MODEL_CAPABILITIES[name]

    def test_high_query_risk_selects_a_strong_sql_model(self):
        decision = routing.select_initial_model(
            a_profile(tier="HIGH", overall_risk=8.0, query_risk=9.0, correctness_risk=4.0))
        best_sql = max(c.sql for c in routing.MODEL_CAPABILITIES.values())
        self.assertGreaterEqual(self.cap(decision.model).sql, best_sql - 0.06,
                                f"{decision.model} is not among the SQL-capable models")

    def test_high_structural_risk_prefers_repository_capability(self):
        decision = routing.select_initial_model(
            a_profile(tier="HIGH", overall_risk=8.0, structural_risk=9.0, repo_files=18000,
                      target_lines=4000, relevant_files=6, dependency_depth=3))
        best_repo = max(c.repository for c in routing.MODEL_CAPABILITIES.values())
        self.assertGreaterEqual(self.cap(decision.model).repository, best_repo - 0.10)

    def test_risk_floors_exclude_weak_models_from_going_first(self):
        weakest = min(routing.MODEL_CAPABILITIES.values(), key=lambda c: c.coding).model_name
        for tier in ("MEDIUM", "HIGH", "VERY_HIGH"):
            decision = routing.select_initial_model(
                a_profile(tier=tier, overall_risk=9.0, query_risk=8.0, correctness_risk=8.0))
            self.assertNotEqual(decision.model, weakest, f"{tier} must not start on the weakest model")
            self.assertNotIn(weakest, [m for m, _ in decision.ranked[:1]])

    def test_uncertainty_never_routes_to_a_weaker_model(self):
        calm = routing.select_initial_model(a_profile(tier="MEDIUM", overall_risk=5.0, query_risk=4.0))
        anxious = routing.select_initial_model(
            a_profile(tier="MEDIUM", overall_risk=5.0, query_risk=4.0, uncertainty=6.0))
        capability = lambda name: (self.cap(name).coding + self.cap(name).sql + self.cap(name).reasoning)
        self.assertGreaterEqual(capability(anxious.model), capability(calm.model) - 1e-9,
                                "uncertainty must never buy a weaker model")

    def test_a_task_too_large_for_a_context_window_rejects_that_model(self):
        windows = sorted({c.context_length for c in routing.MODEL_CAPABILITIES.values()})
        self.assertGreater(len(windows), 1, "needs models with different context windows")
        # Size the task between the small and the large windows, so the check
        # has something to discriminate.
        big = a_profile(tier="LOW", target_lines=15_000, relevant_files=6)
        needed = routing._estimated_context_tokens(big)
        self.assertTrue(windows[0] < needed < windows[-1], f"{needed} tokens does not straddle {windows}")
        decision = routing.select_initial_model(big)
        self.assertGreaterEqual(self.cap(decision.model).context_length, needed,
                                "chose a model that cannot hold the task")

    def test_a_task_too_large_for_every_model_still_returns_a_choice(self):
        # Nothing fits: every model takes the same penalty, so capability still
        # decides and the agent tries rather than refusing.
        impossible = a_profile(tier="LOW", target_lines=600_000, relevant_files=6)
        largest = max(c.context_length for c in routing.MODEL_CAPABILITIES.values())
        self.assertGreater(routing._estimated_context_tokens(impossible), largest)
        decision = routing.select_initial_model(impossible)
        self.assertIsNotNone(decision)
        self.assertIn(decision.model, agent.MODELS)

    def test_cost_decides_only_between_near_equals(self):
        # Two rows identical but for price: the cheaper must win.
        registry = dict(routing.MODEL_CAPABILITIES)
        twin = next(iter(registry.values()))
        routing.MODEL_CAPABILITIES.clear()
        agent.MODELS.setdefault("test/rich", agent.ModelSpec("test/rich", 9.0, 9.0, 1_000_000))
        agent.MODELS.setdefault("test/thrifty", agent.ModelSpec("test/thrifty", 0.2, 0.5, 1_000_000))
        try:
            for name in ("test/rich", "test/thrifty"):
                routing.MODEL_CAPABILITIES[name] = routing.ModelCapabilityProfile(
                    model_name=name, role="test", coding=twin.coding, debugging=twin.debugging,
                    sql=twin.sql, repository=twin.repository, reasoning=twin.reasoning,
                    agentic=twin.agentic, reliability=twin.reliability, confidence=twin.confidence,
                    source_notes="published: synthetic test row")
            decision = routing.select_initial_model(a_profile())
            self.assertEqual(decision.model, "test/thrifty")
            # Now make the dear one clearly better: capability must win.
            routing.MODEL_CAPABILITIES["test/rich"] = routing.ModelCapabilityProfile(
                model_name="test/rich", role="test", coding=0.95, debugging=0.95, sql=0.95,
                repository=0.95, reasoning=0.95, agentic=0.95, reliability=0.90, confidence=0.9,
                source_notes="published: synthetic test row")
            self.assertEqual(routing.select_initial_model(a_profile()).model, "test/rich",
                             "cost must not outrank a real capability gap")
        finally:
            routing.MODEL_CAPABILITIES.clear(); routing.MODEL_CAPABILITIES.update(registry)
            agent.MODELS.pop("test/rich", None); agent.MODELS.pop("test/thrifty", None)

    def test_fallback_order_prefers_a_different_family(self):
        decision = routing.select_initial_model(a_profile(tier="MEDIUM", overall_risk=5.0, query_risk=4.0))
        order = [decision.model] + [m for m, _ in decision.ranked if m != decision.model]
        eligible = [m for m in order if m in routing.MODEL_CAPABILITIES]
        if len({m.split("/")[0] for m in eligible}) > 1:
            self.assertNotEqual(eligible[1].split("/")[0], eligible[0].split("/")[0],
                                "the first fallback should come from another family")

    def test_selection_is_deterministic(self):
        profile = a_profile(tier="HIGH", overall_risk=8.0, query_risk=7.0, correctness_risk=6.0)
        decisions = [routing.select_initial_model(profile) for _ in range(25)]
        self.assertEqual({d.model for d in decisions}, {decisions[0].model})
        self.assertEqual({tuple(m for m, _ in d.ranked) for d in decisions},
                         {tuple(m for m, _ in decisions[0].ranked)})

    def test_selection_makes_no_network_call(self):
        import urllib.request, socket
        def forbidden(*args, **kwargs):
            raise AssertionError("model selection must not touch the network")
        saved = (urllib.request.urlopen, socket.create_connection)
        urllib.request.urlopen, socket.create_connection = forbidden, forbidden
        try:
            self.assertIsNotNone(routing.select_initial_model(a_profile(tier="HIGH", query_risk=8.0)))
        finally:
            urllib.request.urlopen, socket.create_connection = saved

    def test_selects_one_model_not_a_committee(self):
        decision = routing.select_initial_model(a_profile(tier="VERY_HIGH", overall_risk=10.0, query_risk=9.0))
        self.assertIsInstance(decision.model, str)
        self.assertIn(decision.model, agent.MODELS)


class TestSelectionIntegration(unittest.TestCase):
    """The selector picks the first rung; everything after it is unchanged."""

    def test_empty_registry_falls_back_to_the_tier_policy(self):
        registry = dict(routing.MODEL_CAPABILITIES)
        routing.MODEL_CAPABILITIES.clear()
        try:
            self.assertIsNone(routing.select_initial_model(a_profile()))
            route = routing.select_initial_route(a_profile(tier="MEDIUM"))
            self.assertIsNone(route.decision)
            self.assertEqual(route.models, routing.ROUTING_POLICY["MEDIUM"]["models"])
        finally:
            routing.MODEL_CAPABILITIES.update(registry)

    def test_a_selector_exception_falls_back_to_the_tier_policy(self):
        original = routing.select_initial_model
        routing.select_initial_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            route = routing.select_initial_route(a_profile(tier="HIGH"))
        finally:
            routing.select_initial_model = original
        self.assertIsNone(route.decision)
        self.assertEqual(route.models, routing.ROUTING_POLICY["HIGH"]["models"])

    def test_selection_can_be_disabled(self):
        routing.MODEL_SELECTION_ENABLED = False
        try:
            route = routing.select_initial_route(a_profile(tier="HIGH", query_risk=9.0))
        finally:
            routing.MODEL_SELECTION_ENABLED = True
        self.assertIsNone(route.decision)
        self.assertEqual(route.models, routing.ROUTING_POLICY["HIGH"]["models"])
        self.assertEqual(route.reasoning, routing.ROUTING_POLICY["HIGH"]["reasoning"])

    def test_unavailable_models_are_never_selected(self):
        llm = agent.LLM()
        llm.discovered = list(routing.MODEL_CAPABILITIES)
        llm.unsupported = set(list(routing.MODEL_CAPABILITIES)[:2])
        route = routing.select_initial_route(a_profile(tier="HIGH", query_risk=8.0), llm)
        self.assertIsNotNone(route.decision)
        self.assertNotIn(route.decision.model, llm.unsupported)

    def test_reasoning_stays_the_conservative_policy(self):
        for tier in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH"):
            route = routing.select_initial_route(a_profile(tier=tier, query_risk=5.0))
            self.assertEqual(route.reasoning, "low",
                             "reasoning policy is unmeasured; the selector must not change it")

    def test_telemetry_separates_the_prior_from_the_outcome(self):
        repo = make_repo({"a.py": "x = 1\n"})
        instruction = agent.parse_instruction("Fix `a.py`.", repo.root)
        solver = agent.Solver(repo, instruction, None, agent.LLM())
        solver.attempts, solver.final_clean, solver.first_attempt_clean = 2, True, False
        profile = a_profile(tier="HIGH", overall_risk=8.0, query_risk=8.0)
        route = routing.select_initial_route(profile)
        record = routing.telemetry_record(profile, route, solver, solver.llm, "diff", 12.0)
        json.dumps(record)
        for key in ("selection_used", "capability_score", "cost_penalty", "reliability_score",
                    "selection_confidence", "selection_score", "recovery_used"):
            self.assertIn(key, record)
        self.assertTrue(record["recovery_used"], "two attempts means recovery was used")
        self.assertTrue(record["final_success"])



class TestProblemProfiler(unittest.TestCase):
    """The profiler is deterministic and costs no model tokens; these pin the
    signals that move a task between tiers, and the routing contract.

    Moved here from test_units.py when routing left agent.py: the agent no
    longer defines any of these names, so the tests follow the code.
    """

    def profile(self, text: str, files: dict[str, str] | None = None):
        repo = make_repo(files or {"app/q.py": "def run():\n    return Model.objects.all()\n"})
        ins = agent.parse_instruction(text, repo.root)
        targets = agent.locate_targets(repo, ins)
        return routing.profile_task(ins, repo, targets, None), ins

    def test_simple_named_local_change_is_low_or_medium(self):
        p, _ = self.profile("Limit production changes to `app/q.py`, specifically `run()`. "
                            "Run `python -m pytest tests/test_q.py` before finishing.")
        self.assertIn(p.tier, ("LOW", "MEDIUM"))
        self.assertLess(p.overall_risk, routing.RISK_THRESHOLDS["MEDIUM"] + 0.01)

    def test_bounded_query_task_scores_query_risk(self):
        p, _ = self.profile("The importer issues one INSERT per row, an N+1 pattern. The work must be "
                            "bounded: at most 3 queries regardless of input size. Limit production "
                            "changes to `app/q.py`. Run `python -m pytest tests/t.py`.")
        self.assertGreaterEqual(p.query_risk, 5.0)
        self.assertIn("query_optimization", p.task_types)

    def test_correctness_wording_raises_correctness_risk_alone(self):
        p, _ = self.profile("The report double-counts rows. Duplicate tags must count once, ties broken "
                            "by version, NULL groups must appear with zero, ordering is exact. Correctness "
                            "is judged on data the task does not show. Limit production changes to "
                            "`app/q.py`, specifically `run()`. Keep the rest of the file unchanged, "
                            "including imports. Run `python -m pytest tests/t.py`.")
        self.assertGreaterEqual(p.correctness_risk, 5.0)
        self.assertIn("correctness_sensitive", p.task_types)
        self.assertGreaterEqual(p.overall_risk, p.correctness_risk,
                                "max() must not hide one severe dimension")

    def test_unknown_and_unverifiable_task_is_not_cheap(self):
        # No file, no method, no check command, no recognised shape: every
        # unknown must push risk up, never down.
        p, _ = self.profile("Make the dashboard faster.")
        self.assertGreater(p.uncertainty, 0)
        self.assertIn("no check command named: the first edit cannot be verified",
                      "; ".join(p.uncertainty_reasons))
        self.assertNotEqual(p.tier, "LOW")

    def test_overall_risk_is_max_plus_uncertainty_not_mean(self):
        p, _ = self.profile("The vlan utilisation percentage is wrong: it double-counts overlapping "
                            "ranges, must handle NULL and empty groups, and duplicate prefixes count "
                            "once. Limit production changes to `app/q.py`, specifically `run()`. "
                            "Run `python -m pytest tests/t.py`.")
        dims = [p.structural_risk, p.semantic_risk, p.query_risk, p.correctness_risk]
        self.assertAlmostEqual(p.overall_risk,
                               min(12.0, max(dims) + routing.UNCERTAINTY_WEIGHT * p.uncertainty),
                               places=6)
        self.assertGreater(max(dims), sum(dims) / 4, "the test task must have one dominant dimension")

    def test_tiers_are_monotonic_in_risk(self):
        seen = [routing._tier_for(r) for r in (0.0, 3.0, 3.1, 6.0, 6.1, 9.0, 9.1, 12.0)]
        self.assertEqual(seen, ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH",
                                "VERY_HIGH", "VERY_HIGH"])

    def test_route_uses_only_known_models_and_separates_reasoning(self):
        for tier in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH"):
            route = routing.select_initial_route(a_profile(tier=tier))
            self.assertTrue(route.models, tier)
            for m in route.models:
                self.assertIn(m, agent.MODELS, f"{tier} routes to a model outside the roster")
            self.assertIn(route.reasoning, ("low", "medium", "high"))
            self.assertNotEqual(route.reasoning, "off",
                                "reasoning-off is a recovery mechanism, not a route")

    def test_only_measured_routes_differ_from_the_baseline_ladder(self):
        # Every deviation from the baseline ladder must have evidence behind
        # it. Today that is one: VERY_HIGH leads with the family that solved
        # the hardest measured task when the default first choice spun.
        deviations = [tier for tier, policy in routing.ROUTING_POLICY.items()
                      if policy["models"] != list(agent.LADDER[0]) or policy["context_bonus"]]
        self.assertEqual(deviations, ["VERY_HIGH"], f"undocumented deviation: {deviations}")

    def test_low_tier_is_conservative_unless_opted_in(self):
        # Cheap-first on easy tasks is unmeasured, so the default must start on
        # the same rung as MEDIUM.
        self.assertEqual(routing.ROUTING_POLICY["LOW"]["models"],
                         routing.ROUTING_POLICY["MEDIUM"]["models"])

    def test_profiler_never_writes_to_the_repository(self):
        repo = make_repo({"app/q.py": "x = 1\n"})
        before = {f: repo.read(f) for f in repo.files}
        ins = agent.parse_instruction("Fix `app/q.py`.", repo.root)
        routing.profile_task(ins, repo, agent.locate_targets(repo, ins), None)
        self.assertEqual({f: repo.read(f) for f in repo.files}, before)
        self.assertEqual(repo.changed_files(), [])

    def test_explanation_names_the_reasons(self):
        p, _ = self.profile("Eliminate the N+1: at most 2 queries regardless of size. Limit production "
                            "changes to `app/q.py`, specifically `run()`. Run `python -m pytest tests/t.py`.")
        route = routing.select_initial_route(p)
        text = p.explain(route, applied=True)
        self.assertIn("overall", text)
        self.assertIn(p.tier, text)
        self.assertIn(route.models[0], text)
        self.assertTrue(any("bounded" in r or "numeric target" in r for r in p.reasons), p.reasons)


if __name__ == "__main__":
    unittest.main(verbosity=1)
