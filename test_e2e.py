"""End-to-end tests of agent_main with a scripted model and no database.

    python3 test_e2e.py [path-to-pinned-checkout]
    python3 -m unittest test_e2e

With no argument the checkout is resolved from $RIDGES_NETBOX or the local
cache; when there is none the whole case is skipped with a reason rather than
failing every test in setUp.
"""
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import agent, selftest

BENCH = Path("/home/ajh/Documents/ridges-bench/db-engineering")
TASK = BENCH / "pg-netbox-contact-group-counts-001"
TARGET = "netbox/tenancy/models/contacts.py"
# sys.argv[1] under `python -m unittest test_e2e` is the module name, not a
# path, so it is offered as a candidate and rejected unless it really is one.
PRISTINE = selftest.default_checkout(sys.argv[1] if len(sys.argv) > 1 else None)

# The real fix for this sample: an MPTT descendant predicate instead of one
# that only reaches immediate children.
BROKEN = '" AND (cg.id = tenancy_contactgroup.id"\n                " OR cg.parent_id = tenancy_contactgroup.id)"'
FIXED = '" AND cg.lft >= tenancy_contactgroup.lft"\n                " AND cg.lft <= tenancy_contactgroup.rght"'


class ScriptedLLM(agent.LLM):
    """Replays a fixed list of replies instead of calling a provider."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.calls = 0
        self.max_cost = 1.0
        self.spent_estimate = 0.0
        self.unsupported = set()
        self.blocked = set()          # solve()'s InferenceError path reads this
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.per_model = {}
        self.discovered = []
        # Mirror the real client's counters so the telemetry record can read them.
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.no_reasoning = set()
        self.effort = "low"

    def discover_models(self):
        return []

    def roster(self, preferred):
        return list(preferred)

    def report(self):
        return f"cost $0.00000 (scripted) over {self.calls} call(s)"

    def spent(self):
        return 0.0

    def headroom(self):
        return 1.0

    def complete(self, candidates, messages, **kwargs):
        self.seen.append(messages)
        self.tiers = getattr(self, "tiers", []) + [tuple(candidates)]
        self.calls += 1
        if not self.replies:
            raise agent.InferenceError("script exhausted")
        return self.replies.pop(0), candidates[0]


@unittest.skipIf(PRISTINE is None,
                 f"no pinned checkout (looked in $RIDGES_NETBOX and {selftest.CHECKOUT_CACHE})")
class E2E(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = selftest.stage(PRISTINE, TASK, Path(self.tmp.name))
        self.statement = (TASK / "instruction.md").read_text()
        # Every stub below replaces something on the agent module itself, so
        # each is recorded and put back in tearDown. Left in place they outlive
        # this file: `python3 -m unittest discover` runs test_e2e before
        # test_units, and a stubbed LLM and a probe that discovers nothing
        # failed 24 of that suite's tests for reasons that had nothing to do
        # with what they were testing.
        self._original = {
            (agent, "workdir"): agent.workdir,
            (agent, "LLM"): agent.LLM,
            (agent.DatabaseProbe, "_discover"): agent.DatabaseProbe._discover,
            (agent.Checker, "run_task_commands"): agent.Checker.run_task_commands,
            (agent.Checker, "discovered_commands"): agent.Checker.discovered_commands,
            (agent.Checker, "unmeasured_bounded_work"): agent.Checker.unmeasured_bounded_work,
        }
        self._patch_env()

    def tearDown(self):
        for (owner, name), value in self._original.items():
            setattr(owner, name, value)
        self.tmp.cleanup()

    def _patch_env(self):
        agent.workdir = lambda: self.root
        # No database and no Django here; the task's own commands are exercised
        # by their own test below.
        agent.DatabaseProbe._discover = lambda self: None
        agent.Checker.run_task_commands = lambda self: [
            agent.CheckResult("$ task checks", True, "stubbed")
        ]
        # This sample is a bounded-work task, so a reply carrying no `measure`
        # probe now costs a round. That round is real and has its own test
        # below; switching it off here keeps every other case measuring the
        # thing it was written to measure -- the repair loop -- rather than
        # counting one extra call each.
        agent.Checker.unmeasured_bounded_work = lambda self, supplied: None

    def run_agent(self, replies):
        llm = ScriptedLLM(replies)
        agent.LLM = lambda: llm
        patch = agent.agent_main({"problem_statement": self.statement})
        return patch, llm

    @staticmethod
    def edit_reply(search, replace, path=TARGET):
        return json.dumps({
            "action": "edit",
            "diagnosis": "the predicate only reaches immediate children",
            "edits": [{"path": path, "search": search, "replace": replace}],
        })

    def test_single_round_produces_applicable_patch(self):
        patch, llm = self.run_agent([self.edit_reply(BROKEN, FIXED)])
        self.assertTrue(patch.startswith(f"diff --git a/{TARGET}"), patch[:200])
        self.assertEqual(patch.count("diff --git"), 1, "patch must touch exactly one file")
        self.assertIn("cg.lft", patch)
        self.assertEqual(llm.calls, 1)

    def test_working_tree_is_restored_after_run(self):
        original = (self.root / TARGET).read_text()
        self.run_agent([self.edit_reply(BROKEN, FIXED)])
        self.assertEqual((self.root / TARGET).read_text(), original,
                         "the checkout must be left pristine for the checker")

    def test_need_context_round_then_edit(self):
        need = json.dumps({"action": "need_context", "why": "locate the annotation",
                           "requests": [{"kind": "grep", "pattern": "annotate_contacts"}]})
        patch, llm = self.run_agent([need, self.edit_reply(BROKEN, FIXED)])
        self.assertEqual(llm.calls, 2)
        context = llm.seen[1][-1]["content"]
        self.assertIn("Requested context", context)
        self.assertIn("annotate_contacts", context)
        self.assertIn("cg.lft", patch)

    def test_broken_edit_is_repaired_on_retry(self):
        # An unbalanced call, so the file genuinely fails to parse.
        broken_syntax = self.edit_reply("contact_count=RawSQL(", "contact_count=RawSQL((")
        patch, llm = self.run_agent([broken_syntax, self.edit_reply(BROKEN, FIXED)])
        self.assertEqual(llm.calls, 2)
        feedback = llm.seen[1][-1]["content"]
        self.assertIn("did not pass verification", feedback)
        self.assertIn("cg.lft <= tenancy_contactgroup.rght", patch)

    def test_out_of_scope_edit_is_rejected_and_reported(self):
        stray = json.dumps({"action": "edit", "diagnosis": "d", "edits": [
            {"path": "netbox/tenancy/models/__init__.py", "search": "from .contacts import *",
             "replace": "from .contacts import *  # touched"}]})
        patch, llm = self.run_agent([stray, self.edit_reply(BROKEN, FIXED)])
        feedback = llm.seen[1][-1]["content"]
        self.assertIn("permits edits only to", feedback)
        self.assertNotIn("__init__.py", patch)

    def test_unparseable_reply_is_challenged(self):
        patch, llm = self.run_agent(["I think you should add an index.",
                                     self.edit_reply(BROKEN, FIXED)])
        self.assertIn("parseable JSON", llm.seen[1][-1]["content"])
        self.assertTrue(patch)

    def test_failing_checks_still_yield_best_effort_patch(self):
        agent.Checker.run_task_commands = lambda self: [
            agent.CheckResult("$ task checks", False, "AssertionError: 3 != 5")
        ]
        patch, llm = self.run_agent([self.edit_reply(BROKEN, FIXED)] * 3)
        self.assertEqual(llm.calls, 3, "should exhaust its repair rounds")
        self.assertTrue(patch, "a best-effort patch beats returning nothing")
        self.assertIn("AssertionError: 3 != 5", llm.seen[1][-1]["content"])

    def test_gate_slips_do_not_escalate_or_burn_repair_rounds(self):
        # L2: four scope slips in a row, then a good edit. None of them ran the
        # tests, so the model tier must not move and the good edit must still
        # be accepted (a repair-round counter would have given up after 3).
        # Four *different* out-of-scope edits: distinct slips stay on the same
        # tier. (An identical repeat is a stall and is tested separately.)
        strays = [json.dumps({"action": "edit", "diagnosis": "d", "edits": [
            {"path": "netbox/tenancy/models/__init__.py", "search": "from .contacts import *",
             "replace": f"from .contacts import *  # touched {n}"}]}) for n in range(4)]
        patch, llm = self.run_agent(strays + [self.edit_reply(BROKEN, FIXED)])
        self.assertEqual(llm.calls, 5)
        self.assertEqual(len(set(llm.tiers)), 1, "a protocol slip must not change model tier")
        self.assertIn("rule violation, not a wrong diagnosis", llm.seen[1][-1]["content"])
        self.assertIn("cg.lft <= tenancy_contactgroup.rght", patch)

    def test_too_many_gate_slips_gives_up(self):
        stray = json.dumps({"action": "edit", "diagnosis": "d", "edits": [
            {"path": "netbox/tenancy/models/__init__.py", "search": "from .contacts import *",
             "replace": "from .contacts import *  # touched"}]})
        patch, llm = self.run_agent([stray] * 8)
        self.assertEqual(llm.calls, agent.MAX_GATE_SLIPS + 1)

    def test_verified_failure_escalates_tier(self):
        # L3: the task's tests fail -> that is evidence -> next tier of the ladder.
        agent.Checker.run_task_commands = lambda self: [
            agent.CheckResult("$ task checks", False, "AssertionError: 3 != 5")
        ]
        patch, llm = self.run_agent([self.edit_reply(BROKEN, FIXED)] * 3)
        self.assertEqual(llm.calls, 3)
        self.assertNotEqual(llm.tiers[0], llm.tiers[1], "a verified failure must escalate")

    def test_unverifiable_patch_is_adopted_after_one_verify_request(self):
        # No test command anywhere: ask once for `verify`, then take the
        # statically clean patch rather than spending three more calls on it.
        agent.Checker.run_task_commands = lambda self: []
        agent.Checker.discovered_commands = lambda self, changed: []
        patch, llm = self.run_agent([self.edit_reply(BROKEN, FIXED)] * 4)
        self.assertEqual(llm.calls, 2)
        asked = [m for m in llm.seen[1] if m["role"] == "user" and "No test command was available" in m["content"]]
        self.assertEqual(len(asked), 1, "exactly one verify request")
        self.assertIn("cg.lft <= tenancy_contactgroup.rght", patch)

    def test_identical_invalid_edit_moves_to_the_next_model_family(self):
        # Same broken edit twice at temperature 0 -> the second slip is a
        # stall; the third call must go to a different tier, and the feedback
        # must say the reply was identical.
        broken = self.edit_reply("contact_count=RawSQL(", "contact_count=RawSQL((")
        patch, llm = self.run_agent([broken, broken, self.edit_reply(BROKEN, FIXED)])
        self.assertEqual(llm.calls, 3)
        self.assertEqual(llm.tiers[0], llm.tiers[1], "a first slip stays on the same tier")
        self.assertNotEqual(llm.tiers[1], llm.tiers[2], "an identical repeat moves to the next family")
        self.assertIn("byte-identical", llm.seen[2][-1]["content"])
        self.assertIn("cg.lft <= tenancy_contactgroup.rght", patch)


@unittest.skipIf(PRISTINE is None, "no pinned checkout")
class BoundedWorkMustBeMeasured(unittest.TestCase):
    """The round a bounded-work task buys when no measurement was taken.

    The other cases switch this off. Here it is on, because the whole point is
    that a reply which never measures the statement count does not finish the
    run -- which is what happened three times on the device-filter task, twice
    with no probe supplied at all.
    """

    # Deliberately not a subclass of E2E: inheriting it would re-run every case
    # in that class with the measurement switched on, and those cases script a
    # fixed number of replies.
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = selftest.stage(PRISTINE, TASK, Path(self.tmp.name))
        self.statement = (TASK / "instruction.md").read_text()
        self._original = {
            (agent, "workdir"): agent.workdir,
            (agent, "LLM"): agent.LLM,
            (agent.DatabaseProbe, "_discover"): agent.DatabaseProbe._discover,
            (agent.Checker, "run_task_commands"): agent.Checker.run_task_commands,
        }
        agent.workdir = lambda: self.root
        agent.DatabaseProbe._discover = lambda self: None
        agent.Checker.run_task_commands = lambda self: [
            agent.CheckResult("$ task checks", True, "stubbed")]

    def tearDown(self):
        for (owner, name), value in self._original.items():
            setattr(owner, name, value)
        self.tmp.cleanup()

    edit_reply = staticmethod(E2E.edit_reply)
    run_agent = E2E.run_agent

    def test_a_reply_with_no_probe_buys_one_more_attempt(self):
        self.assertTrue(agent.parse_instruction(self.statement, self.root).bounded_work)
        patch, llm = self.run_agent([self.edit_reply(BROKEN, FIXED)] * 2)
        self.assertEqual(llm.calls, 2, "no measurement means the run is not finished")
        asked = [m for m in llm.seen[1]
                 if m["role"] == "user" and "never measured" in m["content"]]
        self.assertEqual(len(asked), 1, "the model is told what was not measured, once")
        self.assertIn("`measure` probe", asked[0]["content"])
        self.assertIn("cg.lft", patch, "and the patch is still returned")

    def test_it_asks_once_and_then_adopts(self):
        # A model that cannot produce a working probe must not loop: one
        # request, then the statically clean patch is the answer.
        patch, llm = self.run_agent([self.edit_reply(BROKEN, FIXED)] * 5)
        self.assertEqual(llm.calls, 2)
        self.assertIn("cg.lft", patch)


if __name__ == "__main__":
    if PRISTINE is None:
        raise SystemExit("usage: test_e2e.py <pinned-netbox-checkout>")
    sys.argv = sys.argv[:1]
    unittest.main(verbosity=2)
