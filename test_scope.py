"""Offline tests for the scope parser, the submit gate, the polling loop,
the round gate, and the cost flag. No inference, no git, no database.

    python3 -m unittest test_scope
"""
from __future__ import annotations

import glob
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.request
from unittest import mock

import agent

HERE = os.path.dirname(os.path.abspath(__file__))

# What the 27 sample instructions say, read once by hand from the dump the
# parser produced and checked against each instruction.md.
EXPECTED = {
    "pg-netbox-bulk-tag-assignment-001": (None, "", 1),
    "pg-netbox-cached-value-index-001": (
        "netbox/extras/migrations/0107_cachedvalue_extras_cachedvalue_object.py", "", 2),
    "pg-netbox-contact-group-counts-001": (
        "netbox/tenancy/models/contacts.py", "ContactGroupManager.annotate_contacts", 2),
    "pg-netbox-ipaddress-device-filter-001": (
        "netbox/ipam/filtersets.py", "IPAddressFilterSet.filter_device", 2),
    "pg-netbox-prefix-hierarchy-annotations-001": (
        "netbox/ipam/querysets.py", "PrefixQuerySet.annotate_hierarchy", 2),
    "pg-netbox-vlangroup-utilization-001": (
        "netbox/ipam/querysets.py", "VLANGroupQuerySet.annotate_utilization", 2),
    "ch-hitmap-top-paths-047-test": ("src/paths.js", "topPaths", 2),
    "ch-metrics-rollup-008-test": (None, "", 2),
    "ch-retention-cohorts-009-test": ("internal/cohorts/retention.go", "CohortRetention", 3),
    "ch-roster-device-state-050-test": ("internal/roster/roster.go", "SiteRoster", 3),
    "ch-stickiness-day-n-retention-048-test": ("src/retention.js", "dayNRetention", 2),
    "ch-tide-revenue-trend-049-test": ("internal/trend/trend.go", "MovingAverage", 3),
    "pg-catalog-facet-counts-002-test": (None, "", 2),
    "pg-clubhouse-lapsed-members-037-test": (
        "lib/clubhouse/reports/lapsed.rb", "Clubhouse::Reports::Lapsed.rows", 2),
    "pg-counter-average-order-036-test": (
        "lib/counter/reports/average_order.rb", "Counter::Reports::AverageOrder.rows", 2),
    "pg-fleet-telemetry-latest-reading-003-test": (
        "internal/telemetry/latest.go", "LatestReadings", 3),
    "pg-ledger-running-balance-005-test": (
        "lib/tallyline/reports/running_balance.rb", "Tallyline::Reports::RunningBalance.rows", 2),
    "pg-ledgerlink-batched-import-029-test": ("src/importer.js", "storeStatement", 2),
    "pg-marketrow-top-per-category-032-test": ("src/top.js", "topProductsPerCategory", 2),
    "pg-nestly-region-listings-027-test": ("src/search.js", "regionListings", 2),
    "pg-packsmith-order-detail-030-test": ("src/detail.js", "orderDetail", 2),
    "pg-pulse-latency-histogram-035-test": ("internal/latency/latency.go", "Histogram", 3),
    "pg-purseline-wallet-statements-031-test": ("src/statements.js", "walletStatements", 2),
    "pg-rooms-slot-availability-033-test": (
        "internal/availability/availability.go", "FreeRooms", 3),
    "pg-shelfmark-outside-category-028-test": ("src/outside.js", "outsideCategory", 2),
    "pg-studioboard-day-board-038-test": (
        "lib/studioboard/reports/day_board.rb", "Studioboard::Reports::DayBoard.rows", 2),
    "pg-trail-audit-feed-034-test": ("internal/feed/feed.go", "Recent", 3),
}


def instructions() -> dict:
    found = {}
    for path in glob.glob(os.path.join(HERE, "fast-tasks", "*", "instruction.md")) + \
            glob.glob(os.path.join(HERE, "local-tasks", "*", "instruction.md")):
        with open(path, encoding="utf-8") as fh:
            found[os.path.basename(os.path.dirname(path))] = fh.read()
    return found


def quiet(func):
    """Run func with the agent's stdout chatter suppressed."""
    def wrapper(*args, **kwargs):
        with mock.patch("sys.stdout", new=io.StringIO()):
            return func(*args, **kwargs)
    return wrapper


class FakeTree:
    """Just enough of Tree for the parser: a root and read()."""
    def __init__(self, root: str) -> None:
        self.root = root

    def read(self, rel: str) -> str:
        full = os.path.join(self.root, rel)
        if not os.path.isfile(full):
            raise agent.ToolFault("no such file: %s" % rel)
        with open(full, encoding="utf-8", errors="replace") as fh:
            return fh.read()


def slurp(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def touch(root: str, rel: str, text: str = "x = 1\n") -> None:
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# The parser against every sample instruction
# ---------------------------------------------------------------------------

class StatementParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = instructions()
        if len(cls.texts) < len(EXPECTED):
            raise unittest.SkipTest("sample tasks are not checked out here")

    def test_every_sample_is_read_as_expected(self):
        for name, (path, symbol, checks) in EXPECTED.items():
            text = self.texts[name]
            root = tempfile.mkdtemp(prefix="scope-")
            try:
                if path:
                    touch(root, path)
                files, missing = agent.named_files(text, root)
                got_symbol = agent.named_symbol(text)[0]
                got_checks = agent.parse_checks(text)
                with self.subTest(task=name):
                    self.assertEqual(files, [path] if path else [])
                    self.assertEqual(missing, [])
                    self.assertEqual(got_symbol, symbol)
                    self.assertEqual(len(got_checks), checks, got_checks)
                    for command in got_checks:
                        head = os.path.basename(command.split()[0])
                        self.assertIn(head, agent.CHECK_RUNNERS, command)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    def test_named_file_that_does_not_exist_lands_in_missing(self):
        text = self.texts["ch-hitmap-top-paths-047-test"]
        root = tempfile.mkdtemp(prefix="scope-")
        try:
            files, missing = agent.named_files(text, root)
            self.assertEqual(files, [])
            self.assertEqual(missing, ["src/paths.js"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_named_file_is_found_one_directory_down(self):
        text = self.texts["ch-hitmap-top-paths-047-test"]
        root = tempfile.mkdtemp(prefix="scope-")
        try:
            touch(root, "app/src/paths.js")
            files, missing = agent.named_files(text, root)
            self.assertEqual(files, ["app/src/paths.js"])
            self.assertEqual(missing, [])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_inline_django_command_is_joined_across_the_line_break(self):
        checks = agent.parse_checks(self.texts["pg-netbox-prefix-hierarchy-annotations-001"])
        self.assertEqual(len(checks), 2)
        test = [c for c in checks if "manage.py test" in c][0]
        self.assertIn("--keepdb --noinput", test)
        self.assertNotIn("\n", test)

    def test_backslash_continued_block_becomes_one_command(self):
        checks = agent.parse_checks(self.texts["pg-netbox-bulk-tag-assignment-001"])
        self.assertEqual(len(checks), 1)
        self.assertNotIn("\\", checks[0])
        self.assertIn("test_create_tagged_item", checks[0])
        self.assertIn("test_update_tagged_item", checks[0])
        # "Also run `ruff check --no-cache` on the file you changed" names no
        # target, so it is advice, not a command.
        self.assertFalse(any(c.startswith("ruff") for c in checks))


class CheckParsing(unittest.TestCase):
    def test_cd_line_prefixes_the_commands_after_it(self):
        text = "Run:\n\n```bash\ncd netbox\npython manage.py test app.tests\nruff check x.py\n```\n"
        self.assertEqual(agent.parse_checks(text),
                         ["cd netbox && python manage.py test app.tests",
                          "cd netbox && ruff check x.py"])

    def test_env_assignment_does_not_hide_the_runner(self):
        text = "```bash\nDJANGO_SETTINGS_MODULE=x.settings pytest tests/a.py\n```"
        self.assertEqual(agent.parse_checks(text),
                         ["DJANGO_SETTINGS_MODULE=x.settings pytest tests/a.py"])

    def test_non_runner_lines_and_prose_are_ignored(self):
        text = "```bash\n$ echo hi\nls -la\n# python is not run here\n```"
        self.assertEqual(agent.parse_checks(text), [])

    def test_at_most_four_commands(self):
        text = "```bash\n" + "\n".join("pytest tests/t%d.py" % i for i in range(9)) + "\n```"
        self.assertEqual(len(agent.parse_checks(text)), agent.CHECK_COMMANDS_MAX)


# ---------------------------------------------------------------------------
# Region resolution per language
# ---------------------------------------------------------------------------

PY_SOURCE = '''import os
from x import y


class A:
    def rows(self):
        return 1

    @property
    def other(self):
        return 2


class B:
    def rows(self):
        return 3


def rows():
    return 4
'''


class RegionResolution(unittest.TestCase):
    def test_python_owner_disambiguates_same_named_methods(self):
        self.assertEqual(agent.resolve_region(PY_SOURCE, "m.py", "B", "rows"), (15, 16, "ast"))
        self.assertEqual(agent.resolve_region(PY_SOURCE, "m.py", "A", "rows"), (6, 7, "ast"))

    def test_python_bare_name_takes_the_module_level_definition(self):
        self.assertEqual(agent.resolve_region(PY_SOURCE, "m.py", "", "rows"), (19, 20, "ast"))

    def test_python_decorator_is_part_of_the_region(self):
        self.assertEqual(agent.resolve_region(PY_SOURCE, "m.py", "A", "other"), (9, 11, "ast"))

    def test_python_ambiguous_or_missing_name_is_whole(self):
        self.assertEqual(agent.resolve_region(PY_SOURCE, "m.py", "", "nope")[2], "whole")
        self.assertEqual(agent.resolve_region("def (:\n", "m.py", "", "rows")[2], "whole")

    def test_python_import_block(self):
        self.assertEqual(agent.import_block(PY_SOURCE, "m.py"), "import os\nfrom x import y")

    def test_javascript_function_and_arrow(self):
        source = ("const db = require('./db');\n\nasync function topPaths(db, { day }) {\n"
                  "  const rows = await db.query('x');\n  return rows;\n}\n\n"
                  "const other = async (a) => {\n  return a;\n};\n\nmodule.exports = { topPaths, other };\n")
        self.assertEqual(agent.resolve_region(source, "src/p.js", "", "topPaths"), (3, 6, "regex"))
        self.assertEqual(agent.resolve_region(source, "src/p.js", "", "other"), (8, 10, "regex"))
        self.assertEqual(agent.import_block(source, "src/p.js"), "const db = require('./db');")

    def test_go_receiver_method_and_multiline_signature(self):
        source = ("package x\n\nimport (\n\t\"context\"\n)\n\nfunc (s *Store) FreeRooms(ctx context.Context,\n"
                  "\tfrom string) ([]Room, error) {\n\treturn nil, nil\n}\n")
        self.assertEqual(agent.resolve_region(source, "a.go", "Store", "FreeRooms"), (7, 10, "regex"))
        self.assertEqual(agent.import_block(source, "a.go"), "package x\n\nimport (\n\t\"context\"\n)")

    def test_ruby_nested_modules_end_by_indentation(self):
        source = ("require 'pg'\n\nmodule Studioboard\n  module Reports\n    class DayBoard\n"
                  "      def self.rows(day:)\n        if day\n          1\n        end\n      end\n"
                  "    end\n  end\nend\n")
        self.assertEqual(agent.resolve_region(source, "d.rb", "Studioboard::Reports::DayBoard", "rows"),
                         (6, 10, "regex"))

    def test_unknown_language_is_whole(self):
        self.assertEqual(agent.resolve_region("fn rows() {}", "a.rs", "", "rows")[2], "whole")

    def test_symbol_that_is_really_a_path_is_not_a_symbol(self):
        self.assertEqual(agent.named_symbol("specifically `src/paths.js`"), ("", "", ""))
        self.assertEqual(agent.named_symbol("specifically `Owner.name()`"),
                         ("Owner.name", "Owner", "name"))

    def test_excerpt_carries_imports_and_the_verbatim_definition(self):
        root = tempfile.mkdtemp(prefix="scope-")
        try:
            touch(root, "m.py", PY_SOURCE)
            scope = quiet(agent.parse_scope)("Edit only `m.py`, specifically `B.rows()`.", FakeTree(root))
            self.assertEqual(scope["files"], ["m.py"])
            self.assertEqual(scope["kind"], "ast")
            self.assertIn("import os\nfrom x import y", scope["excerpt"])
            self.assertIn("    def rows(self):\n        return 3", scope["excerpt"])
            self.assertNotIn("return 4", scope["excerpt"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tree.revert_outside
# ---------------------------------------------------------------------------

class RevertOutside(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="tree-")
        touch(self.root, "src/allowed.py", "a = 1\n")
        touch(self.root, "src/other.py", "b = 1\n")
        touch(self.root, "package.json", "{}\n")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _edit_both_and_add_one(self, tree):
        tree.write("src/allowed.py", "a = 2\n")
        tree.write("src/other.py", "b = 2\n")
        tree.write("src/new.py", "c = 1\n")

    def test_only_paths_outside_the_scope_go_back(self):
        tree = quiet(agent.Tree)(self.root)
        self._edit_both_and_add_one(tree)
        undone = quiet(tree.revert_outside)(["src/allowed.py"])
        self.assertEqual(sorted(undone), ["src/new.py", "src/other.py"])
        self.assertEqual(tree.read("src/allowed.py"), "a = 2\n")
        self.assertEqual(tree.read("src/other.py"), "b = 1\n")
        self.assertFalse(os.path.exists(os.path.join(self.root, "src/new.py")))

    def test_originals_come_from_the_pristine_copy_when_memory_has_none(self):
        tree = quiet(agent.Tree)(self.root)
        tree.originals.clear()
        tree.unkept = set(tree.index)
        self.assertIsNotNone(quiet(tree.make_pristine)())
        # Edits from outside the tree's own write path, as a shell would.
        touch(self.root, "src/allowed.py", "a = 2\n")
        touch(self.root, "src/other.py", "b = 2\n")
        quiet(tree.revert_outside)(["src/allowed.py"])
        self.assertEqual(tree.read("src/other.py"), "b = 1\n")
        self.assertEqual(tree.read("src/allowed.py"), "a = 2\n")

    def test_no_scope_means_nothing_is_reverted(self):
        tree = quiet(agent.Tree)(self.root)
        self._edit_both_and_add_one(tree)
        self.assertEqual(quiet(tree.revert_outside)([]), [])
        self.assertEqual(tree.read("src/other.py"), "b = 2\n")


# ---------------------------------------------------------------------------
# Kit: scope gate and restore_file
# ---------------------------------------------------------------------------

class KitScope(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="kit-")
        touch(self.root, "src/allowed.py", "a = 1\n")
        touch(self.root, "src/other.py", "b = 1\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pool = agent.ShellPool(self.root)
        self.kit = agent.Kit(self.tree, self.pool, agent.Allowance(),
                             scope={"files": ["src/allowed.py"]})

    def tearDown(self):
        self.pool.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_edit_outside_the_scope_is_refused(self):
        with self.assertRaises(agent.ToolFault):
            self.kit.do_edit({"path": "src/other.py", "old": "b = 1", "new": "b = 2"})
        with self.assertRaises(agent.ToolFault):
            self.kit.do_create_file({"path": "src/new.py", "content": "c = 1\n"})
        self.assertEqual(self.tree.read("src/other.py"), "b = 1\n")

    def test_edit_inside_the_scope_goes_through(self):
        self.kit.do_edit({"path": "src/allowed.py", "old": "a = 1", "new": "a = 2"})
        self.assertEqual(self.tree.read("src/allowed.py"), "a = 2\n")

    def test_restore_file_puts_a_file_back(self):
        self.tree.write("src/other.py", "b = 9\n")
        reply = self.kit.do_restore_file({"path": "src/other.py"})
        self.assertIn("back exactly", reply)
        self.assertEqual(self.tree.read("src/other.py"), "b = 1\n")
        self.assertIn("unchanged", self.kit.do_restore_file({"path": "src/other.py"}))


# ---------------------------------------------------------------------------
# Warden: scope faults and the named checks
# ---------------------------------------------------------------------------

class WardenGate(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="warden-")
        touch(self.root, "src/allowed.py", "a = 1\n")
        touch(self.root, "src/other.py", "b = 1\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pool = agent.ShellPool(self.root)

    def tearDown(self):
        self.pool.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def warden(self, checks):
        return agent.Warden(self.tree, self.pool, agent.Allowance(), "",
                            {"files": ["src/allowed.py"], "checks": checks})

    def test_red_check_is_a_fault_carrying_command_and_output(self):
        warden = self.warden(["python3 -c \"print('boom'); raise SystemExit(3)\""])
        self.tree.write("src/allowed.py", "a = 2\n")
        faults = quiet(warden.verdict)()
        self.assertEqual(len(faults), 1)
        self.assertIn("exit 3", faults[0])
        self.assertIn("boom", faults[0])
        self.assertEqual(warden.refusals, 1)

    def test_clean_checks_pass_and_are_not_rerun_on_the_same_tree(self):
        marker = os.path.join(tempfile.mkdtemp(prefix="marker-"), "ran.txt")
        warden = self.warden(["python3 -c \"open(%r, 'a').write('x')\"" % marker])
        self.tree.write("src/allowed.py", "a = 2\n")
        self.assertEqual(quiet(warden.verdict)(), [])
        self.assertEqual(slurp(marker), "x")
        self.assertEqual(quiet(warden.verdict)(), [])
        self.assertEqual(slurp(marker), "x", "the check ran again on an unchanged tree")
        self.tree.write("src/allowed.py", "a = 3\n")
        self.assertEqual(quiet(warden.verdict)(), [])
        self.assertEqual(slurp(marker), "xx")

    def test_edit_outside_scope_is_refused_before_any_check_runs(self):
        marker = os.path.join(tempfile.mkdtemp(prefix="marker-"), "ran.txt")
        warden = self.warden(["python3 -c \"open(%r, 'a').write('x')\"" % marker])
        self.tree.write("src/other.py", "b = 2\n")
        faults = quiet(warden.verdict)()
        self.assertEqual(len(faults), 1)
        self.assertIn("src/other.py", faults[0])
        self.assertFalse(os.path.exists(marker))


# ---------------------------------------------------------------------------
# The driver loop: polling is not a repeat, and money runs out gracefully
# ---------------------------------------------------------------------------

def call(name: str, **args) -> dict:
    return {"id": "c%d" % id(args), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedSeat:
    """Stands in for agent.Seat: hands out scripted replies in order."""
    script: list = []

    def __init__(self, allowance, models=None, patient=True):
        self.allowance = allowance

    def current(self):
        return "openai/gpt-5.6-luna"

    def retire(self, model):
        return False

    def ask(self, messages, tools=None):
        if not ScriptedSeat.script:
            raise agent.Spent("script exhausted")
        calls = ScriptedSeat.script.pop(0)
        self.allowance.charge(self.current(), {"prompt_tokens": 1000, "completion_tokens": 50})
        return {"content": "", "tool_calls": calls, "_usage": {"completion_tokens": 50}}


class DriverLoop(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="drive-")
        touch(self.root, "src/q.py", "def q():\n    return 1\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pool = agent.ShellPool(self.root)
        self.allowance = agent.Allowance()
        self.scope = {"files": ["src/q.py"], "checks": []}
        self.warden = agent.Warden(self.tree, self.pool, self.allowance, "", self.scope)

    def tearDown(self):
        self.pool.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def run_drive(self, script):
        ScriptedSeat.script = list(script)
        with mock.patch.object(agent, "Seat", ScriptedSeat):
            with mock.patch("sys.stdout", new=io.StringIO()) as out:
                try:
                    agent.drive("stmt", self.tree, self.pool, self.allowance,
                                {"targets": []}, "", self.warden, self.scope)
                except agent.Finished:
                    return "finished", out.getvalue()
                return "returned", out.getvalue()

    def test_repeated_polls_do_not_end_the_run(self):
        script = [
            [call("edit", path="src/q.py", old="return 1", new="return 2")],
            [call("bash", command="sleep 2; echo done", background=True)],
        ] + [[call("bash_poll", job="job1", wait=1)]] * 6 + [[call("submit", summary="ok")]]
        outcome, log = self.run_drive(script)
        self.assertEqual(outcome, "finished", log)
        self.assertNotIn("identical replies", log)

    def test_identical_non_poll_replies_still_end_the_run(self):
        script = [[call("read_file", path="src/q.py")]] * 5
        outcome, log = self.run_drive(script)
        self.assertEqual(outcome, "returned")
        self.assertIn("identical replies", log)

    def test_little_money_left_wraps_up_instead_of_stopping(self):
        self.allowance.spent = self.allowance.soft_usd - 0.002
        script = [
            [call("edit", path="src/q.py", old="return 1", new="return 2")],
            [call("submit", summary="ok")],
        ]
        outcome, log = self.run_drive(script)
        self.assertEqual(outcome, "finished", log)
        self.assertIn("wrapping up", log)


# ---------------------------------------------------------------------------
# Seat: the request asks for a quoted cost and books it
# ---------------------------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SeatCost(unittest.TestCase):
    def test_usage_include_is_requested_and_quoted_cost_is_booked(self):
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["body"] = json.loads(request.data.decode("utf-8"))
            reply = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0123}}
            return FakeResponse(json.dumps(reply).encode("utf-8"))

        allowance = agent.Allowance()
        seat = agent.Seat(allowance, models=["openai/gpt-5.6-luna"], patient=False)
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            with mock.patch("sys.stdout", new=io.StringIO()) as out:
                seat.ask([{"role": "user", "content": "x"}])
        self.assertEqual(seen["body"].get("usage"), {"include": True})
        self.assertAlmostEqual(allowance.spent, 0.0123)
        self.assertEqual(allowance.billed, 1)
        self.assertIn("est=$", out.getvalue())


# ---------------------------------------------------------------------------
# Edits are byte-exact: line endings and foreign bytes survive
# ---------------------------------------------------------------------------

def git_check(root: str, patch: str) -> tuple:
    import subprocess
    path = os.path.join(tempfile.mkdtemp(prefix="p-"), "p.diff")
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
        fh.write(patch)
    done = subprocess.run(["git", "apply", "--check", path], cwd=root,
                          capture_output=True, text=True)
    return done.returncode, done.stderr


class ByteEdits(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="bytes-")
        os.makedirs(os.path.join(self.root, "src"))
        with open(os.path.join(self.root, "src/crlf.py"), "wb") as fh:
            fh.write(b"x = 1\r\ny = 2\r\nz = 3\r\n")
        with open(os.path.join(self.root, "src/latin.py"), "wb") as fh:
            fh.write(b"# caf\xe9\nq = 1\nw = 2\n\n\ndef f():\n    return q\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pool = agent.ShellPool(self.root)
        self.kit = agent.Kit(self.tree, self.pool, agent.Allowance())

    def tearDown(self):
        self.pool.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def raw(self, rel: str) -> bytes:
        with open(os.path.join(self.root, rel), "rb") as fh:
            return fh.read()

    def hunk_lines(self, patch: str) -> list:
        return [l for l in patch.split("\n")
                if l[:1] in ("+", "-") and not l.startswith(("---", "+++"))]

    def test_crlf_file_keeps_its_line_endings(self):
        self.kit.do_edit({"path": "src/crlf.py", "old": "y = 2\n", "new": "y = 20\n"})
        self.assertEqual(self.raw("src/crlf.py"), b"x = 1\r\ny = 20\r\nz = 3\r\n")
        patch = quiet(self.tree.diff)()
        self.assertEqual(len(self.hunk_lines(patch)), 2, patch)

    def test_foreign_byte_survives_an_edit_elsewhere(self):
        self.kit.do_edit({"path": "src/latin.py", "old": "q = 1", "new": "q = 10"})
        self.assertEqual(self.raw("src/latin.py"),
                         b"# caf\xe9\nq = 10\nw = 2\n\n\ndef f():\n    return q\n")
        patch = quiet(self.tree.diff)()
        self.assertEqual(len(self.hunk_lines(patch)), 2, patch)

    def test_create_file_round_trips_non_ascii(self):
        self.kit.do_create_file({"path": "src/new.py", "content": "s = 'café'\n"})
        self.assertEqual(self.raw("src/new.py"), "s = 'café'\n".encode("utf-8"))

    def test_both_patches_apply_with_git(self):
        self.kit.do_edit({"path": "src/crlf.py", "old": "y = 2", "new": "y = 20"})
        self.kit.do_edit({"path": "src/latin.py", "old": "q = 1", "new": "q = 10"})
        patch = quiet(self.tree.diff)()
        quiet(self.tree.restore)()
        self.assertEqual(self.raw("src/crlf.py"), b"x = 1\r\ny = 2\r\nz = 3\r\n")
        code, err = git_check(self.root, patch)
        self.assertEqual(code, 0, err)

    def test_compile_and_outline_accept_foreign_bytes(self):
        note = self.kit._compile_check("src/latin.py")
        self.assertEqual(note, "")
        rows = quiet(self.kit.do_outline)({"path": "src/latin.py"})
        self.assertIn("def f", rows)


# ---------------------------------------------------------------------------
# Tool calls: dict arguments and missing ids
# ---------------------------------------------------------------------------

class ToolCalls(unittest.TestCase):
    def test_dict_arguments_are_accepted(self):
        name, args, err = agent.extract_tool_call(
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}})
        self.assertEqual((name, args, err), ("read_file", {"path": "a.py"}, ""))

    def test_missing_id_is_made_up_and_arguments_serialised(self):
        kept = agent.recorded_calls(
            [{"function": {"name": "submit", "arguments": {"summary": "x"}}},
             {"id": "abc", "function": {"name": "read_file", "arguments": "{}"}}],
            turn=7)
        self.assertEqual(kept[0]["id"], "call_7_1")
        self.assertEqual(json.loads(kept[0]["function"]["arguments"]), {"summary": "x"})
        self.assertEqual(kept[1]["id"], "abc")


class DriverToolIds(DriverLoop):
    def test_calls_without_ids_still_finish_and_echo_one_id(self):
        seen = []

        class Capturing(ScriptedSeat):
            def ask(self, messages, tools=None):
                seen.append([dict(m) for m in messages])
                return super().ask(messages, tools)

        edit = call("edit", path="src/q.py", old="return 1", new="return 2")
        del edit["id"]
        ScriptedSeat.script = [[edit], [call("submit", summary="ok")]]
        with mock.patch.object(agent, "Seat", Capturing):
            with mock.patch("sys.stdout", new=io.StringIO()):
                with self.assertRaises(agent.Finished):
                    agent.drive("stmt", self.tree, self.pool, self.allowance,
                                {"targets": []}, "", self.warden, self.scope)
        second = seen[1]
        assistant = [m for m in second if m["role"] == "assistant"][-1]
        tool = [m for m in second if m["role"] == "tool"][-1]
        self.assertTrue(assistant["tool_calls"][0]["id"])
        self.assertEqual(tool["tool_call_id"], assistant["tool_calls"][0]["id"])


# ---------------------------------------------------------------------------
# submit never hands in an empty tree
# ---------------------------------------------------------------------------

class SubmitGate(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="submit-")
        touch(self.root, "src/q.py", "x = 1\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pool = agent.ShellPool(self.root)
        self.allowance = agent.Allowance()
        warden = agent.Warden(self.tree, self.pool, self.allowance, "", {})
        self.kit = agent.Kit(self.tree, self.pool, self.allowance, warden=warden)

    def tearDown(self):
        self.pool.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_empty_tree_is_refused_even_at_the_end_of_the_run(self):
        self.allowance.deadline = __import__("time").time() + 60
        reply = quiet(self.kit.do_submit)({"summary": "done"})
        self.assertIn("Not handed in", reply)

    def test_unreadable_tree_is_refused(self):
        with mock.patch.object(self.tree, "has_changes", return_value=None):
            reply = quiet(self.kit.do_submit)({"summary": "done"})
        self.assertIn("could not read the tree", reply)

    def test_a_real_change_goes_through(self):
        self.tree.write("src/q.py", "x = 2\n")
        with self.assertRaises(agent.Finished):
            quiet(self.kit.do_submit)({"summary": "done"})


# ---------------------------------------------------------------------------
# An incomplete patch says so
# ---------------------------------------------------------------------------

class PatchStatus(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="status-")
        touch(self.root, "src/q.py", "x = 1\n")
        touch(self.root, "package.json", "{}\n")
        with open(os.path.join(self.root, "src/blob.bin"), "wb") as fh:
            fh.write(b"\x00\x01old")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_binary_change_is_reported_as_dropped(self):
        tree = quiet(agent.Tree)(self.root)
        with open(os.path.join(self.root, "src/blob.bin"), "wb") as fh:
            fh.write(b"\x00\x01new")
        tree.write("src/q.py", "x = 2\n")
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            patch = tree.diff()
        self.assertIn("src/q.py", patch)
        self.assertEqual(len(tree.dropped), 1)
        self.assertIn("INCOMPLETE", out.getvalue())

    def test_run_report_says_partial(self):
        def fake_drive(statement, tree, pool, allowance, located, plan, warden, scope=None):
            with open(os.path.join(self.root, "src/blob.bin"), "wb") as fh:
                fh.write(b"\x00\x01new")
            tree.write("src/q.py", "x = 2\n")
            raise agent.Finished("done")

        with mock.patch.dict(os.environ, {"RIDGES_WORKDIR": self.root}):
            with mock.patch.object(agent, "run_locator",
                                   return_value={"targets": [], "note": ""}):
                with mock.patch.object(agent, "run_planner", return_value=""):
                    with mock.patch.object(agent, "drive", fake_drive):
                        with mock.patch("sys.stdout", new=io.StringIO()) as out:
                            patch = agent.agent_main({"problem_statement": "Change q."})
        self.assertIn("x = 2", patch)
        self.assertIn("usable=partial", out.getvalue())
        self.assertIn("INCOMPLETE", out.getvalue())


# ---------------------------------------------------------------------------
# Snapshot and pristine copy stay bounded
# ---------------------------------------------------------------------------

class Snapshot(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="snap-")
        touch(self.root, "small.py", "x = 1\n")
        with open(os.path.join(self.root, "big.dat"), "wb") as fh:
            fh.write(b"a" * (3 * 1024 * 1024))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_big_file_is_hashed_but_not_held(self):
        tree = quiet(agent.Tree)(self.root)
        self.assertIn("big.dat", tree.unkept)
        self.assertNotIn("big.dat", tree.originals)
        self.assertTrue(tree.digests["big.dat"])
        self.assertIn("small.py", tree.originals)

    def test_file_over_the_hash_ceiling_is_tracked_by_size_and_mtime(self):
        with mock.patch.object(agent, "SNAPSHOT_HASH_CEILING", 1024):
            tree = quiet(agent.Tree)(self.root)
        self.assertEqual(tree.digests["big.dat"], "")
        self.assertEqual(tree.changed_paths(), [])
        with open(os.path.join(self.root, "big.dat"), "ab") as fh:
            fh.write(b"b")
        self.assertEqual(tree.changed_paths(), [("big.dat", "modified")])

    def test_pristine_copy_over_budget_leaves_nothing_behind(self):
        tree = quiet(agent.Tree)(self.root)
        with mock.patch.object(agent.time, "monotonic", side_effect=[0.0] + [5.0] * 8):
            result = quiet(tree.make_pristine)(budget=1.0)
        self.assertIsNone(result)
        self.assertIsNone(tree.pristine_dir)

    def test_pristine_copy_holds_only_unheld_files(self):
        tree = quiet(agent.Tree)(self.root)
        where = quiet(tree.make_pristine)()
        self.assertTrue(os.path.isfile(os.path.join(where, "big.dat")))
        self.assertFalse(os.path.exists(os.path.join(where, "small.py")))
        self.assertEqual(tree.original("big.dat"), b"a" * (3 * 1024 * 1024))


# ---------------------------------------------------------------------------
# Scope resolution does not guess between top-level dirs
# ---------------------------------------------------------------------------

class ScopeResolve(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="resolve-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_docs_copy_loses_to_the_package(self):
        touch(self.root, "docs/ipam/x.py")
        touch(self.root, "netbox/ipam/x.py")
        self.assertEqual(agent._resolve_named("ipam/x.py", self.root), "netbox/ipam/x.py")

    def test_project_marker_breaks_a_tie(self):
        touch(self.root, "alpha/ipam/x.py")
        touch(self.root, "beta/ipam/x.py")
        touch(self.root, "beta/manage.py")
        self.assertEqual(agent._resolve_named("ipam/x.py", self.root), "beta/ipam/x.py")

    def test_two_plain_candidates_resolve_to_nothing(self):
        touch(self.root, "alpha/ipam/x.py")
        touch(self.root, "beta/ipam/x.py")
        self.assertEqual(quiet(agent._resolve_named)("ipam/x.py", self.root), "")

    def test_single_candidate_and_direct_hit_still_work(self):
        touch(self.root, "netbox/ipam/x.py")
        self.assertEqual(agent._resolve_named("ipam/x.py", self.root), "netbox/ipam/x.py")
        self.assertEqual(agent._resolve_named("netbox/ipam/x.py", self.root), "netbox/ipam/x.py")


# ---------------------------------------------------------------------------
# The clock is monotonic, anchored to the harness, and shared by the tail
# ---------------------------------------------------------------------------

class Clock(unittest.TestCase):
    def test_a_wall_clock_step_does_not_move_the_deadline(self):
        allowance = agent.Allowance()
        before = allowance.clock_left()
        with mock.patch.object(agent.time, "time", return_value=agent.time.time() - 3600):
            after = allowance.clock_left()
        self.assertAlmostEqual(before, after, delta=1.0)

    def test_shift_moves_both_deadlines_earlier_and_ignores_a_negative_lead(self):
        allowance = agent.Allowance()
        soft, hard = allowance.deadline, allowance.hard_deadline
        allowance.shift(40.0)
        self.assertAlmostEqual(allowance.deadline, soft - 40.0)
        self.assertAlmostEqual(allowance.hard_deadline, hard - 40.0)
        allowance.shift(-5.0)
        self.assertAlmostEqual(allowance.deadline, soft - 40.0)

    def test_anchor_file_age_is_the_lead_within_reason(self):
        folder = tempfile.mkdtemp(prefix="anchor-")
        anchor = os.path.join(folder, "instruction.md")
        touch(folder, "instruction.md", "do it")
        now = os.stat(anchor).st_mtime
        os.utime(anchor, (now - 50, now - 50))
        self.assertAlmostEqual(agent.harness_lead(1500.0, anchor), 50.0, delta=2.0)
        os.utime(anchor, (now - 5000, now - 5000))     # stale
        self.assertEqual(agent.harness_lead(1500.0, anchor), 0.0)
        os.utime(anchor, (now + 100, now + 100))       # from the future
        self.assertEqual(agent.harness_lead(1500.0, anchor), 0.0)
        self.assertEqual(agent.harness_lead(1500.0, anchor + ".missing"), 0.0)
        shutil.rmtree(folder, ignore_errors=True)

    def test_hard_deadline_sits_inside_the_wall(self):
        with mock.patch.dict(os.environ, {"AGENT_TIMEOUT": "600"}):
            allowance = agent.Allowance()
        self.assertAlmostEqual(allowance.hard_left(),
                               600 - agent.HARD_TAIL_RESERVE_SEC, delta=1.0)
        self.assertLess(allowance.clock_left(), allowance.hard_left())

    def test_tail_budgets_stay_inside_the_hard_deadline(self):
        allowance = agent.Allowance()
        allowance.hard_deadline = agent.time.monotonic() + 20.0
        steps = [(0.25, 30.0), (0.5, 60.0), (0.6, 60.0), (0.5, 30.0), (0.9, 45.0)]
        total = 0.0
        for share, cap in steps:
            budget = agent.tail_budget(allowance, share, cap)
            total += budget
            allowance.hard_deadline -= budget      # as if the step used it all
        self.assertLessEqual(total, 20.0 + 3.0 * len(steps))
        allowance.hard_deadline = agent.time.monotonic() - 100.0
        self.assertEqual(agent.tail_budget(allowance, 0.5, 60.0), 3.0)


# ---------------------------------------------------------------------------
# Seat: retries grow, abandoned calls cost, dead routes are dropped,
# and a dead network is waited out rather than called "budget"
# ---------------------------------------------------------------------------

def ok_reply(request=None, timeout=None):
    reply = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    return FakeResponse(json.dumps(reply).encode("utf-8"))


class SeatRetry(unittest.TestCase):
    def seat(self, patient=True, bases=None):
        allowance = agent.Allowance()
        seat = agent.Seat(allowance, models=["openai/gpt-5.6-luna"], patient=patient)
        seat.bases = list(bases or ["http://proxy/api/v1"])
        return allowance, seat

    def test_retry_timeouts_grow_and_abandoned_calls_are_charged(self):
        seen = []

        def timing_out(request, timeout=None):
            seen.append(timeout)
            raise urllib.error.URLError(TimeoutError("slow"))

        allowance, seat = self.seat()
        with mock.patch.object(urllib.request, "urlopen", timing_out), \
                mock.patch.object(agent.time, "sleep"), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            with self.assertRaises(agent.SeatTimedOut):
                seat._attempt("openai/gpt-5.6-luna",
                              [{"role": "user", "content": "x" * 3500}], None)
        self.assertEqual(len(seen), agent.REQUEST_ATTEMPTS)
        self.assertEqual(seen, sorted(seen))
        self.assertGreater(seen[1], seen[0])
        self.assertGreater(allowance.spent, 0.0)
        self.assertEqual(allowance.calls, agent.REQUEST_ATTEMPTS)
        self.assertIn("charged", out.getvalue())

    def test_a_404_base_is_dropped(self):
        def not_found(request, timeout=None):
            if request.full_url.startswith("http://proxy/v1/"):
                raise urllib.error.HTTPError(request.full_url, 404, "nf", {},
                                             io.BytesIO(b"no such route"))
            return ok_reply()

        allowance, seat = self.seat(bases=["http://proxy/v1", "http://proxy/api/v1"])
        with mock.patch.object(urllib.request, "urlopen", not_found), \
                mock.patch("sys.stdout", new=io.StringIO()):
            seat.ask([{"role": "user", "content": "x"}])
        self.assertEqual(seat.bases, ["http://proxy/api/v1"])

    def test_connection_failure_is_waited_out_then_retried(self):
        calls = {"n": 0}

        def flaky(request, timeout=None):
            calls["n"] += 1
            if calls["n"] <= agent.REQUEST_ATTEMPTS:
                raise urllib.error.URLError(ConnectionRefusedError("down"))
            return ok_reply()

        allowance, seat = self.seat()
        naps = []
        with mock.patch.object(urllib.request, "urlopen", flaky), \
                mock.patch.object(agent.time, "sleep", naps.append), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            reply = seat.ask([{"role": "user", "content": "x"}])
        self.assertEqual(reply.get("content"), "hi")
        self.assertIn("unreachable", out.getvalue())
        self.assertIn(agent.SEAT_REFUSED_WAIT_SEC, naps)
        self.assertEqual(seat.models, ["openai/gpt-5.6-luna"], "the model was benched")

    def test_impatient_seat_raises_unreachable(self):
        def down(request, timeout=None):
            raise urllib.error.URLError(ConnectionRefusedError("down"))

        allowance, seat = self.seat(patient=False)
        with mock.patch.object(urllib.request, "urlopen", down), \
                mock.patch("sys.stdout", new=io.StringIO()):
            with self.assertRaises(agent.SeatUnreachable):
                seat.ask([{"role": "user", "content": "x"}])


class CostSync(unittest.TestCase):
    @staticmethod
    def reporting(total):
        def usage(url, timeout=None):
            return FakeResponse(json.dumps({"total_cost_usd": total}).encode("utf-8"))
        return usage

    def test_proxy_total_raises_local_spend_but_never_lowers_it(self):
        allowance = agent.Allowance()
        allowance.spent = 0.05
        with mock.patch.object(urllib.request, "urlopen", self.reporting(0.08)), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            self.assertTrue(allowance.sync("http://proxy"))
        self.assertAlmostEqual(allowance.spent, 0.08)
        self.assertIn("[COST]", out.getvalue())
        with mock.patch.object(urllib.request, "urlopen", self.reporting(0.02)):
            self.assertFalse(allowance.sync("http://proxy"))
        self.assertAlmostEqual(allowance.spent, 0.08)
        self.assertAlmostEqual(allowance.synced_usd, 0.02)

    def test_failing_endpoint_is_asked_once_and_leaves_spend_alone(self):
        asked = {"n": 0}

        def down(url, timeout=None):
            asked["n"] += 1
            raise urllib.error.URLError("down")

        allowance = agent.Allowance()
        allowance.spent = 0.05
        with mock.patch.object(urllib.request, "urlopen", down), \
                mock.patch("sys.stdout", new=io.StringIO()):
            self.assertFalse(allowance.sync("http://proxy"))
            self.assertFalse(allowance.sync("http://proxy"))
        self.assertEqual(asked["n"], 1)
        self.assertAlmostEqual(allowance.spent, 0.05)

    def test_no_proxy_means_no_request(self):
        allowance = agent.Allowance()
        with mock.patch.dict(os.environ, {"SANDBOX_PROXY_URL": ""}):
            with mock.patch.object(urllib.request, "urlopen",
                                   side_effect=AssertionError("asked")):
                self.assertFalse(allowance.sync())


# ---------------------------------------------------------------------------
# The diff of a big file is bounded and still applies with git
# ---------------------------------------------------------------------------

class BigDiff(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="bigdiff-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self, name, text):
        touch(self.root, name, text)
        tree = quiet(agent.Tree)(self.root)
        pristine = tempfile.mkdtemp(prefix="pristine-")
        shutil.copytree(self.root, pristine, dirs_exist_ok=True)
        return tree, pristine

    def test_one_line_edit_in_a_long_file_is_one_small_hunk(self):
        lines = ["line %d\n" % n for n in range(30000)]
        tree, pristine = self.build("big.txt", "".join(lines))
        lines[20000] = "line twenty thousand\n"
        touch(self.root, "big.txt", "".join(lines))
        patch = quiet(tree.diff)()
        self.assertIn("@@ -19998,7 +19998,7 @@", patch)
        self.assertLess(len(patch.splitlines()), 20)
        code, err = git_check(pristine, patch)
        self.assertEqual(code, 0, err)

    def test_scattered_change_over_the_ceiling_is_a_whole_file_hunk(self):
        lines = ["line %d\n" % n for n in range(12000)]
        tree, pristine = self.build("big.txt", "".join(lines))
        changed = [("changed %d\n" % n if n % 2 else line) for n, line in enumerate(lines)]
        touch(self.root, "big.txt", "".join(changed))
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            patch = tree.diff()
        self.assertIn("whole-file hunk", out.getvalue())
        self.assertIn("@@ -1,12000 +1,12000 @@", patch)
        code, err = git_check(pristine, patch)
        self.assertEqual(code, 0, err)

    def test_whole_file_hunk_keeps_a_missing_final_newline(self):
        a = ["a%d\n" % n for n in range(3)]
        b = ["b0\n", "b1\n", "b2"]
        tree, pristine = self.build("f.txt", "".join(a))
        with mock.patch.object(agent, "DIFF_LINES_CEILING", 1):
            section = quiet(agent.file_diff)("f.txt", "modified", "".join(a).encode(),
                                             "".join(b).encode(), "100644", "100644")
        self.assertIn("\\ No newline at end of file", section)
        code, err = git_check(pristine, section)
        self.assertEqual(code, 0, err)

    def test_edits_at_both_ends_still_apply(self):
        lines = ["line %d\n" % n for n in range(500)]
        tree, pristine = self.build("f.txt", "".join(lines))
        lines[0] = "first\n"
        lines[-1] = "last\n"
        lines.insert(250, "middle\n")
        touch(self.root, "f.txt", "".join(lines))
        patch = quiet(tree.diff)()
        code, err = git_check(pristine, patch)
        self.assertEqual(code, 0, err)


# ---------------------------------------------------------------------------
# Stages stop on their clock share; the shell can be stopped; the Warden
# walks once and forgets a skipped check; the driver lets it stand down
# ---------------------------------------------------------------------------

class StageClock(unittest.TestCase):
    def test_locator_stops_once_its_clock_share_is_used(self):
        root = tempfile.mkdtemp(prefix="stage-")
        touch(root, "src/q.py", "def q():\n    return 1\n")
        tree = quiet(agent.Tree)(root)
        pool = agent.ShellPool(root)
        allowance = agent.Allowance()
        allowance.shift(allowance.run_length() * agent.LOCATOR_CLOCK_SHARE + 1)
        ScriptedSeat.script = [[call("read_file", path="src/q.py")]] * 3
        with mock.patch.object(agent, "Seat", ScriptedSeat), \
                mock.patch("sys.stdout", new=io.StringIO()) as out:
            agent.run_locator("find q", tree, pool, allowance, agent.Beacon("locator"))
        self.assertIn("stopped=clock", out.getvalue())
        self.assertEqual(len(ScriptedSeat.script), 3, "a call was made past the share")
        pool.close()
        shutil.rmtree(root, ignore_errors=True)


class ShellStop(unittest.TestCase):
    def test_bash_poll_stop_kills_the_job_and_frees_the_slot(self):
        root = tempfile.mkdtemp(prefix="stop-")
        tree = quiet(agent.Tree)(root)
        pool = agent.ShellPool(root)
        kit = agent.Kit(tree, pool, agent.Allowance())
        try:
            with mock.patch("sys.stdout", new=io.StringIO()):
                kit.do_bash({"command": "echo started; sleep 60", "background": True})
                kit.do_bash({"command": "sleep 60", "background": True})
                with self.assertRaises(agent.ToolFault) as refused:
                    kit.do_bash({"command": "echo third", "background": True})
                self.assertIn("stop=true", str(refused.exception))
                first = min(pool.jobs, key=lambda n: int(n[3:]))
                agent.time.sleep(0.3)
                served = kit.do_bash_poll({"job": first, "stop": True})
                self.assertIn("started", served)
                self.assertIn("stopped", served)
                self.assertNotIn(first, pool.jobs)
                self.assertIn("third", kit.do_bash({"command": "echo third"}))
        finally:
            pool.close()
            shutil.rmtree(root, ignore_errors=True)


class WardenOnce(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="wonce-")
        touch(self.root, "src/allowed.py", "a = 1\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pool = agent.ShellPool(self.root)

    def tearDown(self):
        self.pool.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def warden(self, checks):
        return agent.Warden(self.tree, self.pool, agent.Allowance(), "",
                            {"files": ["src/allowed.py"], "checks": checks})

    def test_one_tree_walk_per_verdict(self):
        warden = self.warden(["true"])
        self.tree.write("src/allowed.py", "a = 2\n")
        with mock.patch.object(self.tree, "changed_paths",
                               wraps=self.tree.changed_paths) as walks:
            self.assertEqual(quiet(warden.verdict)(), [])
        self.assertEqual(walks.call_count, 1)

    def test_a_skipped_check_is_not_recorded_as_passed(self):
        warden = self.warden(["sleep 30"])
        self.tree.write("src/allowed.py", "a = 2\n")
        with mock.patch.object(agent, "CHECK_SEC", 1.0):
            self.assertEqual(quiet(warden.verdict)(), [])
        self.assertIsNone(warden.passed_key)


class DriverRefusals(DriverLoop):
    def test_warden_refusals_do_not_end_the_loop_before_it_stands_down(self):
        self.scope["checks"] = ["false"]
        self.warden = agent.Warden(self.tree, self.pool, self.allowance, "", self.scope)
        script = [[call("edit", path="src/q.py", old="return 1", new="return 2")]] + \
                 [[call("submit", summary="try %d" % n)] for n in range(4)]
        outcome, log = self.run_drive(script)
        self.assertEqual(outcome, "finished", log)
        self.assertNotIn("submit refusals", log)
        self.assertEqual(self.warden.refusals, agent.WARDEN_REFUSALS_MAX)


class Rounds(unittest.TestCase):
    def test_growth_is_priced_fresh_once(self):
        allowance = agent.Allowance()
        model = "openai/gpt-5.6-luna"
        history = [{"growth": 4000, "reply_tokens": 1200}] * 5
        rounds = agent.rounds_left(allowance, model, 100_000, history)
        with mock.patch.dict(agent.MODEL_PRICING,
                             {model: (0.0, agent.MODEL_PRICING[model][1])}):
            free_input = agent.rounds_left(allowance, model, 100_000, history)
        self.assertLess(rounds, free_input)


# ---------------------------------------------------------------------------
# The patch must rebuild the edited tree, not merely apply
# ---------------------------------------------------------------------------

LOSSY = ("diff --git a/src/q.py b/src/q.py\n--- a/src/q.py\n+++ b/src/q.py\n"
         "@@ -1 +1 @@\n-x = 1\n+x = 2\n")


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rt-")
        touch(self.root, "src/q.py", "x = 1\n")
        touch(self.root, "src/gone.py", "gone = 1\n")
        self.tree = quiet(agent.Tree)(self.root)
        self.pristine = tempfile.mkdtemp(prefix="rt-pristine-")
        shutil.copytree(self.root, self.pristine, dirs_exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.pristine, ignore_errors=True)

    def test_a_normal_edit_round_trips(self):
        self.tree.write("src/q.py", "x = 3\n")
        os.remove(os.path.join(self.root, "src/gone.py"))
        touch(self.root, "src/new.py", "fresh = 1\n")
        edited = self.tree.fingerprint()
        self.assertEqual(edited["src/gone.py"], None)
        patch = quiet(self.tree.diff)()
        self.assertEqual(quiet(self.tree.round_trip)(patch, edited), [])

    def test_a_lossy_patch_is_caught_and_rebuilt_whole(self):
        self.tree.write("src/q.py", "x = 3\n")
        edited = self.tree.fingerprint()
        self.assertEqual(git_check(self.pristine, LOSSY)[0], 0, "the lossy patch must apply")
        self.assertEqual(quiet(self.tree.round_trip)(LOSSY, edited), ["src/q.py"])
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            rebuilt = self.tree.diff(whole=["src/q.py"])
        self.assertIn("asked for", out.getvalue())
        self.assertIn("@@ -1 +1 @@", rebuilt)
        self.assertEqual(quiet(self.tree.round_trip)(rebuilt, edited), [])
        self.assertEqual(git_check(self.pristine, rebuilt)[0], 0)

    def test_without_git_the_check_is_skipped(self):
        self.tree.write("src/q.py", "x = 3\n")
        edited = self.tree.fingerprint()
        patch = quiet(self.tree.diff)()
        with mock.patch.object(agent.shutil, "which", return_value=None):
            with mock.patch("sys.stdout", new=io.StringIO()) as out:
                self.assertEqual(self.tree.round_trip(patch, edited), [])
        self.assertIn("skipped", out.getvalue())

    def test_agent_main_reports_the_round_trip(self):
        def fake_drive(statement, tree, pool, allowance, located, plan, warden, scope=None):
            tree.write("src/q.py", "x = 2\n")
            raise agent.Finished("done")

        touch(self.root, "package.json", "{}\n")
        with mock.patch.dict(os.environ, {"RIDGES_WORKDIR": self.root}):
            with mock.patch.object(agent, "run_locator",
                                   return_value={"targets": [], "note": ""}):
                with mock.patch.object(agent, "run_planner", return_value=""):
                    with mock.patch.object(agent, "drive", fake_drive):
                        with mock.patch("sys.stdout", new=io.StringIO()) as out:
                            patch = agent.agent_main({"problem_statement": "Change q."})
        self.assertIn("x = 2", patch)
        self.assertIn("[PATCH] round trip: 1 file(s) reproduced", out.getvalue())
        self.assertIn("usable=yes", out.getvalue())
        self.assertEqual(slurp(os.path.join(self.root, "src/q.py")), "x = 1\n")


# ---------------------------------------------------------------------------
# Memory and CPU: the container's limits shape the children
# ---------------------------------------------------------------------------

GB = 1024 * 1024 * 1024


def fake_cgroup(files: dict):
    def read(path):
        if path in files:
            return files[path]
        raise OSError(path)
    return read


class MemoryGuard(unittest.TestCase):
    def test_cgroup_v2_v1_and_unlimited_reads(self):
        v2 = {"/sys/fs/cgroup/memory.max": str(4 * GB),
              "/sys/fs/cgroup/memory.current": str(2 * GB),
              "/sys/fs/cgroup/cpu.max": "200000 100000"}
        with mock.patch.object(agent, "_read_text", fake_cgroup(v2)):
            self.assertEqual(agent.memory_limit_bytes(), 4 * GB)
            self.assertAlmostEqual(agent.memory_share(), 0.5)
            self.assertEqual(agent.cpu_quota(), 2)
        v1 = {"/sys/fs/cgroup/memory/memory.limit_in_bytes": str(2 * GB),
              "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(GB // 2),
              "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "50000",
              "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}
        with mock.patch.object(agent, "_read_text", fake_cgroup(v1)):
            self.assertEqual(agent.memory_limit_bytes(), 2 * GB)
            self.assertAlmostEqual(agent.memory_share(), 0.25)
            self.assertEqual(agent.cpu_quota(), 1)
        with mock.patch.object(agent, "_read_text", fake_cgroup({"/sys/fs/cgroup/memory.max": "max"})):
            self.assertEqual(agent.memory_limit_bytes(), 0)
            self.assertEqual(agent.memory_share(), -1.0)
            self.assertGreaterEqual(agent.cpu_quota(), 1)

    def test_child_cap_arithmetic(self):
        self.assertEqual(agent.child_cap(4 * GB, 2), (4 * GB - agent.MEMORY_HEADROOM_BYTES) // 2)
        self.assertEqual(agent.child_cap(2 * GB, 1), 2 * GB - agent.MEMORY_HEADROOM_BYTES)
        self.assertEqual(agent.child_cap(GB, 2), agent.CHILD_CAP_FLOOR_BYTES)
        self.assertEqual(agent.child_cap(0, 2), agent.CHILD_MEMORY_BYTES)
        self.assertEqual(agent.child_cap(64 * GB, 1), agent.CHILD_MEMORY_BYTES)

    def test_pool_backs_off_under_memory_pressure(self):
        root = tempfile.mkdtemp(prefix="mem-")
        with mock.patch.object(agent, "memory_limit_bytes", return_value=8 * GB):
            pool = agent.ShellPool(root)
        self.assertEqual(pool.jobs_allowed, agent.BACKGROUND_JOBS_MAX)
        try:
            with mock.patch.object(agent, "memory_share", return_value=0.8):
                with self.assertRaises(agent.ToolFault) as refused:
                    pool.start("sleep 30", background=True)
                self.assertIn("80%", str(refused.exception))
                pool.start("true")                     # foreground is still served
            with mock.patch.object(agent, "memory_share", return_value=0.1):
                pool.start("sleep 30", background=True)
            with mock.patch.object(agent, "memory_share", return_value=0.5):
                with self.assertRaises(agent.ToolFault) as refused:
                    pool.start("sleep 30", background=True)
                self.assertIn("50%", str(refused.exception))
            with mock.patch.object(agent, "memory_share", return_value=0.1):
                pool.start("sleep 30", background=True)
                self.assertEqual(len([j for j in pool.jobs.values() if not j.finished()]), 2)
        finally:
            pool.close()
            shutil.rmtree(root, ignore_errors=True)

    def test_small_container_allows_one_job(self):
        with mock.patch.object(agent, "memory_limit_bytes", return_value=2 * GB):
            pool = agent.ShellPool(tempfile.mkdtemp(prefix="small-"))
        self.assertEqual(pool.jobs_allowed, 1)
        self.assertEqual(pool.child_cap, 2 * GB - agent.MEMORY_HEADROOM_BYTES)

    def test_child_is_niced_and_volunteers_for_the_oom_killer(self):
        root = tempfile.mkdtemp(prefix="hook-")
        job = agent.Shell("sleep 5", root, cap=agent.CHILD_MEMORY_BYTES, cpus=3)
        try:
            pid = job.process.pid
            self.assertEqual(slurp("/proc/%d/oom_score_adj" % pid).strip(), str(agent.CHILD_OOM_SCORE_ADJ))
            mine = os.getpriority(os.PRIO_PROCESS, 0)
            self.assertEqual(os.getpriority(os.PRIO_PROCESS, pid), min(19, mine + agent.CHILD_NICE))
        finally:
            job.stop()
            shutil.rmtree(root, ignore_errors=True)

    def test_child_env_follows_the_cpu_count(self):
        root = tempfile.mkdtemp(prefix="env-")
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("GOMAXPROCS", "MAKEFLAGS", "MAKEOPTS", "CARGO_BUILD_JOBS",
                              "npm_config_jobs", "BUNDLE_JOBS")}
        with mock.patch.dict(os.environ, clean, clear=True):
            job = agent.Shell("env", root, cpus=3)
            done, out = job.wait(10)
        job.stop()
        shutil.rmtree(root, ignore_errors=True)
        self.assertTrue(done)
        self.assertIn("GOMAXPROCS=3", out)
        self.assertIn("MAKEFLAGS=-j3", out)


if __name__ == "__main__":
    unittest.main()
