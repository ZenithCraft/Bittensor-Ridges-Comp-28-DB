"""Unit tests for agent.py's parsing, editing, verification and patching."""
import json, os, re, sys, tempfile, textwrap, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import agent


def make_repo(files: dict[str, str]) -> agent.Repository:
    root = Path(tempfile.mkdtemp())
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return agent.Repository(root)


class TestJSONExtraction(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(agent.extract_json('```json\n{"action": "edit"}\n```')["action"], "edit")

    def test_bare_with_prose(self):
        content = 'Here is my fix:\n{"action": "edit", "edits": []}\nHope that helps.'
        self.assertEqual(agent.extract_json(content)["action"], "edit")

    def test_braces_inside_strings(self):
        content = '{"action": "edit", "edits": [{"path": "a.py", "replace": "d = {\\"k\\": 1}"}]}'
        self.assertEqual(agent.extract_json(content)["edits"][0]["replace"], 'd = {"k": 1}')

    def test_no_json(self):
        self.assertIsNone(agent.extract_json("I could not determine the fix."))


class TestApplyEdits(unittest.TestCase):
    def setUp(self):
        self.repo = make_repo({"q.py": "def run():\n    return Model.objects.all()\n"})

    def test_exact(self):
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "q.py", "search": "Model.objects.all()", "replace": "Model.objects.only('id')"}])
        self.assertEqual((changed, errors), (["q.py"], []))
        self.assertIn("only('id')", self.repo.read("q.py"))

    def test_trailing_whitespace_tolerated(self):
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "q.py", "search": "def run():   \n    return Model.objects.all()",
             "replace": "def run():\n    return Model.objects.none()"}])
        self.assertEqual(errors, [])
        self.assertIn("none()", self.repo.read("q.py"))

    def test_missing_search_reports_error(self):
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "q.py", "search": "nonexistent", "replace": "x"}])
        self.assertEqual(changed, [])
        self.assertIn("not found", errors[0])

    def test_ambiguous_search_rejected(self):
        repo = make_repo({"d.py": "x = 1\nx = 1\n"})
        changed, errors = agent.apply_edits(repo, [{"path": "d.py", "search": "x = 1", "replace": "x = 2"}])
        self.assertEqual(changed, [])
        self.assertIn("appears 2 times", errors[0])

    def test_path_traversal_refused(self):
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "../escape.py", "new_file": True, "content": "x"}])
        self.assertEqual(changed, [])
        self.assertIn("outside the repository", errors[0])

    def test_new_file(self):
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "migrations/0002_index.py", "new_file": True, "content": "# migration\n"}])
        self.assertEqual((changed, errors), (["migrations/0002_index.py"], []))

    def test_revert_restores_and_removes(self):
        agent.apply_edits(self.repo, [{"path": "q.py", "search": "all()", "replace": "none()"}])
        agent.apply_edits(self.repo, [{"path": "new.py", "new_file": True, "content": "x\n"}])
        self.repo.revert_all()
        self.assertIn("all()", (self.repo.root / "q.py").read_text())
        self.assertFalse((self.repo.root / "new.py").exists())


class TestPatch(unittest.TestCase):
    def test_modify_and_apply(self):
        repo = make_repo({"a/b.py": "one\ntwo\nthree\n"})
        repo.write("a/b.py", "one\nTWO\nthree\n")
        patch = agent.build_patch(repo, ["a/b.py"])
        self.assertTrue(patch.startswith("diff --git a/a/b.py b/a/b.py"))
        applies, detail = agent.verify_patch_applies(patch, repo)
        self.assertTrue(applies, detail)

    def test_new_file_patch_applies(self):
        repo = make_repo({"keep.py": "x\n"})
        repo.write("made.py", "brand new\n")
        patch = agent.build_patch(repo, ["made.py"])
        self.assertIn("new file mode", patch)
        self.assertTrue(agent.verify_patch_applies(patch, repo)[0], patch)

    def test_missing_trailing_newline(self):
        repo = make_repo({"c.py": "alpha\nbeta"})
        repo.write("c.py", "alpha\ngamma")
        patch = agent.build_patch(repo, ["c.py"])
        self.assertIn("\\ No newline at end of file", patch)
        self.assertTrue(agent.verify_patch_applies(patch, repo)[0], patch)

    def test_unchanged_file_yields_no_patch(self):
        repo = make_repo({"d.py": "same\n"})
        repo.write("d.py", "same\n")
        self.assertEqual(agent.build_patch(repo, ["d.py"]), "")


SOURCE = '''\
from django.db.models import Count

class Manager:
    def annotate_things(self):
        return self.annotate(total=Count("thing"))

    def other(self):
        return self
'''


class TestVerifier(unittest.TestCase):
    def build(self, statement: str):
        repo = make_repo({"m.py": SOURCE})
        instruction = agent.parse_instruction(statement, repo.root)
        return repo, agent.Checker(repo, instruction), instruction

    def test_scope_rejects_stray_file(self):
        repo, checker, _ = self.build("You may edit only `m.py`.")
        result = checker.check_scope(["m.py", "other.py"])
        self.assertFalse(result.passed)
        self.assertIn("other.py", result.detail)

    def test_syntax_failure_detected(self):
        repo, checker, _ = self.build("Fix it.")
        repo.write("m.py", "def broken(:\n")
        self.assertFalse(checker.check_syntax(["m.py"]).passed)

    def test_single_method_allows_in_body_edit(self):
        repo, checker, ins = self.build(
            "Limit changes to `m.py`, specifically `Manager.annotate_things()`. "
            "Keep its signature and the rest of the file unchanged, including imports.")
        self.assertTrue(ins.single_method)
        repo.write("m.py", SOURCE.replace('Count("thing")', 'Count("thing", distinct=True)'))
        self.assertTrue(checker.check_single_method(["m.py"]).passed)

    def test_single_method_rejects_import_edit(self):
        repo, checker, _ = self.build(
            "Limit changes to `m.py`, specifically `Manager.annotate_things()`. "
            "Keep its signature and the rest of the file unchanged, including imports.")
        repo.write("m.py", SOURCE.replace("from django.db.models import Count",
                                          "from django.db.models import Count, Q"))
        result = checker.check_single_method(["m.py"])
        self.assertFalse(result.passed)
        self.assertIn("imports", result.detail)

    def test_style_constraint_rejects_comprehension(self):
        repo, checker, ins = self.build(
            "Limit changes to `m.py`, specifically `Manager.annotate_things()`. Write the method "
            "as plain ORM expressions: no Python loops, comprehensions, lambdas, exception "
            "handling, or context managers inside it.")
        self.assertIn("comprehensions", ins.style_constraints)
        repo.write("m.py", SOURCE.replace(
            'return self.annotate(total=Count("thing"))',
            'return self.annotate(total=sum([x for x in range(3)]))'))
        result = checker.check_style(["m.py"])
        self.assertFalse(result.passed)
        self.assertIn("ListComp", result.detail)

    def test_style_constraint_passes_clean_orm(self):
        repo, checker, _ = self.build(
            "Limit changes to `m.py`, specifically `Manager.annotate_things()`. Write the method as "
            "plain ORM expressions: no Python loops, comprehensions, lambdas, exception handling, "
            "or context managers inside it.")
        repo.write("m.py", SOURCE.replace('Count("thing")', 'Count("thing", distinct=True)'))
        self.assertTrue(checker.check_style(["m.py"]).passed)


class TestContextRequests(unittest.TestCase):
    def test_grep_and_read(self):
        repo = make_repo({"x.py": "line one\nSELECT * FROM t\nline three\n"})
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.targets, probe.notes = [], []
        answer = agent.fulfil_requests(repo, probe, [
            {"kind": "grep", "pattern": "SELECT"},
            {"kind": "read_file", "path": "x.py", "start": 1, "end": 2},
        ])
        self.assertIn("x.py:2", answer)
        self.assertIn("line one", answer)

    def test_destructive_sql_refused(self):
        repo = make_repo({"x.py": "x\n"})
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.targets, probe.notes = [], []
        answer = agent.fulfil_requests(repo, probe, [{"kind": "sql", "query": "DROP TABLE users"}])
        self.assertIn("refused", answer)


class TestRuntimeLoading(unittest.TestCase):
    """The Harbor runtime execs agent.py without registering it in sys.modules.

    Reproduces ridges_harbor/ridges_miner_runtime.py:_load_agent_module exactly.
    A module-level construct that resolves its own module (dataclasses' KW_ONLY
    probe under string annotations) crashes at import there but not under a
    normal `import agent`, so only this shape of test catches it.
    """

    def test_loads_the_way_the_runtime_does(self):
        import importlib.util
        path = Path(__file__).parent / "agent.py"
        spec = importlib.util.spec_from_file_location("ridges_miner_agent", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.pop("ridges_miner_agent", None)
        try:
            spec.loader.exec_module(module)          # no sys.modules registration
        except Exception as exc:
            self.fail(f"agent.py fails to import under the Harbor runtime loader: {exc!r}")
        self.assertTrue(callable(getattr(module, "agent_main", None)),
                        "agent_main must exist after loading")

class TestUnverifiedNotClean(unittest.TestCase):
    """A patch nothing executed must never report as verified.

    `all([])` is True, so before this an instruction whose test commands we
    failed to parse produced zero checks -- and the agent shipped an untested
    patch claiming "all checks passed". That is what broke prefix-hierarchy.
    """

    def make(self, checks):
        return agent.Candidate(patch="diff --git a/x b/x\n", checks=checks, diagnosis="d")

    def test_no_checks_at_all_is_not_clean(self):
        self.assertFalse(self.make([]).clean)

    def test_static_checks_only_is_not_clean(self):
        checks = [agent.CheckResult("syntax", True, "parsed"),
                  agent.CheckResult("scope", True, "ok")]
        candidate = self.make(checks)
        self.assertFalse(candidate.verified, "nothing ran against the database")
        self.assertFalse(candidate.clean)

    def test_a_passing_task_command_makes_it_clean(self):
        checks = [agent.CheckResult("syntax", True, "parsed"),
                  agent.CheckResult("$ python manage.py test x", True, "OK")]
        self.assertTrue(self.make(checks).clean)

    def test_a_skipped_task_command_does_not_count(self):
        # `verified=False` is what the skip site now sets. It used to be
        # inferred by looking for "skipped" in the detail text, which meant the
        # signal lived in prose and only covered this one check.
        checks = [agent.CheckResult("$ python manage.py test x", True, "skipped: out of time",
                                    verified=False)]
        self.assertFalse(self.make(checks).clean)

    def test_a_check_that_could_not_run_blocks_clean(self):
        # The failure this exists for: a query-count probe printed nothing,
        # `query scaling` reported success, clean went true, and the agent
        # returned after one attempt with three quarters of its budget and two
        # repair rounds unspent -- on a patch one redundant statement away from
        # passing. Nothing failed; something simply was not measured.
        checks = [agent.CheckResult("syntax", True, "parsed"),
                  agent.CheckResult("$ python manage.py test x", True, "OK"),
                  agent.CheckResult("query scaling", True, "the probe produced no query count",
                                    verified=False)]
        candidate = self.make(checks)
        self.assertTrue(candidate.verified, "the tests did run")
        self.assertEqual([c.name for c in candidate.unmeasured], ["query scaling"])
        self.assertFalse(candidate.clean, "an unmeasured check must not read as done")

    def test_everything_measured_is_clean(self):
        checks = [agent.CheckResult("syntax", True, "parsed"),
                  agent.CheckResult("$ python manage.py test x", True, "OK"),
                  agent.CheckResult("query scaling", True, "bounded: 3 at N=1, 3 at N=10")]
        self.assertTrue(self.make(checks).clean)

    def test_a_failing_task_command_is_not_clean(self):
        checks = [agent.CheckResult("$ python manage.py test x", False, "AssertionError")]
        self.assertFalse(self.make(checks).clean)


class TestVerifierContract(unittest.TestCase):
    """Mirror the grader's structural audit of the edited method.

    Any of these makes the whole problem score zero however correct the SQL is,
    so they must be caught before the patch is returned -- not discovered from
    a checker we cannot see.
    """

    SRC = ('from django.db.models import Count\n\n'
           'class M:\n'
           '    def annotate_things(self):\n'
           '        return self.annotate(total=Count("t"))\n')

    def build(self, body):
        repo = make_repo({"m.py": self.SRC})
        ins = agent.parse_instruction(
            "Limit changes to `m.py`, specifically `M.annotate_things()`. "
            "Keep its signature and the rest of the file unchanged.", repo.root)
        self.assertTrue(ins.single_method)
        repo.write("m.py", self.SRC.replace('        return self.annotate(total=Count("t"))\n', body))
        return agent.Checker(repo, ins).check_method_contract(["m.py"])

    def test_clean_edit_passes(self):
        self.assertTrue(self.build('        return self.annotate(total=Count("t", distinct=True))\n').passed)

    def test_import_inside_method_rejected(self):
        r = self.build('        from django.db.models import Q\n        return self.annotate(x=Q())\n')
        self.assertFalse(r.passed); self.assertIn("import inside the method", r.detail)

    def test_lambda_rejected(self):
        r = self.build('        f = lambda x: x\n        return self.annotate(total=f(1))\n')
        self.assertFalse(r.passed); self.assertIn("Lambda", r.detail)

    def test_nested_function_rejected(self):
        r = self.build('        def helper():\n            return 1\n        return self.annotate(total=helper())\n')
        self.assertFalse(r.passed); self.assertIn("nested function", r.detail)

    def test_getattr_rejected(self):
        r = self.build('        return self.annotate(total=getattr(self, "x"))\n')
        self.assertFalse(r.passed); self.assertIn("getattr", r.detail)

    def test_dunder_attribute_rejected(self):
        r = self.build('        return self.annotate(total=self.__class__)\n')
        self.assertFalse(r.passed); self.assertIn("__class__", r.detail)

    def test_while_loop_rejected(self):
        r = self.build('        while True:\n            break\n        return self\n')
        self.assertFalse(r.passed); self.assertIn("While", r.detail)


class TestVerifierContractInheritsOriginal(unittest.TestCase):
    """Only constructs the edit INTRODUCES are violations.

    vlangroup-utilization's method begins with `from .models import VLAN` in the
    original. Flagging it rejected the correct one-token fix, drove three repair
    rounds, and shipped a patch that removed count_related and failed ruff.
    """

    SRC = ('from django.db.models import Count, F, Value\n'
           'from utilities.query import count_related\n\n'
           'class Q:\n'
           '    def annotate_utilization(self):\n'
           '        from .models import VLAN\n\n'
           '        return self.annotate(\n'
           '            vlan_count=count_related(VLAN, "group"),\n'
           '            utilization=F("vlan_count") * 100 / F("total"),\n'
           '        )\n')

    def check(self, new_src):
        repo = make_repo({"q.py": self.SRC})
        ins = agent.parse_instruction(
            "Limit changes to `q.py`, specifically `Q.annotate_utilization()`. "
            "Keep its signature and the rest of the file unchanged.", repo.root)
        repo.write("q.py", new_src)
        return agent.Checker(repo, ins).check_method_contract(["q.py"])

    def test_pre_existing_local_import_is_not_a_violation(self):
        fixed = self.SRC.replace('* 100 /', '* 100.0 /')          # the gold fix
        r = self.check(fixed)
        self.assertTrue(r.passed, r.detail)

    def test_newly_added_local_import_is_a_violation(self):
        added = self.SRC.replace('        from .models import VLAN\n',
                                 '        from .models import VLAN\n        from django.db.models import Q\n')
        r = self.check(added)
        self.assertFalse(r.passed)
        self.assertIn("import inside the method", r.detail)
        self.assertIn("django.db.models", r.detail)
        self.assertNotIn(".models import VLAN", r.detail, "the pre-existing import must not be reported")


class TestContextCompaction(unittest.TestCase):
    """Consumed investigation payloads must shrink; the newest turn must not."""

    def test_old_payloads_trimmed_newest_kept(self):
        big = "# Requested context\n\n" + "x" * 5000
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": big},
                {"role": "assistant", "content": "{}"},
                {"role": "user", "content": big}]
        agent.Solver._compact_old_context(msgs)
        self.assertLess(len(msgs[1]["content"]), 1000, "older payload should be trimmed")
        self.assertIn("trimmed", msgs[1]["content"])
        self.assertEqual(len(msgs[3]["content"]), len(big), "newest turn must be untouched")

    def test_small_payloads_untouched(self):
        small = "# Requested context\n\nshort"
        msgs = [{"role": "user", "content": small}, {"role": "user", "content": "later"}]
        agent.Solver._compact_old_context(msgs)
        self.assertEqual(msgs[0]["content"], small)


class TestAdoptConstraints(unittest.TestCase):
    """The model's reading of the edit boundary is used only when ours is empty."""

    def solver(self, statement):
        repo = make_repo({"app/q.py": "def f():\n    return 1\n",
                          "app/other.py": "x = 1\n"})
        ins = agent.parse_instruction(statement, repo.root)
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.targets, probe.notes = [], []
        llm = agent.LLM.__new__(agent.LLM)
        return agent.Solver(repo, ins, probe, llm), ins

    def test_adopts_existing_files_when_static_found_nothing(self):
        solver, ins = self.solver("Make the slow thing fast.")
        self.assertEqual(ins.edit_only, [])
        solver._adopt_constraints({"editable_files": ["app/q.py", "app/missing.py", "../etc/passwd"]})
        self.assertEqual(ins.edit_only, ["app/q.py"], "only real, in-repo paths are accepted")

    def test_ignored_when_static_parsing_found_something(self):
        solver, ins = self.solver("You may edit only `app/other.py`. Make it fast.")
        self.assertEqual(ins.edit_only, ["app/other.py"])
        solver._adopt_constraints({"editable_files": ["app/q.py"]})
        self.assertEqual(ins.edit_only, ["app/other.py"], "deterministic parsing must win")

    def test_method_bound_needs_corroboration(self):
        solver, ins = self.solver("Make the slow thing fast.")
        solver._adopt_constraints({"bounded_to_method": "Q.f"})
        self.assertFalse(ins.single_method, "instruction never mentions f: bound must be refused")
        solver2, ins2 = self.solver("The f() helper is slow; fix it.")
        solver2._adopt_constraints({"bounded_to_method": "Q.f"})
        self.assertTrue(ins2.single_method)
        self.assertEqual(ins2.method_hint, "f")

    def test_garbage_is_ignored(self):
        solver, ins = self.solver("Make it fast.")
        for bad in (None, "x", 42, [], {"editable_files": "notalist"}):
            solver._adopt_constraints(bad)
        self.assertEqual(ins.edit_only, [])
        self.assertFalse(ins.single_method)


class TestInstructionText(unittest.TestCase):
    """Where the problem statement comes from, and in what order."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "app"
        self.root.mkdir()
        self.harness = Path(self.tmp.name) / "installed-agent" / "instruction.md"
        self.harness.parent.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_payload_wins_over_every_file(self):
        self.harness.write_text("harness")
        (self.root / "instruction.md").write_text("repo")
        got = agent._instruction_text({"problem_statement": "payload"}, self.root, self.harness)
        self.assertEqual(got, "payload")

    def test_harness_copy_beats_repo_local_file(self):
        self.harness.write_text("harness")
        (self.root / "instruction.md").write_text("repo docs, not the task")
        got = agent._instruction_text({}, self.root, self.harness)
        self.assertEqual(got, "harness")

    def test_repo_local_file_is_the_last_resort(self):
        (self.root / "instruction.md").write_text("repo")
        got = agent._instruction_text({}, self.root, self.harness)
        self.assertEqual(got, "repo")

    def test_directory_named_instruction_md_is_skipped(self):
        (self.root / "instruction.md").mkdir()
        with self.assertRaises(agent.MissingInstruction):
            agent._instruction_text({}, self.root, self.harness)

    def test_blank_payload_value_falls_through(self):
        self.harness.write_text("harness")
        got = agent._instruction_text({"problem_statement": "   "}, self.root, self.harness)
        self.assertEqual(got, "harness")


class TestDatabaseProbe(unittest.TestCase):
    """Discovery must find the connection wherever the app keeps it, and must
    never hand the model a dead candidate."""

    def make(self, files: dict[str, str], env: dict[str, str] | None = None, engine="unknown"):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        saved = dict(os.environ)
        os.environ.pop("RIDGES_PROBE_NO_PING", None)
        for k in [k for k in os.environ if k.endswith(("_URL", "HOST", "PORT", "USER", "PASSWORD", "DB", "DATABASE"))]:
            os.environ.pop(k, None)
        os.environ.update(env or {}); os.environ["RIDGES_PROBE_NO_PING"] = "1"
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        repo = agent.Repository(root)
        ins = agent.Instruction(text=""); ins.engine = engine
        return agent.DatabaseProbe(repo, ins)

    def test_dsn_literal_in_source_with_driver_scheme(self):
        probe = self.make({"app/db.py":
            'DEFAULT_URL = "postgresql+psycopg://wayfare_app:wayfare-dev-8b3f@postgres:5432/wayfare_dev"\n'
            'def url():\n    return os.environ.get("WAYFARE_DATABASE_URL", DEFAULT_URL)\n'})
        t = probe.targets[0]
        self.assertEqual((t.engine, t.user, t.host, t.port, t.database),
                         ("postgresql", "wayfare_app", "postgres", "5432", "wayfare_dev"))
        self.assertTrue(t.dsn.startswith("postgresql://"), "psql rejects postgresql+psycopg://")

    def test_env_default_arguments_in_go_style_client(self):
        probe = self.make({"internal/chclient/client.go":
            'package chclient\n// talks to ClickHouse over HTTP\n'
            'func New() *Client { return &Client{\n'
            '  Host: envOr("DRIFTWOOD_CLICKHOUSE_HOST", "clickhouse"),\n'
            '  Port: envOr("DRIFTWOOD_CLICKHOUSE_PORT", "8123"),\n'
            '  User: envOr("DRIFTWOOD_CLICKHOUSE_USER", "retention_app"),\n'
            '  Password: envOr("DRIFTWOOD_CLICKHOUSE_PASSWORD", "retention-dev"),\n'
            '  Database: envOr("DRIFTWOOD_CLICKHOUSE_DATABASE", "retention")}}\n'})
        t = probe.targets[0]
        self.assertEqual((t.engine, t.user, t.host, t.port, t.database),
                         ("clickhouse", "retention_app", "clickhouse", "8123", "retention"))

    def test_docstring_placeholder_does_not_become_the_database(self):
        probe = self.make({"app/db.py":
            '"""Statements use the ``{name:Type}`` placeholder syntax."""\n'
            'DEFAULTS = {"host": "clickhouse", "port": "8123", "user": "gw_app", '
            '"password": "gw-dev", "database": "gatewatch"}\n'})
        self.assertEqual(probe.targets[0].database, "gatewatch")

    def test_prefixed_env_url_and_family(self):
        probe = self.make({"x.py": "print(1)"},
                          env={"WELCOME_DATABASE_URL": "postgres://w:pw@dbhost:5433/welcome_dev"})
        t = probe.targets[0]
        self.assertEqual((t.engine, t.host, t.port, t.database, t.priority),
                         ("postgresql", "dbhost", "5433", "welcome_dev", 1))
        probe = self.make({"x.py": "print(1)"},
                          env={"BEACON_CLICKHOUSE_HOST": "ch1", "BEACON_CLICKHOUSE_USER": "u",
                               "BEACON_CLICKHOUSE_PASSWORD": "p", "BEACON_CLICKHOUSE_DATABASE": "beacon"})
        t = probe.targets[0]
        self.assertEqual((t.engine, t.host, t.port, t.user, t.database), ("clickhouse", "ch1", "8123", "u", "beacon"))

    def test_unrelated_host_families_are_ignored(self):
        probe = self.make({"x.py": "print(1)"}, env={"SMTP_HOST": "mail", "REDIS_HOST": "redis"})
        self.assertEqual(probe.targets, [])

    def test_dotenv_file_is_read_and_expanded(self):
        probe = self.make({".env": 'PGHOST=${DB_HOST_OVERRIDE:-pgbox}\nPGUSER=app\nPGPASSWORD=s\nPGDATABASE=appdb\n',
                           "x.py": "print(1)"})
        t = probe.targets[0]
        self.assertEqual((t.engine, t.host, t.user, t.database), ("postgresql", "pgbox", "app", "appdb"))

    def test_dead_candidates_are_dropped_by_ping(self):
        probe = self.make({"app/db.py": 'URL = "postgres://a:b@example-host:5432/app"\n'
                                        'URL2 = "postgres://a:b@postgres:5432/app"\n'})
        self.assertEqual(len(probe.targets), 2)
        os.environ.pop("RIDGES_PROBE_NO_PING")
        probe.sql = lambda query, target=None, timeout=60: "?column?\n1\n(1 row)" if target.host == "postgres" else "could not connect"
        probe.targets.sort(key=lambda t: t.priority)
        alive = [t for t in probe.targets if probe._ping(t)]
        self.assertEqual([t.host for t in alive], ["postgres"])

    def test_app_python_prefers_the_apps_venv(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".venv/bin").mkdir(parents=True); (root / ".venv/bin/python").write_text("")
        self.assertEqual(agent.app_python(root), str(root / ".venv/bin/python"))
        self.assertEqual(agent.app_python(Path(tmp.name) / "nowhere"), agent.sys.executable)


class TestConsolidatedUpgrades(unittest.TestCase):
    """The 21 small changes agreed on 2026-09-05, each pinned by one case."""

    def test_edit_only_window_stops_at_the_sentence(self):
        root = Path(tempfile.mkdtemp())
        for rel in ("app/q.py", "tests/test_q.py"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True); (root / rel).write_text("x = 1\n")
        ins = agent.parse_instruction("Limit production changes to `app/q.py`. Do not change "
                                      "`tests/test_q.py` or fixtures.", root)
        self.assertEqual(ins.edit_only, ["app/q.py"])

    def test_fenced_block_becomes_one_script_without_lint_lines(self):
        text = ("Run these checks before finishing:\n\n```bash\nexport APP_DB=test\n"
                "python -m pytest tests/test_x.py -q\nruff check --no-cache app/q.py\n```\n")
        ins = agent.parse_instruction(text, Path(tempfile.mkdtemp()))
        self.assertEqual(ins.commands[0], "set -e\nexport APP_DB=test\npython -m pytest tests/test_x.py -q")
        self.assertIn("ruff check --no-cache app/q.py", ins.commands)
        self.assertEqual(ins.lint_paths, ["app/q.py"])
        single = agent.parse_instruction("```bash\npython -m pytest -q\n```", Path(tempfile.mkdtemp()))
        self.assertEqual(single.commands, ["python -m pytest -q"])

    def test_numeric_targets_are_extracted(self):
        ins = agent.parse_instruction("The operation must issue at most 3 queries regardless of size, "
                                      "and read no more than 50,000 rows.", Path(tempfile.mkdtemp()))
        self.assertEqual(ins.targets["max_queries"], 3)
        self.assertEqual(ins.targets["max_read_rows"], 50000)
        self.assertEqual(agent.parse_instruction("Fix the query.", Path(tempfile.mkdtemp())).targets, {})

    def test_crlf_file_round_trips_through_repository_and_patch(self):
        root = Path(tempfile.mkdtemp())
        (root / "w.py").write_bytes(b"one\r\ntwo\r\nthree\r\n")
        repo = agent.Repository(root)
        self.assertEqual(repo.read("w.py"), "one\r\ntwo\r\nthree\r\n")
        repo.write("w.py", "one\r\nTWO\r\nthree\r\n")
        patch = agent.build_patch(repo, ["w.py"])
        removed = [l for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")]
        self.assertEqual(len(removed), 1, "only the changed line may appear in the diff")
        self.assertTrue(agent.verify_patch_applies(patch, repo)[0], patch)
        repo.revert_all()
        self.assertEqual((root / "w.py").read_bytes(), b"one\r\ntwo\r\nthree\r\n")

    def test_syntax_gate_catches_compile_time_errors(self):
        repo = make_repo({"m.py": "def f():\n    return g(a=1, a=2)\n"})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        result = checker.check_syntax(["m.py"])
        self.assertFalse(result.passed)
        self.assertIn("repeated", result.detail)

    def test_single_method_rejects_change_below_the_method(self):
        repo = make_repo({"m.py": SOURCE})
        ins = agent.parse_instruction("Limit changes to `m.py`, specifically `Manager.annotate_things()`. "
                                      "Keep its signature and the rest of the file unchanged.", repo.root)
        repo.write("m.py", SOURCE.replace('Count("thing")', 'Count("thing", distinct=True)') + "\nEXTRA = 1\n")
        result = agent.Checker(repo, ins).check_single_method(["m.py"])
        self.assertFalse(result.passed)

    def test_verified_candidate_outranks_unverified(self):
        ok = agent.CheckResult
        unverified = agent.Candidate("p", [ok("scope", True, ""), ok("syntax", True, ""), ok("ruff", True, "")], "")
        verified = agent.Candidate("p", [ok("scope", True, ""), ok("$ pytest", True, "3 passed"),
                                         ok("ruff", False, "E501")], "")
        self.assertGreater(verified.score, unverified.score)

    def test_duplicate_request_is_answered_as_already_shown(self):
        repo = make_repo({"a.py": "x = 1\n"})
        served: set[str] = set()
        first = agent.fulfil_requests(repo, None, [{"kind": "read_file", "path": "a.py"}], served=served)
        second = agent.fulfil_requests(repo, None, [{"kind": "read_file", "path": "a.py"}], served=served)
        self.assertIn("x = 1", first)
        self.assertIn("already shown", second)
        self.assertNotIn("x = 1", second)

    def test_read_file_by_symbol_returns_the_definition(self):
        repo = make_repo({"m.py": SOURCE})
        out = agent.fulfil_requests(repo, None, [{"kind": "read_file", "path": "m.py", "symbol": "other"}])
        self.assertIn("def other", out)
        self.assertNotIn("annotate_things", out.split("def other")[0].split("\n")[-3:])
        missing = agent.fulfil_requests(repo, None, [{"kind": "read_file", "path": "m.py", "symbol": "nope"}])
        self.assertIn("no definition named", missing)

    def test_schema_request_uses_the_probe(self):
        class P:
            def schema_for(self, names, limit=8):
                return "columns: " + ",".join(names)
        out = agent.fulfil_requests(make_repo({"a.py": ""}), P(), [{"kind": "schema", "tables": ["t1", "t2"]}])
        self.assertIn("columns: t1,t2", out)

    def test_discovered_commands_for_a_node_project(self):
        repo = make_repo({"package.json": '{"scripts": {"test": "node --test"}}', "src/q.js": "x"})
        self.assertEqual(agent.Checker(repo, agent.Instruction(text="")).discovered_commands(["src/q.js"]),
                         ["npm test --silent"])

    def test_cached_and_reasoning_tokens_are_reported(self):
        llm = agent.LLM()
        llm._usage_unavailable = True
        llm._record_usage("m", {"prompt_tokens": 100, "completion_tokens": 40, "cost": 0.001,
                                "prompt_tokens_details": {"cached_tokens": 80},
                                "completion_tokens_details": {"reasoning_tokens": 30}})
        self.assertIn("100 prompt (80 cached)", llm.report())
        self.assertIn("40 completion (30 reasoning)", llm.report())

    def test_query_scaling_honours_an_absolute_limit(self):
        repo = make_repo({"manage.py": "", "a.py": ""})
        ins = agent.Instruction(text=""); ins.targets = {"max_queries": 2}
        checker = agent.Checker(repo, ins)
        # Save and restore rather than delete-and-reload. importlib.reload
        # rebinds agent.MODELS to a fresh dict, while routing_lab still holds
        # the old one -- so running this file in the same process as
        # test_routing made the selector see an empty roster and return None.
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(a, 0, 'RIDGES_QC{"1": 3, "10": 3}\n')
        try:
            result = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertFalse(result.passed)
        self.assertIn("at most 2", result.detail)


class TestAgentCarriesNoRouting(unittest.TestCase):
    """Routing was measured, found unproven, and moved to routing_lab.

    If it comes back it must come back whole and switched on deliberately, not
    as dead computation a graded run pays for and no miner can read.
    """

    def test_the_agent_neither_imports_nor_defines_routing(self):
        self.assertNotIn("import routing_lab", Path(agent.__file__).read_text())
        for name in ("ROUTING_MODE", "ROUTING_POLICY", "MODEL_CAPABILITIES",
                     "select_initial_model", "profile_task", "TaskProfile"):
            self.assertFalse(hasattr(agent, name), f"agent still defines {name}")

    def test_the_ladder_is_the_only_thing_choosing_models(self):
        repo = make_repo({"a.py": "x = 1\n"})
        solver = agent.Solver(repo, agent.parse_instruction("Fix `a.py`.", repo.root), None, agent.LLM())
        self.assertFalse(hasattr(solver, "route"))
        self.assertEqual([solver.tier_for(n) for n in range(3)], list(agent.LADDER))


class TestProtectedPathsNeedPermission(unittest.TestCase):
    """A path the instruction MENTIONS is not a path it PERMITS.

    parse_instruction collects every path-looking token in the prose without
    regard to what its sentence says, so "Do not change tests/x.py" and a
    `python manage.py test ...` invocation both land in named_paths. While
    check_protected treated named_paths as permission, both of those exempted
    a file from the never-edit rule -- a prohibition read as a licence, on the
    one category that scores zero for the whole problem.
    """

    def checker(self, text: str, files: dict[str, str]):
        repo = make_repo(files)
        return agent.Checker(repo, agent.parse_instruction(text, repo.root)), repo

    def test_a_prohibition_does_not_grant_permission(self):
        v, _ = self.checker(
            "Fix the query in `app/q.py`. Do not change `app/tests/test_q.py`.",
            {"app/q.py": "x = 1\n", "app/tests/test_q.py": "y = 2\n"})
        self.assertIn("app/tests/test_q.py", v.instruction.named_paths,
                      "the parser still collects it as ranking evidence")
        result = v.check_protected(["app/tests/test_q.py"])
        self.assertFalse(result.passed, "a forbidden test file must never be editable")
        self.assertIn("however the instruction is worded", result.detail)

    def test_a_command_invocation_does_not_grant_permission(self):
        # The netbox prefix-hierarchy statement writes its checks inline rather
        # than fenced, which put netbox/manage.py into named_paths.
        v, _ = self.checker(
            "Limit production changes to `app/q.py`. Run `python manage.py test app` before finishing.",
            {"app/q.py": "x = 1\n", "manage.py": "z = 3\n"})
        self.assertFalse(v.check_protected(["manage.py"]).passed)

    def test_an_explicit_permission_still_allows_a_migration(self):
        # The cached-value-index sample: migrations are legitimately edited
        # when the instruction says so.
        v, _ = self.checker(
            "You may edit only `app/migrations/0107_x.py`. Run `ruff check --no-cache "
            "app/migrations/0107_x.py` before finishing.",
            {"app/migrations/0107_x.py": "ops = []\n"})
        self.assertTrue(v.check_protected(["app/migrations/0107_x.py"]).passed)

    def test_a_migration_without_permission_is_refused(self):
        v, _ = self.checker("Fix the slow query in `app/q.py`.",
                             {"app/q.py": "x = 1\n", "app/migrations/0107_x.py": "ops = []\n"})
        self.assertFalse(v.check_protected(["app/migrations/0107_x.py"]).passed)


class TestStyleConstraintsSurviveWrapping(unittest.TestCase):
    """Constraints are two-word phrases and instructions are hard-wrapped.

    Measured on the six netbox samples: five state the same style clause, and
    the one that happened to wrap between "exception" and "handling" lost that
    construct from check_style entirely -- silently, since nothing downstream
    knows a constraint was meant to be there.
    """

    CLAUSE = ("Write the method as plain ORM expressions: no Python loops, "
              "comprehensions, lambdas, exception handling, or context managers inside it.")

    def constraints(self, clause: str) -> list[str]:
        repo = make_repo({"a.py": "x = 1\n"})
        return agent.parse_instruction(f"Fix `a.py`. {clause}", repo.root).style_constraints

    def test_prohibition_clauses_survive_a_wrap_between_do_and_not(self):
        # "Do\nnot change ..." cost four clauses across three of the six netbox
        # samples, and both of them on the vlangroup task.
        repo = make_repo({"a.py": "x = 1\n"})
        for clause in ("Do not change models, fields, or tests.",
                       "Do\nnot change models, fields, or tests."):
            parsed = agent.parse_instruction(f"Fix `a.py`. {clause}", repo.root)
            self.assertEqual(parsed.forbidden, ["models, fields, or tests"], clause)

    def test_a_method_bound_survives_a_wrap(self):
        repo = make_repo({"a.py": "x = 1\n"})
        for clause in ("Keep its signature and the rest of the file unchanged.",
                       "Keep its signature and the\nrest of the file unchanged."):
            self.assertTrue(agent.parse_instruction(f"Fix `a.py`. {clause}",
                                                    repo.root).single_method, clause)

    def test_a_query_target_survives_a_wrap(self):
        repo = make_repo({"a.py": "x = 1\n"})
        for clause in ("Use at most 3 queries.", "Use at\nmost 3 queries."):
            self.assertEqual(agent.parse_instruction(f"Fix `a.py`. {clause}",
                                                     repo.root).targets.get("max_queries"), 3, clause)

    def test_the_same_clause_parses_the_same_however_it_wraps(self):
        flat = self.constraints(self.CLAUSE)
        self.assertIn("exception handling", flat)
        for split_at in ("exception handling", "context managers", "no Python loops"):
            wrapped = self.CLAUSE.replace(split_at, split_at.replace(" ", "\n", 1), 1)
            self.assertEqual(self.constraints(wrapped), flat,
                             f"a line break inside {split_at!r} changed the parsed constraints")


class TestSchemaNeverReportsAnError(unittest.TestCase):
    """Every failure path in sql() returns its complaint as text.

    Those strings are truthy, so a failed lookup used to be rendered under a
    "Live schema (columns and existing indexes)" header -- the prompt claiming
    it had read the database when it had not.
    """

    def probe(self, reply: str):
        repo = make_repo({"a.py": "x = 1\n"})
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.repo, probe.notes = repo, []
        probe.instruction = agent.Instruction(text="")
        probe.targets = [agent.DatabaseTarget(engine="postgresql", host="db", port="5432",
                                              user="u", database="d")]
        probe.sql = lambda query, **kw: reply
        return probe

    def test_a_driver_complaint_is_not_a_schema(self):
        for reply in ("[psql not installed]", "[no postgres driver]",
                      "[no database target discovered]", "[clickhouse query failed: refused]"):
            self.assertEqual(self.probe(reply).schema_for(["vlangroup"]), "", reply)

    def test_real_rows_still_render(self):
        out = self.probe("vlangroup | id | integer").schema_for(["vlangroup"])
        self.assertIn("vlangroup | id | integer", out)


class TestOnlyAVerifiedUrlReachesThePrompt(unittest.TestCase):
    """The prompt names a database only after this agent has connected to it.

    Discovery scrapes every host/port pair out of a file that mentions
    "postgres" somewhere, which on the netbox samples yields a Redis candidate
    and an unexpanded env var alongside the real one. Two things keep those out
    of the prompt: the port filter drops what cannot be a database before any
    connection is attempted, and SELECT 1 has to answer for the rest.
    """

    SETTINGS = (
        'DATABASES = {"default": {"ENGINE": "postgresql", "HOST": "db", "PORT": "5432",\n'
        '                         "USER": "app", "PASSWORD": "pw", "NAME": "netbox"}}\n'
        'REDIS = {"host": "cache", "port": "6379", "db": "0"}\n'
        'TASKS = {"host": "queue", "port": "TASKS_REDIS_PORT", "name": "getatt"}\n')

    def probe(self, answers):
        repo = make_repo({"config/database.py": self.SETTINGS, "q.py": "x = 1\n"})
        ins = agent.parse_instruction("Fix `q.py`.", repo.root)
        agent.DatabaseProbe._from_network = lambda self: None      # no real sockets
        agent.DatabaseProbe._ping = lambda self, target: answers.get(target.host, False)
        try:
            return repo, ins, agent.DatabaseProbe(repo, ins)
        finally:
            del agent.DatabaseProbe._from_network, agent.DatabaseProbe._ping

    def test_a_redis_port_is_never_even_tried(self):
        _, _, probe = self.probe({"db": True})
        tried = {url for url, _ in probe.attempts}
        self.assertFalse(any(":6379" in url for url in tried), tried)
        self.assertFalse(any("TASKS_REDIS_PORT" in url for url in tried), tried)

    def test_the_prompt_carries_the_verified_url_and_only_that(self):
        repo, ins, probe = self.probe({"db": True})
        self.assertEqual([agent.target_url(t) for t in probe.targets],
                         ["postgresql://app@db:5432/netbox"])
        rendered = agent.render_evidence(repo, ins, [("q.py", [])], probe)
        self.assertIn("postgresql://app@db:5432/netbox", rendered)
        self.assertIn("answered SELECT 1", rendered)
        self.assertNotIn("pw", rendered, "the password must not travel to a provider")

    def test_nothing_answers_means_the_prompt_names_no_database(self):
        repo, ins, probe = self.probe({})
        self.assertEqual(probe.targets, [])
        self.assertNotIn("live database",
                         agent.render_evidence(repo, ins, [("q.py", [])], probe))


class TestErrorHints(unittest.TestCase):
    def test_f401_and_cardinality_are_explained(self):
        text = ("F401 [*] `django.db.models.expressions.RawSQL` imported but unused\n"
                "psycopg.errors.CardinalityViolation: more than one row returned by a subquery "
                "used as an expression")
        hints = agent.error_hints(text)
        self.assertEqual(len(hints), 2)
        self.assertIn("must still use that name", hints[0])
        self.assertIn("exactly one row", hints[1])

    def test_compiler_wording_for_repeated_keyword_is_explained(self):
        hints = agent.error_hints("m.py:47: keyword argument repeated: vrf__isnull")
        self.assertEqual(len(hints), 1)
        self.assertIn("separate Q objects", hints[0])

    def test_wrong_api_on_app_object_is_explained(self):
        hints = agent.error_hints("AttributeError: 'Client' object has no attribute 'query'")
        self.assertEqual(len(hints), 1)
        self.assertIn("application's own classes", hints[0])

    def test_package_map_lists_sibling_apis_but_not_tests(self):
        repo = make_repo({
            "trailmix/reports/sessions.py": "def session_stats(client, day):\n    return []\n",
            "trailmix/db.py": "class Client:\n    def rows(self, statement):\n        pass\n    def command(self, s):\n        pass\n\ndef client():\n    return Client()\n",
            "trailmix/cli.py": "def main():\n    pass\n",
            "tests/test_sessions.py": "def test_x():\n    pass\n",
        })
        out = agent.package_map(repo, "trailmix/reports/sessions.py")
        self.assertIn("trailmix/db.py: class Client, client", out)
        self.assertIn("trailmix/cli.py: main", out)
        self.assertNotIn("test_sessions", out)
        self.assertNotIn("sessions.py:", out, "the target itself is not a neighbour")

    def test_no_hint_for_unknown_output(self):
        self.assertEqual(agent.error_hints("AssertionError: 3 != 5"), [])


class TestInferenceLoop(unittest.TestCase):
    """The retry loop must escalate caps without burning attempts, price each
    call against the remaining budget, and not retry policy refusals."""

    def make_llm(self, replies, headroom=1.0):
        llm = agent.LLM()
        llm.openrouter_key = "k"; llm.sandbox_proxy = ""; llm.local_base = ""
        llm._usage_unavailable = True
        llm.max_cost = headroom
        llm.sent = []
        llm.reasoning_flags = []
        llm.roster = lambda preferred: list(preferred)     # no roster expansion in tests
        def fake(url, key, model, messages, temperature, timeout, cap=None, reasoning=True):
            llm.sent.append((model, cap, messages[-1]["content"]))
            llm.reasoning_flags.append(reasoning)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        llm._call_openai_style = fake
        agent.time.sleep = lambda s: None
        return llm

    def test_runaway_hands_the_turn_to_the_next_family_with_reasoning_on(self):
        empty = agent.InferenceError("completion truncated by max_tokens before any content")
        llm = self.make_llm([empty, '{"action":"edit"}'])
        content, model = llm.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m2")
        self.assertEqual(llm.reasoning_flags, [True, True], "the second family keeps its reasoning")
        self.assertIn("m1", llm.no_reasoning)
        # Later calls put the spun model last and, when reached, run it without reasoning.
        llm.replies_ = None
        llm2 = self.make_llm(['ok'])
        llm2.no_reasoning.add("m1")
        content, model = llm2.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m2")
        self.assertEqual(llm2.reasoning_flags, [True])

    def test_every_family_spinning_falls_back_to_no_reasoning(self):
        empty = agent.InferenceError("completion truncated by max_tokens before any content")
        llm = self.make_llm([empty, empty, "ok"])
        content, model = llm.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m1")
        self.assertEqual(llm.reasoning_flags, [True, True, False])
        self.assertEqual(llm.sent[-1][2], agent.NUDGE, "the reasoning-off retry carries the nudge")

    def test_empty_without_reasoning_moves_to_next_model(self):
        empty = agent.InferenceError("completion truncated by max_tokens before any content")
        llm = self.make_llm([empty, "ok"])
        llm.no_reasoning.update({"m1", "m2"})
        content, model = llm.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m2")
        self.assertIn("m1", llm.blocked)

    def test_all_models_unaffordable_is_budget_exhausted_not_transient(self):
        llm = self.make_llm(["ok"], headroom=0.001)
        with self.assertRaises(agent.BudgetExhausted):
            llm.complete(["moonshotai/kimi-k2.6"], [{"role": "user", "content": "q"}])

    def test_connection_reset_becomes_retryable_inference_error(self):
        llm = agent.LLM()
        def boom(request, timeout=None, context=None):
            raise ConnectionResetError(104, "Connection reset by peer")
        original = agent.urllib.request.urlopen
        agent.urllib.request.urlopen = boom
        try:
            with self.assertRaises(agent.InferenceError):
                llm._post("https://x/", {}, {}, 5)
        finally:
            agent.urllib.request.urlopen = original

    def test_transport_failure_is_retried_then_next_model(self):
        reset = agent.InferenceError("transport failure: ConnectionResetError: reset")
        llm = self.make_llm([reset, reset, "ok"])
        content, model = llm.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m2")
        self.assertEqual([m for m, _, _ in llm.sent], ["m1", "m1", "m2"])

    def test_documented_proxy_route_is_tried_first_and_retired_when_unreachable(self):
        llm = self.make_llm(["ok"])
        llm.sandbox_proxy = "http://sandbox-proxy:80"
        urls = []
        real = llm._call_openai_style
        def by_url(url, key, model, messages, temperature, timeout, cap=None, reasoning=True):
            urls.append(url)
            if url.startswith("http://sandbox-proxy"):
                raise agent.InferenceError("transport failure: URLError: name not known")
            return real(url, key, model, messages, temperature, timeout, cap, reasoning)
        llm._call_openai_style = by_url
        content, model = llm.complete(["m1"], [{"role": "user", "content": "q"}])
        self.assertEqual(urls, ["http://sandbox-proxy:80/api/v1/chat/completions", agent.OPENROUTER_URL])
        self.assertIn("http://sandbox-proxy:80/api/v1/chat/completions", llm.dead_routes)
        # A model-level 404 is not a dead route: it propagates to the per-model handling.
        llm2 = self.make_llm([agent.InferenceError("HTTP 404: model not supported"), "ok"])
        llm2.sandbox_proxy = "http://sandbox-proxy:80"
        content, model = llm2.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m2")
        self.assertEqual(llm2.dead_routes, set())

    def test_403_is_not_retried(self):
        denied = agent.InferenceError('HTTP 403: {"success": false, "error": "Access denied by security policy."}')
        llm = self.make_llm([denied, "ok"])
        content, model = llm.complete(["m1", "m2"], [{"role": "user", "content": "q"}])
        self.assertEqual(model, "m2")
        self.assertEqual([m for m, _, _ in llm.sent], ["m1", "m2"])
        self.assertIn("m1", llm.blocked)

    def test_cap_shrinks_to_fit_remaining_budget(self):
        model = "moonshotai/kimi-k2.6"          # $4.00 per M output tokens
        llm = self.make_llm(["ok"], headroom=0.05)
        cap = llm.affordable_cap(model, [{"role": "user", "content": "x" * 350}], 40000)
        self.assertLess(cap, 40000)
        self.assertGreaterEqual(cap, agent.MIN_COMPLETION_CAP)
        # 6000 tokens at $4/M is $0.024; with $0.01 left not even that fits.
        llm.max_cost = 0.01
        self.assertIsNone(llm.affordable_cap(model, [{"role": "user", "content": "x"}], 40000))

    def test_unaffordable_models_are_skipped_not_called(self):
        # An unknown slug is priced conservatively, so at $0.01 of headroom
        # neither candidate may be called; the loop must say so, not guess.
        llm = self.make_llm(["ok"], headroom=0.01)
        with self.assertRaises(agent.BudgetExhausted):
            llm.complete(["moonshotai/kimi-k2.6", "unknown/free-model"], [{"role": "user", "content": "q"}])
        self.assertEqual(llm.sent, [])
        llm = self.make_llm(["ok"], headroom=0.29)
        self.assertLessEqual(llm.affordable_cap("unknown/free-model", [{"role": "user", "content": "q"}], 12000), 12000)


    def test_reasoning_effort_is_a_setting_not_a_literal(self):
        """`effort` is read from the client, and reasoning=False switches the
        field to {"enabled": False} -- the only form DeepSeek honours on this
        route. Measured: reasoning.max_tokens is ignored, so this is the whole
        mechanism behind the runaway recovery above."""
        llm = agent.LLM()
        self.assertEqual(llm.effort, "low")
        sent = {}
        llm._post = lambda url, payload, headers, timeout: sent.update(payload) or {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
        llm.effort = "medium"
        llm._call_openai_style("http://x", "k", "m", [{"role": "user", "content": "q"}], 0, 5, 100)
        self.assertEqual(sent["reasoning"], {"effort": "medium"})
        llm._call_openai_style("http://x", "k", "m", [{"role": "user", "content": "q"}], 0, 5, 100,
                               reasoning=False)
        self.assertEqual(sent["reasoning"], {"enabled": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFilesystemFailuresNeverLoseTheRun(unittest.TestCase):
    """A filesystem problem must become feedback, never an escaped exception.

    Every call these exercise sits inside Solver.solve() with nothing between
    it and agent_main's blanket `except Exception`, which logs a traceback and
    returns "". So an OSError here does not degrade the answer -- it discards a
    patch that may already have passed every check, and scores the task zero.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text("value = 1\n")
        self.repo = agent.Repository(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_file_under_a_file_is_an_edit_error_not_a_crash(self):
        # `app.py/child.py` -- mkdir raises NotADirectoryError. The model chose
        # this path, so it is the model that has to be told.
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "app.py/child.py", "new_file": True, "content": "x = 1\n"}])
        self.assertEqual(changed, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("could not be created", errors[0])

    def test_write_onto_a_directory_is_an_edit_error_not_a_crash(self):
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "m.py").write_text("a = 1\n")
        # Editing a path that is a directory: IsADirectoryError on open.
        self.repo._cache["pkg"] = "a = 1\n"          # make read() return text
        changed, errors = agent.apply_edits(self.repo, [
            {"path": "pkg", "search": "a = 1", "replace": "a = 2"}])
        self.assertEqual(changed, [])
        self.assertTrue(errors and "could not be written" in errors[0], errors)

    def test_revert_reports_and_continues_when_a_file_cannot_be_restored(self):
        self.repo.write("app.py", "value = 2\n")
        self.repo.write("other.py", "value = 3\n")
        broken = self.root / "app.py"
        broken.unlink()
        broken.mkdir()                               # restoring it must fail
        self.repo.revert_all()                       # must not raise
        # The reachable file is still restored: one bad path does not abort the
        # rest of the rollback, which is what leaves a checkout dirty.
        self.assertFalse((self.root / "other.py").exists())

    def test_build_patch_omits_an_unreadable_file_instead_of_raising(self):
        self.repo.write("app.py", "value = 2\n")
        self.repo.write("keep.py", "kept = 1\n")
        (self.root / "app.py").unlink()
        (self.root / "app.py").mkdir()               # exists(), but unreadable
        patch = agent.build_patch(self.repo, ["app.py", "keep.py"])
        self.assertNotIn("a/app.py", patch)
        self.assertIn("keep.py", patch)

    def test_apply_check_that_cannot_stage_does_not_fail_the_patch(self):
        self.repo.write("app.py", "value = 2\n")
        patch = agent.build_patch(self.repo, ["app.py"])
        original = agent.write_source

        def refuse(path, text):
            raise OSError(28, "No space left on device")

        agent.write_source = refuse
        try:
            applies, detail = agent.verify_patch_applies(patch, self.repo)
        finally:
            agent.write_source = original
        self.assertTrue(applies, "a check that cannot run must not condemn the patch")
        self.assertIn("could not run", detail)


class TestConfigReadsTolerateMissingFiles(unittest.TestCase):
    """The probe reads whatever the application left lying around. Almost none
    of it exists on any given repo, so absence is the normal case."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app.py").write_text("x = 1\n")
        self.repo = agent.Repository(self.root)
        self.instruction = agent.parse_instruction("Fix the query.", self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_dotenv_files_at_all(self):
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.repo, probe.instruction = self.repo, self.instruction
        self.assertIsInstance(probe._load_env_files(), dict)

    def test_a_dotenv_that_is_a_directory_is_skipped(self):
        (self.root / ".env").mkdir()
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.repo, probe.instruction = self.repo, self.instruction
        self.assertIsInstance(probe._load_env_files(), dict)

    def test_instruction_falls_through_a_directory_named_instruction_md(self):
        (self.root / "instruction.md").mkdir()
        with self.assertRaises(agent.MissingInstruction):
            agent._instruction_text({}, self.root, harness_copy=self.root / "nope.md")

    def test_app_python_survives_an_unreadable_manage_py(self):
        (self.root / "manage.py").mkdir()
        self.assertTrue(agent.app_python(self.root))


class TestPromptFollowsItsOwnDesignRules(unittest.TestCase):
    """The prompt-design rules, as checks rather than as intentions.

    Each of these was a real defect at some point, and each is invisible in
    review: a template that is not valid JSON still reads like JSON, and an
    instruction that has drifted below the input data still reads like an
    instruction.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "shop.py").write_text("def total():\n    return 1\n")
        self.repo = agent.Repository(self.root)
        self.instruction = agent.parse_instruction(
            "# Fix it\n\nThe count is wrong in `shop.py`.", self.root)
        self.probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        self.probe.targets, self.probe.notes, self.probe.attempts = [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def _prompt(self):
        return agent.render_evidence(self.repo, self.instruction,
                                     [("shop.py", [1])], self.probe)

    def test_every_json_template_in_the_protocol_parses(self):
        # The protocol used to show `"verify": ["a " "b"]` -- Python's adjacent
        # string concatenation, which is not JSON. A model copying the shape it
        # was given produced an unparseable reply and burned a round on it.
        blocks = re.findall(r'^\{"action".*?^ \]\}', agent.EDIT_PROTOCOL, re.S | re.M)
        self.assertGreaterEqual(len(blocks), 3, "expected both templates and a worked example")
        for block in blocks:
            json.loads(block)          # raises, with the offset, if it ever breaks again

    def test_the_prompt_opens_with_the_instruction_element(self):
        # Instructions first: the statement is input data and says nothing
        # about this protocol, so a prompt that opens with it opens with no
        # instruction at all.
        self.assertTrue(self._prompt().startswith("# Instructions"))

    def test_elements_appear_in_order(self):
        prompt = self._prompt() + "\n\n# Response format\n\n"
        order = [prompt.index(h) for h in
                 ("# Instructions", "# Task statement", "# Relevant source", "# Response format")]
        self.assertEqual(order, sorted(order))

    def test_sections_are_separated_by_a_rule(self):
        # A statement carries its own headings, so a heading alone does not
        # mark where it stops and this agent's own findings start.
        self.assertIn(f"\n\n{agent.PromptBuilder.SEPARATOR}\n\n# Task statement", self._prompt())

    def test_the_worked_example_is_not_a_bench_task_answer(self):
        # An example drawn from a real task hands the model that task's answer.
        for token in ("VLANGroup", "annotate_utilization", "annotate_hierarchy", "netbox/"):
            self.assertNotIn(token, agent.EDIT_PROTOCOL)

    def test_uncertainty_has_a_prescribed_fallback(self):
        self.assertIn("need_context", agent.DIRECTIVE)
        self.assertIn("unsure", agent.DIRECTIVE.lower())

    def test_reply_limits_are_countable(self):
        # "Keep the reply short" is not a limit the model can measure against.
        self.assertRegex(agent.EDIT_PROTOCOL, r"at most \d+ sentence")
        self.assertRegex(agent.EDIT_PROTOCOL, r"under [\d,]+ characters")


class TestDatabaseDiscoveryIsSecondaryAndCorrect(unittest.TestCase):
    """Discovery costs a connect timeout per candidate, and only targets[0] is
    ever read. Both facts are easy to regress and neither shows up as an
    error -- one as seconds, the other as a silently missing schema."""

    def _probe(self, targets):
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.attempts = []
        probe.targets = targets
        probe.answered = set()
        return probe

    def test_verification_stops_at_the_first_candidate_that_answers(self):
        good = agent.DatabaseTarget(engine="postgresql", priority=0, host="db",
                                    port="5432", database="app")
        rest = [agent.DatabaseTarget(engine="postgresql", priority=4, host=f"h{n}",
                                     port="5432", database="app") for n in range(3)]
        probe = self._probe([good] + rest)
        probe._ping = lambda target: target.host == "db"
        self.assertEqual(probe._first_reachable(probe.targets), [good])
        # One ping, not four: the three behind it were never dialled.
        self.assertEqual(len(probe.attempts), 1)

    def test_dead_candidates_ahead_of_a_live_one_are_all_tried(self):
        dead = [agent.DatabaseTarget(engine="postgresql", priority=n, host=f"h{n}",
                                     port="5432", database="app") for n in range(2)]
        good = agent.DatabaseTarget(engine="postgresql", priority=5, host="db",
                                    port="5432", database="app")
        probe = self._probe(dead + [good])
        probe._ping = lambda target: target.host == "db"
        self.assertEqual(probe._first_reachable(probe.targets), [good])
        self.assertEqual([ok for _, ok in probe.attempts], [False, False, True])

    def test_a_redis_database_index_is_not_read_as_a_database_name(self):
        # NetBox's configuration.py puts REDIS {'DATABASE': 0} below DATABASES
        # {'NAME': 'netbox_dev'}. A bare DATABASE key outranks the weak NAME, so
        # the scrape produced .../0 -- unconnectable, and the live schema went
        # missing without a word.
        config = textwrap.dedent("""
            DATABASES = {'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': 'netbox_dev', 'USER': 'solver',
                'PASSWORD': 'pw', 'HOST': 'postgres', 'PORT': 5432,
            }}
            REDIS = {'tasks': {'HOST': 'redis', 'PORT': 6379, 'DATABASE': 0},
                     'caching': {'HOST': 'redis', 'PORT': 6379, 'DATABASE': 1}}
        """)
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.targets, probe.notes, probe._env = [], [], {}
        probe.instruction = agent.parse_instruction("fix the query", Path("."))
        probe._scrape_fields("netbox/netbox/configuration.py", config, 3)
        self.assertTrue(probe.targets, "the postgres block should still be found")
        self.assertEqual(agent.target_url(probe.targets[0]),
                         "postgresql://solver@postgres:5432/netbox_dev")


class TestSlicerPicksTheMethodTheInstructionNamed(unittest.TestCase):
    """When a method name repeats, the class decides which one is shown.

    netbox/ipam/filtersets.py defines filter_device three times. Matching the
    bare name picked whichever came first in the file, so the model was shown
    the right method by luck. On that task the luck held; the point is that it
    was luck, and a task ordered the other way loses silently -- the model
    cannot repair a fix aimed at a method it was never shown.
    """

    SOURCE = textwrap.dedent("""
        class AlphaFilterSet:
            def filter_device(self, qs):
                return qs.filter(a=1)

        class TargetFilterSet:
            def filter_device(self, qs):
                return qs.filter(b=2)
    """) + "\n# padding\n" * 400

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "filtersets.py").write_text(self.SOURCE)
        self.repo = agent.Repository(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _label(self, class_hint):
        instruction = agent.Instruction(text="x", method_hint="filter_device",
                                        class_hint=class_hint)
        pieces = agent.slice_around(self.repo, "filtersets.py", [], instruction,
                                    budget_lines=20)
        return pieces[0].label if pieces else ""

    def test_the_named_class_wins_over_file_order(self):
        self.assertIn("TargetFilterSet.filter_device", self._label("TargetFilterSet"))

    def test_the_first_definition_is_still_used_without_a_class_hint(self):
        self.assertIn("AlphaFilterSet.filter_device", self._label(None))

    def test_a_class_the_file_does_not_define_falls_back_to_the_bare_name(self):
        # A hint naming a class that lives elsewhere must still find the method
        # rather than showing nothing at all.
        self.assertIn("AlphaFilterSet.filter_device", self._label("SomeOtherFilterSet"))


class TestMethodContractMatchesTheGraders(unittest.TestCase):
    """The graded method contract, checked before the grader gets to.

    Measured against all six verify.py files: everything they forbid beyond
    _FORBIDDEN_NODES is already covered by check_style, because the instruction
    carrying that grader states the rule in prose. `Raise` is the one exception
    -- four of six reject it, no instruction mentions it, and until now neither
    gate looked.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = agent.Repository(self.root)
        self.instruction = agent.parse_instruction(
            "Fix `Q.run()`. Change only that method and keep its signature.", self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _check(self, original, current):
        (self.root / "m.py").write_text(original)
        self.repo._snapshots["m.py"] = original
        self.repo._cache["m.py"] = current
        (self.root / "m.py").write_text(current)
        checker = agent.Checker(self.repo, self.instruction)
        return checker.check_method_contract(["m.py"])

    def test_a_raise_the_edit_introduced_is_rejected(self):
        before = "class Q:\n    def run(self):\n        return 1\n"
        after = "class Q:\n    def run(self):\n        raise ValueError('x')\n"
        result = self._check(before, after)
        self.assertFalse(result.passed)
        self.assertIn("Raise", result.detail)

    def test_a_raise_that_was_already_there_is_not_held_against_the_edit(self):
        # The grader accommodates what it shipped; flagging pre-existing code
        # sends a correct fix into repair rounds it cannot win.
        before = "class Q:\n    def run(self):\n        raise ValueError('x')\n"
        after = "class Q:\n    def run(self):\n        raise ValueError('y')\n"
        self.assertTrue(self._check(before, after).passed)

    def test_the_node_budget_never_undercuts_the_method_it_starts_from(self):
        # bulk-tag-assignment's `add` is 224 nodes before any edit and its
        # grader allows 400. A flat 240 would reject a winning patch over 16
        # nodes of headroom, so the floor comes from the original.
        budget = agent.Checker._node_budget
        self.assertEqual(budget(23), 240, "a small method gets the strictest budget")
        self.assertGreater(budget(224), 240, "a large method is not squeezed below itself")
        self.assertLessEqual(budget(350), agent.Checker._MAX_METHOD_NODES)

    def test_byte_limit_matches_the_strictest_grader(self):
        self.assertEqual(agent.Checker._MAX_METHOD_BYTES, 4500)


class TestMigrationContract(unittest.TestCase):
    """Migration tasks get no method contract -- `single_method` is false, so
    check_method_contract never runs -- yet cached-value-index has the
    strictest grader in the set. The rule those graders encode is one thing
    said many ways: an edit may change the VALUES inside the operations, never
    the structure around them.
    """

    ORIGINAL = textwrap.dedent("""\
        from django.db import migrations, models


        class Migration(migrations.Migration):
            dependencies = [
                ('extras', '0106_bookmark_user_cascade_deletion'),
            ]

            operations = [
                migrations.AddIndex(
                    model_name='cachedvalue',
                    index=models.Index(fields=['object_type'], name='extras_cachedvalue_object'),
                ),
            ]
        """)
    PATH = "extras/migrations/0107_cachedvalue_object.py"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / self.PATH).parent.mkdir(parents=True)
        (self.root / self.PATH).write_text(self.ORIGINAL)
        self.repo = agent.Repository(self.root)
        self.instruction = agent.parse_instruction("Repair the migration.", self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _gate(self, current):
        self.repo._snapshots[self.PATH] = self.ORIGINAL
        self.repo._cache[self.PATH] = current
        (self.root / self.PATH).write_text(current)
        return agent.Checker(self.repo, self.instruction).check_migration_contract([self.PATH])

    def test_widening_the_index_fields_is_allowed(self):
        # The one thing the task actually asks for.
        fixed = self.ORIGINAL.replace("fields=['object_type']",
                                      "fields=['object_type', 'object_id']")
        self.assertTrue(self._gate(fixed).passed, self._gate(fixed).detail)

    def test_a_second_operation_is_rejected(self):
        self.assertFalse(self._gate(
            self.ORIGINAL.replace("    ]\n", "        migrations.RunSQL('SELECT 1'),\n    ]\n", 1)
        ).passed)

    def test_swapping_the_operation_type_is_rejected(self):
        self.assertFalse(self._gate(
            self.ORIGINAL.replace("migrations.AddIndex", "migrations.RemoveIndex")).passed)

    def test_changing_the_dependency_is_rejected(self):
        self.assertFalse(self._gate(
            self.ORIGINAL.replace("0106_bookmark_user_cascade_deletion", "0999_other")).passed)

    def test_an_added_import_is_rejected(self):
        self.assertFalse(self._gate("import os\n" + self.ORIGINAL).passed)

    def test_an_added_helper_is_rejected(self):
        self.assertFalse(self._gate(self.ORIGINAL + "\n\ndef helper():\n    return 1\n").passed)

    def test_a_non_migration_file_is_not_gated(self):
        # The gate keys on the path; ordinary source keeps its own contract.
        self.assertTrue(agent.Checker(self.repo, self.instruction)
                        .check_migration_contract(["app/models.py"]).passed)


class TestQueryProbeRunsWhereTheTablesAre(unittest.TestCase):
    """The probe measures the database work a change performs.

    It had never once succeeded. `manage.py shell` opens the default database,
    which in these projects has no tables -- migrations are applied to the one
    Django builds for a test run -- so every probe died on `relation "..." does
    not exist`, `query scaling` reported no count, and an optimisation task
    could ship with the number it is judged on never taken.
    """

    def script(self, setup="from a.models import M", call="list(M.objects.all()[:N])",
               result=""):
        return agent.Checker._DJANGO_PROBE.format(
            setup="\n".join(f"        {l}" for l in setup.splitlines()) or "        pass",
            call="\n".join(f"                {l}" for l in call.splitlines()),
            result=f"        _fp = _fingerprint({result})" if result else "        pass",
            small=1, large=10)

    def test_the_generated_script_is_valid_python(self):
        # The blocks are nested, so setup and call sit at different depths. An
        # off-by-one here is a SyntaxError the agent would report as "the probe
        # produced no query count" -- indistinguishable from the bug above.
        import ast as _ast
        for setup, call in (("", "list(M.objects.all()[:N])"),
                            ("x = 1", "qs = M.objects.filter(n=N)\nlist(qs)"),
                            ("from a.models import M\nrows = [M(i) for i in range(3)]",
                             "list(M.objects.filter(n=N))")):
            _ast.parse(self.script(setup, call))

    def test_it_repoints_at_the_test_database(self):
        script = self.script()
        self.assertIn('_cfg["NAME"] = (_cfg.get("TEST") or {}).get("NAME")', script)
        self.assertIn("connection.close()", script)
        self.assertLess(script.index('_cfg["NAME"]'), script.index("CaptureQueriesContext(connection)"),
                        "the connection must be repointed before it is used")

    def test_it_measures_inside_a_transaction_and_rolls_back(self):
        # A Django TestCase wraps each test in a transaction, so the hidden
        # test's count is taken under those conditions; measuring outside one
        # would count Django's per-statement transaction management too. The
        # rollback then leaves the database as it was found.
        script = self.script()
        self.assertIn("with transaction.atomic():", script)
        self.assertIn("transaction.set_rollback(True)", script)

    def test_a_reported_count_is_still_read(self):
        repo = make_repo({"manage.py": "", "a.py": ""})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(
            a, 0, 'RIDGES_QC{"1": 3, "10": 3}\n')
        try:
            result = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertTrue(result.passed)
        self.assertTrue(result.verified, "a real count is a real measurement")
        self.assertIn("bounded: 3 queries at N=1, 3 at N=10", result.detail)

    def test_a_traceback_is_reported_as_unmeasured_with_its_cause(self):
        repo = make_repo({"manage.py": "", "a.py": ""})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(
            a, 0, 'Traceback (most recent call last):\npsycopg.errors.UndefinedTable: '
                  'relation "dcim_device" does not exist\n')
        try:
            result = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertTrue(result.passed, "nothing about the code failed")
        self.assertFalse(result.verified, "but nothing was measured either")
        self.assertIn("UndefinedTable", result.detail, "the cause must reach the log")


class TestBaselineTurnsACountIntoADelta(unittest.TestCase):
    """"Three queries" does not say whether the change helped.

    An optimisation task is graded on the difference the edit makes, and the
    agent only ever saw the number after it. Measuring the same probe either
    side of the change answers the question the count alone cannot: was this
    better than what was there? These tests drive the probe off the actual file
    contents, so a baseline that did not really revert the edit shows up as the
    wrong number rather than passing quietly.
    """

    def fixture(self):
        repo = make_repo({"manage.py": "", "app.py": "SLOW\n"})
        repo.write("app.py", "FAST\n")
        checker = agent.Checker(repo, agent.Instruction(text="", kinds={"bounded_queries"}))
        return repo, checker

    def counting_run(self, repo, calls, counts={"SLOW": 12, "FAST": 3}):
        """A run_command that answers from whatever app.py currently says."""
        def run(*a, **k):
            body = (repo.root / "app.py").read_text().strip()
            calls.append(body)
            n = counts[body]
            return agent.subprocess.CompletedProcess(
                a, 0, 'RIDGES_QC{"1": %d, "10": %d}\n' % (min(n, 3), n))
        return run

    def measure(self, repo, checker, calls, **kw):
        original = agent.run_command
        agent.run_command = self.counting_run(repo, calls, **kw)
        try:
            return checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original

    def test_the_baseline_is_taken_against_the_reverted_code(self):
        repo, checker = self.fixture()
        calls = []
        result = self.measure(repo, checker, calls)
        self.assertEqual(calls, ["FAST", "SLOW"],
                         "the edit is measured first, then the code it replaced")
        self.assertTrue(result.passed)
        self.assertIn("12 -> 3 queries at N=10", result.detail)

    def test_the_edit_is_put_back_afterwards(self):
        # The whole patch lives on disk. A baseline that reverts and does not
        # restore would hand build_patch an empty diff.
        repo, checker = self.fixture()
        self.measure(repo, checker, [])
        self.assertEqual((repo.root / "app.py").read_text(), "FAST\n")
        self.assertEqual(repo.changed_files(), ["app.py"])
        self.assertEqual(repo.original("app.py"), "SLOW\n",
                         "and the original snapshot still says what we started from")

    def test_the_edit_is_put_back_even_when_the_baseline_probe_dies(self):
        repo, checker = self.fixture()
        original = agent.run_command
        state = {"n": 0}

        def run(*a, **k):
            state["n"] += 1
            if state["n"] == 1:
                return agent.subprocess.CompletedProcess(a, 0, 'RIDGES_QC{"1": 3, "10": 3}\n')
            raise OSError("probe died")

        agent.run_command = run
        try:
            with self.assertRaises(OSError):
                checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertEqual((repo.root / "app.py").read_text(), "FAST\n")

    def test_it_is_measured_once_and_reused(self):
        # It costs a shell run. Three repair rounds must not cost three of them.
        repo, checker = self.fixture()
        calls = []
        self.measure(repo, checker, calls)
        self.measure(repo, checker, calls)
        self.assertEqual(calls, ["FAST", "SLOW", "FAST"], "the second round reuses the baseline")

    def test_a_baseline_that_cannot_be_taken_is_not_retaken(self):
        repo, checker = self.fixture()
        original = agent.run_command
        calls = []

        def run(*a, **k):
            body = (repo.root / "app.py").read_text().strip()
            calls.append(body)
            out = 'RIDGES_QC{"1": 3, "10": 3}\n' if body == "FAST" else "Traceback: boom\n"
            return agent.subprocess.CompletedProcess(a, 0, out)

        agent.run_command = run
        try:
            first = checker.measure_query_scaling({"call": "f(N)"})
            second = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertEqual(calls, ["FAST", "SLOW", "FAST"])
        for result in (first, second):
            self.assertTrue(result.passed)
            self.assertIn("bounded: 3 queries at N=1", result.detail,
                          "without a baseline the check reports what it did measure")

    def test_a_change_that_does_more_work_fails_a_bounded_work_task(self):
        # Bounded and worse are not the same verdict. This one scales fine and
        # is still a regression on the one number the task is about.
        repo, checker = self.fixture()
        result = self.measure(repo, checker, [], counts={"SLOW": 2, "FAST": 3})
        self.assertFalse(result.passed)
        self.assertIn("increased the database work", result.detail)
        self.assertIn("2 queries at N=10 before it, 3 after", result.detail)

    def test_doing_more_work_is_allowed_when_the_task_never_asked_for_less(self):
        # A correctness fix may legitimately cost a statement. Only a task whose
        # subject is the count gets failed for raising it.
        repo = make_repo({"manage.py": "", "app.py": "SLOW\n"})
        repo.write("app.py", "FAST\n")
        checker = agent.Checker(repo, agent.Instruction(text=""))
        result = self.measure(repo, checker, [], counts={"SLOW": 2, "FAST": 3})
        self.assertTrue(result.passed)
        self.assertIn("2 -> 3 queries at N=10", result.detail)

    def test_a_failure_carries_the_before_numbers(self):
        # The model reads this detail to decide what to try next; "still 4, was
        # 12" and "still 4, was 4" call for very different second attempts.
        repo = make_repo({"manage.py": "", "app.py": "SLOW\n"})
        repo.write("app.py", "FAST\n")
        checker = agent.Checker(repo, agent.Instruction(text="", targets={"max_queries": 2}))
        result = self.measure(repo, checker, [])
        self.assertFalse(result.passed)
        self.assertIn("at most 2 queries", result.detail)
        self.assertIn("12 at N=10", result.detail)

    def test_no_edit_means_no_baseline(self):
        # Nothing to revert, so nothing to compare against; the check still runs.
        repo = make_repo({"manage.py": "", "app.py": "SLOW\n"})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        calls = []
        result = self.measure(repo, checker, calls)
        self.assertEqual(calls, ["SLOW"], "one probe run, and nothing to revert for a second")
        self.assertIn("3 queries at N=1, 12 at N=10", result.detail)
        self.assertNotIn("Before the edit", result.detail)


class TestTheRewriteMustReturnTheSameRows(unittest.TestCase):
    """Fewer statements returning different rows is not an optimisation.

    The scaling check answers "how much work?" and nothing else, so a rewrite
    that halves the query count by dropping half the rows passed it. The task's
    own tests are no defence either: they passed before the edit, against the
    code the edit replaced. The baseline run is already happening, so comparing
    a value across it costs nothing but the expression the model supplies.
    """

    def fingerprint(self):
        """The function as it is actually shipped, lifted out of the template."""
        source = agent.Checker._DJANGO_PROBE
        block = source[source.index("def _fingerprint"):source.index("counts = {{}}")]
        namespace = {}
        exec(block.replace("{{", "{").replace("}}", "}"), namespace)
        return namespace["_fingerprint"]

    def test_the_fingerprint_is_order_insensitive(self):
        # A filter rewrite is not required to preserve ordering, and the task's
        # own tests cover it when it is. Failing on order would be noise.
        fp = self.fingerprint()
        self.assertEqual(fp(["b", "a"]), fp(["a", "b"]))

    def test_the_fingerprint_survives_values_it_cannot_iterate(self):
        fp = self.fingerprint()
        self.assertEqual(fp(7), ["7"])
        self.assertEqual(fp("ab"), ["'ab'"], "a string is one value, not two")

        class Hostile:
            def __iter__(self):
                raise RuntimeError("no")

        self.assertEqual(len(fp(Hostile())), 1, "a probe must not die inside its own reporting")

    def test_the_generated_script_is_valid_python_with_a_result(self):
        import ast as _ast
        probe = TestQueryProbeRunsWhereTheTablesAre()
        _ast.parse(probe.script(result="sorted(x.name for x in qs)"))
        self.assertIn("_fp = _fingerprint(sorted(x.name for x in qs))",
                      probe.script(result="sorted(x.name for x in qs)"))

    def fixture(self, **instruction):
        repo = make_repo({"manage.py": "", "app.py": "SLOW\n"})
        repo.write("app.py", "FAST\n")
        return repo, agent.Checker(repo, agent.Instruction(text="", **instruction))

    def measure(self, repo, checker, rows, calls=None, counts={"SLOW": 12, "FAST": 3},
                result="names"):
        """Answer both probe runs from whatever app.py currently says."""
        def run(*a, **k):
            body = (repo.root / "app.py").read_text().strip()
            if calls is not None:
                calls.append(body)
            n = counts[body]
            return agent.subprocess.CompletedProcess(
                a, 0, 'RIDGES_QC{"1": %d, "10": %d}\nRIDGES_QR%s\n'
                      % (min(n, 3), n, json.dumps(rows[body])))
        original = agent.run_command
        agent.run_command = run
        try:
            return checker.measure_query_scaling({"call": "f(N)", "result": result})
        finally:
            agent.run_command = original

    def test_the_same_rows_either_side_pass_and_say_so(self):
        repo, checker = self.fixture()
        result = self.measure(repo, checker, {"SLOW": ["a", "b"], "FAST": ["b", "a"]})
        self.assertTrue(result.passed)
        self.assertIn("12 -> 3 queries at N=10", result.detail)
        self.assertIn("`result` unchanged (2 item(s))", result.detail)

    def test_dropped_rows_fail_even_though_the_count_improved(self):
        repo, checker = self.fixture()
        result = self.measure(repo, checker, {"SLOW": ["a", "b", "c"], "FAST": ["a", "b"]})
        self.assertFalse(result.passed)
        self.assertIn("`result` changed", result.detail)
        self.assertIn("produced 3 item(s), this version produces 2", result.detail)
        self.assertIn("only before: 'c'".replace("'", ""), result.detail.replace("'", ""))

    def test_the_failure_names_its_own_benign_cause(self):
        # The likeliest mismatch is not a bug: the two runs are separate
        # transactions, so an expression built out of auto-assigned ids differs
        # even when the rows are identical. Say so, or the model spends a repair
        # round rewriting correct code.
        repo, checker = self.fixture()
        result = self.measure(repo, checker, {"SLOW": ["1"], "FAST": ["4"]})
        self.assertFalse(result.passed)
        self.assertIn("database ids", result.detail)
        self.assertIn("does not rewind", result.detail)

    def test_it_never_displaces_a_count_failure(self):
        # The counts are certain; this comparison depends on an expression the
        # model wrote. When both are wrong, report the one that cannot be wrong.
        repo, checker = self.fixture(targets={"max_queries": 2})
        result = self.measure(repo, checker, {"SLOW": ["a"], "FAST": ["b"]})
        self.assertFalse(result.passed)
        self.assertIn("at most 2 queries", result.detail)
        self.assertNotIn("`result` changed", result.detail)

    def test_a_corrected_expression_gets_a_fresh_baseline(self):
        # The model's answer to a spurious mismatch is a better `result`.
        # Comparing it against a fingerprint taken with the old expression would
        # compare two different things, so the expression is part of the key.
        repo, checker = self.fixture()
        calls = []
        first = self.measure(repo, checker, {"SLOW": ["1"], "FAST": ["4"]}, calls, result="ids")
        self.assertFalse(first.passed)
        second = self.measure(repo, checker, {"SLOW": ["a"], "FAST": ["a"]}, calls, result="names")
        self.assertTrue(second.passed)
        self.assertEqual(calls, ["FAST", "SLOW", "FAST", "SLOW"])

    def test_no_result_expression_means_no_comparison(self):
        repo, checker = self.fixture()
        original = agent.run_command
        scripts = []

        def run(cmd, *a, **k):
            scripts.append(cmd[-1])
            body = (repo.root / "app.py").read_text().strip()
            return agent.subprocess.CompletedProcess(
                cmd, 0, 'RIDGES_QC{"1": 3, "10": %d}\n' % (12 if body == "SLOW" else 3))

        agent.run_command = run
        try:
            result = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertTrue(result.passed)
        self.assertNotIn("`result`", result.detail)
        self.assertNotIn("_fp = _fingerprint(", scripts[0],
                         "nothing to evaluate, so nothing emitted")


class TestTheGraderStartsWhateverTheTaskShips(unittest.TestCase):
    """grade.py could not grade a ClickHouse task at all.

    Its compose overlay named postgres and redis outright, so on a task whose
    stack is a single clickhouse service it wrote a file referring to services
    that do not exist -- and the true grader, the one harness that reports the
    real reward, was unavailable on a third of the corpus. Reading the names
    out of the task's own compose file needs no list to keep up to date.
    """

    def setUp(self):
        import grade
        self.grade = grade

    def write(self, body):
        path = Path(tempfile.mkdtemp()) / "docker-compose.yaml"
        path.write_text(textwrap.dedent(body))
        return path

    def test_it_reads_a_postgres_stack(self):
        compose = self.write("""
            services:
              postgres:
                image: postgres:16
                healthcheck:
                  test: ["CMD", "pg_isready"]
              redis:
                image: redis
                healthcheck:
                  test: ["CMD", "redis-cli", "ping"]
        """)
        self.assertEqual(self.grade.sidecars(compose), {"postgres": True, "redis": True})

    def test_it_reads_a_clickhouse_stack(self):
        compose = self.write("""
            services:
              clickhouse:
                build:
                  context: .
                healthcheck:
                  test: ["CMD-SHELL", "clickhouse-client --query 'SELECT 1'"]
        """)
        self.assertEqual(self.grade.sidecars(compose), {"clickhouse": True})

    def test_the_verifier_is_not_its_own_sidecar(self):
        compose = self.write("""
            services:
              verifier:
                build:
                  context: .
              clickhouse:
                image: clickhouse
        """)
        self.assertEqual(self.grade.sidecars(compose), {"clickhouse": False})

    def test_a_service_without_a_healthcheck_is_only_waited_on_for_start(self):
        # depends_on: service_healthy on a service that never reports health
        # hangs until the timeout, which reads as a broken task rather than a
        # broken overlay.
        overlay = self.grade.overlay_for({"clickhouse": False, "postgres": True})
        self.assertIn("clickhouse:\n        condition: service_started", overlay)
        self.assertIn("postgres:\n        condition: service_healthy", overlay)

    def test_a_stack_with_no_sidecars_declares_no_dependencies(self):
        self.assertNotIn("depends_on", self.grade.overlay_for({}))

    def test_the_real_tasks_resolve(self):
        for task, expected in (("fast-tasks/pg-netbox-ipaddress-device-filter-001",
                                {"postgres", "redis"}),
                               ("local-tasks/ch-metrics-rollup-008-test", {"clickhouse"})):
            compose = Path(task) / "tests" / "docker-compose.yaml"
            if not compose.is_file():
                continue                      # the corpus is not always checked out
            self.assertEqual(set(self.grade.sidecars(compose)), expected, task)


class TestAnUnreachableDatabaseIsCheapToDiscover(unittest.TestCase):
    """Neither client bounds name resolution, and it cost 85 seconds.

    psql and urlopen both start their timeout after the hostname resolves, so
    an unresolvable host costs the resolver's own timeout -- ~21s per call.
    schema_for makes four, which turned a one-millisecond stage into 85 seconds
    of a 1500-second budget. Every ClickHouse task reaches its database over
    HTTP, so this was the ClickHouse side of a hazard the TCP probe had already
    been hardened against.
    """

    def probe(self, *, open_port):
        probe = agent.DatabaseProbe.__new__(agent.DatabaseProbe)
        probe.targets = [agent.DatabaseTarget(engine="clickhouse", host="nowhere",
                                              port="8123", database="metrics")]
        probe._refused = {}
        probe.knocks = []
        probe._tcp_open = lambda host, port, timeout=1.0: (
            probe.knocks.append((host, port)) or open_port)
        return probe

    def test_a_closed_port_is_reported_rather_than_waited_on(self):
        probe = self.probe(open_port=False)
        answer = probe.sql("SHOW CREATE TABLE request_events")
        self.assertIn("unreachable", answer)
        self.assertFalse(agent.DatabaseProbe._usable(answer),
                         "and it must never be rendered as if it were schema")

    def test_a_host_that_just_refused_is_not_knocked_on_again(self):
        # schema_for alone makes four calls; each one paying the bound again
        # buys nothing, because a host that just refused will refuse the next.
        probe = self.probe(open_port=False)
        for _ in range(4):
            probe.sql("SHOW CREATE TABLE request_events")
        self.assertEqual(len(probe.knocks), 1)

    def test_a_database_that_comes_back_is_used_again(self):
        # Remembered for a window, not for the run: a sidecar that restarts
        # mid-task must not be written off for the rest of it.
        probe = self.probe(open_port=False)
        probe.sql("SELECT 1")
        probe._refused[("nowhere", "8123")] -= agent.DatabaseProbe._REFUSAL_WINDOW + 1
        probe._tcp_open = lambda host, port, timeout=1.0: True
        probe._clickhouse = lambda target, query, timeout: "back\n"
        self.assertEqual(probe.sql("SELECT 1"), "back\n")
        self.assertEqual(probe._refused, {}, "and the refusal is forgotten")

    def test_an_open_port_is_queried_normally(self):
        probe = self.probe(open_port=True)
        probe._clickhouse = lambda target, query, timeout: "metrics\n"
        self.assertEqual(probe.sql("SHOW TABLES"), "metrics\n")


class TestDiscoveryCannotResurrectAnExcludedModel(unittest.TestCase):
    """The roster put a model back that measurement had thrown out.

    LADDER leaves deepseek-v4-flash out on evidence: two real tasks, no content
    either time even with the cap removed, 551s spent producing nothing. But
    roster() appends every model the platform allows, and in production
    discover_models succeeds -- so the excluded model came back at position 4
    on every round, each appearance burning a whole completion cap to return
    nothing. Locally discovery usually finds nothing and the built-in roster is
    used, so it never appeared: invisible where it is tested, live where it is
    graded.
    """

    def llm(self, discovered):
        llm = agent.LLM.__new__(agent.LLM)
        llm.discovered = list(discovered)
        llm.unsupported = set()
        return llm

    def test_an_excluded_model_is_not_tried_even_when_the_platform_allows_it(self):
        allowed = sorted(list(agent.MODELS) + ["deepseek/deepseek-v4-flash"])
        self.assertNotIn("deepseek/deepseek-v4-flash",
                         self.llm(allowed).roster(agent.LADDER[0]))

    def test_it_is_not_tried_when_discovery_finds_nothing_either(self):
        self.assertNotIn("deepseek/deepseek-v4-flash",
                         self.llm([]).roster(agent.LADDER[0]))

    def test_an_excluded_model_named_in_a_tier_is_still_dropped(self):
        # Belt and braces: the filter is on the roster, so a tier that named
        # one could not smuggle it back in either.
        roster = self.llm([]).roster(["deepseek/deepseek-v4-flash",
                                      "deepseek/deepseek-v4-pro-0813"])
        self.assertNotIn("deepseek/deepseek-v4-flash", roster)
        self.assertEqual(roster[0], "deepseek/deepseek-v4-pro-0813")

    def test_the_tail_is_ordered_by_capability_not_by_name(self):
        # MODELS is written strongest-first. Sorting by name put gemma
        # (coding 43, agentic 14) ahead of kimi-k2.6 (coding 62).
        roster = self.llm(sorted(agent.MODELS)).roster(agent.LADDER[0])
        self.assertLess(roster.index("moonshotai/kimi-k2.6"),
                        roster.index("google/gemma-4-31b-it"))

    def test_a_model_the_table_does_not_know_is_tried_last(self):
        # Discovery exists so a newly allowed model can be used; it just does
        # not get to outrank one whose capability has been measured.
        roster = self.llm(list(agent.MODELS) + ["aaa/brand-new"]).roster(agent.LADDER[0])
        self.assertEqual(roster[-1], "aaa/brand-new")

    def test_every_allowed_model_still_reaches_the_roster(self):
        allowed = list(agent.MODELS) + ["aaa/brand-new", "zzz/other"]
        self.assertEqual(set(self.llm(allowed).roster(agent.LADDER[0])), set(allowed))


class TestSingleMethodGatesEveryLanguage(unittest.TestCase):
    """21 of 21 non-Python tasks state the rule; the gate fired on none of them.

    `single_method` parses true on every JavaScript, Go and Ruby task in the
    corpus, and every one of their graders enforces it with a source digest and
    a signature check. But check_single_method skipped any file that was not
    Python, so those tasks ran with four checks where a Python task runs eight.

    Validated against all 21 reference solutions: accepted 21, rejected 0, with
    definitions actually resolved for all 21 -- so it cannot reject a correct
    patch on this corpus, and it is not passing them vacuously either.
    """

    JS = ("const db = require('./db');\n"
          "\n"
          "async function report(a, b) {\n"
          "  const rows = await db.query('SELECT 1');\n"
          "  return rows;\n"
          "}\n"
          "\n"
          "module.exports = { report };\n")
    GO = ("package feed\n"
          "\n"
          "import \"database/sql\"\n"
          "\n"
          "func Recent(db *sql.DB) ([]Entry, error) {\n"
          "\trows, err := db.Query(\"SELECT 1\")\n"
          "\treturn nil, err\n"
          "}\n")
    RB = ("module Reports\n"
          "  module_function\n"
          "\n"
          "  def rows(on:)\n"
          "    Member.where(day: on).pluck(:id)\n"
          "  end\n"
          "end\n")

    def gate(self, name, text, edited, *, frozen_imports=True):
        repo = make_repo({name: text})
        repo.write(name, edited)
        checker = agent.Checker(repo, agent.Instruction(
            text="", single_method=True,
            style_constraints=["names the file does not import"] if frozen_imports else []))
        return checker.check_single_method([name])

    # -- the edit the task asks for ---------------------------------------
    def test_an_edit_inside_the_function_passes(self):
        for name, text, old, new in (
                ("app.js", self.JS, "SELECT 1", "SELECT 2"),
                ("feed.go", self.GO, "SELECT 1", "SELECT 2"),
                ("rows.rb", self.RB, "pluck(:id)", "distinct.pluck(:id)")):
            self.assertTrue(self.gate(name, text, text.replace(old, new)).passed, name)

    # -- the edit the grader fails you for --------------------------------
    def test_touching_the_imports_is_caught_in_every_language(self):
        # "use only names the file already requires" -- adding one is the
        # single most common way to lose these tasks.
        for name, text, old, new in (
                ("app.js", self.JS, "const db = require('./db');",
                 "const db = require('./db');\nconst util = require('util');"),
                ("feed.go", self.GO, 'import "database/sql"',
                 'import (\n\t"database/sql"\n\t"fmt"\n)'),
                ("rows.rb", self.RB, "module Reports", "require 'date'\nmodule Reports")):
            result = self.gate(name, text, text.replace(old, new))
            self.assertFalse(result.passed, f"{name} should have been caught")

    def test_a_task_that_permits_imports_is_left_alone(self):
        # One task reads "unchanged apart from imports it genuinely needs" and
        # its reference solution adds two. Enforcing the rule everywhere would
        # reject the correct answer, so only what the statement says is
        # enforced -- this is the case that caught the over-reach.
        edited = self.JS.replace("const db = require('./db');",
                                 "const db = require('./db');\nconst util = require('util');")
        self.assertTrue(self.gate("app.js", self.JS, edited, frozen_imports=False).passed)

    def test_changing_code_below_the_function_is_caught(self):
        result = self.gate("app.js", self.JS,
                           self.JS.replace("module.exports = { report };",
                                           "module.exports = { report, extra: 1 };"))
        self.assertFalse(result.passed)

    # -- and it never guesses ---------------------------------------------
    def test_a_file_it_cannot_read_is_skipped_rather_than_failed(self):
        # The finder outside Python is approximate. An arrow function or a
        # wrapped signature it cannot see must not fail a correct patch, so
        # finding nothing means "cannot judge", never "violation".
        opaque = "const f = (a) => a + 1;\nconst g = (b) => b * 2;\n"
        self.assertEqual(agent.Checker._definitions("x.js", opaque), [])
        self.assertTrue(self.gate("x.js", opaque, opaque.replace("+ 1", "+ 2")).passed)

    def test_ruby_definitions_find_the_method_by_indentation(self):
        found = agent.ruby_definitions(self.RB)
        self.assertIn(("rows", 4, 6, "function"), found)

    def test_ruby_definitions_survive_a_file_it_cannot_match(self):
        self.assertEqual(agent.ruby_definitions("def broken(\n"), [])


class TestAFailingProbeSaysWhatToFix(unittest.TestCase):
    """Three runs of one task shipped unmeasured for the same avoidable reason.

    The probe was never silent -- it raised, every time, with the cause in the
    text. But the agent opened with "the probe produced no query count", which
    reads as "the harness could not run it", and the repair message then asked
    for a probe the model had already supplied. So it sent the same broken one
    back. The exception was buried under a traceback whose middle gets elided.

    Diagnosed from a real run: the model passed the filterset's own parameter
    name where the model field was wanted.
    """

    REAL = ("158 objects imported automatically (use -v 2 for details).\n"
            "\nTraceback (most recent call last):\n"
            '  File "<string>", line 28, in <module>\n'
            '  File "/app/netbox/ipam/filtersets.py", line 795, in filter_device\n'
            "    devices = Device.objects.filter(**{'{}__in'.format(name): value})\n"
            "django.core.exceptions.FieldError: Cannot resolve keyword 'device_id' "
            "into field. Choices are: airflow, asset_tag, id, name, site, virtual_chassis\n")

    def test_it_leads_with_the_exception_not_the_traceback(self):
        detail = agent.Checker._probe_failed(self.REAL)
        self.assertLess(detail.index("FieldError"), 300,
                        "the cause must be readable before the traceback")
        self.assertIn("Cannot resolve keyword 'device_id'", detail)

    def test_it_says_the_probe_is_wrong_not_the_patch(self):
        # The distinction the agent kept failing to draw. A raised probe is no
        # evidence at all about the edit, and telling the model otherwise sent
        # it off to rewrite correct code.
        detail = agent.Checker._probe_failed(self.REAL)
        self.assertIn("Fix the probe, not the patch", detail)
        self.assertIn("says nothing about whether the edit", detail)

    def test_the_full_output_still_follows(self):
        self.assertIn("158 objects imported", agent.Checker._probe_failed(self.REAL))

    def test_a_chained_traceback_reports_the_final_exception(self):
        chained = ("Traceback (most recent call last):\n"
                   "ValueError: bad input\n"
                   "\nThe above exception was the direct cause of:\n"
                   "django.db.utils.ProgrammingError: relation does not exist\n")
        detail = agent.Checker._probe_failed(chained)
        self.assertIn("ProgrammingError", detail[:300])
        self.assertNotIn("ValueError", detail[:300], "the final cause, not the first")

    def test_output_with_no_exception_keeps_the_plain_wording(self):
        detail = agent.Checker._probe_failed("nothing useful here\n")
        self.assertIn("produced no query count", detail)
        self.assertNotIn("Fix the probe", detail)

    def test_silence_still_reports_the_exit_status(self):
        # The other half of the same problem: when there really is no output,
        # the exit status is all there is, and it must survive.
        detail = agent.Checker._probe_failed("[the probe wrote nothing and exited -9; "
                                             "a negative status is a signal]")
        self.assertIn("exited -9", detail)


class TestAnEmptyResultIsNotEvidence(unittest.TestCase):
    """The differential confirmed "unchanged" on two empty results.

    Caught by a live run, not by reading. The model's probe selected no rows,
    so the guard at the top of the method returned early and it counted three
    statements where the real selection issues four -- under the stated limit,
    reported as bounded. The fingerprint was empty on both sides, `[] == []`,
    so the differential agreed the rows were unchanged without having seen one.
    Every check passed, the run declared itself clean and stopped at 43% of its
    budget, and the patch failed the hidden query-count test.

    Two empty results prove nothing, and a count taken down a short-circuit
    path is not the count the task is graded on.
    """

    def measure(self, repo, checker, fingerprint, counts=(3, 3)):
        payload = 'RIDGES_QC{"1": %d, "10": %d}\nRIDGES_QR%s\n' % (
            counts[0], counts[1], json.dumps(fingerprint))
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(a, 0, payload)
        try:
            return checker.measure_query_scaling({"call": "f(N)", "result": "rows"})
        finally:
            agent.run_command = original

    def checker(self, **targets):
        repo = make_repo({"manage.py": "", "app.py": "x\n"})
        return repo, agent.Checker(repo, agent.Instruction(text="", targets=targets))

    def test_an_empty_result_is_reported_as_unmeasured(self):
        repo, checker = self.checker(max_queries=3)
        result = self.measure(repo, checker, [])
        self.assertTrue(result.passed, "nothing about the code failed")
        self.assertFalse(result.verified, "but nothing was measured either")
        self.assertIn("selected no rows", result.detail)

    def test_it_never_claims_the_rows_are_unchanged(self):
        # The exact sentence the failing run printed: "`result` unchanged
        # (0 item(s))". It must not be possible to say that of nothing.
        repo, checker = self.checker()
        self.assertNotIn("unchanged", self.measure(repo, checker, []).detail)

    def test_a_count_under_the_limit_does_not_rescue_an_empty_probe(self):
        # This is what made the run stop early: 3 <= 3 read as success.
        repo, checker = self.checker(max_queries=3)
        self.assertFalse(self.measure(repo, checker, [], counts=(3, 3)).verified)

    def test_it_says_to_exercise_the_task_s_own_entry_point(self):
        repo, checker = self.checker()
        detail = self.measure(repo, checker, []).detail
        self.assertIn("the way the task's own tests do", detail)

    def test_a_probe_that_selected_rows_is_measured_normally(self):
        repo, checker = self.checker(max_queries=3)
        result = self.measure(repo, checker, ["<IPAddress: 10.0.0.1/24>"])
        self.assertTrue(result.passed)
        self.assertTrue(result.verified)

    def test_a_probe_with_no_result_expression_is_unaffected(self):
        # `result` is optional; absent is not the same as empty.
        repo, checker = self.checker(max_queries=3)
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(
            a, 0, 'RIDGES_QC{"1": 3, "10": 3}\n')
        try:
            result = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertTrue(result.verified)

    def test_clickhouse_has_the_same_guard(self):
        repo = make_repo({"package.json": "{}", "app.js": "x\n"})

        class Probe:
            targets = [agent.DatabaseTarget(engine="clickhouse", host="ch", port="8123")]
            def available(self): return True
            def sql(self, query, **kw):
                return "2026-09-09 07:45:12.345\n" if "now64" in query else "1\t900\t0\n"

        checker = agent.Checker(repo, agent.Instruction(text=""))
        checker.probe = Probe()
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(a, 0, "   \n")
        try:
            result = checker.measure_query_scaling({"command": "node run.js"})
        finally:
            agent.run_command = original
        self.assertFalse(result.verified)
        self.assertIn("printed nothing", result.detail)


class TestClickHouseIntrospectionIsReadable(unittest.TestCase):
    """The read-only gate refused every ClickHouse introspection query.

    `\\bSYSTEM\\b` in the mutation denylist put its word boundary on the dot in
    `system.query_log`, so every SELECT against ClickHouse's own tables was
    refused as if it were the `SYSTEM` admin command. Those tables are where a
    ClickHouse engineer looks and where this agent reads work from, and reading
    them mutates nothing -- while PostgreSQL's pg_indexes and
    information_schema were readable the whole time.
    """

    def test_the_introspection_tables_can_be_read(self):
        for query in ("SELECT count() FROM system.query_log",
                      "SELECT name, engine FROM system.tables WHERE database = 'metrics'",
                      "SELECT * FROM system.parts WHERE active",
                      "SELECT sum(read_rows) FROM system . query_log",
                      "EXPLAIN indexes = 1 SELECT * FROM system.tables"):
            self.assertTrue(agent.is_read_only_sql(query), query)

    def test_the_system_command_is_still_refused(self):
        # It is followed by a verb, not a dot, so the lookahead lets it through
        # to the denylist exactly as before.
        for query in ("SYSTEM FLUSH LOGS", "SYSTEM DROP REPLICA 'r'",
                      "SYSTEM SHUTDOWN", "SYSTEM RESTART REPLICA t"):
            self.assertFalse(agent.is_read_only_sql(query), query)

    def test_nothing_else_about_the_gate_moved(self):
        for query in ("SELECT 1; SYSTEM SHUTDOWN",
                      "WITH x AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM x",
                      "DROP TABLE t", "UPDATE t SET a = 1", "OPTIMIZE TABLE t FINAL",
                      "TRUNCATE TABLE t", "ATTACH TABLE t"):
            self.assertFalse(agent.is_read_only_sql(query), query)


class TestClickHouseWorkIsMeasuredLikePostgres(unittest.TestCase):
    """ClickHouse tasks were graded on a number nothing measured.

    The agent could read a ClickHouse schema and EXPLAIN against one, but the
    only work measurement it had was Django's CaptureQueriesContext -- so on
    every ClickHouse task, which in this corpus is a Node application with no
    manage.py, the check that catches a rewrite doing more work than the code
    it replaced simply did not exist. ClickHouse publishes the same evidence in
    system.query_log; these tests drive it from the server side, around a shell
    command, and off the actual file contents, so a baseline that did not
    really revert shows up as the wrong number.
    """

    MARKER = "2026-09-09 07:45:12.345"

    def fixture(self, *, statement="", reads=(9_000_000, 12_000), out=("rows", "rows"),
                user="metrics_app"):
        repo = make_repo({"package.json": "{}", "app.js": "SLOW\n"})
        repo.write("app.js", "FAST\n")
        checker = agent.Checker(repo, agent.Instruction(text=statement))
        checker.probe = self.probe(repo, reads, out, user)
        return repo, checker

    def probe(self, repo, reads, out, user):
        """A ClickHouse probe whose query_log answers from what is on disk."""
        test = self

        class Probe:
            def __init__(self):
                self.targets = [agent.DatabaseTarget(
                    engine="clickhouse", host="clickhouse", port="8123",
                    user=user, database="metrics")]
                self.asked = []

            def available(self):
                return True

            def sql(self, query, **kw):
                self.asked.append(query)
                if "now64" in query:
                    return test.MARKER + "\n"
                if query.startswith("SYSTEM"):
                    return ""
                body = (repo.root / "app.js").read_text().strip()
                rows = reads[0] if body == "SLOW" else reads[1]
                return f"1\t{rows}\t30\n"

        return Probe()

    def run_command(self, repo, out, calls=None):
        def run(command, **kw):
            body = (repo.root / "app.js").read_text().strip()
            if calls is not None:
                calls.append(body)
            return agent.subprocess.CompletedProcess(
                command, 0, out[0] if body == "SLOW" else out[1])
        return run

    def measure(self, repo, checker, out=("rows", "rows"), calls=None, command="node run.js"):
        original = agent.run_command
        agent.run_command = self.run_command(repo, out, calls)
        try:
            return checker.measure_query_scaling({"command": command})
        finally:
            agent.run_command = original

    # -- the query that reads the measurement ----------------------------
    def test_it_excludes_the_agents_own_statements(self):
        # SYSTEM FLUSH LOGS is issued after the watermark and is itself logged,
        # so without this every measurement counts one statement the
        # application never made.
        query = agent.Checker._CH_WORK
        self.assertIn("query NOT ILIKE 'SYSTEM %'", query)
        self.assertIn("positionCaseInsensitive(query, 'system.') = 0", query)
        self.assertIn("type = 'QueryFinish'", query)

    def test_the_watermark_comes_from_the_server_not_this_container(self):
        # The agent's clock and the database's need not agree, and a skewed
        # watermark either drops the queries being measured or picks up
        # somebody else's.
        repo, checker = self.fixture()
        self.measure(repo, checker)
        self.assertIn("now64", checker.probe.asked[0])
        self.assertIn(f"toDateTime64('{self.MARKER}', 3)", checker.probe.asked[2])

    def test_it_scopes_the_measurement_to_the_application_user(self):
        repo, checker = self.fixture(user="metrics_app")
        self.measure(repo, checker)
        self.assertIn("AND user = 'metrics_app'", checker.probe.asked[2])

    def test_a_user_that_is_not_a_plain_name_is_left_out_rather_than_injected(self):
        repo, checker = self.fixture(user="bob'; DROP")
        self.measure(repo, checker)
        self.assertNotIn("DROP", checker.probe.asked[2])

    # -- the verdicts ----------------------------------------------------
    def test_fewer_rows_for_the_same_output_passes_and_reports_the_delta(self):
        repo, checker = self.fixture()
        result = self.measure(repo, checker)
        self.assertTrue(result.passed)
        self.assertIn("9,000,000 -> 12,000 rows read", result.detail)

    def test_more_rows_for_the_same_output_fails_without_classifying_the_task(self):
        # No correctness argument can justify returning exactly what the old
        # code returned while reading more to do it, so this needs no view on
        # what kind of task it is -- and the instruction here says nothing.
        repo, checker = self.fixture(reads=(12_000, 9_000_000))
        result = self.measure(repo, checker)
        self.assertFalse(result.passed)
        self.assertIn("reads more data", result.detail)
        self.assertIn("12,000 -> 9,000,000 rows read", result.detail)

    def test_a_few_percent_more_is_noise_and_is_not_a_failure(self):
        # Part merges move granule boundaries; two runs of an unchanged query
        # need not read byte-identical row counts.
        repo, checker = self.fixture(reads=(100_000, 105_000))
        self.assertTrue(self.measure(repo, checker).passed)

    def test_changed_output_fails_even_though_the_work_fell(self):
        repo, checker = self.fixture()
        result = self.measure(repo, checker, out=("a b c", "a b"))
        self.assertFalse(result.passed)
        self.assertIn("altered what the command returns", result.detail)
        self.assertIn("before: a b c", result.detail)
        self.assertIn("after:  a b", result.detail)
        self.assertIn("9,000,000 -> 12,000 rows read", result.detail,
                      "and the numbers still travel with it")

    def test_whitespace_alone_is_not_a_changed_result(self):
        repo, checker = self.fixture()
        self.assertTrue(self.measure(repo, checker, out=("a b\n", "  a   b  ")).passed)

    # -- the baseline ----------------------------------------------------
    def test_the_baseline_runs_the_command_against_the_reverted_code(self):
        repo, checker = self.fixture()
        calls = []
        self.measure(repo, checker, calls=calls)
        self.assertEqual(calls, ["FAST", "SLOW"])

    def test_the_edit_is_put_back_afterwards(self):
        repo, checker = self.fixture()
        self.measure(repo, checker)
        self.assertEqual((repo.root / "app.js").read_text(), "FAST\n")
        self.assertEqual(repo.changed_files(), ["app.js"])

    def test_it_is_measured_once_and_reused(self):
        repo, checker = self.fixture()
        calls = []
        self.measure(repo, checker, calls=calls)
        self.measure(repo, checker, calls=calls)
        self.assertEqual(calls, ["FAST", "SLOW", "FAST"])

    # -- when it cannot measure ------------------------------------------
    def test_a_command_that_fails_is_unmeasured_not_a_failing_patch(self):
        repo, checker = self.fixture()
        original = agent.run_command
        agent.run_command = lambda command, **kw: agent.subprocess.CompletedProcess(
            command, 1, "MODULE_NOT_FOUND")
        try:
            result = checker.measure_query_scaling({"command": "node run.js"})
        finally:
            agent.run_command = original
        self.assertTrue(result.passed, "nothing about the code failed")
        self.assertFalse(result.verified, "but nothing was measured either")
        self.assertIn("MODULE_NOT_FOUND", result.detail)

    def test_a_command_that_never_reached_the_database_says_so(self):
        repo, checker = self.fixture()
        checker.probe.sql = lambda query, **kw: (
            self.MARKER if "now64" in query else "0\t0\t0\n")
        result = self.measure(repo, checker, out=("x", "x"))
        self.assertFalse(result.verified)
        self.assertIn("logged no queries", result.detail)

    # -- dispatch --------------------------------------------------------
    def test_a_django_project_still_takes_the_django_path(self):
        repo = make_repo({"manage.py": "", "a.py": ""})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        original = agent.run_command
        agent.run_command = lambda *a, **k: agent.subprocess.CompletedProcess(
            a, 0, 'RIDGES_QC{"1": 3, "10": 3}\n')
        try:
            result = checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original
        self.assertIn("bounded: 3 queries", result.detail)

    def test_a_command_is_ignored_when_the_database_is_not_clickhouse(self):
        repo = make_repo({"package.json": "{}"})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        self.assertIsNone(checker.measure_query_scaling({"command": "node run.js"}),
                          "no ClickHouse means no query_log to read this from")


class TestThePromptAsksForTheProbeTheEngineCanRun(unittest.TestCase):
    """Which probe to ask for is the engine's business, not the task shape's.

    A ClickHouse task that happens to read as bounded_queries would otherwise
    be asked for a Django `call`, and there is no shell in a Node application
    to run one -- an ask that cannot be answered, rather than one that merely
    was not needed.
    """

    def findings(self, statement, engine):
        repo = make_repo({"a.py": ""})
        instruction = agent.parse_instruction(statement, repo.root)

        class Probe:
            targets = [agent.DatabaseTarget(engine=engine, host="db", port="5432")]

            def available(self):
                return True

        builder = agent.PromptBuilder(repo, instruction, [], Probe())
        return builder._agent_findings()

    BOUNDED = ("The importer issues one statement per row in a loop; the query count "
               "must not grow with the number of rows.")

    def test_postgres_is_asked_for_the_django_probe(self):
        found = self.findings(self.BOUNDED, "postgresql")
        self.assertIn("selection sizes", found)
        self.assertNotIn("measure.command", found)

    def test_clickhouse_is_never_asked_for_a_shell_it_does_not_have(self):
        found = self.findings(self.BOUNDED, "clickhouse")
        self.assertNotIn("selection sizes", found)

    def test_clickhouse_is_asked_for_a_command_when_the_task_wants_less_read(self):
        found = self.findings("Answer from the rollup so the rows it reads stay "
                              "proportional to the days on screen.", "clickhouse")
        self.assertIn("measure.command", found)


class TestClickHouseWorkMustBeMeasured(unittest.TestCase):
    """The ClickHouse arm of the same guard: asking for the right thing.

    On PostgreSQL the unmeasured number is how many statements are issued; on
    ClickHouse it is how much each one reads. Asking a Node application for a
    Django `call` would be asking for something it cannot give.
    """

    def checker(self, statement, *, clickhouse=True):
        repo = make_repo({"package.json": "{}"})
        checker = agent.Checker(repo, agent.Instruction(text=statement))

        class Probe:
            targets = [agent.DatabaseTarget(engine="clickhouse" if clickhouse else "postgresql",
                                            host="db", port="8123")]

            def available(self):
                return True

        checker.probe = Probe()
        return checker

    def test_it_asks_when_the_task_wants_the_query_to_read_less(self):
        checker = self.checker("Answer from the rollup so the rows it reads stay "
                               "proportional to the days on screen.")
        result = checker.unmeasured_bounded_work({})
        self.assertIsNotNone(result)
        self.assertFalse(result.verified)
        self.assertIn("`measure` command", result.detail)

    def test_it_is_silent_once_a_command_is_supplied(self):
        checker = self.checker("the rows it reads stay proportional to the days on screen")
        self.assertIsNone(checker.unmeasured_bounded_work({"command": "node run.js"}))

    def test_it_is_silent_on_a_task_that_only_wants_different_rows(self):
        # Four of the six ClickHouse statements in the corpus are like this.
        # Asking them for a work probe spends a round on a number nobody grades.
        checker = self.checker("Return one row per cohort with the retention percentage.")
        self.assertIsNone(checker.unmeasured_bounded_work({}))


class TestASilentProbeStillReportsWhatItCan(unittest.TestCase):
    """A probe that says nothing was the one failure carrying no evidence.

    Every path through the script prints something -- the counts, or the
    traceback the script catches -- and run_command labels its own timeouts and
    exec failures. So empty output means the process died without writing, and
    a real run reported "the probe produced no query count. Output was:"
    followed by nothing at all. The exit status is the only thing left, and it
    separates a signal from an interpreter that could not start.
    """

    def probe_with(self, returncode, stdout):
        repo = make_repo({"manage.py": "", "a.py": ""})
        checker = agent.Checker(repo, agent.Instruction(text=""))
        original = agent.run_command
        agent.run_command = lambda cmd, **k: agent.subprocess.CompletedProcess(
            cmd, returncode, stdout)
        try:
            return checker.measure_query_scaling({"call": "f(N)"})
        finally:
            agent.run_command = original

    def test_silence_reports_the_exit_status(self):
        result = self.probe_with(-9, "")
        self.assertFalse(result.verified)
        self.assertIn("exited -9", result.detail)
        self.assertIn("signal", result.detail)

    def test_it_names_the_interpreter_when_the_command_could_not_run(self):
        self.assertIn("could not run", self.probe_with(127, "").detail)

    def test_output_that_exists_is_still_what_gets_reported(self):
        result = self.probe_with(1, "Traceback: UndefinedTable")
        self.assertIn("UndefinedTable", result.detail)
        self.assertNotIn("wrote nothing", result.detail)


class TestBoundedWorkMustBeMeasured(unittest.TestCase):
    """On a task about statement count, supplying no probe is not "done".

    measure_query_scaling returns None when there is nothing to run, so no check
    was appended and `clean` went true. That is the same hole as a probe that
    fails, reached by the other route -- and it is the route that actually
    occurred: two of three runs on the device-filter task supplied no probe, so
    the mechanism built to catch a failing one never had anything to catch.
    """

    def checker(self, text, files=None):
        repo = make_repo(files or {"manage.py": "", "app/q.py": "x = 1\n"})
        return agent.Checker(repo, agent.parse_instruction(text, repo.root))

    BOUNDED = ("Fix `app/q.py`. The work must not grow with the number of rows: "
               "it issues one query per row in a loop today.")
    PLAIN = "Fix the wrong count in `app/q.py`."

    def test_the_predicate_matches_statement_count_tasks_only(self):
        self.assertTrue(self.checker(self.BOUNDED).instruction.bounded_work)
        self.assertFalse(self.checker(self.PLAIN).instruction.bounded_work)
        stated = self.checker("Fix `app/q.py`. Use at most 3 queries.")
        self.assertTrue(stated.instruction.bounded_work, "an explicit ceiling counts too")

    def test_no_probe_on_a_bounded_task_is_unmeasured(self):
        result = self.checker(self.BOUNDED).unmeasured_bounded_work({})
        self.assertIsNotNone(result)
        self.assertTrue(result.passed, "nothing about the code failed")
        self.assertFalse(result.verified, "but the number was never taken")

    def test_a_supplied_probe_leaves_the_verdict_to_the_measurement(self):
        self.assertIsNone(self.checker(self.BOUNDED).unmeasured_bounded_work({"call": "f(N)"}))

    def test_an_ordinary_task_is_not_asked_for_a_probe(self):
        self.assertIsNone(self.checker(self.PLAIN).unmeasured_bounded_work({}))

    def test_a_project_with_no_runner_is_not_asked_for_what_it_cannot_give(self):
        checker = self.checker(self.BOUNDED, {"src/q.js": "x", "package.json": "{}"})
        self.assertTrue(checker.instruction.bounded_work)
        self.assertIsNone(checker.unmeasured_bounded_work({}),
                          "no Django runner: there is no measurement to ask for")

    def test_it_blocks_clean(self):
        checks = [agent.CheckResult("syntax", True, "parsed"),
                  agent.CheckResult("$ manage.py test x", True, "OK"),
                  self.checker(self.BOUNDED).unmeasured_bounded_work({})]
        candidate = agent.Candidate(patch="diff --git a/x b/x\n", checks=checks, diagnosis="d")
        self.assertTrue(candidate.verified, "the tests did run")
        self.assertFalse(candidate.clean, "a bounded task with no measurement is not finished")
