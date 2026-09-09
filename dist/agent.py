"""Ridges miner agent for the database query engineering category.

    def agent_main(input: dict) -> str   # returns a unified diff

Runs inside the task container with the application repository at the workdir and a
live database reachable from it. Only the unified diff returned travels any further:
the patch is applied to an untouched checkout elsewhere and the tests are re-run
there. So the agent may experiment freely here -- run the suite, run EXPLAIN, read
the schema -- and the diff must stay minimal and confined to the files the
instruction names.

Standard library only: an arbitrary application container is not guaranteed to have
anything else. Built from agent.py by build_upload.py; edit that, not this.
"""
import ast
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
# The host may exec this module without registering it in sys.modules, which breaks
# anything resolving a class back to its defining module -- dataclasses' KW_ONLY probe
# is the one that bites. Give them something real.
if __name__ not in sys.modules:
    sys.modules[__name__] = types.ModuleType(__name__)
AGENT_START = time.monotonic()
DEFAULT_AGENT_TIMEOUT = 1500.0
TIMEOUT_SAFETY_MARGIN = 120.0
DEFAULT_MAX_COST_USD = 0.29
COST_TARGET_USD = 0.05
USAGE_URL = 'http://sandbox-proxy:80/api/v1/usage'
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

@dataclass(frozen=True)
class ModelSpec:
    slug: str
    usd_per_m_in: float
    usd_per_m_out: float
    context: int
MODELS: dict[str, ModelSpec] = {'deepseek/deepseek-v4-pro-0813': ModelSpec('deepseek/deepseek-v4-pro-0813', 1.12, 3.36, 1048576), 'qwen/qwen3.8-27b': ModelSpec('qwen/qwen3.8-27b', 0.42, 3.0, 1000000), 'moonshotai/kimi-k2.6': ModelSpec('moonshotai/kimi-k2.6', 0.95, 4.0, 262144), 'moonshotai/kimi-k2.7-code': ModelSpec('moonshotai/kimi-k2.7-code', 0.66, 3.4, 262144), 'minimax/minimax-m3': ModelSpec('minimax/minimax-m3', 0.3, 1.2, 1048576), 'qwen/qwen3.6-35b-a3b': ModelSpec('qwen/qwen3.6-35b-a3b', 0.1, 0.9, 262144), 'google/gemma-4-31b-it': ModelSpec('google/gemma-4-31b-it', 0.09, 0.34, 262144)}
# Escalation ladder, strongest model first. A retry after a failed attempt moves one
# rung up and switches model family: a second opinion from the same architecture tends
# to repeat the same mistake.
LADDER: list[list[str]] = [['deepseek/deepseek-v4-pro-0813', 'qwen/qwen3.8-27b', 'minimax/minimax-m3'], ['qwen/qwen3.8-27b', 'moonshotai/kimi-k2.6', 'deepseek/deepseek-v4-pro-0813'], ['moonshotai/kimi-k2.6', 'moonshotai/kimi-k2.7-code', 'qwen/qwen3.8-27b']]
MAX_REPAIR_ROUNDS = 3
MAX_GATE_SLIPS = 4
MAX_CONTEXT_ROUNDS = 6
COMPLETION_CAP = 12000
MIN_COMPLETION_CAP = 6000
PRICE_SAFETY = 1.3
NUDGE = 'Your previous reply contained no answer text -- the reasoning used up the whole token budget. Answer now with the JSON object only.'
FORCE_MODEL = os.getenv('RIDGES_DB_AGENT_MODEL', '').strip()
SOURCE_SUFFIXES = {'.py', '.sql', '.go', '.rb', '.java', '.kt', '.ts', '.tsx', '.js', '.jsx', '.rs', '.php', '.cs', '.scala', '.ex', '.exs', '.c', '.cpp', '.h', '.hpp', '.yml', '.yaml', '.toml', '.json', '.xml', '.hql', '.erb', '.ini', '.cfg', '.properties', '.prisma', '.env'}
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env', 'dist', 'build', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'site-packages', '.tox', 'target', 'vendor', 'coverage', '.next', '.idea', '.cache'}
MAX_INDEXED_FILES = 20000

def log(message: str) -> None:
    elapsed = time.monotonic() - AGENT_START
    print(f'[db-agent {elapsed:7.1f}s] {message}', flush=True)

def remaining_seconds() -> float:
    try:
        budget = float(os.getenv('AGENT_TIMEOUT') or DEFAULT_AGENT_TIMEOUT)
    except ValueError:
        budget = DEFAULT_AGENT_TIMEOUT
    return budget - TIMEOUT_SAFETY_MARGIN - (time.monotonic() - AGENT_START)
PROJECT_MARKERS = ('manage.py', 'pyproject.toml', 'setup.py', 'requirements.txt', 'package.json', 'go.mod', 'Gemfile', 'pom.xml', 'build.gradle', 'composer.json', 'Cargo.toml', 'mix.exs', 'Makefile', 'schema', 'src')

def _looks_like_project(path: Path) -> bool:
    try:
        return any(((path / marker).exists() for marker in PROJECT_MARKERS))
    except OSError:
        return False

def workdir() -> Path:
    override = os.getenv('RIDGES_WORKDIR')
    if override and Path(override).is_dir():
        return Path(override).resolve()
    cwd = Path(os.getcwd())
    if cwd.is_dir() and cwd.resolve() not in (Path('/'), Path('/installed-agent')) and _looks_like_project(cwd):
        return cwd.resolve()
    for candidate in ('/app', '/repo', '/workspace', '/src'):
        path = Path(candidate)
        if path.is_dir() and _looks_like_project(path):
            return path.resolve()
    chosen = cwd.resolve() if cwd.is_dir() and cwd.resolve() != Path('/') else Path('/app')
    return chosen

def app_python(root: Path) -> str:
    manage = root / 'manage.py'
    if manage.is_file():
        try:
            first = manage.read_text(errors='replace').splitlines()[:1]
        except OSError:
            first = []
        if first and first[0].startswith('#!'):
            exe = first[0][2:].split()[-1]
            if exe.startswith('/') and Path(exe).exists() and ('env' not in Path(exe).name):
                return exe
    for candidate in (root / '.venv/bin/python', root / 'venv/bin/python', Path('/opt/venv/bin/python'), Path('/app/.venv/bin/python')):
        if candidate.exists():
            return str(candidate)
    return sys.executable

def run_command(command: Sequence[str] | str, *, cwd: Path | None=None, timeout: float=300.0, env: dict[str, str] | None=None) -> subprocess.CompletedProcess:
    shell = isinstance(command, str)
    merged = os.environ.copy()
    merged['PYTHONDONTWRITEBYTECODE'] = '1'
    if env:
        merged.update(env)
    shown = command if shell else ' '.join((str(part) for part in command))
    started = time.monotonic()
    try:
        done = subprocess.run(command, shell=shell, cwd=str(cwd or workdir()), env=merged, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ''
        if isinstance(output, bytes):
            output = output.decode('utf-8', 'replace')
        done = subprocess.CompletedProcess(command, 124, output + f'\n[timeout after {timeout}s]')
    except (OSError, ValueError) as exc:
        done = subprocess.CompletedProcess(command, 127, f'[could not run: {exc}]')
    return done

def truncate(text: str, limit: int, *, head_ratio: float=0.4) -> str:
    if text is None:
        return ''
    if len(text) <= limit:
        return text
    head = int(limit * head_ratio)
    tail = limit - head
    return f'{text[:head]}\n... [{len(text) - limit} characters elided] ...\n{text[-tail:]}'

class InferenceError(RuntimeError):
    pass

class BudgetExhausted(RuntimeError):
    pass

class LLM:

    def __init__(self) -> None:
        self.run_id = os.getenv('EVALUATION_RUN_ID') or os.getenv('RUN_ID') or ''
        self.sandbox_proxy = (os.getenv('SANDBOX_PROXY_URL') or '').rstrip('/')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY') or ''
        self.local_key = os.getenv('RIDGES_INFERENCE_API_KEY') or ''
        self.local_base = (os.getenv('RIDGES_INFERENCE_BASE_URL') or '').rstrip('/')
        try:
            self.max_cost = float(os.getenv('RIDGES_MAX_COST_USD') or DEFAULT_MAX_COST_USD)
        except ValueError:
            self.max_cost = DEFAULT_MAX_COST_USD
        self.spent_estimate = 0.0
        self.calls = 0
        self.unsupported: set[str] = set()
        self._usage_unavailable = False
        self.blocked: set[str] = set()
        self.no_reasoning: set[str] = set()
        self._second_pass = False
        self.discovered: list[str] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.per_model: dict[str, float] = {}
        self._insecure_ctx = ssl.create_default_context()
        # Certificate verification off for one retry only. In production a transparent proxy
        # intercepts openrouter.ai; when its CA is not in the container trust store an otherwise
        # healthy request fails verification. The retry is used only after an SSLError and the
        # hop is loopback-local. Normal calls verify.
        self._insecure_ctx.check_hostname = False
        self._insecure_ctx.verify_mode = ssl.CERT_NONE
        self.dead_routes: set[str] = set()
        self.effort = 'minimal'

    def routes(self) -> list[str]:
        found: list[str] = []
        if self.sandbox_proxy and self.openrouter_key:
            found.append(f'{self.sandbox_proxy}/api/v1/chat/completions')
        if self.openrouter_key:
            found.append(OPENROUTER_URL)
        if self.local_base and self.local_key:
            found.append(f'{self.local_base}/chat/completions')
        return found

    def discover_models(self) -> list[str]:
        endpoints = []
        if self.sandbox_proxy:
            endpoints += [f'{self.sandbox_proxy}/api/inference-models', f'{self.sandbox_proxy}/api/v1/models']
        endpoints += ['http://sandbox-proxy:80/api/inference-models', 'http://sandbox-proxy:80/api/v1/models']
        for url in endpoints:
            try:
                request = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(request, timeout=8) as response:
                    payload = json.loads(response.read().decode('utf-8'))
            except Exception:
                continue
            rows = payload.get('data') if isinstance(payload, dict) else payload
            if not isinstance(rows, list) or not rows:
                continue
            names: list[str] = []
            for row in rows:
                if isinstance(row, str):
                    names.append(row)
                elif isinstance(row, dict):
                    name = row.get('name') or row.get('id') or row.get('external_name')
                    if name:
                        names.append(name)
                        spec = MODELS.get(name)
                        cin = row.get('cost_usd_per_million_input_tokens')
                        cout = row.get('cost_usd_per_million_output_tokens')
                        if cin is not None and cout is not None:
                            MODELS[name] = ModelSpec(name, float(cin), float(cout), int(row.get('max_input_tokens') or (spec.context if spec else 128000)))
            if names:
                log(f'discovered {len(names)} allowed model(s) from {url}')
                self.discovered = names
                return names
        return []

    def roster(self, preferred: Sequence[str]) -> list[str]:
        allowed = self.discovered
        if allowed:
            ordered = [m for m in preferred if m in allowed]
            ordered += sorted((m for m in allowed if m not in ordered))
            return ordered or sorted(allowed)
        return list(preferred) + [name for name in MODELS if name not in preferred]

    def spent(self) -> float:
        reported = self._usage()
        if reported is not None:
            return reported
        return self.spent_estimate

    def _usage(self) -> float | None:
        if self._usage_unavailable:
            return None
        if not self.sandbox_proxy:
            self._usage_unavailable = True
            return None
        try:
            request = urllib.request.Request(USAGE_URL, method='GET')
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode('utf-8'))
            value = payload.get('total_cost_usd')
            return float(value) if value is not None else None
        except Exception:
            self._usage_unavailable = True
            return None

    def headroom(self) -> float:
        return max(0.0, self.max_cost - self.spent())

    def _post(self, url: str, payload: dict, headers: dict, timeout: float) -> dict:
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(url, data=body, headers=headers, method='POST')
        try:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except ssl.SSLError:
                with urllib.request.urlopen(request, timeout=timeout, context=self._insecure_ctx) as response:
                    return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')[:500]
            raise InferenceError(f'HTTP {exc.code}: {detail}') from exc
        except urllib.error.URLError as exc:
            raise InferenceError(f'transport failure: {exc.reason}') from exc
        except (OSError, ValueError) as exc:
            raise InferenceError(f'transport failure: {exc.__class__.__name__}: {exc}') from exc

    def _call_via_routes(self, model: str, messages: list[dict], temperature: float, timeout: float, cap: int, reasoning: bool) -> str:
        live = [r for r in self.routes() if r not in self.dead_routes]
        last: InferenceError | None = None
        for url in live:
            key = self.local_key if url.startswith(self.local_base or '\x00') else self.openrouter_key
            try:
                return self._call_openai_style(url, key, model, messages, temperature, timeout, cap, reasoning=reasoning)
            except InferenceError as exc:
                text = str(exc)
                unreachable = 'transport failure' in text or ('HTTP 404' in text and 'model' not in text.lower()) or 'HTTP 502' in text or ('HTTP 503' in text)
                if unreachable and len(live) > 1:
                    self.dead_routes.add(url)
                    log(f'inference route unreachable, retired for this run: {url} ({truncate(text, 120)})')
                    last = exc
                    continue
                raise
        if self.sandbox_proxy:
            return self._call_sandbox_proxy(model, messages, temperature, timeout)
        raise last or InferenceError('no inference transport configured')

    def _call_sandbox_proxy(self, model: str, messages: list[dict], temperature: float, timeout: float) -> str:
        payload = {'run_id': self.run_id, 'evaluation_run_id': self.run_id, 'model': model, 'temperature': temperature, 'messages': messages}
        data = self._post(f'{self.sandbox_proxy}/api/inference', payload, {'Content-Type': 'application/json'}, timeout)
        if isinstance(data, str):
            return data
        return data.get('content') or ''

    def _call_openai_style(self, url: str, key: str, model: str, messages: list[dict], temperature: float, timeout: float, max_tokens: int | None=None, reasoning: bool=True) -> str:
        payload = {'model': model, 'temperature': temperature, 'messages': messages, 'reasoning': {'effort': self.effort} if reasoning else {'enabled': False}}
        if max_tokens:
            payload['max_tokens'] = max_tokens
        data = self._post(url, payload, {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}, timeout)
        self._record_usage(model, data.get('usage') or {})
        try:
            choice = data['choices'][0]
            content = choice['message'].get('content') or ''
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f'malformed completion: {truncate(json.dumps(data), 300)}') from exc
        if not content.strip() and choice.get('finish_reason') == 'length':
            raise InferenceError('completion truncated by max_tokens before any content')
        return content

    def _record_usage(self, model: str, usage: dict) -> None:
        prompt = float(usage.get('prompt_tokens') or 0)
        completion = float(usage.get('completion_tokens') or 0)
        self.prompt_tokens += int(prompt)
        self.completion_tokens += int(completion)
        details = usage.get('prompt_tokens_details') or {}
        self.cached_tokens += int(details.get('cached_tokens') or 0)
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
        self.reasoning_tokens += int(reasoning)
        reported = usage.get('cost')
        if reported is not None:
            cost = float(reported)
        else:
            spec = MODELS.get(model)
            if not spec:
                return
            cost = prompt / 1000000.0 * spec.usd_per_m_in + completion / 1000000.0 * spec.usd_per_m_out
        self.spent_estimate += cost
        self.per_model[model] = self.per_model.get(model, 0.0) + cost

    def report(self) -> str:
        rows = [f'  {model:32} ${cost:.5f}' for model, cost in sorted(self.per_model.items(), key=lambda item: -item[1])]
        source = 'proxy' if self._usage() is not None else 'provider-reported'
        return f'cost ${self.spent():.5f} ({source}) over {self.calls} call(s), {self.prompt_tokens} prompt ({self.cached_tokens} cached) + {self.completion_tokens} completion ({self.reasoning_tokens} reasoning) tokens' + ('\n' + '\n'.join(rows) if rows else '')

    def complete(self, candidates: Sequence[str], messages: list[dict], *, temperature: float=0.0, timeout: float=240.0) -> tuple[str, str]:
        if self.headroom() <= 0.005:
            raise BudgetExhausted(f'cost cap reached (~${self.spent():.4f} of ${self.max_cost:.2f})')
        ordered = list(candidates) if FORCE_MODEL else self.roster(candidates)
        usable = [name for name in ordered if name not in self.unsupported]
        if not usable:
            raise BudgetExhausted('no allowed model is usable with this key: every candidate returned 404 (not allowed) or 402 (insufficient credit)')
        last_error: Exception | None = None
        usable.sort(key=lambda m: m in self.no_reasoning)
        skipped_for_budget = False
        for model in usable:
            if model in self.unsupported:
                continue
            transient = 0
            while transient < 2:
                try:
                    cap = self.affordable_cap(model, messages, COMPLETION_CAP)
                    if cap is None:
                        log(f'{model}: even a {MIN_COMPLETION_CAP}-token reply would exceed the remaining budget (~${self.headroom():.3f}); skipping')
                        skipped_for_budget = True
                        break
                    reasoning = model not in self.no_reasoning
                    outgoing = messages
                    if not reasoning:
                        outgoing = messages + [{'role': 'user', 'content': NUDGE}]
                    if remaining_seconds() < timeout + 60:
                        raise BudgetExhausted('wall-clock budget cannot cover another completion')
                    self.calls += 1
                    before = (self.prompt_tokens, self.completion_tokens, self.cached_tokens, self.reasoning_tokens, self.spent_estimate)
                    content = ''
                    content = self._call_via_routes(model, outgoing, temperature, timeout, cap, reasoning)
                    if content and content.strip():
                        log(f'inference ok: {model} ({len(content)} chars, ~${self.spent_estimate:.4f})')
                        return (content, model)
                    last_error = InferenceError('empty completion')
                except InferenceError as exc:
                    last_error = exc
                    text = str(exc)
                    lowered = text.lower()
                    if 'truncated by max_tokens' in lowered or 'empty completion' in lowered:
                        if reasoning:
                            self.no_reasoning.add(model)
                            log(f'{model} spent the whole cap reasoning; trying the next model family with reasoning on, this one answers without it from now on')
                        else:
                            self.unsupported.add(model)
                            self.blocked.add(model)
                            log(f'{model} returns no content even without reasoning; moving on')
                        break
                    if '403' in text or 'access denied' in lowered:
                        self.unsupported.add(model)
                        self.blocked.add(model)
                        log(f'model refused by policy on this route: {model}')
                        break
                    if '404' in text or 'not supported' in lowered or 'no allowed providers' in lowered:
                        self.unsupported.add(model)
                        self.blocked.add(model)
                        break
                    if '402' in text or 'more credits' in lowered or 'insufficient' in lowered:
                        self.unsupported.add(model)
                        self.blocked.add(model)
                        log(f'model unaffordable on this key: {model}')
                        break
                    if '429' in text and 'cost' in lowered:
                        raise BudgetExhausted(text) from exc
                    transient += 1
                    log(f'inference retry ({model}, attempt {transient}): {truncate(text, 200)}')
                    time.sleep(2 + 3 * (transient - 1))
        spun = [m for m in usable if m in self.no_reasoning and m not in self.unsupported]
        ran_dry = last_error is not None and ('empty' in str(last_error).lower() or 'truncated by max_tokens' in str(last_error).lower())
        if spun and ran_dry and (not self._second_pass):
            self._second_pass = True
            try:
                return self.complete(candidates, messages, temperature=temperature, timeout=timeout)
            finally:
                self._second_pass = False
        if skipped_for_budget and last_error is None:
            raise BudgetExhausted(f'remaining budget (~${self.headroom():.3f}) cannot cover another completion')
        raise InferenceError(f'all models failed; last error: {last_error}')

    def affordable_cap(self, model: str, messages: list[dict], cap: int) -> int | None:
        spec = MODELS.get(model) or ModelSpec(model, 2.0, 8.0, 128000)
        prompt_tokens = sum((len(str(m.get('content', ''))) for m in messages)) / 3.5
        prompt_cost = prompt_tokens / 1000000.0 * spec.usd_per_m_in * PRICE_SAFETY
        room = self.headroom() * 0.95 - prompt_cost
        if room <= 0:
            return None
        max_tokens = int(room / (spec.usd_per_m_out * PRICE_SAFETY) * 1000000.0)
        if max_tokens < MIN_COMPLETION_CAP:
            return None
        if max_tokens < cap:
            log(f'{model}: shrinking completion cap {cap} -> {max_tokens} to stay in budget')
            return max_tokens
        return cap
# What KIND of problem the statement describes, which selects the evidence gathered and
# the checks run. It never selects a fix: no branch here leads to stored patch text, and
# the label is not shown to the model. The statement itself is sent verbatim.
PROBLEM_PATTERNS: dict[str, tuple[str, ...]] = {'bounded_queries': ('\\bN\\+1\\b', 'bounded number', 'grows with', 'scale[sd]? with', 'bulk[_ ]?(create|update|insert)', 'per[- ]row', 'in a loop', 'round[- ]trip', 'query count', 'number of (SQL |)queries'), 'index_or_plan': ('\\bindex\\b', '\\bindexes\\b', '\\bEXPLAIN\\b', '\\bbuffers\\b', 'sequential scan', 'seq scan', 'full scan', '\\bslow\\b', '\\btimeout\\b', 'query plan', 'selective', 'partial index'), 'result_correctness': ('wrong (count|result|value|number)', 'double[- ]count', 'incorrect', 'off by', 'duplicate rows', 'missing rows', 'should return', 'percentage', 'utilization', 'aggregat', '\\bcounts?\\b'), 'authoring': ('\\bauthor\\b', '\\bwrite\\b a query', '\\bimplement\\b', '\\badd\\b an? annotation', 'annotate', 'currently (returns|raises|is) (not|un)implemented'), 'orm_layer': ('\\bORM\\b', 'Django', 'SQLAlchemy', 'queryset', 'QuerySet', 'select_related', 'prefetch_related', 'ActiveRecord', 'Ecto', 'GORM', 'Prisma', 'query builder', 'manager method'), 'raw_sql': ('\\bRawSQL\\b', 'raw SQL', '\\.raw\\(', '\\bSELECT\\b', '\\bJOIN\\b', '\\bCTE\\b', 'WITH RECURSIVE', 'window function', '\\.sql\\b'), 'migration': ('\\bmigration\\b', 'ALTER TABLE', 'schema change', 'AddIndex', 'RunSQL', 'alembic', '\\bDDL\\b'), 'clickhouse': ('ClickHouse', 'MergeTree', 'ReplacingMergeTree', 'materiali[sz]ed view', '\\bPREWHERE\\b', 'ORDER BY key', '\\bsharding key\\b', 'Distributed\\(', '\\bpartition(ing|)\\b key')}
ENGINE_PATTERNS = {'clickhouse': ('clickhouse', 'mergetree', 'prewhere', 'clickhouse_driver', 'chdb'), 'postgresql': ('postgres', 'postgresql', 'psycopg', 'pg_', '\\bpsql\\b', '::regclass')}

@dataclass
class Instruction:
    text: str
    kinds: list[str] = field(default_factory=list)
    engine: str = 'unknown'
    named_paths: list[str] = field(default_factory=list)
    edit_only: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    lint_paths: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    method_hint: str | None = None
    class_hint: str | None = None
    single_method: bool = False
    style_constraints: list[str] = field(default_factory=list)
    traced_hint: list[str] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    unparsed: list[str] = field(default_factory=list)

    @property
    def primary_kind(self) -> str:
        return self.kinds[0] if self.kinds else 'general'
SHELL_FENCES = frozenset({'bash', 'sh', 'shell', 'zsh', 'console', 'shell-session', 'terminal'})

def _fenced_blocks(text: str, *, tagged: bool=False):
    found = [(match.group(1).lower(), match.group(2).strip()) for match in re.finditer('```(\\w*)\\n(.*?)```', text, re.DOTALL)]
    return found if tagged else [body for _, body in found]

def _split_shell_commands(block: str) -> list[str]:
    joined = re.sub('\\\\\\n\\s*', ' ', block)
    commands = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        commands.append(line)
    return commands

def _looks_like_path(token: str) -> bool:
    if not token or len(token) > 200 or ' ' in token:
        return False
    if token.startswith(('http://', 'https://')):
        return False
    suffix = re.search('(\\.[A-Za-z0-9]{1,5})$', token)
    if not suffix:
        return False
    return '/' in token or suffix.group(1).lower() in SOURCE_SUFFIXES

def normalize_repo_path(raw: str) -> str | None:
    if not raw:
        return None
    candidate = raw.strip().strip('`').replace('\\', '/')
    while candidate.startswith('./'):
        candidate = candidate[2:]
    path = Path(candidate)
    if path.is_absolute() or '..' in path.parts or (not candidate):
        return None
    return path.as_posix()

class InstructionParser:

    def __init__(self, text: str, root: Path) -> None:
        self.text = text
        self.flat = re.sub('\\s+', ' ', text)
        self.prose = re.sub('```.*?```', ' ', text, flags=re.DOTALL)
        self.lowered = text.lower()
        self.root = root
        self.parsed = Instruction(text=text)

    def parse(self) -> Instruction:
        for rule in (self._classify, self._detect_engine, self._collect_paths, self._read_permissions, self._read_prohibitions, self._read_method_bound, self._read_style_rules, self._read_commands, self._read_lint_paths, self._collect_identifiers, self._read_numeric_targets, self._resolve_paths, self._audit_extraction):
            rule()
        parsed = self.parsed
        return parsed

    def _classify(self) -> None:
        scores: list[tuple[int, str]] = []
        for kind, patterns in PROBLEM_PATTERNS.items():
            hits = sum((1 for pattern in patterns if re.search(pattern, self.flat, re.IGNORECASE)))
            if hits:
                scores.append((hits, kind))
        self.parsed.kinds = [kind for _, kind in sorted(scores, reverse=True)]

    def _detect_engine(self) -> None:
        scores = {engine: sum((len(re.findall(pattern, self.lowered)) for pattern in patterns)) for engine, patterns in ENGINE_PATTERNS.items()}
        best = max(scores, key=lambda key: scores[key])
        if scores[best]:
            self.parsed.engine = best

    def _collect_paths(self) -> None:
        seen: set[str] = set()
        tokens = re.findall('`([^`\\n]+)`', self.prose) + re.findall('(?<![\\w`/])([\\w./-]+/[\\w./-]+\\.\\w{1,5})', self.prose)
        for token in tokens:
            token = token.strip().strip(',.;:')
            if token in seen or not _looks_like_path(token):
                continue
            seen.add(token)
            self.parsed.named_paths.append(token)
    _PERMISSION = re.compile('(?:you\\s+may\\s+(?:only\\s+)?(?:edit|modify|change)(?:\\s+only)?|(?:edit|modify|change)\\s+only|limit\\s+(?:production\\s+|source\\s+|)changes\\s+to|restrict\\s+(?:your\\s+|)(?:changes|edits)\\s+to|the\\s+only\\s+file\\s+you\\s+may\\s+(?:edit|change|modify)|confine\\s+(?:your\\s+|)(?:changes|edits)\\s+to)', re.IGNORECASE)

    def _read_permissions(self) -> None:
        for match in self._PERMISSION.finditer(self.text):
            window = re.split('\\.\\s|\\n\\s*\\n', self.text[match.end():match.end() + 240], maxsplit=1)[0]
            for token in re.findall('([\\w][\\w./-]*\\.\\w{1,5})', window):
                if _looks_like_path(token) and token not in self.parsed.edit_only:
                    self.parsed.edit_only.append(token)

    def _read_prohibitions(self) -> None:
        for match in re.finditer('Do\\s+not\\s+(?:change|modify|edit|add|touch)\\s+([^.]{0,300})\\.', self.flat, re.IGNORECASE):
            self.parsed.forbidden.append(' '.join(match.group(1).split()))
    _METHOD_BOUND = re.compile('(?:change|modify|edit)\\s+only\\s+that\\s+(?:method|function)|only\\s+that\\s+(?:method|function)|keep\\s+(?:its|the)\\s+signature|(?:the\\s+)?rest\\s+of\\s+(?:its|the)\\s+file\\s+unchanged|bounded\\s+to\\s+one\\s+method|specifically\\s+`[\\w.]+\\(\\)`', re.IGNORECASE)
    _METHOD_NAME = re.compile('`(?:(?P<cls>[A-Za-z_]\\w*)\\.)?(?P<name>[A-Za-z_]\\w*)\\(\\)`|(?:method|function)\\s+`(?:(?P<cls2>[A-Za-z_]\\w*)\\.)?(?P<name2>[A-Za-z_]\\w*)`')

    def _read_method_bound(self) -> None:
        self.parsed.single_method = bool(self._METHOD_BOUND.search(self.flat))
        best_rank = -1
        for match in self._METHOD_NAME.finditer(self.text):
            name = match.group('name') or match.group('name2')
            qualifier = match.group('cls') or match.group('cls2')
            if not name:
                continue
            preceding = self.text[max(0, match.start() - 60):match.start()].lower()
            rank = 2 if qualifier else 0
            if re.search('specifically|namely|the method|change only|limit .{0,40}to', preceding):
                rank += 3
            if rank > best_rank:
                best_rank = rank
                self.parsed.class_hint = qualifier
                self.parsed.method_hint = name
    _FORBIDDEN_CONSTRUCTS = {'loops': 'loops?', 'comprehensions': 'comprehensions?', 'lambdas': 'lambdas?', 'exception handling': 'exception handling|try/except', 'context managers': 'context managers?', 'raw SQL': 'raw SQL'}
    _NO_NEW_NAMES = re.compile('use only names (?:the file|it) already imports|only names .{0,24}already imports|without adding (?:any )?(?:new )?imports|do not add (?:any )?(?:new )?imports', re.IGNORECASE)
    _NO_MATERIALISE = re.compile('do not materiali[sz]e|keep .{0,40}database-backed|must not (?:be )?(?:fetch|load|materiali[sz]e)', re.IGNORECASE)

    def _read_style_rules(self) -> None:
        constraints = self.parsed.style_constraints
        if re.search('no Python loops|without (?:a |)loops?|plain ORM expressions', self.flat, re.IGNORECASE):
            clause = re.search('no Python[^.]{0,200}\\.', self.flat, re.IGNORECASE)
            haystack = clause.group(0) if clause else self.flat
            for label, pattern in self._FORBIDDEN_CONSTRUCTS.items():
                if re.search(pattern, haystack, re.IGNORECASE):
                    constraints.append(label)
        if self._NO_NEW_NAMES.search(self.flat):
            constraints.append('names the file does not import')
        if self._NO_MATERIALISE.search(self.flat):
            constraints.append('materialising rows in Python')
    _RUNNER = re.compile('^\\s*(python|python3|pytest|ruff|flake8|mypy|manage\\.py|\\./|npm|yarn|pnpm|go |cargo|bundle|mvn|gradle|make|psql|clickhouse|tox|nose|rspec|phpunit|node|ruby|deno|bun|php|dotnet|mix\\b|elixir|perl|jest|vitest|java\\b|swift|dart)', re.IGNORECASE | re.MULTILINE)
    _LINTER = re.compile('^\\s*(ruff|flake8|pylint|mypy|black|eslint|gofmt|rubocop)\\b')

    def _read_commands(self) -> None:
        for tag, block in _fenced_blocks(self.text, tagged=True):
            if tag not in SHELL_FENCES and (not (self._RUNNER.match(block) or self._RUNNER.search(block))):
                continue
            lines = _split_shell_commands(block)
            lint_lines = [line for line in lines if self._LINTER.match(line)]
            run_lines = [line for line in lines if not self._LINTER.match(line)]
            if len(run_lines) == 1:
                self.parsed.commands.append(run_lines[0])
            elif run_lines:
                self.parsed.commands.append('set -e\n' + '\n'.join(run_lines))
            self.parsed.commands.extend(lint_lines)
        if not self.parsed.commands:
            for span in re.findall('`([^`]+)`', self.text, re.DOTALL):
                candidate = ' '.join(span.split())
                if self._RUNNER.match(candidate) and len(candidate) > 12:
                    self.parsed.commands.append(candidate)

    def _read_lint_paths(self) -> None:
        for command in self.parsed.commands:
            if not self._LINTER.match(command):
                continue
            for token in command.split():
                if _looks_like_path(token) and token not in self.parsed.lint_paths:
                    self.parsed.lint_paths.append(token)
    _STOP_IDENTIFIERS = {'do_not', 'make_sure', 'the_same', 'read_only', 'task_toml'}

    def _collect_identifiers(self) -> None:
        found: list[str] = []
        found += re.findall('\\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\\b', self.prose)
        found += re.findall('\\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\\b', self.prose)
        for token in re.findall('`([A-Za-z_][\\w.]*)`', self.prose):
            if not _looks_like_path(token):
                found.append(token.split('.')[-1])
        ordered: list[str] = []
        for token in found:
            if token.lower() in self._STOP_IDENTIFIERS or len(token) < 4 or token in ordered:
                continue
            ordered.append(token)
        self.parsed.identifiers = ordered[:40]

    def _read_numeric_targets(self) -> None:
        match = re.search('(?:at\\s+most|no\\s+more\\s+than|a\\s+maximum\\s+of|not\\s+exceed|≤|<=)\\s*(\\d+)\\s+(?:SQL\\s+|database\\s+)?(?:quer(?:y|ies)|statements?|round[-\\s]trips?)', self.flat, re.IGNORECASE) or re.search('\\b(\\d+)\\s+(?:SQL\\s+|database\\s+)?quer(?:y|ies)\\s+(?:in\\s+total|total|regardless|for\\s+the\\s+whole)', self.flat, re.IGNORECASE)
        if match:
            self.parsed.targets['max_queries'] = int(match.group(1))
        match = re.search('(?:read|scan)s?\\s+(?:at\\s+most|no\\s+more\\s+than|fewer\\s+than|under)\\s+([\\d,]+)\\s+rows', self.flat, re.IGNORECASE)
        if match:
            self.parsed.targets['max_read_rows'] = int(match.group(1).replace(',', ''))

    def _resolve_paths(self) -> None:
        self.parsed.named_paths = [path for path in self.parsed.named_paths if (self.root / path).exists() or (self.root / path).parent.is_dir()]
        create = [path for path in self.parsed.named_paths if not (self.root / path).exists()]
        if create:
            self.parsed.targets['create_paths'] = create

    def _audit_extraction(self) -> None:
        shell_fences = [body for tag, body in _fenced_blocks(self.text, tagged=True) if tag in SHELL_FENCES or self._RUNNER.search(body)]
        if shell_fences and (not self.parsed.commands):
            self.parsed.unparsed.append("this agent could not parse the commands out of the instruction's shell block: copy every check it names into `verify`, verbatim, or the patch can only be checked statically and nothing will run it")
        if self.parsed.single_method and (not self.parsed.method_hint):
            self.parsed.unparsed.append('the instruction bounds the change to one method but names no symbol this agent could resolve: put the method in `constraints.bounded_to_method` as `Class.method` so the edit can be checked against it')
        about_files = any((not re.match('\\s*that\\s+(method|function)', self.text[match.end():match.end() + 24], re.IGNORECASE) for match in self._PERMISSION.finditer(self.text)))
        if about_files and (not (self.parsed.edit_only or self.parsed.lint_paths)):
            self.parsed.unparsed.append('the instruction restricts which files may change but this agent could not read the paths out of that sentence: list them in `constraints.editable_files`')

def parse_instruction(text: str, root: Path) -> Instruction:
    return InstructionParser(text, root).parse()
QUERY_SIGNALS: tuple[tuple[str, float], ...] = (('\\bRawSQL\\b|\\.raw\\(|raw_sql|text\\(\\s*[\\"\']|find_by_sql|\\$queryRaw|\\$executeRaw|knex\\.raw\\(', 3.0), ('cursor\\.execute|connection\\.cursor|executemany|execute_batch|\\.QueryRow(Context|)\\(|\\.Query(Context|)\\(|\\.Exec(Context|)\\(|sql\\.DB|sqlx\\.', 2.0), ('\\bSELECT\\b', 1.2), ('\\bFROM\\s+\\w', 0.8), ('\\bJOIN\\b', 1.0), ('\\bWHERE\\b', 0.8), ('\\bGROUP\\s+BY\\b|\\bORDER\\s+BY\\b|\\bHAVING\\b|\\bLIMIT\\s+BY\\b|\\bPREWHERE\\b', 1.0), ('\\bINSERT\\s+INTO\\b|\\bUPDATE\\b.+\\bSET\\b|\\bDELETE\\s+FROM\\b', 2.5), ('\\bOVER\\s*\\(|\\bPARTITION\\s+BY\\b|\\bWITH\\s+RECURSIVE\\b|\\bLATERAL\\b|\\bEXISTS\\s*\\(', 1.5), ('\\.annotate\\(|\\.aggregate\\(|Subquery\\(|OuterRef\\(|\\bWindow\\(|\\bExists\\(', 2.0), ('\\.select_related\\(|\\.prefetch_related\\(|\\.only\\(|\\.defer\\(|\\.values(_list|)\\(', 1.5), ('bulk_create|bulk_update|\\.update\\(|\\.delete\\(', 1.5), ('session\\.query|select\\(|join\\(|\\.filter\\(|\\.exclude\\(|\\.order_by\\(', 1.0), ('session\\.(execute|scalars|scalar)\\(|sa\\.(select|func|text|case)\\(|joinedload\\(|selectinload\\(|\\.subquery\\(', 1.5), ('knex\\(|\\.whereIn\\(|\\.whereRaw\\(|\\.leftJoin\\(|\\.innerJoin\\(|\\.groupBy\\(|\\.havingRaw\\(|\\.select\\(|\\.insert\\(', 1.5), ('prisma\\.\\w+\\.(findMany|findFirst|findUnique|create|update|upsert|delete|aggregate|groupBy|count)\\(', 2.0), ('createQueryBuilder\\(|getRepository\\(|leftJoinAndSelect\\(|\\.getMany\\(|\\.getRawMany\\(', 2.0), ('db\\.(Where|Preload|Joins|Select|Find|First|Model|Raw|Exec)\\(|pool\\.(Query|QueryRow|Exec)\\(|pgx\\.', 1.5), ('\\.includes\\(|\\.joins\\(|\\.left_joins\\(|\\.group\\(|\\.pluck\\(|\\.find_each\\(|\\.where\\(|\\.where\\.not\\(', 1.0), ('@Query\\(|createQuery\\(|createNativeQuery\\(|JOIN FETCH|dsl\\.select|\\.fetch\\(', 1.5), ('MergeTree|SETTINGS\\s+\\w+|clickhouse|client\\.(query|execute|command|query_df|insert)\\(|windowFunnel|argMax|uniqExact|quantiles?', 2.5), ('CREATE\\s+(UNIQUE\\s+|)INDEX|AddIndex|RemoveIndex|models\\.Index\\(|Index\\(fields|@@index|add_index|USING\\s+(gin|gist|brin|btree)', 2.5))
QUERY_PATTERNS: tuple[tuple[re.Pattern, float], ...] = tuple(((re.compile(pattern), weight) for pattern, weight in QUERY_SIGNALS))

def query_density(text: str) -> float:
    raw = sum((weight * len(pattern.findall(text)) for pattern, weight in QUERY_PATTERNS))
    return raw / (len(text.splitlines()) ** 0.5 + 4.0)

@dataclass
class RepoFile:
    path: Path
    relative: str
    text: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

def read_source(path: Path) -> str:
    with open(path, encoding='utf-8', newline='') as handle:
        return handle.read()

def write_source(path: Path, text: str) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)

class Repository:

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: list[str] = []
        self._cache: dict[str, str] = {}
        self._snapshots: dict[str, str | None] = {}
        self._index()

    def _index(self) -> None:
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted((name for name in dirnames if name not in SKIP_DIRS and (not name.startswith('.'))))
            for name in sorted(filenames):
                if Path(name).suffix.lower() not in SOURCE_SUFFIXES:
                    continue
                full = Path(dirpath) / name
                try:
                    if full.is_symlink() or full.stat().st_size > 2000000:
                        continue
                except OSError:
                    continue
                self.files.append(str(full.relative_to(self.root)))
                count += 1
                if count >= MAX_INDEXED_FILES:
                    log(f'index truncated at {MAX_INDEXED_FILES} files')
                    return
        log(f'indexed {len(self.files)} source files under {self.root}')

    def read(self, relative: str) -> str | None:
        if relative in self._cache:
            return self._cache[relative]
        path = self.root / relative
        try:
            text = read_source(path)
        except (OSError, UnicodeDecodeError):
            return None
        self._cache[relative] = text
        return text

    def snapshot(self, relative: str) -> None:
        if relative in self._snapshots:
            return
        path = self.root / relative
        try:
            self._snapshots[relative] = read_source(path) if path.exists() else None
        except (OSError, UnicodeDecodeError):
            self._snapshots[relative] = None

    def write(self, relative: str, text: str) -> None:
        self.snapshot(relative)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_source(path, text)
        self._cache[relative] = text

    def revert(self, relative: str) -> None:
        if relative not in self._snapshots:
            return
        original = self._snapshots[relative]
        path = self.root / relative
        try:
            if original is None:
                path.unlink(missing_ok=True)
                self._cache.pop(relative, None)
            else:
                write_source(path, original)
                self._cache[relative] = original
        except OSError as exc:
            log(f'WARNING: could not restore {relative} ({exc}); the checkout may not be pristine')

    def revert_all(self) -> None:
        for relative in list(self._snapshots):
            self.revert(relative)

    def original(self, relative: str) -> str | None:
        if relative in self._snapshots:
            return self._snapshots[relative]
        return self.read(relative)

    def changed_files(self) -> list[str]:
        changed = []
        for relative, original in self._snapshots.items():
            current = None
            path = self.root / relative
            if path.exists():
                try:
                    current = read_source(path)
                except (OSError, UnicodeDecodeError):
                    continue
            if current != original:
                changed.append(relative)
        return sorted(changed)
STOPWORDS = frozenset('\na an and are as at be been before but by can do does for from has have if in into is it\nits may must no not of on only or should so than that the their then there these this\nthose to use used using was were when which while will with without you your work run\nalso any each every same such keep make change changed unchanged rest file files line\nlines method function name names instruction task check checks finishing following\nbecome becomes became becoming stay stays staying stayed remain remains remaining\ncurrently instead both being handful number numbers added adding still now exact exactly\ncorrect correctly wrong incorrect incorrectly slow slowly fast expensive cheap large small\nmany few several some most more less least first last new old one two three per well very\njust even quite rather already again once twice however therefore because since although\nwhether either neither between within across through against toward towards during after\nunder over above below behind result results returns returned returning gives given give\ntakes taken take shows shown show needs needed need wants want expected expects expect\n'.split())
ROLE_WORDS = frozenset('\nmigration index manager queryset filter filterset serializer view signal model cache\nsearch api admin form util service repository dao schema query sql handler router\ncontroller resource endpoint task job worker command middleware backend store\n'.split())
_TEST_PATH = re.compile('(^|/)(tests?|specs?|__tests__|fixtures?)(/|$)|(^|/)test_[^/]*$|_test\\.\\w+$|\\.(test|spec)\\.\\w+$|conftest|fixture')
_SEED_PATH = re.compile('(^|/|_)seeds?(_|\\.|/|$)|sample_data|(^|/)data/')
_PATH_NOISE = frozenset('\npy js ts tsx jsx go rb rs java kt php sql yml yaml toml json index init main src lib app\ninternal pkg cmd core utils util common base\n'.split())

def _path_tokens(relative: str) -> set[str]:
    return {t for t in re.split('[/._\\-]+', relative.lower()) if t and t not in _PATH_NOISE}

def _token_match(word: str, tokens: set[str]) -> bool:
    stem = word.rstrip('s')
    return any((t.rstrip('s') == stem or t.startswith(stem) for t in tokens))
MENTIONED_PATH_BONUS = 15.0
ROLE_WORD_BONUS = 12.0

class Ranker:

    def __init__(self, repo: Repository, instruction: Instruction) -> None:
        self.repo = repo
        self.instruction = instruction
        self.keywords = self._keywords()
        self.idf = self._document_frequencies()
        absent = math.log(1.0 + max(1, len(self.repo.files)))
        present = [w for w in self.keywords if self.idf.get(w, absent) < absent]
        self.keywords = present

    def _keywords(self) -> list[str]:
        prose = re.sub('```.*?```', ' ', self.instruction.text, flags=re.DOTALL)
        counts: dict[str, int] = {}
        for word in re.findall('[A-Za-z_][A-Za-z0-9_]{3,}', prose):
            lowered = word.lower()
            if lowered in STOPWORDS:
                continue
            counts[lowered] = counts.get(lowered, 0) + 1
        for identifier in self.instruction.identifiers:
            lowered = identifier.lower()
            counts[lowered] = counts.get(lowered, 0) + 3
        if self.instruction.method_hint:
            counts[self.instruction.method_hint.lower()] = 12
        if self.instruction.class_hint:
            counts[self.instruction.class_hint.lower()] = 12
        self._counts = counts
        ordered = sorted(counts, key=lambda word: (-counts[word], word))
        return ordered[:30]

    def _document_frequencies(self) -> dict[str, float]:
        total = max(1, len(self.repo.files))
        frequency = {word: 0 for word in self.keywords}
        for relative in self.repo.files:
            text = self.repo.read(relative)
            if text is None:
                continue
            lowered = text.lower()
            for word in self.keywords:
                if word in lowered:
                    frequency[word] += 1
        return {word: math.log(1.0 + total / (1.0 + count)) for word, count in frequency.items()}

    def score(self, relative: str) -> tuple[float, list[int]]:
        text = self.repo.read(relative)
        if text is None:
            return (0.0, [])
        lines = text.splitlines()
        lowered = text.lower()
        hot: dict[int, float] = {}
        signal = 0.0
        for number, line in enumerate(lines, start=1):
            line_score = 0.0
            for pattern, weight in QUERY_PATTERNS:
                if pattern.search(line):
                    line_score += weight
            lowered_line = line.lower()
            for word in self.keywords[:12]:
                if word in lowered_line:
                    line_score += min(3.0, self.idf.get(word, 0.0))
            if line_score:
                hot[number] = line_score
                signal += line_score
        density = signal / (len(lines) ** 0.5 + 4.0)
        topical = 0.0
        for word in self.keywords:
            occurrences = lowered.count(word)
            if occurrences:
                topical += min(occurrences, 4) * self.idf.get(word, 0.0)
        topical /= len(lines) ** 0.5 + 4.0
        symbols = ' '.join((match.group(2).lower() for match in re.finditer('^\\s*(class|def|func|function|type|struct)\\s+(\\w+)', text, re.MULTILINE)))
        symbol_score = sum((4.0 * self.idf.get(word, 0.0) for word in self.keywords if len(word) >= 4 and word in symbols))
        path_lower = relative.lower()
        path_score = 0.0
        tokens = _path_tokens(relative)
        for word in self.keywords:
            if len(word) >= 4 and _token_match(word, tokens):
                path_score += 3.0 * self.idf.get(word, 0.0)
                if word.rstrip('s') in ROLE_WORDS:
                    path_score += ROLE_WORD_BONUS
        for hint, weight in (('manager', 6.0), ('queryset', 6.0), ('repositor', 6.0), ('migration', 4.0), ('query', 4.0), ('dao', 4.0), ('sql', 4.0), ('model', 2.0), ('filters', 2.0), ('store', 1.5), ('search', 1.5)):
            if hint in path_lower:
                path_score += weight
        if relative in self.instruction.named_paths:
            path_score += MENTIONED_PATH_BONUS
        total = 3.0 * density + 8.0 * topical + symbol_score + path_score
        penalty = 1.0
        if _TEST_PATH.search(path_lower):
            penalty = 0.1
        elif _SEED_PATH.search(path_lower):
            penalty = 0.3
        if '/migrations/' in path_lower and 'migration' not in self.keywords:
            penalty *= 0.5
        total *= penalty
        top = sorted(hot, key=lambda number: hot[number], reverse=True)[:12]
        return (total, sorted(top))

class CallGraph:
    _DEF = re.compile('^\\s*(?:class|def|async def|func|function)\\s+(\\w+)', re.MULTILINE)
    _CALL = re.compile('\\b([A-Za-z_]\\w{2,})\\s*\\(')
    _ATTR = re.compile('\\.([A-Za-z_]\\w{2,})\\b')
    _TEST_PATH = re.compile('(^|/)(tests?|testing|spec|fixtures?|conftest\\.py)(/|$)|(^|/)test_[^/]*$|_test\\.[a-z]+$', re.IGNORECASE)

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.defined_in: dict[str, list[str]] = {}
        self.defined_at: dict[str, list[tuple[str, int]]] = {}
        self.mentions: dict[str, set[str]] = {}
        self._weights: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        for relative in self.repo.files:
            if self._TEST_PATH.search(relative):
                continue
            text = self.repo.read(relative)
            if text is None:
                continue
            for match in self._DEF.finditer(text):
                name = match.group(1)
                self.defined_in.setdefault(name, []).append(relative)
                self.defined_at.setdefault(name, []).append((relative, text.count('\n', 0, match.start()) + 1))
            used = set(self._CALL.findall(text)) | set(self._ATTR.findall(text))
            self.mentions[relative] = used

    def query_weight(self, relative: str) -> float:
        if relative not in self._weights:
            self._weights[relative] = query_density(self.repo.read(relative) or '')
        return self._weights[relative]

    def seeds_from_keywords(self, keywords: Sequence[str], limit: int=40) -> list[str]:
        wanted = [word for word in keywords if len(word) >= 3]
        hits: list[tuple[int, str]] = []
        for symbol, files in self.defined_in.items():
            lowered = symbol.lower()
            matched = sum((1 for word in wanted if word in lowered))
            if matched:
                weight = max((self.query_weight(f) for f in files[:3]), default=0.0)
                hits.append((matched * 100 + int(weight), symbol))
        hits.sort(key=lambda hit: (-hit[0], hit[1]))
        return [symbol for _, symbol in hits[:limit]]

def locate_targets(repo: Repository, instruction: Instruction, limit: int=6) -> list[tuple[str, list[int]]]:
    ranker = Ranker(repo, instruction)
    explicit = [path for path in instruction.edit_only or instruction.lint_paths if (repo.root / path).exists()]
    if explicit:
        results = [(path, ranker.score(path)[1]) for path in explicit[:limit]]
        log(f'instruction names editable paths: {[path for path, _ in results]}')
        return results
    scored: list[tuple[float, str, list[int]]] = []
    for relative in repo.files:
        value, hot = ranker.score(relative)
        if value > 0:
            scored.append((value, relative, hot))
    traced: dict[str, float] = {}
    try:
        graph = CallGraph(repo)
        seeds = list(instruction.identifiers)
        if instruction.method_hint:
            seeds.append(instruction.method_hint)
        if instruction.class_hint:
            seeds.append(instruction.class_hint)
        resolved = [name for name in seeds if name in graph.defined_in]
        if not resolved:
            resolved = graph.seeds_from_keywords(ranker.keywords)
            log(f'no named symbol resolved; seeding the trace from vocabulary: {resolved[:6]}')
        traced = graph.trace(resolved)
    except Exception as exc:
        log(f'call-graph tracing skipped: {exc}')
    scored.sort(key=lambda item: (-item[0], item[1]))
    log('keywords: ' + ', '.join(ranker.keywords[:10]))
    if traced:
        reached = sorted(traced, key=lambda k: (-traced[k], k))[:4]
        instruction.traced_hint = [r for r in reached if r not in {path for _, path, _ in scored[:limit]}][:3]
        log(f'call-graph reached {len(traced)} query-bearing file(s); strongest: {reached}')
    log('top candidates: ' + ', '.join((f'{path} ({value:.0f})' for value, path, _ in scored[:limit])))
    instruction.candidate_scores = [value for value, _, _ in scored[:limit]]
    top = scored[:limit]
    margin = (top[0][0] - top[1][0]) / top[0][0] if len(top) > 1 and top[0][0] else 1.0
    return [(path, hot) for _, path, hot in top]

@dataclass
class Slice:
    relative: str
    start: int
    end: int
    label: str

    def render(self, repo: Repository) -> str:
        text = repo.read(self.relative) or ''
        lines = text.splitlines()
        chunk = lines[self.start - 1:self.end]
        numbered = '\n'.join((f'{self.start + offset:5d}| {line}' for offset, line in enumerate(chunk)))
        return f'--- {self.relative} lines {self.start}-{self.end} ({self.label}) ---\n{numbered}'

def python_definitions(text: str) -> list[tuple[str, int, int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found: list[tuple[str, int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{prefix}{child.name}'
                kind = 'class' if isinstance(child, ast.ClassDef) else 'function'
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                found.append((name, start, child.end_lineno or child.lineno, kind))
                walk(child, f'{name}.')
    walk(tree, '')
    return found
_BLOCK_START = re.compile('^\\s*(?:(?:public|private|protected|static|final|async|export|pub|def|func|function|class|module|type|impl|interface|struct)\\b|[\\w<>\\[\\], ]+\\s+\\w+\\s*\\()')

def generic_blocks(text: str) -> list[tuple[str, int, int, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, int, int, str]] = []
    for index, line in enumerate(lines):
        if not _BLOCK_START.match(line) or '{' not in line:
            continue
        depth = 0
        for end_index in range(index, min(len(lines), index + 400)):
            depth += lines[end_index].count('{') - lines[end_index].count('}')
            if depth <= 0 and end_index > index:
                name = re.sub('[^\\w]+', ' ', line).strip()[:60]
                blocks.append((name, index + 1, end_index + 1, 'block'))
                break
    return blocks

def slice_around(repo: Repository, relative: str, hot_lines: Sequence[int], instruction: Instruction, *, budget_lines: int=320) -> list[Slice]:
    text = repo.read(relative)
    if text is None:
        return []
    total = len(text.splitlines())
    if total <= budget_lines:
        return [Slice(relative, 1, total, 'whole file')]
    if relative.endswith('.py'):
        definitions = python_definitions(text)
    else:
        definitions = generic_blocks(text)
    chosen: list[Slice] = []
    used: set[tuple[int, int]] = set()
    if instruction.method_hint:
        methods = [(name, start, end, kind) for name, start, end, kind in definitions if kind != 'class' and name.split('.')[-1] == instruction.method_hint]
        qualified = f'{instruction.class_hint}.{instruction.method_hint}'
        hit = next((m for m in methods if m[0] == qualified), None) or next(iter(methods), None)
        if hit:
            name, start, end, kind = hit
            chosen.append(Slice(relative, start, end, f'{kind} {name}'))
            used.add((start, end))
    for line in hot_lines:
        enclosing = [(name, start, end, kind) for name, start, end, kind in definitions if start <= line <= end and kind != 'class']
        if enclosing:
            name, start, end, kind = min(enclosing, key=lambda item: item[2] - item[1])
        else:
            name, start, end, kind = ('context', max(1, line - 25), min(total, line + 25), 'window')
        if (start, end) in used:
            continue
        used.add((start, end))
        chosen.append(Slice(relative, start, end, f'{kind} {name}'))
    if not chosen:
        return [Slice(relative, 1, min(total, budget_lines), 'file head')]
    if any((piece.label.startswith(('function', 'block')) for piece in chosen)):
        named = [piece for piece in chosen if not piece.label.startswith('window')]
        if named:
            chosen = named
    chosen.sort(key=lambda item: item.start)
    spent = 0
    kept: list[Slice] = []
    for piece in chosen:
        span = piece.end - piece.start + 1
        if spent + span > budget_lines * 2 and kept:
            break
        kept.append(piece)
        spent += span
    return kept

def package_map(repo: Repository, relative: str, limit: int=25) -> str:
    parts = Path(relative).parts
    if len(parts) < 2:
        return ''
    package = str(Path(relative).parent)
    while True:
        siblings = [f for f in repo.files if f.startswith(package + '/') and f != relative and (not _TEST_PATH.search(f.lower())) and ('/migrations/' not in f) and (Path(f).stem != '__init__')]
        if len(siblings) >= 2 or '/' not in package:
            break
        package = str(Path(package).parent)
    depth = package.count('/') + 1
    lines: list[str] = []
    for other in sorted(siblings, key=lambda f: (f.count('/') - depth, f)):
        if not other.startswith(package + '/') or other == relative:
            continue
        text = repo.read(other)
        if text is None:
            continue
        defs = python_definitions(text) if other.endswith('.py') else generic_blocks(text)
        names = [f"{('class ' if kind == 'class' else '')}{name}" for name, _, _, kind in defs if '.' not in name][:6]
        if names:
            lines.append(f"  {other}: {', '.join(names)}")
        if len(lines) >= limit:
            lines.append('  ...')
            break
    return 'other modules in this package (request one with read_file if the target uses their objects):\n' + '\n'.join(lines) if lines else ''

def file_outline(repo: Repository, relative: str, limit: int=80) -> str:
    text = repo.read(relative)
    if text is None:
        return ''
    if relative.endswith('.py'):
        definitions = python_definitions(text)
    else:
        definitions = generic_blocks(text)
    if not definitions:
        return ''
    rows = [f'  L{start:<5} {kind:8} {name}' for name, start, _, kind in definitions[:limit]]
    imports = ''
    if relative.endswith('.py'):
        heads = [line for line in text.splitlines()[:80] if line.startswith(('import ', 'from ')) or re.match('^\\s+(import|from)\\s', line)]
        if heads:
            imports = '\nimports available in this file:\n' + '\n'.join((f'  {line.strip()}' for line in heads[:40]))
    return f'outline of {relative}:\n' + '\n'.join(rows) + imports

@dataclass
class DatabaseTarget:
    engine: str
    priority: int = 5
    dsn: str | None = None
    host: str = ''
    port: str = ''
    user: str = ''
    password: str = ''
    database: str = ''

def target_url(target: 'DatabaseTarget', *, password: bool=False) -> str:
    credentials = target.user
    if credentials and password and target.password:
        credentials += f':{target.password}'
    return f"{target.engine}://{(credentials + '@' if credentials else '')}{target.host}:{target.port}/{target.database}"

class DatabaseProbe:

    def __init__(self, repo: Repository, instruction: Instruction) -> None:
        self.repo = repo
        self.instruction = instruction
        self.targets: list[DatabaseTarget] = []
        self.notes: list[str] = []
        self._discover()

    def _discover(self) -> None:
        self.attempts: list[tuple[str, bool]] = []
        self._env = self._load_env_files()
        self._from_env()
        self._from_django()
        self._from_source()
        offline = bool(os.getenv('RIDGES_PROBE_NO_PING'))
        if not self.targets and (not offline):
            self._from_network()
        if offline:
            pass
        elif self.targets:
            self.targets = self._first_reachable(self.targets)
            if not self.targets:
                self._from_network()
                self.targets = self._first_reachable(self.targets)
        if self.targets:
            for target in self.targets:
                log(f'database: {target_url(target)} (SELECT 1 answered)')
        else:
            log('no database credentials discovered; proceeding on static evidence')
    _NOT_A_DATABASE_PORT = {'6379', '6380', '11211', '27017', '27018', '9200', '9300', '5672', '15672', '2181', '9092', '80', '443', '3000', '8000', '8080'}

    def _add(self, target: DatabaseTarget) -> None:
        if not target.host or not re.fullmatch('[\\w.-]+', target.host):
            return
        if target.port and (not target.port.isdigit()):
            return
        if target.port in self._NOT_A_DATABASE_PORT:
            return
        if target.user and (not re.fullmatch('[\\w.$-]+', target.user)):
            return
        if target.password and (not re.fullmatch('[^\\s\\"\'<>{}$]+', target.password)):
            return
        key = (target.engine, target.host, target.port, target.database, target.user)
        for existing in self.targets:
            if (existing.engine, existing.host, existing.port, existing.database, existing.user) == key:
                if target.priority < existing.priority:
                    existing.priority = target.priority
                    self.targets.sort(key=lambda item: item.priority)
                return
        if self.instruction.engine != 'unknown' and target.engine != self.instruction.engine:
            target.priority += 10
        self.targets.append(target)
        self.targets.sort(key=lambda item: item.priority)
    _ENV_FILES = ('.env', '.env.local', '.env.development', '.env.test', '.env.example')

    def _load_env_files(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for name in self._ENV_FILES:
            path = self.repo.root / name
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip().removeprefix('export ').strip()
                merged.setdefault(key, value.strip().strip('"').strip("'"))
        merged.update(os.environ)
        return merged

    def _expand(self, value: str) -> str:
        env = self._env

        def sub(match: re.Match) -> str:
            return env.get(match.group('name'), match.group('default') or '')
        value = re.sub('\\$\\{(?P<name>\\w+)(?::-(?P<default>[^}]*))?\\}', sub, value)
        value = re.sub('\\$(?P<name>[A-Za-z_]\\w*)(?P<default>)', sub, value)
        value = re.sub('env\\(\\s*["\\\'](?P<name>\\w+)["\\\']\\s*\\)(?P<default>)', sub, value)
        return value
    _URL_KEY = re.compile('(DATABASE_URL|DATABASE_URI|DB_URL|DSN|CLICKHOUSE_URL|POSTGRES_URL|POSTGRESQL_URL)$')
    _PART_KEY = re.compile('^(?P<prefix>.*?)_?(?P<part>HOST|HOSTNAME|PORT|HTTP_PORT|USER|USERNAME|PASSWORD|PASS|DB|DATABASE|DBNAME|NAME)$')

    def _from_env(self) -> None:
        for key, value in self._env.items():
            if self._URL_KEY.search(key) and value:
                self._add_url(self._expand(value), priority=1)
        families: dict[str, dict[str, str]] = {}
        for key, value in self._env.items():
            match = self._PART_KEY.match(key)
            if match and value:
                families.setdefault(match.group('prefix'), {})[match.group('part')] = value
        for prefix, parts in families.items():
            upper = prefix.upper()
            if upper in ('PG',) or 'POSTGRES' in upper or 'PGSQL' in upper:
                engine = 'postgresql'
            elif 'CLICKHOUSE' in upper or upper.endswith('CH') or upper == 'CH':
                engine = 'clickhouse'
            elif 'DB' in upper or 'DATABASE' in upper or 'SQL' in upper:
                engine = 'postgresql'
            else:
                continue
            host = parts.get('HOST') or parts.get('HOSTNAME')
            if not host:
                continue
            self._add(DatabaseTarget(engine=engine, priority=1, host=self._expand(host), port=parts.get('PORT') or parts.get('HTTP_PORT') or ('8123' if engine == 'clickhouse' else '5432'), user=parts.get('USER') or parts.get('USERNAME') or '', password=parts.get('PASSWORD') or parts.get('PASS') or '', database=parts.get('DB') or parts.get('DATABASE') or parts.get('DBNAME') or parts.get('NAME') or ''))
    _URL = re.compile('(?P<scheme>[a-z][\\w+.-]*)://(?:(?P<user>[^:@/\\s]+)(?::(?P<password>[^@/\\s]*))?@)?(?P<host>[\\w.-]+)(?::(?P<port>\\d+))?(?:/(?P<database>[\\w.-]*))?(?P<query>\\?[^\\s\\"\'`]*)?', re.IGNORECASE)

    def _add_url(self, url: str, priority: int) -> None:
        match = self._URL.match(url.strip())
        if not match:
            return
        scheme = match.group('scheme').lower().split('+')[0]
        if scheme in ('postgres', 'postgresql', 'pgsql', 'pg'):
            engine, default_port = ('postgresql', '5432')
        elif scheme in ('clickhouse', 'clickhouses', 'ch', 'chs'):
            engine, default_port = ('clickhouse', '8123')
        else:
            return
        host = match.group('host') or 'localhost'
        port = match.group('port') or default_port
        user, password = (match.group('user') or '', match.group('password') or '')
        database = (match.group('database') or '').strip('/')
        auth = f'{user}:{password}@' if user else ''
        dsn = f"{engine}://{auth}{host}:{port}/{database}{match.group('query') or ''}"
        self._add(DatabaseTarget(engine=engine, priority=priority, dsn=dsn if engine == 'postgresql' else None, host=host, port=port, user=user, password=password, database=database))

    def _from_django(self) -> None:
        manage = self._find_manage_py()
        if not manage:
            return
        script = "import json;from django.conf import settings;print('RIDGES_DB'+json.dumps({k:{kk:str(vv) for kk,vv in v.items() if kk in ('ENGINE','NAME','USER','PASSWORD','HOST','PORT')} for k,v in settings.DATABASES.items()}))"
        result = run_command([app_python(self.repo.root), str(manage), 'shell', '-c', script], timeout=120)
        match = re.search('RIDGES_DB(\\{.*\\})', result.stdout or '')
        if not match:
            return
        try:
            databases = json.loads(match.group(1))
        except json.JSONDecodeError:
            return
        for name, config in databases.items():
            engine_string = (config.get('ENGINE') or '').lower()
            engine = 'clickhouse' if 'clickhouse' in engine_string else 'postgresql'
            self._add(DatabaseTarget(engine=engine, priority=0, host=config.get('HOST') or 'localhost', port=str(config.get('PORT') or ('8123' if engine == 'clickhouse' else '5432')), user=config.get('USER') or '', password=config.get('PASSWORD') or '', database=config.get('NAME') or ''))
        self.notes.append(f'django databases: {sorted(databases)}')

    def _find_manage_py(self) -> Path | None:
        for relative in self.repo.files:
            if Path(relative).name == 'manage.py' and relative.count('/') <= 2:
                return self.repo.root / relative
        return None
    _DSN_LITERAL = re.compile('\\b(?:postgres(?:ql)?(?:\\+\\w+)?|pgsql|clickhouses?(?:\\+\\w+)?)://[^\\s\\"\'`<>]{6,200}', re.IGNORECASE)
    _ENV_DEFAULT = re.compile('\\(\\s*[\\"\']\\w*?(?P<part>HOST|HOSTNAME|PORT|HTTP_PORT|USER|USERNAME|PASSWORD|PASS|DB|DATABASE|DBNAME|DB_NAME|NAME)[\\"\']\\s*,\\s*[\\"\'](?P<value>[^\\"\']*)[\\"\']\\s*\\)')
    _JS_DEFAULT = re.compile('process\\.env\\.\\w*?(?P<part>HOST|PORT|USER|USERNAME|PASSWORD|DB|DATABASE)\\s*(?:\\|\\||\\?\\?)\\s*[\\"\'](?P<value>[^\\"\']*)[\\"\']')
    _KEY_VALUE = re.compile('(?<![\\w.@-])[\\"\']?(?P<part>host|hostname|port|http_port|user|username|password|db_?name|database|dbname|name)[\\"\']?\\s*[:=]\\s*(?:[\\"\'](?P<quoted>[^\\"\'\\n]*)[\\"\']|(?P<bare>[\\w.-]+)(?!\\s*[\\(\\[]))', re.IGNORECASE)
    _PART_ALIASES = {'HOSTNAME': 'HOST', 'HTTP_PORT': 'PORT', 'USERNAME': 'USER', 'PASS': 'PASSWORD', 'DB': 'DATABASE', 'DBNAME': 'DATABASE', 'DB_NAME': 'DATABASE', 'NAME': 'DATABASE'}

    def _from_source(self) -> None:
        candidates: list[tuple[int, str, str]] = []
        for relative in self.repo.files:
            lowered_path = relative.lower()
            text = self.repo.read(relative)
            if text is None or len(text) > 400000:
                continue
            lowered = text.lower()
            if 'postgres' not in lowered and 'clickhouse' not in lowered:
                continue
            if _TEST_PATH.search(lowered_path) or 'example' in lowered_path or 'sample' in lowered_path:
                priority = 6
            elif re.search('(^|/)(db|database|conn(ection)?|client|store|settings|config)', lowered_path):
                priority = 3
            else:
                priority = 4
            candidates.append((priority, relative, text))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for priority, relative, text in candidates[:80]:
            for literal in self._DSN_LITERAL.findall(text):
                if '${' in literal or '%s' in literal or '{' in literal:
                    literal = self._expand(literal)
                    if '{' in literal or '$' in literal:
                        continue
                self._add_url(literal.rstrip('.,;)'), priority=priority)
        for priority, relative, text in candidates[:80]:
            self._scrape_fields(relative, text, priority)

    def _scrape_fields(self, relative: str, text: str, priority: int) -> None:
        text = self._DSN_LITERAL.sub(' ', text)
        lowered = text.lower()
        engine = 'clickhouse' if 'clickhouse' in lowered else 'postgresql'
        parts: dict[str, str] = {}
        weak_name: str | None = None
        for pattern in (self._ENV_DEFAULT, self._JS_DEFAULT):
            for match in pattern.finditer(text):
                raw = match.group('part').upper()
                if raw == 'NAME':
                    weak_name = weak_name or match.group('value')
                    continue
                part = self._PART_ALIASES.get(raw, raw)
                parts.setdefault(part, match.group('value'))
        if re.search('(^|/)(db|database|conn(ection)?|client|store|settings|config)', relative.lower()) or re.search('DATABASES\\s*=|DEFAULTS\\s*=|connection|datasource', text):
            for match in self._KEY_VALUE.finditer(text):
                raw = match.group('part').upper().replace('_', '')
                value = match.group('quoted') if match.group('quoted') is not None else match.group('bare')
                if not value:
                    continue
                if raw == 'NAME':
                    weak_name = weak_name or value
                    continue
                part = self._PART_ALIASES.get(raw, raw)
                if part == 'DATABASE' and value.isdigit():
                    continue
                if part in ('HOST', 'PORT', 'USER', 'PASSWORD', 'DATABASE'):
                    parts.setdefault(part, value)
        if 'DATABASE' not in parts and weak_name:
            parts['DATABASE'] = weak_name
        host = parts.get('HOST')
        if not host or host.lower() in ('true', 'false', 'none', 'null') or host.isdigit():
            return
        if parts.get('DATABASE', '').lower() in ('true', 'false', 'none', 'null'):
            parts.pop('DATABASE')
        self._add(DatabaseTarget(engine=engine, priority=priority, host=self._expand(host), port=parts.get('PORT') or ('8123' if engine == 'clickhouse' else '5432'), user=parts.get('USER') or '', password=parts.get('PASSWORD') or '', database=parts.get('DATABASE') or ''))
    # Last-resort discovery of the application's own database, used only when the repository
    # and environment name none. The task ships a live database and the app's config holds
    # the credentials; these conventional names and default logins are the fallback for when
    # that config could not be read. Every candidate must answer SELECT 1 to be used.
    _PROBE_HOSTS = ('localhost', '127.0.0.1', 'db', 'database', 'postgres', 'postgresql', 'pg', 'clickhouse', 'ch', 'clickhouse-server')
    _PROBE_CREDS = {'postgresql': (5432, (('postgres', 'postgres'), ('postgres', ''), ('app', 'app'))), 'clickhouse': (8123, (('default', ''), ('clickhouse', 'clickhouse'), ('app', 'app')))}

    @staticmethod
    def _tcp_open(host: str, port: int, timeout: float=1.0) -> bool:
        import socket
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        def attempt() -> bool:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                return False
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(attempt).result(timeout=timeout)
        except FutureTimeout:
            return False
        finally:
            pool.shutdown(wait=False)

    def _from_network(self) -> None:
        wanted = [self.instruction.engine] if self.instruction.engine in self._PROBE_CREDS else list(self._PROBE_CREDS)
        deadline = time.monotonic() + 15.0
        for engine in wanted:
            port, creds = self._PROBE_CREDS[engine]
            for host in self._PROBE_HOSTS:
                if time.monotonic() > deadline:
                    break
                if not self._tcp_open(host, port):
                    continue
                for user, password in creds:
                    target = DatabaseTarget(engine=engine, priority=8, host=host, port=str(port), user=user, password=password, database='postgres' if engine == 'postgresql' else 'default')
                    if self._ping(target):
                        if engine == 'postgresql':
                            names = self.sql("SELECT datname FROM pg_database WHERE NOT datistemplate AND datname <> 'postgres' ORDER BY 1", target=target, timeout=10)
                            first = next((n.strip() for n in names.splitlines() if re.fullmatch('[\\w-]+', n.strip()) and n.strip() != 'datname'), '')
                            if first:
                                target.database = first
                        self._add(target)
                        break
                break

    def _first_reachable(self, candidates: list[DatabaseTarget]) -> list[DatabaseTarget]:
        for target in candidates:
            if self._verify(target):
                return [target]
            log(f'database candidate unreachable, dropped: {target_url(target)}')
        return []

    def _ping(self, target: DatabaseTarget) -> bool:
        out = self.sql('SELECT 1', target=target, timeout=8)
        return bool(re.search('^\\s*1\\s*$', out or '', re.MULTILINE))

    def _verify(self, target: DatabaseTarget) -> bool:
        ok = self._ping(target)
        self.attempts.append((target_url(target), ok))
        return ok

    def available(self) -> bool:
        return bool(self.targets)

    def sql(self, query: str, *, target: DatabaseTarget | None=None, timeout: float=60.0) -> str:
        chosen = target or (self.targets[0] if self.targets else None)
        if chosen is None:
            return '[no database target discovered]'
        started = time.monotonic()
        out = self._clickhouse(chosen, query, timeout) if chosen.engine == 'clickhouse' else self._postgres(chosen, query, timeout)
        return out

    def _postgres(self, target: DatabaseTarget, query: str, timeout: float) -> str:
        dsn = target.dsn or f'postgresql://{target.user}:{target.password}@{target.host}:{target.port}/{target.database}'
        if shutil.which('psql'):
            result = run_command(['psql', dsn, '-X', '-A', '-F', ' | ', '-c', query], timeout=timeout)
            if result.returncode == 0:
                return result.stdout
            error = result.stdout
        else:
            error = '[psql not installed]'
        return self._postgres_via_python(target, query, timeout) or error

    # Runs a read-only query through the app's own driver when psql is absent. Investigation
    # queries are filtered by is_read_only_sql before they reach here.
    def _postgres_via_python(self, target: DatabaseTarget, query: str, timeout: float) -> str:
        script = f'\nimport json, sys\ntry:\n    import psycopg\n    connect = psycopg.connect\nexcept Exception:\n    try:\n        import psycopg2 as psycopg\n        connect = psycopg.connect\n    except Exception:\n        sys.exit("[no postgres driver]")\nwith connect({target.dsn!r} or "dbname={target.database} user={target.user} "\n             "password={target.password} host={target.host} port={target.port}") as conn:\n    with conn.cursor() as cur:\n        cur.execute({query!r})\n        rows = cur.fetchall()\nfor row in rows[:200]:\n    print(" | ".join("" if v is None else str(v) for v in row))\n'
        result = run_command([sys.executable, '-c', script], timeout=timeout)
        return result.stdout if result.returncode == 0 else ''

    def _clickhouse(self, target: DatabaseTarget, query: str, timeout: float) -> str:
        if shutil.which('clickhouse-client'):
            command = ['clickhouse-client', '--host', target.host, '--query', query]
            if target.user:
                command += ['--user', target.user]
            if target.password:
                command += ['--password', target.password]
            if target.database:
                command += ['--database', target.database]
            result = run_command(command, timeout=timeout)
            if result.returncode == 0:
                return result.stdout
        http_port = '8123' if target.port in ('9000', '9440', '') else target.port
        url = f'http://{target.host}:{http_port}/?database={target.database}'
        try:
            request = urllib.request.Request(url, data=query.encode('utf-8'), method='POST')
            if target.user:
                import base64
                token = base64.b64encode(f'{target.user}:{target.password}'.encode()).decode()
                request.add_header('Authorization', f'Basic {token}')
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode('utf-8', 'replace')
        except Exception as exc:
            return f'[clickhouse query failed: {exc}]'
    _QUERY_ERROR = re.compile('^\\s*\\[(?:no |psql |clickhouse |could not )', re.IGNORECASE)

    @classmethod
    def _usable(cls, output: str) -> bool:
        return bool(output and output.strip() and (not cls._QUERY_ERROR.match(output)))

    def schema_for(self, names: Iterable[str], limit: int=8) -> str:
        if not self.available():
            return ''
        target = self.targets[0]
        wanted = [name for name in names if re.fullmatch('[A-Za-z_][\\w]{2,}', name)][:limit]
        if not wanted:
            return ''
        blocks: list[str] = []
        if target.engine == 'postgresql':
            pattern = '|'.join((re.escape(name.lower()) for name in wanted))
            columns = self.sql(f"SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name ~ '{pattern}' ORDER BY table_name, ordinal_position LIMIT 400")
            indexes = self.sql(f"SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public' AND tablename ~ '{pattern}' ORDER BY tablename LIMIT 200")
            if self._usable(columns):
                blocks.append('columns (table | column | type):\n' + truncate(columns, 4000))
            if self._usable(indexes):
                blocks.append('indexes (table | name | definition):\n' + truncate(indexes, 4000))
        else:
            for name in wanted[:4]:
                ddl = self.sql(f'SHOW CREATE TABLE {name}')
                if self._usable(ddl) and 'failed' not in ddl[:40]:
                    blocks.append(truncate(ddl, 2500))
        rendered = '\n\n'.join(blocks)
        return rendered

    def clickhouse_work(self, query: str) -> str:
        if not self.available() or self.targets[0].engine != 'clickhouse':
            return ''
        marker = f'ridges_probe_{int(time.time() * 1000)}'
        self.sql(f"SELECT 1 AS {marker} SETTINGS log_comment = '{marker}'")
        self.sql(f"{query} SETTINGS log_comment = '{marker}'")
        self.sql('SYSTEM FLUSH LOGS')
        stats = self.sql(f"SELECT read_rows, read_bytes, result_rows, query_duration_ms, ProfileEvents['SelectedParts'] AS parts, ProfileEvents['SelectedMarks'] AS marks FROM system.query_log WHERE log_comment = '{marker}' AND type = 'QueryFinish' ORDER BY event_time DESC LIMIT 1")
        first = stats.strip().splitlines()[0] if stats.strip() else ''
        if not first or not re.match('^\\s*\\d+\\s*\\|', first):
            return ''
        return 'read_rows | read_bytes | result_rows | duration_ms | parts | marks\n' + truncate(stats, 800)

    def explain(self, query: str) -> str:
        if not self.available():
            return ''
        target = self.targets[0]
        if target.engine == 'postgresql':
            return truncate(self.sql(f'EXPLAIN (ANALYZE, BUFFERS, SUMMARY OFF) {query}'), 4000)
        return truncate(self.sql(f'EXPLAIN indexes = 1 {query}'), 4000)

def _diff_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)

def _annotate_no_newline(diff: Iterable[str]) -> list[str]:
    output: list[str] = []
    for line in diff:
        if line.startswith(('---', '+++', '@@', 'diff ', 'new file', 'index ')):
            output.append(line if line.endswith('\n') else line + '\n')
            continue
        if line.endswith('\n'):
            output.append(line)
        else:
            output.append(line + '\n')
            output.append('\\ No newline at end of file\n')
    return output

def build_patch(repo: Repository, files: Sequence[str]) -> str:
    parts: list[str] = []
    for relative in files:
        original = repo.original(relative)
        path = repo.root / relative
        try:
            current = read_source(path) if path.exists() else None
        except (OSError, UnicodeDecodeError) as exc:
            log(f'cannot read {relative} to diff it ({exc}); omitted from the patch')
            continue
        if current == original:
            continue
        if original is None:
            header = f'diff --git a/{relative} b/{relative}\nnew file mode 100644\n'
            from_label, to_label = ('/dev/null', f'b/{relative}')
            original = ''
        elif current is None:
            header = f'diff --git a/{relative} b/{relative}\ndeleted file mode 100644\n'
            from_label, to_label = (f'a/{relative}', '/dev/null')
            current = ''
        else:
            header = f'diff --git a/{relative} b/{relative}\n'
            from_label, to_label = (f'a/{relative}', f'b/{relative}')
        body = difflib.unified_diff(_diff_lines(original), _diff_lines(current), fromfile=from_label, tofile=to_label, n=3)
        rendered = ''.join(_annotate_no_newline(body))
        if rendered.strip():
            parts.append(header + rendered)
    patch = ''.join(parts)
    return patch

def verify_patch_applies(patch: str, repo: Repository) -> tuple[bool, str]:
    if not patch.strip():
        return (False, 'empty patch')
    if not shutil.which('git'):
        return (True, 'git unavailable; skipped apply check')
    staging = Path('/tmp/ridges-patch-check')
    touched = re.findall('^diff --git a/(\\S+) b/\\S+$', patch, re.MULTILINE)
    patch_file = Path('/tmp/ridges-candidate.diff')
    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        for relative in touched:
            original = repo.original(relative)
            if original is None:
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_source(destination, original)
        patch_file.write_text(patch, encoding='utf-8')
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        log(f'could not stage the patch for the apply check ({exc}); skipping it')
        return (True, f'apply check could not run: {exc}')
    result = run_command(['git', 'apply', '--check', '-v', str(patch_file)], cwd=staging, timeout=60)
    shutil.rmtree(staging, ignore_errors=True)
    return (result.returncode == 0, truncate(result.stdout, 1500))

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

# Everything this agent can check for itself before committing to a patch: scope, syntax,
# the constraints the instruction states, and the repository's own tests. Ordered
# cheapest-first so a syntax slip never costs a six-minute test run.
class Checker:

    def __init__(self, repo: Repository, instruction: Instruction, candidates: Sequence[str]=(), probe: 'DatabaseProbe | None'=None) -> None:
        self.repo = repo
        self.instruction = instruction
        self.probe = probe
        self.candidates = list(candidates)

    def run_all(self, changed: Sequence[str], *, include_tests: bool=True) -> list[CheckResult]:
        results = [self.check_scope(changed)]
        results.append(self.check_protected(changed))
        results.append(self.check_syntax(changed))
        if self.instruction.single_method:
            results.append(self.check_single_method(changed))
        if self.instruction.style_constraints:
            results.append(self.check_style(changed))
        if self.instruction.single_method:
            results.append(self.check_method_contract(changed))
        if any((self._MIGRATION_PATH.search(p) for p in changed)):
            results.append(self.check_migration_contract(changed))
        if all((check.passed for check in results)):
            results.extend(self.check_lint(changed))
            if include_tests:
                if not self.instruction.commands:
                    found = self.discovered_commands(changed)
                    if found:
                        log(f"instruction named no checks; running the repository's own: {found}")
                        self.instruction.commands = found
                results.extend(self.run_task_commands())
        return results
    _NEVER_EDIT_ABSOLUTE = re.compile('(^|/)(tests?|testing|spec|fixtures?|conftest\\.py)(/|$)|(^|/)test_[^/]*$|_test\\.[a-z]+$|(^|/)(setup|conftest|manage)\\.py$|\\.lock$', re.IGNORECASE)
    _NEVER_EDIT_UNLESS_PERMITTED = re.compile('(^|/)migrations(/|$)|\\.(cfg|ini)$', re.IGNORECASE)
    _NEVER_EDIT = re.compile(_NEVER_EDIT_ABSOLUTE.pattern + '|' + _NEVER_EDIT_UNLESS_PERMITTED.pattern, re.IGNORECASE)

    def allowed_paths(self) -> set[str] | None:
        if self.instruction.edit_only:
            return set(self.instruction.edit_only)
        if self.instruction.lint_paths:
            return set(self.instruction.lint_paths)
        inferred = set(self.candidates) | set(self.instruction.named_paths)
        return inferred or None

    def check_protected(self, changed: Sequence[str]) -> CheckResult:
        permitted = set(self.instruction.edit_only) | set(self.instruction.lint_paths)
        forbidden = [p for p in changed if self._NEVER_EDIT_ABSOLUTE.search(p)]
        unpermitted = [p for p in changed if p not in permitted and self._NEVER_EDIT_UNLESS_PERMITTED.search(p)]
        if forbidden:
            return CheckResult('protected paths', False, f'tests, fixtures and project scripts are not yours to change, however the instruction is worded; these were modified: {forbidden}. Fix the production code instead: a change to a test does not fix the behaviour the test describes.')
        if unpermitted:
            return CheckResult('protected paths', False, f'migrations and project config may only be changed when the instruction explicitly permits that file; these were modified without one: {unpermitted}. Fix the production code instead.')
        return CheckResult('protected paths', True, 'no test or fixture files touched')

    def check_scope(self, changed: Sequence[str]) -> CheckResult:
        allowed = self.allowed_paths()
        if allowed is None:
            return CheckResult('scope', True, f'changed: {list(changed)}')
        stray = [path for path in changed if path not in allowed]
        if stray:
            return CheckResult('scope', False, f'the instruction permits edits only to {sorted(allowed)}, but these files were modified: {stray}')
        return CheckResult('scope', True, f'changed: {list(changed)}')

    def check_syntax(self, changed: Sequence[str]) -> CheckResult:
        problems = []
        for relative in changed:
            if not relative.endswith('.py'):
                continue
            text = self.repo.read(relative)
            if text is None:
                continue
            try:
                compile(text, relative, 'exec', dont_inherit=True)
            except SyntaxError as exc:
                problems.append(f'{relative}:{exc.lineno}: {exc.msg}')
            except (ValueError, RecursionError) as exc:
                problems.append(f'{relative}: {exc}')
        if problems:
            return CheckResult('syntax', False, '; '.join(problems))
        return CheckResult('syntax', True, 'parsed')

    def check_single_method(self, changed: Sequence[str]) -> CheckResult:
        for relative in changed:
            if not relative.endswith('.py'):
                continue
            original = self.repo.original(relative)
            current = self.repo.read(relative)
            if original is None or current is None:
                continue
            try:
                current_defs = python_definitions(current)
            except Exception:
                continue
            original_lines = original.splitlines(keepends=True)
            current_lines = current.splitlines(keepends=True)
            matcher = difflib.SequenceMatcher(None, original_lines, current_lines, autojunk=False)
            edited = [op for op in matcher.get_opcodes() if op[0] != 'equal']
            if not edited:
                continue
            first_line = min((op[3] for op in edited)) + 1
            last_line = max((op[4] for op in edited))
            enclosing = [(name, start, end) for name, start, end, kind in current_defs if kind != 'class' and start <= first_line and (last_line <= end)]
            if not enclosing:
                return CheckResult('single-method', False, f'{relative}: edits at lines {first_line}-{last_line} fall outside a single function body; the instruction bounds the change to one method, so imports and every other line in the file must stay byte-identical')
            name, start, end = min(enclosing, key=lambda item: item[2] - item[1])
            if original_lines[:start - 1] != current_lines[:start - 1]:
                return CheckResult('single-method', False, f'{relative}: source above {name} changed')
            before = next(((s0, e0) for n0, s0, e0, k0 in python_definitions(original) if n0 == name and k0 != 'class'), None)
            if before and original_lines[before[1]:] != current_lines[end:]:
                return CheckResult('single-method', False, f'{relative}: source below {name} changed')
        return CheckResult('single-method', True, 'confined to one method')
    _STYLE_NODES: dict[str, tuple[type, ...]] = {'loops': (ast.For, ast.AsyncFor, ast.While), 'comprehensions': (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp), 'lambdas': (ast.Lambda,), 'exception handling': (ast.Try,), 'context managers': (ast.With, ast.AsyncWith)}

    def check_style(self, changed: Sequence[str]) -> CheckResult:
        forbidden: tuple[type, ...] = tuple((node for label in self.instruction.style_constraints for node in self._STYLE_NODES.get(label, ())))
        problems: list[str] = []
        for relative in changed:
            if not relative.endswith('.py'):
                continue
            original = self.repo.original(relative) or ''
            current = self.repo.read(relative) or ''
            try:
                tree = ast.parse(current)
            except SyntaxError:
                continue
            edited = self._edited_line_range(original, current)
            if edited is None:
                continue
            first, last = edited
            for node in ast.walk(tree):
                line = getattr(node, 'lineno', None)
                if line is None or not first <= line <= last:
                    continue
                if forbidden and isinstance(node, forbidden):
                    problems.append(f'{relative}:{line}: {type(node).__name__}')
                if 'raw SQL' in self.instruction.style_constraints and isinstance(node, ast.Name) and (node.id in {'RawSQL', 'raw'}):
                    problems.append(f'{relative}:{line}: raw SQL is ruled out for this change')
                if 'materialising rows in Python' in self.instruction.style_constraints and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id in {'list', 'tuple', 'set', 'sorted'}):
                    problems.append(f'{relative}:{line}: {node.func.id}() pulls the rows into Python; the instruction requires the work stay in the database')
            if 'names the file does not import' in self.instruction.style_constraints:
                problems.extend(self._undefined_names(relative, tree, first, last))
        if problems:
            return CheckResult('style constraints', False, f"the instruction forbids {', '.join(self.instruction.style_constraints)} in the changed code, but found: " + '; '.join(sorted(set(problems))[:10]))
        return CheckResult('style constraints', True, f'none of {self.instruction.style_constraints} present')

    @staticmethod
    def _undefined_names(relative: str, tree: ast.AST, first: int, last: int) -> list[str]:
        bound: set[str] = set(dir(__builtins__) if isinstance(__builtins__, type(ast)) else __builtins__) | {'self', 'cls', '__name__'}
        edited: list[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split('.')[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                if first <= node.lineno <= last or node.lineno <= first <= (node.end_lineno or 0):
                    args = node.args if not isinstance(node, ast.ClassDef) else None
                    for arg in args.posonlyargs + args.args + args.kwonlyargs if args else []:
                        bound.add(arg.arg)
                    for extra in (args.vararg, args.kwarg) if args else ():
                        if extra:
                            bound.add(extra.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, (ast.comprehension,)):
                for target in ast.walk(node.target):
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            if getattr(node, 'lineno', None) is not None and first <= node.lineno <= last:
                edited.append(node)
        missing: list[str] = []
        for node in edited:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and (node.id not in bound):
                message = f'{relative}:{node.lineno}: `{node.id}` is not imported or defined in this file, and the instruction allows only names it already has'
                if message not in missing:
                    missing.append(message)
        return missing

    @staticmethod
    def _edited_line_range(original: str, current: str) -> tuple[int, int] | None:
        matcher = difflib.SequenceMatcher(None, original.splitlines(keepends=True), current.splitlines(keepends=True), autojunk=False)
        edited = [op for op in matcher.get_opcodes() if op[0] != 'equal']
        if not edited:
            return None
        return (min((op[3] for op in edited)) + 1, max(max((op[4] for op in edited)), 1))
    # Names an edit may not introduce into a method the instruction bounds. Pre-existing uses
    # are not flagged -- see the inherited-violation check below -- so a correct fix is never
    # rejected for code it did not write.
    _DANGEROUS_NAMES = {'__import__', 'breakpoint', 'compile', 'eval', 'exec', 'getattr', 'globals', 'locals', 'open', 'setattr', 'vars'}
    _FORBIDDEN_NODES = (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete, ast.Global, ast.Lambda, ast.Match, ast.Nonlocal, ast.Raise, ast.While, ast.Yield, ast.YieldFrom)
    _MAX_METHOD_BYTES = 4500
    _MIN_METHOD_NODES = 240
    _MAX_METHOD_NODES = 400
    _NODE_HEADROOM = 100

    @classmethod
    def _node_budget(cls, original_nodes: int | None) -> int:
        if original_nodes is None:
            return cls._MAX_METHOD_NODES
        return max(cls._MIN_METHOD_NODES, min(cls._MAX_METHOD_NODES, original_nodes + cls._NODE_HEADROOM))

    def _method_violations(self, method: ast.FunctionDef) -> list[str]:
        found: list[str] = []
        for node in ast.walk(ast.Module(body=method.body, type_ignores=[])):
            if isinstance(node, ast.FunctionDef) and node is not method:
                found.append(f'nested function definition `{node.name}`')
            elif isinstance(node, self._FORBIDDEN_NODES):
                found.append(f'{type(node).__name__} is not allowed')
            elif isinstance(node, ast.ImportFrom):
                found.append(f"import inside the method: from {node.module} import {', '.join((a.name for a in node.names))}")
            elif isinstance(node, ast.Import):
                found.append(f"import inside the method: import {', '.join((a.name for a in node.names))}")
            elif isinstance(node, ast.Name) and (node.id in self._DANGEROUS_NAMES or '__' in node.id):
                found.append(f'forbidden name {node.id}')
            elif isinstance(node, ast.Attribute) and '__' in node.attr:
                found.append(f'forbidden attribute {node.attr}')
        return found

    @staticmethod
    def _method_named(tree: ast.AST, name: str) -> ast.FunctionDef | None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        return None

    def check_method_contract(self, changed: Sequence[str]) -> CheckResult:
        problems: list[str] = []
        for relative in changed:
            if not relative.endswith('.py'):
                continue
            current = self.repo.read(relative) or ''
            original = self.repo.original(relative) or ''
            try:
                tree = ast.parse(current)
                original_tree = ast.parse(original)
            except SyntaxError:
                continue
            edited = self._edited_line_range(original, current)
            if edited is None:
                continue
            first, last = edited
            enclosing = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.lineno <= first and ((n.end_lineno or n.lineno) >= last)]
            if not enclosing:
                continue
            method = min(enclosing, key=lambda n: (n.end_lineno or 0) - n.lineno)
            body = '\n'.join(current.splitlines()[method.lineno - 1:method.end_lineno])
            if len(body.encode()) > self._MAX_METHOD_BYTES:
                problems.append(f'method is {len(body.encode())} bytes (limit {self._MAX_METHOD_BYTES})')
            before_method = self._method_named(original_tree, method.name)
            original_nodes = len(list(ast.walk(ast.Module(body=before_method.body, type_ignores=[])))) if before_method else None
            budget = self._node_budget(original_nodes)
            nodes = list(ast.walk(ast.Module(body=method.body, type_ignores=[])))
            if len(nodes) > budget:
                problems.append(f'method has {len(nodes)} AST nodes (limit {budget})')
            inherited = set(self._method_violations(before_method)) if before_method else set()
            for violation in self._method_violations(method):
                if violation not in inherited:
                    problems.append(f'{relative}: {violation}' + (' -- use only names the file already imports' if violation.startswith('import inside') else ''))
        if problems:
            return CheckResult('method contract', False, 'these break the structural constraints the task sets on the change, however correct the query is: ' + '; '.join(sorted(set(problems))[:8]))
        return CheckResult('method contract', True, 'method body within the stated limits')
    _MIGRATION_PATH = re.compile('(^|/)migrations/', re.IGNORECASE)

    @classmethod
    def _dotted(cls, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f'{cls._dotted(node.value)}.{node.attr}'
        return type(node).__name__

    @classmethod
    def _call_shape(cls, node: ast.AST):
        if not isinstance(node, ast.Call):
            return None
        keywords = tuple(sorted(((keyword.arg, cls._call_shape(keyword.value)) for keyword in node.keywords)))
        return (cls._dotted(node.func), len(node.args), keywords)

    @classmethod
    def _migration_shape(cls, text: str) -> dict:
        tree = ast.parse(text)
        shape: dict = {'statements': tuple((type(n).__name__ for n in tree.body))}
        shape['imports'] = tuple((ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            shape['class'] = node.name
            shape['bases'] = tuple((cls._dotted(b) for b in node.bases))
            shape['body'] = tuple((type(n).__name__ for n in node.body))
            for statement in node.body:
                if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)):
                    continue
                name = statement.targets[0].id
                if name == 'operations' and isinstance(statement.value, (ast.List, ast.Tuple)):
                    shape['operations'] = tuple((cls._call_shape(e) for e in statement.value.elts))
                elif name == 'dependencies':
                    try:
                        shape['dependencies'] = repr(ast.literal_eval(statement.value))
                    except (ValueError, SyntaxError):
                        shape['dependencies'] = ast.dump(statement.value)
                shape.setdefault('assigned', set()).add(name)
        shape['assigned'] = tuple(sorted(shape.get('assigned', ())))
        return shape
    _MIGRATION_BYTE_HEADROOM = 600

    def check_migration_contract(self, changed: Sequence[str]) -> CheckResult:
        problems: list[str] = []
        for relative in changed:
            if not (relative.endswith('.py') and self._MIGRATION_PATH.search(relative)):
                continue
            original = self.repo.original(relative)
            current = self.repo.read(relative)
            if original is None or current is None:
                continue
            try:
                before, after = (self._migration_shape(original), self._migration_shape(current))
            except SyntaxError:
                continue
            for key in sorted(set(before) | set(after)):
                if before.get(key) != after.get(key):
                    problems.append(f"{relative}: the migration's {key} changed")
            budget = max(1600, len(original.encode()) + self._MIGRATION_BYTE_HEADROOM)
            if len(current.encode()) > budget:
                problems.append(f'{relative}: {len(current.encode())} bytes exceeds the bounded migration budget ({budget})')
        if problems:
            return CheckResult('migration contract', False, 'a migration may change the values inside its operations, not its structure. These change its structure, however correct the index is: ' + '; '.join(sorted(set(problems))[:8]))
        return CheckResult('migration contract', True, 'structure preserved')

    def check_lint(self, changed: Sequence[str]) -> list[CheckResult]:
        python_files = [path for path in changed if path.endswith('.py')]
        if not python_files or not shutil.which('ruff'):
            return []
        result = run_command(['ruff', 'check', '--no-cache', *python_files], timeout=120)
        return [CheckResult('ruff', result.returncode == 0, truncate(result.stdout, 2000))]
    _DJANGO_PROBE = '\nimport json\nfrom django.db import connection\nfrom django.test.utils import CaptureQueriesContext\n{setup}\ncounts = {{}}\nfor _n in ({small}, {large}):\n    N = _n\n    with CaptureQueriesContext(connection) as _ctx:\n        {call}\n    counts[_n] = len(_ctx)\nprint("RIDGES_QC" + json.dumps(counts))\nprint("RIDGES_QS" + json.dumps([q["sql"] for q in _ctx.captured_queries[:6]]))\n'

    def measure_query_scaling(self, probe: dict) -> CheckResult | None:
        setup = (probe.get('setup') or '').strip()
        call = (probe.get('call') or '').strip()
        if not call:
            return None
        manage = next((p for p in self.repo.files if Path(p).name == 'manage.py' and p.count('/') <= 2), None)
        if not manage:
            return None
        try:
            small = int(probe.get('small') or 1)
            large = int(probe.get('large') or 10)
        except (TypeError, ValueError):
            small, large = (1, 10)
        script = self._DJANGO_PROBE.format(setup='\n'.join((f'{line}' for line in setup.splitlines())), call='\n        '.join(call.splitlines()), small=small, large=large)
        result = run_command([app_python(self.repo.root), manage, 'shell', '-c', script], timeout=min(300.0, max(60.0, remaining_seconds() - 200)))
        match = re.search('RIDGES_QC(\\{.*\\})', result.stdout or '')
        if not match:
            return CheckResult('query scaling', True, f'probe did not report a count; treating as unmeasured. {truncate(result.stdout, 600)}')
        try:
            counts = {int(k): int(v) for k, v in json.loads(match.group(1)).items()}
        except (ValueError, json.JSONDecodeError):
            return CheckResult('query scaling', True, 'probe output unreadable')
        low, high = (counts.get(small), counts.get(large))
        if low is None or high is None:
            return CheckResult('query scaling', True, f'incomplete measurement: {counts}')
        limit = self.instruction.targets.get('max_queries')
        failure = None
        if limit and high > limit:
            failure = f'the instruction allows at most {limit} queries; measured {low} at N={small} and {high} at N={large}.'
        elif high > low + 1:
            failure = f'query count still grows with input: {low} queries at N={small}, {high} at N={large}. The work must be bounded -- fold the per-item statements into one set-based query.'
        if failure:
            return CheckResult('query scaling', False, failure + self._explain_captured(result.stdout or ''))
        return CheckResult('query scaling', True, f'bounded: {low} queries at N={small}, {high} at N={large}' + (f' (limit {limit})' if limit else ''))

    def _explain_captured(self, stdout: str) -> str:
        match = re.search('RIDGES_QS(\\[.*\\])', stdout)
        if not match or self.probe is None or (not self.probe.available()):
            return ''
        try:
            statements = json.loads(match.group(1))
        except json.JSONDecodeError:
            return ''
        shown: list[str] = []
        for sql in statements[:3]:
            if is_read_only_sql(sql):
                shown.append(f'$ EXPLAIN {truncate(sql, 300)}\n{truncate(self.probe.explain(sql), 1500)}')
        return '\n\nStatements issued at N=large, with their plans:\n' + '\n\n'.join(shown) if shown else ''

    def discovered_commands(self, changed: Sequence[str]) -> list[str]:
        root = self.repo.root
        for relative in changed:
            path = Path(relative)
            parts = path.parts
            manage = next((p for p in self.repo.files if Path(p).name == 'manage.py' and p.count('/') <= 2), None)
            if manage and path.suffix == '.py':
                app = next((parts[i] for i in range(len(parts) - 2, -1, -1) if (root / Path(*parts[:i + 1]) / 'tests').is_dir() or (root / Path(*parts[:i + 1]) / 'tests.py').is_file()), None)
                if app:
                    return [f'{app_python(root)} {manage} test {app} --keepdb --noinput']
            if path.suffix == '.py':
                for i in range(len(parts) - 1, 0, -1):
                    candidate = root / Path(*parts[:i]) / 'tests'
                    if candidate.is_dir():
                        return [f'{app_python(root)} -m pytest {candidate.relative_to(root)} -x -q']
            if path.suffix == '.go':
                return [f'go test ./{path.parent.as_posix()}/...']
            if path.suffix == '.rs':
                return ['cargo test']
            if path.suffix == '.rb':
                return ['bundle exec rspec']
            if path.suffix in ('.ex', '.exs') and (root / 'mix.exs').is_file():
                return ['mix test']
            if path.suffix in ('.js', '.ts', '.tsx', '.jsx') and (root / 'package.json').is_file():
                try:
                    scripts = json.loads((root / 'package.json').read_text()).get('scripts', {})
                except (OSError, json.JSONDecodeError):
                    scripts = {}
                if 'test' in scripts:
                    return ['npm test --silent']
                return ['node --test']
            if path.suffix in ('.java', '.kt'):
                if (root / 'pom.xml').is_file():
                    return ['mvn -q test']
                if (root / 'build.gradle').is_file() or (root / 'build.gradle.kts').is_file():
                    return ['gradle test -q']
            if path.suffix == '.php' and (root / 'vendor/bin/phpunit').exists():
                return ['vendor/bin/phpunit']
            if path.suffix == '.cs':
                return ['dotnet test']
        return []

    def selected_commands(self) -> list[str]:
        commands = []
        for command in self.instruction.commands:
            if command.startswith('ruff '):
                continue
            commands.append(command)
        return commands

    def run_task_commands(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for command in self.selected_commands():
            if remaining_seconds() < 180:
                results.append(CheckResult(f'$ {command}', True, 'skipped: out of time'))
                continue
            log(f'running task check: {command}')
            result = run_command(command, timeout=min(600.0, max(60.0, remaining_seconds() - 120)))
            passed = result.returncode == 0
            results.append(CheckResult(f'$ {command}', passed, truncate(result.stdout, 6000, head_ratio=0.25)))
            if not passed:
                break
        return results

def summarise(checks: Sequence[CheckResult]) -> str:
    lines = []
    for check in checks:
        lines.append(f"[{('PASS' if check.passed else 'FAIL')}] {check.name}")
        if not check.passed and check.detail:
            lines.append(truncate(check.detail, 5000, head_ratio=0.3))
    return '\n'.join(lines)
SYSTEM_PROMPT = "You are a database query engineer. You fix, author, and optimise the queries a real application issues against PostgreSQL or ClickHouse, working inside the application's own repository: raw SQL, ORM code, or query-builder code.\n\nHow you work:\n\n* Edit production query code, and write the fix so it holds for data you have not seen. Implement the general rule the task states, in terms of the columns and relations it names, rather than anything that happens to suit the rows in front of you.\n* Reduce the database work the query performs -- statements issued, rows and buffers touched, index usage. Remove work the query genuinely does not need, rather than moving it somewhere less visible.\n* Keep everything outside the blast radius the instruction sets byte-identical, imports included, and build the fix from names already in scope.\n* Make the smallest change that fixes the underlying cause.\n\nReasoning you should apply, by symptom:\n\n* Work that grows with input size -- a statement per element, per row, or per iteration -- becomes one set-based statement: a single bulk insert/update, one `IN`/`ANY` predicate, a join, or a CTE. Compute the set difference in the database, and keep any signal/callback contract firing exactly once with the same payload.\n* A slow or unselective plan usually means the predicate the application actually issues is not the one the index serves. Match index column order and partiality to the real predicate, including equality columns first.\n* Wrong aggregates over a hierarchy or a many-to-many usually mean fan-out: rows multiplied by a join. Fix it with DISTINCT on the counted key, a subquery/lateral, or a nested-set/recursive descendant predicate, keeping the correction in the database rather than in application code.\n* Percentages and ratios should be computed in the database with explicit numeric casting and a zero-denominator guard.\n* On ClickHouse, favour the primary key order and PREWHERE, prefer set-based expressions over per-row subqueries, and remember that JOIN semantics and nullability differ from PostgreSQL. An `explain` request there also returns read_rows, read_bytes, selected parts and marks -- the amount of data the query actually touched, so check it fell. Build a series with numbers(N) or arrayJoin(range(...)) rather than by selecting from system.numbers: a report should not depend on the server's own introspection tables.\n\nYou answer only with a single JSON object, described in the user message."
DIRECTIVE = '# Instructions\n\nDiagnose the database defect described in the task statement below, then reply with one JSON object in the format given at the end of this message.\n\nWork in this order:\n\n1. Read the task statement. It is the authority on what to change, which files you may edit, and which checks to run.\n2. Locate the code that issues the query in question, using the source provided below.\n3. Name the database-level cause in at most two sentences.\n4. Write the smallest edit that fixes that cause.\n5. Reply with the JSON object.\n\nWhen the statement does not name the file to edit and the provided source does not settle which file issues the query, reply with the `need_context` object and request what would settle it. Request context whenever you are unsure rather than editing a file you have not read.'
EDIT_PROTOCOL = 'Reply with ONE JSON object and nothing else. Begin the reply with `{` and end it with `}`. Two shapes are allowed.\n\nTo gather more evidence before deciding (the agent tells you how many rounds remain; an unnamed target allows more than a named one, and each failed attempt grants another):\n\n{"action": "need_context",\n "why": "<one sentence>",\n "requests": [\n   {"kind": "read_file", "path": "<repo-relative path>", "start": 1, "end": 200},\n   {"kind": "grep", "pattern": "<python regex>", "path_filter": "<optional substring>"},\n   {"kind": "sql", "query": "<read-only statement to run against the live database>"},\n   {"kind": "explain", "query": "<SELECT ... to EXPLAIN on the live database>"},\n   {"kind": "read_file", "path": "<repo-relative path>", "symbol": "<def or class name: returns that definition whole>"},\n   {"kind": "schema", "tables": ["<table name>", "..."]},\n   {"kind": "callers", "symbol": "<function or class name: who defines and who references it>"},\n   {"kind": "count", "setup": "<Django shell setup>", "call": "<one line using N>", "small": 1, "large": 10}\n ]}\n`count` measures the query count at two sizes BEFORE you edit -- use it on bounded-work tasks so you know the number you are trying to change.\n\nTo make the change:\n\n{"action": "edit",\n "diagnosis": "<the database-level cause, one or two sentences>",\n "verify": ["<any shell command the instruction says to run before finishing, copied verbatim; [] if it names none>"],\n "constraints": {"editable_files": ["<repo-relative paths the instruction allows you to change>"],\n                 "bounded_to_method": "<Class.method the instruction restricts the change to, or null>"},\n "edits": [\n   {"path": "<repo-relative path>",\n    "search": "<exact contiguous text from the current file, unique within it>",\n    "replace": "<replacement text>"}\n ]}\n\nA complete `edit` reply, to copy the shape of:\n\n{"action": "edit",\n "diagnosis": "share_pct divides two integer columns, so the fraction is truncated before Round() runs.",\n "verify": ["python manage.py test shop.tests.test_reports --keepdb --noinput"],\n "constraints": {"editable_files": ["shop/reports/querysets.py"],\n                 "bounded_to_method": "OrderQuerySet.annotate_share"},\n "edits": [\n   {"path": "shop/reports/querysets.py",\n    "search": "        return self.annotate(\\n            share_pct=Round(F(\'paid\') * 100 / F(\'total\'), 2),",\n    "replace": "        return self.annotate(\\n            share_pct=Round(F(\'paid\') * 100.0 / F(\'total\'), 2),"}\n ]}\n\nRules for edits:\n* `search` must reproduce the existing file byte for byte, including indentation. Include 2 to 5 surrounding lines, enough to appear exactly once in the file.\n* Give the smallest `search`/`replace` pair that expresses the change -- one edit per distinct change, each covering the lines that change plus that much context.\n* Change only what the fix requires. Leave docstrings, comments, formatting, blank lines and import order exactly as they are unless the task asks for them to change -- an unnecessary edit is a way to fail a scope check, never a way to pass one.\n* Keep `diagnosis` to at most 2 sentences and the whole reply under 2000 characters unless the edit itself is longer. Reason as far as naming the cause and writing the edit; deliberation past that point is billed and is not read.\n* `constraints` is read only when the instruction named no file and no method: state what it DOES allow, exactly as written. The agent enforces it against your own edits, so claim only what the instruction grants.\n* After a failed attempt you may request context again -- the failure output usually points at something worth reading before the next edit.\n* `verify` matters: those commands are run against the live database and their output comes back to you if they fail. Copy every check the instruction names, wherever it states them -- fenced block, inline text, or prose.\n* When the task is about work that must not grow with input size, add a probe so the agent can measure it before submitting:\n    "measure": {"setup": "<imports and fixture creation, Django shell>",\n                "call": "<one line exercising the change, using N as the size>",\n                "small": 1, "large": 10}\n  Use `N` as the selection size in `call`. The agent runs it at both sizes and tells you the query counts; if they grow with N the fix is not bounded.\n* To create a new file instead, use {"path": ..., "new_file": true, "content": "<full text>"}.\n* Escape newlines properly -- the whole reply must parse as JSON.'

class PromptBuilder:

    def __init__(self, repo: Repository, instruction: Instruction, candidates: Sequence[tuple[str, list[int]]], probe: DatabaseProbe) -> None:
        self.repo = repo
        self.instruction = instruction
        self.candidates = candidates
        self.probe = probe
        self.named_target = bool(instruction.edit_only or instruction.lint_paths)
    CHARS_PER_TOKEN = 3.5
    SEPARATOR = '###'

    def build(self) -> str:
        named = ('directive', 'task statement', 'what this agent found', 'relevant source', 'live schema')
        sections = (DIRECTIVE, self._task_instruction(), self._agent_findings(), self._relevant_source(), self._live_schema())
        prompt = f'\n\n{self.SEPARATOR}\n\n'.join((s for s in sections if s))
        return prompt

    def _task_instruction(self) -> str:
        return f'# Task statement (the authority on what to change)\n\n{self.instruction.text.strip()}'

    def _agent_findings(self) -> str:
        facts: list[str] = []
        if self.probe.available():
            facts.append(f'live database (this agent connected to it and it answered SELECT 1): {target_url(self.probe.targets[0])}')
        if not (self.instruction.edit_only or self.instruction.lint_paths or self.instruction.named_paths):
            facts.append("the instruction names no file to edit: the candidates below are this agent's ranking, not the task's -- identify which one actually issues the query, and fill `constraints` with what the instruction does permit")
        if 'bounded_queries' in self.instruction.kinds or self.instruction.targets.get('max_queries'):
            facts.append('supply a `measure` probe with your edit: this agent runs it at two selection sizes and reports the query count back to you, so a fix that still scales with input is caught before you finish')
        facts.extend(self.instruction.unparsed)
        if self.instruction.targets.get('create_paths'):
            facts.append(f"these paths named by the instruction do not exist yet: {self.instruction.targets['create_paths']} -- create them with a new_file edit")
        if not facts:
            return ''
        return '# What this agent found (not stated in the instruction)\n\n' + '\n'.join((f'- {fact}' for fact in facts))

    def _imports_are_frozen(self) -> bool:
        return self.instruction.single_method and 'names the file does not import' in self.instruction.style_constraints

    def _budget_for(self, position: int, length: int) -> int:
        if self.named_target:
            return 320
        if length <= 140:
            return length + 10
        return 160 if position == 0 else 40

    def _relevant_source(self) -> str:
        blocks: list[str] = []
        for position, (relative, hot) in enumerate(self.candidates):
            length = len((self.repo.read(relative) or '').splitlines())
            budget = self._budget_for(position, length)
            outline = file_outline(self.repo, relative, limit=80 if self.named_target else 20)
            if outline:
                blocks.append(outline)
            if position == 0 and (not self._imports_are_frozen()):
                neighbours = package_map(self.repo, relative)
                if neighbours:
                    blocks.append(neighbours)
            pieces = slice_around(self.repo, relative, hot or [], self.instruction, budget_lines=budget)
            kept = pieces if budget > 40 else pieces[:1]
            for piece in kept:
                blocks.append(piece.render(self.repo))
        return f'{self._source_header()}\n\n' + '\n\n'.join(blocks)

    def _source_header(self) -> str:
        header = '# Relevant source'
        if self.instruction.traced_hint:
            header += f"\n\nReference tracing from the task's vocabulary also reached these files, not shown below; request one with need_context if the candidates above do not contain the query: {self.instruction.traced_hint}"
        if not self.named_target and len(self.candidates) > 1:
            header += '\n\nThe instruction does not name a file. These are the strongest candidates, best first; identify which one actually issues the query in question -- request more of it with need_context if the excerpt is not enough.'
        return header

    def _live_schema(self) -> str:
        schema = self.probe.schema_for(_table_candidates(self.repo, self.instruction, self.candidates))
        return '# Live schema (columns and existing indexes)\n\n' + schema if schema else ''

def render_evidence(repo: Repository, instruction: Instruction, candidates: Sequence[tuple[str, list[int]]], probe: DatabaseProbe) -> str:
    return PromptBuilder(repo, instruction, candidates, probe).build()

def _table_candidates(repo: Repository, instruction: Instruction, candidates: Sequence[tuple[str, list[int]]]) -> list[str]:
    names: list[str] = []
    for relative, _ in candidates:
        text = repo.read(relative) or ''
        names.extend(re.findall('\\bFROM\\s+([A-Za-z_][\\w.]*)', text, re.IGNORECASE))
        names.extend(re.findall('\\bJOIN\\s+([A-Za-z_][\\w.]*)', text, re.IGNORECASE))
        names.extend(re.findall('\\b(?:db_table|table_name)\\s*=\\s*[\'\\"]([\\w.]+)[\'\\"]', text))
    names.extend(instruction.identifiers)
    ordered: list[str] = []
    for name in names:
        clean = name.split('.')[-1].strip('"')
        if clean and clean not in ordered:
            ordered.append(clean)
    return ordered[:12]

def extract_json(content: str) -> dict | None:
    candidates: list[str] = []
    fenced = re.findall('```(?:json)?\\s*\\n(.*?)```', content, re.DOTALL)
    candidates.extend(fenced)
    candidates.append(content)
    for text in candidates:
        text = text.strip()
        start = text.find('{')
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start:index + 1])
                        return payload
                    except json.JSONDecodeError:
                        break
    return None
_READ_ONLY_START = re.compile('^\\s*(?:\\(\\s*)*(SELECT|WITH|EXPLAIN|SHOW|DESCRIBE|DESC|TABLE|VALUES)\\b', re.IGNORECASE)
_MUTATING = re.compile('\\b(INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE|CREATE|ALTER|DROP|TRUNCATE|RENAME|GRANT|REVOKE|COPY|VACUUM|ANALYZE\\s+\\w|REINDEX|CLUSTER|LOCK|SET\\s+(?!TRANSACTION)|RESET|DO|CALL|EXECUTE|OPTIMIZE|ATTACH|DETACH|KILL|SYSTEM|INTO\\s+OUTFILE|pg_terminate|pg_cancel|lo_)\\b', re.IGNORECASE)

def is_read_only_sql(query: str) -> bool:
    body = re.sub('--[^\\n]*|/\\*.*?\\*/', ' ', query, flags=re.DOTALL).strip()
    if not body or ';' in body.rstrip(';'):
        return False
    if not _READ_ONLY_START.match(body):
        return False
    if re.match('^\\s*(SHOW|DESCRIBE|DESC)\\b', body, re.IGNORECASE):
        return True
    return not _MUTATING.search(body)

def _request_key(kind: str, request: dict) -> str:
    fields = {k: v for k, v in request.items() if k != 'kind'}
    return kind + ':' + json.dumps(fields, sort_keys=True, default=str)

def fulfil_requests(repo: Repository, probe: DatabaseProbe, requests: Sequence[dict], graph: 'CallGraph | None'=None, checker: 'Checker | None'=None, served: 'set[str] | None'=None) -> str:
    blocks: list[str] = []
    for request in requests[:6]:
        kind = (request.get('kind') or '').lower()
        key = _request_key(kind, request)
        if served is not None:
            if key in served:
                blocks.append(f'{kind}: already shown in an earlier round -- it has not changed; request something new or reply with the edit')
                continue
            served.add(key)
        try:
            if kind == 'callers':
                symbol = (request.get('symbol') or '').strip()
                if not symbol or graph is None:
                    blocks.append(f"callers: {('no symbol given' if not symbol else 'unavailable')}")
                    continue
                defined = graph.defined_in.get(symbol, [])
                where = [f'{f}:{line}' for f, line in graph.defined_at.get(symbol, [])[:6]]
                referenced = sorted((f for f, names in graph.mentions.items() if symbol in names and f not in defined))
                blocks.append(f"callers {symbol}:\n  defined at: {where or 'nowhere found'}\n  referenced by {len(referenced)} file(s): {referenced[:20]}")
                continue
            if kind == 'count':
                if checker is None:
                    blocks.append('count: unavailable')
                    continue
                measured = checker.measure_query_scaling(request)
                blocks.append('count: ' + (measured.detail if measured else 'needs a Django project and a `call` using N'))
                continue
            if kind == 'schema':
                tables = request.get('tables') or request.get('table') or []
                if isinstance(tables, str):
                    tables = [tables]
                detail = probe.schema_for(tables, limit=8) if probe and tables else ''
                blocks.append('schema ' + ', '.join(map(str, tables)) + ':\n' + (detail or '[no database, or no such tables]'))
                continue
            if kind == 'read_file':
                relative = normalize_repo_path(request.get('path') or '') or ''
                text = repo.read(relative) if relative else None
                if text is None:
                    blocks.append(f'read_file {relative}: not found')
                    continue
                lines = text.splitlines()
                symbol = (request.get('symbol') or '').strip()
                if symbol:
                    defs = python_definitions(text) if relative.endswith('.py') else generic_blocks(text)
                    hit = next(((st, en) for name, st, en, kind_ in defs if kind_ != 'class' and name.split('.')[-1] == symbol), None) or next(((st, en) for name, st, en, kind_ in defs if re.search(f'\\b{re.escape(symbol)}\\b', name)), None)
                    if hit is None:
                        blocks.append(f'read_file {relative}: no definition named {symbol!r}; definitions here: {[d[0] for d in defs][:30]}')
                        continue
                    request = dict(request, start=max(1, hit[0] - 3), end=min(len(lines), hit[1] + 3))
                start = max(1, int(request.get('start') or 1))
                end = min(len(lines), int(request.get('end') or min(len(lines), start + 200)))
                body = '\n'.join((f'{number:5d}| {lines[number - 1]}' for number in range(start, end + 1)))
                blocks.append(f'read_file {relative} lines {start}-{end}:\n{truncate(body, 12000)}')
            elif kind == 'grep':
                pattern = request.get('pattern') or ''
                path_filter = request.get('path_filter') or ''
                compiled = re.compile(pattern)
                hits: list[str] = []
                for relative in repo.files:
                    if path_filter and path_filter not in relative:
                        continue
                    text = repo.read(relative)
                    if text is None or not compiled.search(text):
                        continue
                    for number, line in enumerate(text.splitlines(), start=1):
                        if compiled.search(line):
                            hits.append(f'{relative}:{number}: {line.strip()[:200]}')
                            if len(hits) >= 60:
                                break
                    if len(hits) >= 60:
                        break
                blocks.append(f'grep {pattern!r}:\n' + ('\n'.join(hits) or 'no matches'))
            elif kind in ('sql', 'explain'):
                query = (request.get('query') or '').strip().rstrip(';')
                if not is_read_only_sql(query):
                    blocks.append(f'{kind}: refused -- investigation queries must be read-only (SELECT / WITH ... SELECT / EXPLAIN / SHOW / DESCRIBE)')
                    continue
                if kind == 'explain':
                    output = probe.explain(query)
                    work = probe.clickhouse_work(query)
                    if work:
                        output += '\n\ndata actually read (system.query_log):\n' + work
                else:
                    output = truncate(probe.sql(query), 4000)
                blocks.append(f"{kind} {truncate(query, 400)}:\n{output or '[no output]'}")
            else:
                blocks.append(f'unsupported request kind: {kind!r}')
        except Exception as exc:
            blocks.append(f'{kind} request failed: {exc}')
    answer = '\n\n'.join(blocks) if blocks else 'no context returned'
    return answer

def apply_edits(repo: Repository, edits: Sequence[dict]) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    errors: list[str] = []
    for edit in edits:
        raw_path = edit.get('path') or ''
        if not raw_path:
            errors.append("an edit is missing its 'path'")
            continue
        relative = normalize_repo_path(raw_path)
        if relative is None:
            errors.append(f'refusing to edit path outside the repository: {raw_path}')
            continue
        if edit.get('new_file'):
            content = edit.get('content')
            if not isinstance(content, str):
                errors.append(f"{relative}: new_file edit has no 'content' string")
                continue
            try:
                repo.write(relative, content)
            except OSError as exc:
                errors.append(f'{relative}: could not be created ({exc}). Check the directory part of the path -- a component of it may be a file.')
                continue
            changed.append(relative)
            continue
        text = repo.read(relative)
        if text is None:
            errors.append(f'{relative}: file not found')
            continue
        search = edit.get('search')
        replace = edit.get('replace')
        if not isinstance(search, str) or not isinstance(replace, str):
            errors.append(f"{relative}: edit needs both 'search' and 'replace' strings")
            continue
        occurrences = text.count(search)
        if occurrences == 0:
            relaxed = _relaxed_find(text, search)
            if relaxed is None:
                errors.append(f"{relative}: the 'search' text was not found. It must reproduce the current file byte for byte, including indentation.")
                continue
            start, end = relaxed
            updated = text[:start] + replace + text[end:]
        elif occurrences > 1:
            errors.append(f"{relative}: the 'search' text appears {occurrences} times; make it unique")
            continue
        else:
            updated = text.replace(search, replace, 1)
        if updated == text:
            errors.append(f'{relative}: the edit is a no-op')
            continue
        try:
            repo.write(relative, updated)
        except OSError as exc:
            errors.append(f'{relative}: could not be written ({exc})')
            continue
        if relative not in changed:
            changed.append(relative)
    return (changed, errors)

def _relaxed_find(text: str, search: str) -> tuple[int, int] | None:

    def normalise(value: str) -> list[str]:
        return [line.rstrip() for line in value.splitlines()]
    haystack = text.splitlines(keepends=True)
    needle = normalise(search)
    if not needle:
        return None
    flat = [line.rstrip() for line in haystack]
    for index in range(len(flat) - len(needle) + 1):
        if flat[index:index + len(needle)] == needle:
            start = sum((len(line) for line in haystack[:index]))
            end = start + sum((len(line) for line in haystack[index:index + len(needle)]))
            replaced = ''.join(haystack[index:index + len(needle)])
            if replaced.endswith('\n') and (not search.endswith('\n')):
                end -= 1
            return (start, end)
    return None

@dataclass
class Candidate:
    patch: str
    checks: list[CheckResult]
    diagnosis: str

    @property
    def score(self) -> tuple[int, int, int]:
        passed = sum((1 for check in self.checks if check.passed))
        failed = sum((1 for check in self.checks if not check.passed))
        return (int(self.verified), -failed, passed)

    @property
    def verified(self) -> bool:
        return any((c.name.startswith('$ ') and c.passed and ('skipped' not in c.detail) for c in self.checks))

    @property
    def clean(self) -> bool:
        return bool(self.patch.strip()) and all((check.passed for check in self.checks)) and self.verified

class Solver:

    def __init__(self, repo: Repository, instruction: Instruction, probe: DatabaseProbe, llm: LLM) -> None:
        self.repo = repo
        self.instruction = instruction
        self.probe = probe
        self.llm = llm
        self.checker = Checker(repo, instruction, candidates=[], probe=probe)
        self.best: Candidate | None = None
        self._graph_cache: CallGraph | None = None
        self._served: set[str] = set()
        self.attempts = 0
        self.first_attempt_clean: bool | None = None
        self.final_clean = False
        self.failures_seen = 0
        self.slips_seen = 0
        self.stalls_seen = 0
        self.stop_reason = ''

    @property
    def _graph(self) -> CallGraph:
        if self._graph_cache is None:
            self._graph_cache = CallGraph(self.repo)
        return self._graph_cache

    def tier_for(self, failures: int) -> list[str]:
        if FORCE_MODEL:
            return [FORCE_MODEL]
        index = min(failures, len(LADDER) - 1)
        capped = self.llm.spent() > COST_TARGET_USD and index > 1
        if capped:
            index = 1
        return LADDER[index]

    def solve(self, candidates: Sequence[tuple[str, list[int]]]) -> str:
        self.checker.candidates = [path for path, _ in candidates]
        evidence = render_evidence(self.repo, self.instruction, candidates, self.probe)
        log(f'evidence bundle: {len(evidence)} characters')
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': f'{evidence}\n\n{PromptBuilder.SEPARATOR}\n\n# Response format\n\n{EDIT_PROTOCOL}'}]
        named_target = bool(self.instruction.edit_only or self.instruction.lint_paths)
        context_budget = 1 if named_target else 3
        context_rounds = 0
        failures = 0
        rounds = 0
        stalls = 0
        slips = 0
        unverified_retries = 0
        stalled = 0
        last_edit_sig = None
        max_rounds = MAX_REPAIR_ROUNDS + MAX_CONTEXT_ROUNDS + MAX_GATE_SLIPS + 2
        while failures < MAX_REPAIR_ROUNDS and rounds < max_rounds:
            rounds += 1
            if remaining_seconds() < 200:
                log('stopping: wall-clock budget nearly spent')
                break
            try:
                content, model = self.llm.complete(self.tier_for(failures + stalled), self._trim(messages))
            except BudgetExhausted as exc:
                log(f'stopping: {exc}')
                break
            except InferenceError as exc:
                permanent = bool(self.llm.blocked) and self.llm.blocked >= self.llm.unsupported
                if permanent or stalls >= 2 or remaining_seconds() < 420:
                    log(f'inference failed, giving up: {exc}')
                    break
                stalls += 1
                pause = 30 * stalls
                log(f'inference failed (looks transient), retrying the roster in {pause}s: {truncate(str(exc), 200)}')
                time.sleep(pause)
                self.llm.unsupported -= self.llm.blocked
                continue
            payload = extract_json(content)
            if payload is None:
                messages.append({'role': 'assistant', 'content': truncate(content, 2000)})
                messages.append({'role': 'user', 'content': 'That reply did not contain a parseable JSON object. Reply with exactly one JSON object in the documented shape.'})
                continue
            action = (payload.get('action') or '').lower()
            if action == 'need_context':
                if context_rounds >= min(context_budget, MAX_CONTEXT_ROUNDS):
                    messages.append({'role': 'assistant', 'content': json.dumps(payload)[:2000]})
                    messages.append({'role': 'user', 'content': "No investigation rounds remain. Reply with the 'edit' object using what you already have."})
                    continue
                context_rounds += 1
                requests = payload.get('requests') or []
                log(f"model requested context ({context_rounds}/{context_budget}): {[r.get('kind') for r in requests]}")
                answers = fulfil_requests(self.repo, self.probe, requests, graph=self._graph, checker=self.checker, served=self._served)
                remaining = min(context_budget, MAX_CONTEXT_ROUNDS) - context_rounds
                self._compact_old_context(messages)
                messages.append({'role': 'assistant', 'content': json.dumps(payload)})
                messages.append({'role': 'user', 'content': f'# Requested context\n\n{truncate(answers, 30000)}\n\n' + (f"You may request context {remaining} more time(s), or reply with the 'edit' object." if remaining else "Now reply with the 'edit' object.")})
                continue
            edits = payload.get('edits') or []
            if not edits:
                messages.append({'role': 'assistant', 'content': truncate(content, 1500)})
                messages.append({'role': 'user', 'content': "No edits were provided. Reply with an 'edit' object containing at least one search/replace edit."})
                continue
            diagnosis = str(payload.get('diagnosis') or '')
            edit_sig = hashlib.sha1(json.dumps(edits, sort_keys=True, default=str).encode()).hexdigest()
            repeated = edit_sig == last_edit_sig
            last_edit_sig = edit_sig
            self._adopt_constraints(payload.get('constraints'))
            if not self.instruction.commands:
                supplied = [c for c in payload.get('verify') or [] if isinstance(c, str) and 8 < len(c) < 400]
                if supplied:
                    log(f'no checks parsed from the instruction; using the {len(supplied)} the model extracted: {supplied}')
                    self.instruction.commands = supplied
            log(f'attempt {failures + 1} via {model}: {truncate(diagnosis, 300)}')
            self.attempts += 1
            self.repo.revert_all()
            changed, errors = apply_edits(self.repo, edits)
            if errors and (not changed):
                messages.append({'role': 'assistant', 'content': json.dumps(payload)[:4000]})
                messages.append({'role': 'user', 'content': 'None of the edits applied:\n' + '\n'.join(errors) + '\n\nRe-read the source shown above and reply with corrected edits.'})
                continue
            include_tests = remaining_seconds() > 400
            checks = self.checker.run_all(changed, include_tests=include_tests)
            if include_tests and all((c.passed for c in checks)):
                measured = self.checker.measure_query_scaling(payload.get('measure') or {})
                if measured is not None:
                    checks.append(measured)
                    log(f'query scaling: {measured.detail}')
            patch = build_patch(self.repo, changed)
            applies, apply_detail = verify_patch_applies(patch, self.repo)
            checks.append(CheckResult('patch applies cleanly', applies, apply_detail))
            candidate = Candidate(patch=patch, checks=checks, diagnosis=diagnosis)
            kept = self.best is None or candidate.score >= self.best.score
            if kept:
                self.best = candidate
            log('checks:\n' + summarise(checks))
            if self.first_attempt_clean is None:
                self.first_attempt_clean = candidate.clean
            if candidate.clean:
                log("all checks passed (including the task's own tests)")
                self.final_clean = True
                return patch
            runtime_failed = [c for c in checks if not c.passed and c.name.startswith('$ ') or (not c.passed and c.name == 'query scaling')]
            unverified = not candidate.verified and all((c.passed for c in checks))
            if unverified:
                log('WARNING: the change passed every static check but nothing ran it against the database -- no usable test command was found')
            self._compact_old_context(messages)
            feedback = self._failure_message(errors, checks)
            messages.append({'role': 'assistant', 'content': json.dumps(payload)[:4000]})
            self.repo.revert_all()
            if runtime_failed:
                failures += 1
                self.failures_seen = failures
                context_budget += 1
                messages.append({'role': 'user', 'content': feedback})
                continue
            if unverified:
                unverified_retries += 1
                if unverified_retries > 1:
                    log('no verification available after one request; adopting the statically clean patch as best effort')
                    return patch
                feedback += '\n\nNo test command was available to exercise this change. Supply the checks the instruction names in `verify`, or name the test module for the code you changed, so the patch can be verified. If the task truly names none, reply with the same edit and an empty verify.'
                messages.append({'role': 'user', 'content': feedback})
                continue
            slips += 1
            self.slips_seen = slips
            if slips > MAX_GATE_SLIPS:
                log(f'giving up after {slips} protocol slips without a verified attempt')
                break
            if repeated:
                stalled += 1
                self.stalls_seen = stalled
                log(f'identical edit repeated; handing the retry to the next model family (stall {stalled})')
                feedback += '\n\nYour reply was byte-identical to the previous one, which failed this same check. Do not resend it: change the construct that the check names.'
            feedback += '\n\nThis was a rule violation, not a wrong diagnosis: the tests did not run. Keep your analysis; fix only what the failing check names.'
            messages.append({'role': 'user', 'content': feedback})
        return self.best.patch if self.best else ''

    def _adopt_constraints(self, supplied) -> None:
        ins = self.instruction
        if not isinstance(supplied, dict):
            return
        static_found = bool(ins.edit_only or ins.lint_paths or ins.named_paths or ins.single_method)
        if static_found:
            return
        files = supplied.get('editable_files') or []
        accepted = []
        for raw in files if isinstance(files, list) else []:
            relative = normalize_repo_path(str(raw))
            if relative and (self.repo.root / relative).is_file():
                accepted.append(relative)
        if accepted:
            ins.edit_only = accepted[:4]
            self.checker.instruction = ins
            log(f"scope adopted from the model's reading of the instruction: {ins.edit_only}")
        bound = supplied.get('bounded_to_method')
        if isinstance(bound, str) and bound.strip():
            name = bound.strip().split('.')[-1].rstrip('()')
            if re.search(f'\\b{re.escape(name)}\\b', ins.text):
                ins.single_method = True
                ins.method_hint = name
                if '.' in bound:
                    ins.class_hint = bound.strip().split('.')[0]
                log(f'method bound adopted from the model, corroborated by the instruction: {bound}')
            else:
                log(f'ignored model-supplied method bound {bound!r}: not mentioned in the instruction')

    def _failure_message(self, errors: Sequence[str], checks: Sequence[CheckResult]) -> str:
        parts = ['The change was applied but did not pass verification.']
        if errors:
            parts.append('Edit problems:\n' + '\n'.join(errors))
        failures = [check for check in checks if not check.passed]
        for check in failures:
            parts.append(f'## {check.name}\n{truncate(check.detail, 6000, head_ratio=0.3)}')
        hints = error_hints('\n'.join((check.detail for check in failures)))
        if hints:
            parts.append('## What this output means\n' + '\n'.join((f'- {h}' for h in hints)))
        parts.append("Diagnose what this output says about the database work being done, then reply with a corrected 'edit' object. The edits replace the ORIGINAL file contents shown earlier -- your previous attempt has been reverted. Do not weaken or edit tests, and do not special-case fixture values.")
        return '\n\n'.join(parts)

    @staticmethod
    def _compact_old_context(messages: list[dict]) -> None:
        marker = '# Requested context'
        saved = 0
        for message in messages[:-1]:
            content = message.get('content') or ''
            if message.get('role') == 'user' and content.startswith(marker) and (len(content) > 1200):
                head = content[:600].rstrip()
                message['content'] = f'{head}\n\n[... {len(content) - 600:,} characters of already-consumed context trimmed; request again if needed]'
                saved += len(content) - len(message['content'])

    @staticmethod
    def _trim(messages: list[dict], keep: int=6) -> list[dict]:
        if len(messages) <= keep + 2:
            return messages
        trimmed = messages[:2] + messages[-keep:]
        return trimmed
# Translations of failure output the models keep misreading, matched against the test
# output of the run rather than against the problem statement. Each is a fact about SQL
# or the ORM, true of any repository; none names a task, a file or an expected value.
ERROR_HINTS: tuple[tuple[str, str], ...] = (('F401 .*imported but unused', 'F401: your change stopped using a name the file imports. Where the instruction says to keep imports unchanged, removing the import is not an option -- the code you write must still use that name.'), ('more than one row returned by a subquery used as an expression', 'A correlated subquery used as an annotation must return exactly one row. Aggregate the whole correlated set: no GROUP BY on a column that varies per matched row (in the ORM, `.values()` before `.annotate(Count)` groups by that column). Group on a constant or write the COUNT in SQL.'), ('null value in column .* violates not-null constraint|AssertionError: None != 0', 'An aggregate subquery yields NULL when nothing matches. Rows with no matches must get 0: COALESCE(..., 0) around the count, using only what the file already imports.'), ('invalid-syntax: Duplicate keyword argument|keyword argument repeated', 'The same keyword was passed twice in one call. To put two conditions on one field, use separate Q objects joined with & or |, e.g. Q(field__isnull=True) & Q(field=OuterRef(...)), or compare with OuterRef/F -- never repeat the keyword.'), ("Cannot resolve keyword '(\\w+)' into field", "That field name does not exist on the model; check the model's actual field names in the evidence before guessing another."), ("AttributeError: '(\\w+)' object has no attribute '(\\w+)'", "That object is one of the application's own classes, not a library client. Find its definition (the package map above, `callers`, or grep for `class <Name>`) and call a method it actually defines, with the statement format it expects."), ('NotSupportedError|not supported by this database backend', "The expression is not available on this database engine; use the engine's native operators or a RawSQL fallback."))

def error_hints(text: str) -> list[str]:
    hints = []
    for pattern, hint in ERROR_HINTS:
        if re.search(pattern, text) and hint not in hints:
            hints.append(hint)
    return hints

class MissingInstruction(RuntimeError):
    pass
HARNESS_INSTRUCTION = Path('/installed-agent/instruction.md')

def _instruction_text(payload: dict, root: Path, harness_copy: Path=HARNESS_INSTRUCTION) -> str:
    for key in ('problem_statement', 'instruction', 'problem', 'task', 'prompt'):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for location in (harness_copy, root / 'instruction.md'):
        if location.is_file():
            try:
                return location.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                continue
    raise MissingInstruction('no problem statement supplied')

# Entry point. Parses the statement, indexes and ranks the repository, connects to the
# live database, then loops: ask the model, apply its edits, run the repository's own
# checks, feed failures back. The patch returned is always built from edits applied
# during this run; there is no path that returns one from anywhere else.
def agent_main(input: dict) -> str:
    repo: Repository | None = None
    try:
        root = workdir()
        text = _instruction_text(input or {}, root)
        log(f"model={FORCE_MODEL or 'ladder'} workdir={root} timeout={os.getenv('AGENT_TIMEOUT', '?')}s budget=${os.getenv('RIDGES_MAX_COST_USD', DEFAULT_MAX_COST_USD)}")
        instruction = parse_instruction(text, root)
        log(f'kind={instruction.primary_kind} engine={instruction.engine} single_method={instruction.single_method} commands={len(instruction.commands)}')
        repo = Repository(root)
        targets = locate_targets(repo, instruction)
        if not targets:
            log('no candidate files found; falling back to instruction-named paths')
            targets = [(path, []) for path in instruction.named_paths]
        probe = DatabaseProbe(repo, instruction)
        if instruction.engine == 'unknown' and probe.available():
            instruction.engine = probe.targets[0].engine
            log(f'engine resolved from the live database: {instruction.engine}')
        llm = LLM()
        if not FORCE_MODEL:
            llm.discover_models()
        solver = Solver(repo, instruction, probe, llm)
        patch = solver.solve(targets)
        if not patch.strip():
            log('no patch produced')
            return ''
        log(f"returning patch: {len(patch)} bytes, {len(re.findall('^diff --git', patch, re.MULTILINE))} file(s)")
        log(llm.report())
        return patch
    except MissingInstruction as exc:
        log(f'no work to do: {exc}')
        return ''
    except Exception:
        log('agent failed:\n' + traceback.format_exc())
        return ''
    finally:
        if repo is not None:
            try:
                repo.revert_all()
            except Exception:
                pass
