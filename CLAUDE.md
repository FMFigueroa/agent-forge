# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`agent-forge` is a multi-agent LinkedIn content factory powered by Claude. A one-line topic fans out to a team of specialized agents (Orchestrator → Researcher → Drafter → Editor → Hashtag Specialist → Image Prompter) and produces a polished post with full observability (traces, cost, LLM-as-judge evals).

**Status**: early scaffolding. CLI/API/agents/observability/evals packages exist but are stubs — the only implemented surface is `cli.py` (version + unimplemented `generate`/`serve`).

## Commands

Package management is **uv** (not pip/poetry). Always prefix runs with `uv run`.

```bash
uv sync                           # install deps (respects uv.lock)
uv sync --all-extras              # include dev group

uv run agent-forge version        # entry point (src/agent_forge/cli.py)
uv run agent-forge generate "..." # stub
uv run agent-forge serve          # stub (FastAPI)

uv run ruff check .               # lint (E,F,I,N,UP,B,SIM,RUF; E501 ignored)
uv run ruff format .              # format (line-length=100)
uv run ruff format --check .      # CI-mode format check
uv run pyright                    # type check (standard mode, py3.12)
uv run pytest -v                  # all tests
uv run pytest tests/test_version.py::test_version_is_set  # single test
uv run pytest --cov=agent_forge   # with coverage
```

CI (`.github/workflows/ci.yml`) runs ruff check + ruff format --check + pyright + pytest on every push/PR to `main`. All must pass.

## Architecture

### Layout — `src/` layout, package = `agent_forge`

```
src/agent_forge/
  cli.py              # typer entry point (exposed as `agent-forge` script)
  config.py           # pydantic-settings, loads .env, single `settings` singleton
  agents/             # stub — will hold Agent base class + specialists
  api/                # stub — FastAPI server
  observability/      # stub — tracer, cost tracker (SQLite via SQLModel)
  evals/              # stub — LLM-as-judge + golden set
```

### Model tiering (`config.py`)

Three model slots, configured by env var and consumed by agents based on job complexity:

| Setting | Default | Purpose |
|---|---|---|
| `ORCHESTRATOR_MODEL` | `claude-opus-4-7` | Planning, routing, reconciliation |
| `SPECIALIST_MODEL` | `claude-sonnet-4-6` | Research, drafting, editing |
| `FAST_MODEL` | `claude-haiku-4-5-20251001` | Hashtags, short image prompts |

When adding a new agent, pick the tier that matches the task — don't hardcode model IDs, read from `settings`.

### Observability contract (design intent)

Every Claude call must flow through the tracer so that **per-agent latency, token usage (incl. cache hits), and USD cost** land in SQLite (or Postgres via `DB_URL`). When implementing the `Agent` base class, the call path should be: `Agent.run()` → wraps `anthropic` SDK call → emits trace row + cost row keyed by agent + run_id. Do not call the SDK directly from agent subclasses.

### Evals contract (design intent)

Outputs are graded by an LLM-as-judge across hook strength, clarity, persona fit, engagement. A golden set of ~20 prompts guards against regressions when prompts/models change — run evals after any prompt or model edit.

## Conventions

- **Python 3.12+ required** (`.python-version`, `requires-python`). Use modern syntax (`X | Y`, `list[int]`, PEP 695 generics if helpful).
- **Type everything.** Pyright runs in `standard` mode over `src` and `tests`; CI will fail on errors.
- **Line length 100** (ruff), but `E501` is ignored — don't manually wrap long strings.
- **Tests are async-friendly**: `asyncio_mode = "auto"` — `async def test_...` works without decorators.
- Entry point is wired via `[project.scripts] agent-forge = "agent_forge.cli:app"`. New CLI commands go on `cli.app` as typer commands.

## Commits (project-specific overrides)

Per global rules: **never** add `Co-Authored-By` footers. Every commit message must end with `🚀 Powered by Claude Opus 4.7`.
