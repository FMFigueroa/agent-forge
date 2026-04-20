# 🔨 agent-forge

> Multi-agent LinkedIn content factory powered by Claude — with built-in observability, cost tracking, and LLM-as-judge evals.

`agent-forge` is a production-grade reference implementation of a **multi-agent system** that turns a one-line topic into a polished LinkedIn post (text + carousel + image prompts) by orchestrating specialized Claude agents.

It's also a **showcase of senior GenAI engineering practices**: every agent call is traced, every token is costed, every output is evaluated.

## ✨ Why this exists

Most "multi-agent" demos out there are toy projects with no observability, no cost tracking, and no quality gates. `agent-forge` answers the questions that actually matter when you ship AI in production:

- **What is each agent doing right now?** → live trace viewer
- **How much did this run cost?** → per-agent token & USD breakdown
- **Is the output any good?** → LLM-as-judge evals + golden regression set
- **Can I trust it across runs?** → eval-driven prompt versioning

## 🧠 The agent team

| Agent | Model | Job |
|-------|-------|-----|
| **Orchestrator** | Opus 4.7 | Plans the workflow, routes work, reconciles outputs |
| **Researcher** | Sonnet 4.6 | Web search + summarize relevant context |
| **Drafter** | Sonnet 4.6 | Long-form post draft from research |
| **Editor** | Sonnet 4.6 | Refines hook, structure, voice |
| **Hashtag Specialist** | Haiku 4.5 | Picks 3-5 hashtags optimized for reach |
| **Image Prompter** | Haiku 4.5 | Generates carousel slide prompts |

## 🚀 Quick start

```bash
# Install uv if you don't have it
brew install uv

# Clone & install
git clone https://github.com/FMFigueroa/agent-forge.git
cd agent-forge
uv sync

# Configure
cp .env.example .env
# edit .env → add your ANTHROPIC_API_KEY

# Generate a post
uv run agent-forge generate "The hidden cost of context windows"
```

## 📊 Observability

Every run produces a trace with:
- **Per-agent latency** (P50/P95)
- **Token usage** (input/output, with prompt-cache hits)
- **USD cost** (per agent, per run)
- **Tool calls** (when applicable)

Traces persist to SQLite locally; switch to Postgres for production via `DB_URL`.

## ✅ Evals

Quality is measured by an **LLM-as-judge** that rates outputs across:
- Hook strength (does the first line stop the scroll?)
- Argument clarity
- Persona fit
- Engagement potential

A **golden set** of 20 prompts catches regressions when prompts/models change.

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────┐
│  CLI / FastAPI / Next.js UI                          │
└────────────────┬─────────────────────────────────────┘
                 ↓
    ┌────────────────────┐
    │  Orchestrator      │  ← Opus, plans the workflow
    └────┬───────────────┘
         ↓ delegate
    ┌────────────────────────────────────────┐
    │ Researcher → Drafter → Editor → ...    │
    └────────────────┬───────────────────────┘
                     ↓
    ┌────────────────────────────────────────┐
    │  Observability Layer                   │
    │  - Tracer (every Claude call → SQLite) │
    │  - Cost tracker (tokens × pricing)     │
    │  - LLM-as-judge evals                  │
    └────────────────────────────────────────┘
```

## 🛠 Tech stack

- **Python 3.12** + [`uv`](https://github.com/astral-sh/uv) (fast package manager)
- [`anthropic`](https://github.com/anthropics/anthropic-sdk-python) SDK with prompt caching + tool use
- [`fastapi`](https://fastapi.tiangolo.com/) for the HTTP API
- [`sqlmodel`](https://sqlmodel.tiangolo.com/) for trace storage
- [`typer`](https://typer.tiangolo.com/) + [`rich`](https://rich.readthedocs.io/) for the CLI
- [`ruff`](https://docs.astral.sh/ruff/) + [`pyright`](https://microsoft.github.io/pyright/) for code quality

## 📍 Status

🚧 **Early scaffolding.** Active development — follow [@FMFigueroa](https://github.com/FMFigueroa) for updates.

Roadmap (in order):
- [x] Project scaffolding & tooling
- [ ] Base `Agent` class with tracing
- [ ] Orchestrator + 2 specialist agents (Researcher, Drafter)
- [ ] Trace storage (SQLite via SQLModel)
- [ ] Cost tracker
- [ ] CLI flow end-to-end
- [ ] LLM-as-judge eval framework
- [ ] Golden regression set
- [ ] FastAPI server
- [ ] Next.js dashboard with live trace viewer
- [ ] Docker compose for one-command deploy

## 📝 License

MIT — use it, fork it, ship it.

## 👤 Author

Built by **[Felix M. Figueroa](https://github.com/FMFigueroa)** — building production AI systems at [@leonobitech](https://github.com/leonobitech).

If you're hiring **Senior GenAI Engineers** and this resonates, [let's chat](mailto:felixmanuelfigueroa@gmail.com).
