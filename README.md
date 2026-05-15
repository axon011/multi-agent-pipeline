# Multi-Agent Research & Report Pipeline

> **Stack:** LangGraph · CrewAI · FastAPI · Pydantic · Langfuse · Docker · GitHub Actions
> **Live demo:** _add your Fly URL here after `fly deploy`_

3-agent orchestration pipeline (**Planner → Researcher → Writer**) wired
across **two interchangeable engines** — a LangGraph typed-state machine
and a CrewAI sequential crew — selectable per request. Strict role
boundaries are enforced via Pydantic structured outputs. Produces
**2,000+ word research reports with verifiable source citations** drawn
from arXiv, Wikipedia, GitHub, Hacker News, and DuckDuckGo via a
per-question router. Deployed as an async FastAPI service with Docker
containerization, GitHub Actions CI/CD, and Langfuse observability on
every agent and LLM call.

![demo](docs/demo.gif)
*(Drop a 30-second screen recording at `docs/demo.gif` after first deploy.)*

---

## Why this is interesting

- **Two engines, one SSE contract.** Send `"engine": "langgraph"` (default)
  or `"engine": "crew"` and get identical `start / stage / plan / routing
  / sources / research / complete` event shapes back. LangGraph runs a
  typed `PipelineState` through a compiled graph; CrewAI runs sequential
  `Crew.kickoff` calls wrapped in manual phase orchestration so the SSE
  stream still emits per-stage deltas. Same Researcher node is shared
  between both engines — the divergence is only in planning and writing.
- **Strict role boundaries.** Each agent's output is parsed into a Pydantic
  schema (`PipelineState`, `ResearchReport`, `Source`) before the next
  agent runs. The Researcher cannot fabricate sources because the schema
  enforces a structured `list[Source]` with `title` / `url` / `snippet`
  fields; the Writer cannot leak between citations because the report is
  validated as `ResearchReport` with `key_findings`, `summary`,
  `full_report`, `word_count`, and a `sources` list.
- **Per-question source routing.** A rule-based router inspects each
  planner-generated sub-question (alongside the original topic) and picks
  2–4 sources from arXiv / Wikipedia / GitHub / HN / DDG. "Best vector DB
  libraries" hits GitHub; "what is GraphRAG" hits Wikipedia; "latest agent
  frameworks 2026" hits Hacker News. The routing decision for each
  sub-question is streamed back to the UI so you can see *why* a source
  was chosen.
- **Verifiable citations.** Every claim in the final report carries `[1]
  [2] [3]` markers that map to the `sources` array — citations are not
  free-text, they are array indices into a structured-output Pydantic
  list, which makes them tamper-evident.
- **Hybrid LLM stack with cost control.** Planner runs on Claude (Sonnet
  default, togglable to **Opus** via the `use_opus_planner` UI flag);
  Researcher and Writer run on a cheaper OpenAI-compatible model
  (GLM-5-turbo by default via Z.ai). Set `LLM_BACKEND=claude` to route
  the whole pipeline through the Claude CLI subscription for local
  testing without API spend; the CrewAI engine bridges Claude CLI into
  CrewAI agents via `app/crew/llm.py`.
- **Async fan-out, throttled by backend.** The Researcher dispatches all
  N sub-questions in parallel via `asyncio.gather` against rate-friendly
  HTTP-API LLMs; an `asyncio.Semaphore(1)` serializes calls automatically
  when the backend is the Claude CLI (which can't be spawned
  concurrently on Windows event loops). Each CLI call is pushed to
  `asyncio.to_thread` so uvicorn's loop isn't blocked.
- **Live SSE pipeline view.** Stage transitions, the per-question routing
  panel, and previewed source URLs all stream to the browser via
  Server-Sent Events before the Writer composes the final report.
- **Langfuse observability.** Every agent and LLM call is wrapped in
  `@observe` / `trace_llm` spans capturing input, output, model, latency,
  routing decisions, and source counts. The instrumentation degrades to
  a no-op when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` aren't set,
  so the pipeline runs unchanged without an account. Module:
  `app/observability.py`.

## Architecture

```
                 ┌──────────────────┐
   POST /research│   Planner        │  Claude Sonnet/Opus  (or GLM in cloud mode)
   /stream  ───▶ │  (3-5 questions) │  engine ∈ {langgraph, crew}
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │   Router         │  rule-based; topic + question
                 │  per question    │
                 └────────┬─────────┘
                          ▼
        ┌─────────┬─────────┬─────────┬──────────┬─────────┐
        │  arXiv  │   Wiki  │ GitHub  │   HN     │   DDG   │   parallel,
        │  (paper)│ (concept)│ (code) │ (trend)  │(fallback)│   per question
        └────┬────┴────┬────┴────┬────┴────┬─────┴────┬────┘
             └─────────┴─────────┴─────────┴──────────┘
                                ▼
                 ┌──────────────────┐
                 │   Researcher     │  GLM-5-turbo (or Claude in local mode)
                 │ cite-and-synthe- │  asyncio.gather across questions
                 │ size per question│  (shared across both engines)
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │     Writer       │  GLM-5-turbo (or Claude in local mode)
                 │ assemble report  │
                 └────────┬─────────┘
                          ▼
                  Markdown + sources
                  streamed via SSE
                          │
                          ▼
                  Langfuse traces
                  (agents + LLM calls)
```

## Quickstart (local)

```bash
git clone https://github.com/axon011/multi-agent-pipeline
cd multi-agent-pipeline
pip install -r requirements.txt

# .env — pick ONE of the two configurations below

# (A) GLM-5-turbo via Z.ai — cheapest, requires top-up at z.ai
LLM_API_KEY=...
LLM_BASE_URL=https://api.z.ai/api/coding/paas/v4
LLM_MODEL=glm-5-turbo
LLM_BACKEND=glm                                # default

# (B) Claude via subscription (local only — needs `claude` CLI logged in)
LLM_BACKEND=claude

# Optional — Langfuse observability (no-op if unset)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# run
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/`.

The OpenAI-compatible variables work for any provider with that API
shape: GLM, OpenAI, OpenRouter, Together, Groq, etc.

## API

### `POST /research/stream`  (Server-Sent Events)

```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H 'Content-Type: application/json' \
  -d '{
        "topic": "GraphRAG explained",
        "depth": "brief",
        "use_opus_planner": true,
        "engine": "langgraph"
      }'
```

`engine` accepts `"langgraph"` (default) or `"crew"`. Both produce the
same event sequence: `start`, `stage`, `plan`, `routing`, `sources`,
`research`, `complete` (or `error`). Each event's payload is a JSON
object — see `app/routes/research.py` for the schema.

### `POST /research/`  (synchronous)

Returns the final `ResearchReport` (topic, summary, key_findings,
full_report, sources, word_count) once the pipeline completes. Honors
the same `engine` field.

## Deploy (Fly.io)

```bash
# Windows: iwr https://fly.io/install.ps1 | iex
# macOS:   brew install flyctl

fly auth login
fly launch --copy-config --no-deploy           # accepts fly.toml
fly secrets set LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=glm-5-turbo
# Optional Langfuse secrets
fly secrets set LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=...
fly deploy
```

The included `fly.toml` runs on a 1-CPU 512 MB shared VM and auto-stops
when idle, so the free tier covers a portfolio demo. Drop the resulting
URL into the badge at the top of this README.

> **Note:** `LLM_BACKEND=claude` only works locally — the Claude CLI
> isn't authenticated on a remote host. Use the OpenAI-compatible
> backend (GLM, OpenAI, OpenRouter, …) for cloud deploys. CrewAI engine
> works in both modes; LangGraph engine works in both modes.

## Tech

Python 3.11 · FastAPI · LangGraph · CrewAI · LangChain ·
langchain-claude-code · ChatOpenAI (any OpenAI-compatible provider) ·
Pydantic v2 · Langfuse · httpx · ddgs · arXiv API · Wikipedia API ·
GitHub Search API · HN/Algolia API.

## Files worth reading

- `app/graph/pipeline.py` — LangGraph wiring + state-to-report
- `app/crew/pipeline.py` — CrewAI sequential crews (planner + writer)
- `app/crew/llm.py` — Claude CLI bridge into CrewAI
- `app/agents/{planner,researcher,writer}.py` — the three nodes (shared between engines)
- `app/tools/router.py` — keyword-based source selection
- `app/tools/sources.py` — five free-API clients
- `app/observability.py` — Langfuse spans (no-op fallback)
- `app/routes/research.py` — sync + SSE endpoints (both engines)
- `app/static/index.html` — the live demo UI
