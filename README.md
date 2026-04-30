# Multi-Agent Research & Report Pipeline

> **Stack:** LangGraph · FastAPI · Pydantic · Docker · GitHub Actions
> **Live demo:** _add your Fly URL here after `fly deploy`_

3-agent orchestration pipeline (**Planner → Researcher → Writer**) built on
LangGraph state machines with **strict role boundaries enforced via Pydantic
structured outputs**. Produces **2,000+ word research reports with verifiable
source citations** drawn from arXiv, Wikipedia, GitHub, Hacker News, and
DuckDuckGo via a per-question router. Deployed as an async FastAPI service
with Docker containerization and GitHub Actions CI/CD — push to production
with zero manual intervention.

![demo](docs/demo.gif)
*(Drop a 30-second screen recording at `docs/demo.gif` after first deploy.)*

---

## Why this is interesting

- **Strict role boundaries.** Each agent's output is parsed into a Pydantic
  schema (`PipelineState`, `ResearchReport`, `Source`) before the next
  agent runs. The Researcher cannot fabricate sources because the schema
  enforces a structured `list[Source]` with `title` / `url` / `snippet`
  fields; the Writer cannot leak between citations because the report is
  validated as `ResearchReport` with `key_findings`, `summary`,
  `full_report`, `word_count`, and a `sources` list.
- **Per-question source routing.** A rule-based router inspects each
  planner-generated sub-question (alongside the original topic) and picks
  2–3 sources from arXiv / Wikipedia / GitHub / HN / DDG. "Best vector DB
  libraries" hits GitHub; "what is GraphRAG" hits Wikipedia; "latest agent
  frameworks 2026" hits Hacker News. The routing decision for each
  sub-question is streamed back to the UI so you can see *why* a source
  was chosen.
- **Verifiable citations.** Every claim in the final report carries `[1]
  [2] [3]` markers that map to the `sources` array — citations are not
  free-text, they are array indices into a structured-output Pydantic
  list, which makes them tamper-evident.
- **Hybrid LLM stack with cost control.** Planner runs on Claude (Sonnet
  default, togglable to **Opus 4.6** via UI flag); Researcher and Writer
  run on a cheaper OpenAI-compatible model (GLM-5-turbo by default).
  `LLM_BACKEND=claude` routes the whole pipeline through the Claude CLI
  for local testing without API spend.
- **Async fan-out, throttled by backend.** The Researcher dispatches all
  N sub-questions in parallel via `asyncio.gather` against rate-friendly
  HTTP-API LLMs; an `asyncio.Semaphore(1)` serializes calls automatically
  when the backend is the Claude CLI (which can't be spawned
  concurrently).
- **Live SSE pipeline view.** Stage transitions, the per-question routing
  panel, and previewed source URLs all stream to the browser via
  Server-Sent Events before the Writer composes the final report.
- **Langfuse observability.** Every agent and LLM call is wrapped in
  `@observe` / `trace_llm` spans capturing input, output, model, latency,
  routing decisions, and source counts. The instrumentation degrades to
  a no-op when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` aren't set,
  so the pipeline runs unchanged without an account.

## Architecture

```
                 ┌──────────────────┐
   POST /research│   Planner        │  Claude Sonnet/Opus  (or GLM in cloud mode)
   /stream  ───▶ │  (3-5 questions) │
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
                 │ size per question│
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │     Writer       │  GLM-5-turbo (or Claude in local mode)
                 │ assemble report  │
                 └────────┬─────────┘
                          ▼
                  Markdown + sources
                  streamed via SSE
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

# (B) Claude via subscription (local only — needs `claude` CLI logged in)
LLM_BACKEND=claude

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
        "use_opus_planner": true
      }'
```

Stream events: `start`, `stage`, `plan`, `routing`, `sources`,
`research`, `complete`. Each event's payload is a JSON object — see
`app/routes/research.py` for the schema.

### `POST /research/`  (synchronous)

Returns the final `ResearchReport` (topic, summary, key_findings,
full_report, sources, word_count) once the pipeline completes.

## Deploy (Fly.io)

```bash
# Windows: iwr https://fly.io/install.ps1 | iex
# macOS:   brew install flyctl

fly auth login
fly launch --copy-config --no-deploy           # accepts fly.toml
fly secrets set LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL=glm-5-turbo
fly deploy
```

The included `fly.toml` runs on a 1-CPU 512 MB shared VM and auto-stops
when idle, so the free tier covers a portfolio demo. Drop the resulting
URL into the badge at the top of this README.

> **Note:** `LLM_BACKEND=claude` only works locally — the Claude CLI
> isn't authenticated on a remote host. Use the OpenAI-compatible
> backend (GLM, OpenAI, OpenRouter, …) for cloud deploys.

## Tech

Python 3.11 · FastAPI · LangGraph · LangChain · langchain-claude-code ·
ChatOpenAI (any OpenAI-compatible provider) · httpx · ddgs · arXiv API
· Wikipedia API · GitHub Search API · HN/Algolia API.

## Files worth reading

- `app/graph/pipeline.py` — LangGraph wiring + state-to-report
- `app/agents/{planner,researcher,writer}.py` — the three nodes
- `app/tools/router.py` — keyword-based source selection
- `app/tools/sources.py` — five free-API clients
- `app/routes/research.py` — sync + SSE endpoints
- `app/static/index.html` — the live demo UI
