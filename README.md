# Multi-Agent Research Pipeline

A LangGraph-orchestrated research agent that breaks a topic into questions,
fans out to **arXiv, Wikipedia, GitHub, Hacker News, and DuckDuckGo** in
parallel based on a per-question router, and synthesises a cited markdown
report. Streams every stage live to the browser over Server-Sent Events.

> **Live demo:** _add your Fly URL here after `fly deploy`_

![demo](docs/demo.gif)
*(Drop a 30-second screen recording at `docs/demo.gif` after first deploy.)*

---

## What's interesting about it

- **Per-question source routing.** A rule-based router inspects each
  sub-question (alongside the original topic) and picks 2–3 sources from
  arXiv / Wikipedia / GitHub / HN / DDG. "Best vector DB libraries" hits
  GitHub; "what is GraphRAG" hits Wikipedia; "latest agent frameworks
  2026" hits Hacker News. The routing decisions are streamed back to the
  UI so you can see *why* each source was chosen.
- **Hybrid LLM stack.** Planner runs on Claude (Sonnet by default,
  togglable to **Opus 4.6**); researcher and writer run on a cheaper
  OpenAI-compatible model (GLM-5-turbo by default). Toggle the entire
  stack to Claude with `LLM_BACKEND=claude` for local testing.
- **All-free sources.** No paid search API. arXiv and Wikipedia have
  no rate limits; GitHub gives 60 unauth requests/hour; HN/Algolia is
  unlimited; DDG is best-effort with a worldwide-region filter and a
  geo-junk heuristic for foreign e-commerce noise.
- **Live SSE pipeline view.** Stages light up as they run; the routing
  panel shows source pills per question; the sources panel previews
  fetched URLs before the writer composes the final report.

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
