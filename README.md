# Multi-Agent Research & Report Pipeline

> **Stack:** LangGraph · CrewAI · FastAPI · Pydantic · Langfuse · Docker · GitHub Actions

4-agent orchestration pipeline (**Planner → Researcher → Writer →
Critic**) wired across **two interchangeable engines** — a LangGraph
typed-state machine with a conditional repair edge and a CrewAI
sequential crew — selectable per request. The Planner and Critic return
**schema-validated Pydantic objects** (native function calling on
OpenAI-compatible models, validated JSON with a retry on the Claude
CLI). The Critic audits every citation against its source snippet and
triggers a one-shot Writer repair pass when grounding is too low, making
the pipeline **self-correcting**. Reports cite sources drawn from arXiv,
Wikipedia, GitHub, Hacker News, and DuckDuckGo via a per-question
router. Served as an async FastAPI app with SSE streaming, a Dockerfile,
GitHub Actions CI, and Langfuse tracing on every agent and LLM call.
Ships with a YAML-driven eval harness that scores runs on citation
validity, grounding, source diversity, and routing precision.

---

## Why this is interesting

- **Self-correcting via the Critic agent.** After the Writer, a fourth
  `critic` node extracts every `[N]`-cited sentence, looks up the cited
  snippets in `sources`, and asks the worker LLM for a per-claim verdict
  (`supported / partial / unsupported / no_citation`). The aggregate
  grounding score is exposed on `ResearchReport.critique`. If the score
  drops below `REPAIR_THRESHOLD` (0.75) **and** there is at least one
  `unsupported` claim, a conditional LangGraph edge routes to a
  `writer_repair` node that rewrites only the flagged claims — capped
  at one repair pass so cost is bounded. Disable via `enable_critic:
  false` in the request body. See `app/agents/critic.py`.
- **Two engines, one SSE contract.** Send `"engine": "langgraph"` (default)
  or `"engine": "crew"` and get identical `start / stage / plan / routing
  / sources / research / critique / complete` event shapes back.
  LangGraph runs a typed `PipelineState` through a compiled graph with
  the conditional critic→repair edge; CrewAI runs sequential
  `Crew.kickoff` calls wrapped in manual phase orchestration so the SSE
  stream still emits per-stage deltas. The Researcher and Critic are
  shared between both engines — the divergence is only in planning and
  writing.
- **Eval harness built on the same Critic.** `python -m eval.run` pushes
  a curated 10-topic set (in `eval/topics.yaml`, spanning the five
  routing categories) through the pipeline and scores each run on six
  metrics — `citation_validity`, `citation_grounding` (reuses the
  Critic), `source_diversity`, `routing_precision`, `word_count`,
  `elapsed_seconds`. Outputs a per-topic JSON dump and an aggregate
  markdown table. See `eval/README.md`.
- **Schema-validated agent contracts.** The Planner returns a
  `ResearchPlan` and the Critic returns a `CritiqueOutput` — Pydantic
  models, not free text parsed with regexes. `app/structured.py` picks
  the strategy per backend: `with_structured_output(method=
  "function_calling")` for OpenAI-compatible models, and for the Claude
  CLI (which runs tools internally and can't surface tool calls) the
  JSON Schema goes in the prompt, the reply is validated, and one retry
  is made with the validation error fed back. Sources are never
  LLM-generated: they come straight from the search APIs as
  `list[Source]`, and the final response is validated as
  `ResearchReport`. The Writer and repair pass still emit markdown.
- **Per-question source routing.** A rule-based router inspects each
  planner-generated sub-question (alongside the original topic) and picks
  2–4 sources from arXiv / Wikipedia / GitHub / HN / DDG. "Best vector DB
  libraries" hits GitHub; "what is GraphRAG" hits Wikipedia; "latest agent
  frameworks 2026" hits Hacker News. The routing decision for each
  sub-question is streamed back to the UI so you can see *why* a source
  was chosen.
- **Checkable citations.** Each Researcher answer cites its own search
  results, and those local `[N]` markers are rewritten to indices into
  the merged, deduplicated `sources` array before the Writer sees them.
  Every `[N]` in the final report can therefore be resolved to a source
  URL and snippet — which is exactly what the Critic and the
  `citation_validity` eval metric check.
- **Depth-aware output.** `depth: "brief"` plans 3 questions and asks
  for 2–3 analysis paragraphs; `"detailed"` plans 5 and asks for 5–7.
- **Hybrid LLM stack with cost control.** When a logged-in Claude CLI is
  on PATH, the Planner runs on Claude (Sonnet default, togglable to
  **Opus** via the `use_opus_planner` UI flag); otherwise it uses the
  worker model. Researcher, Writer and Critic run on a cheaper
  OpenAI-compatible model (GLM-5-turbo by default via Z.ai). Set `LLM_BACKEND=claude` to route
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
  `@observe` / `trace_llm` spans capturing input, output, the model
  actually used for that backend, latency,
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
                 ┌──────────────────┐
                 │     Critic       │  audit each [N] claim against
                 │ verdict per claim│  its source snippet → score
                 └────────┬─────────┘
                          ▼
                 ┌── score<0.75 ──┐
                 │                │
                 ▼                ▼
       ┌────────────────┐    Markdown + sources +
       │ Writer repair  │    critique streamed via SSE
       │ (one pass max) │           │
       └────────┬───────┘           │
                ▼                   ▼
              re-critique     Langfuse traces
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
        "engine": "langgraph",
        "enable_critic": true
      }'
```

`engine` accepts `"langgraph"` (default) or `"crew"`. Both produce the
same event sequence: `start`, `stage`, `plan`, `routing`, `sources`,
`research`, `critique`, `complete` (or `error`). The `critique` event
carries `{score, supported, unsupported, total_claims, repaired}`. Set
`enable_critic: false` to skip the audit + repair entirely. Each
event's payload is a JSON object — see `app/routes/research.py` for the
schema.

### `POST /research/`  (synchronous)

Returns the final `ResearchReport` (topic, summary, key_findings,
full_report, sources, word_count, critique) once the pipeline
completes. Honors the same `engine` and `enable_critic` fields. The
`critique` field is `null` when `enable_critic=false`.

## Eval

```bash
# Defaults: langgraph engine, brief depth, critic on
python -m eval.run

# Smoke run
python -m eval.run --limit 3

# CrewAI engine
python -m eval.run --engine crew
```

Outputs land in `eval_output/<timestamp>/`:

- `summary.md` — markdown table + aggregate metrics
- `summary.json` — machine-readable metrics
- `<topic-slug>.json` — full report + per-topic metrics

Six metrics per topic: `citation_validity` (free), `citation_grounding`
(reuses the Critic), `source_diversity`, `routing_precision`,
`word_count`, `elapsed_seconds`. See `eval/README.md` for details and
cost notes.

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
when idle.

> **Note:** `LLM_BACKEND=claude` only works locally — the Claude CLI
> isn't authenticated on a remote host. Use the OpenAI-compatible
> backend (GLM, OpenAI, OpenRouter, …) for cloud deploys. The LangGraph
> engine works with either backend. The CrewAI engine's Planner and
> Writer currently always run through the Claude CLI
> (`app/crew/llm.py`), so use `engine: "langgraph"` on cloud deploys.

## Tech

Python 3.11 · FastAPI · LangGraph (conditional edges + typed state) ·
CrewAI · LangChain · langchain-claude-code · ChatOpenAI (any
OpenAI-compatible provider) · Pydantic v2 · Langfuse · httpx · ddgs ·
PyYAML · arXiv API · Wikipedia API · GitHub Search API · HN/Algolia
API.

## Files worth reading

- `app/graph/pipeline.py` — LangGraph wiring with conditional critic→repair edge
- `app/crew/pipeline.py` — CrewAI sequential crews + appended critic phase
- `app/crew/llm.py` — Claude CLI bridge into CrewAI
- `app/agents/{planner,researcher,writer,critic}.py` — the four nodes
- `app/agents/critic.py` — claim extraction, structured verdicts, repair loop
- `app/structured.py` — schema-validated output across both LLM backends
- `app/tools/router.py` — keyword-based source selection
- `app/tools/sources.py` — five free-API clients
- `app/observability.py` — Langfuse spans (no-op fallback)
- `app/routes/research.py` — sync + SSE endpoints (both engines)
- `app/static/index.html` — the live demo UI
- `eval/run.py` — eval harness CLI
- `eval/metrics.py` — six per-topic metrics (free + LLM-judge via Critic)
- `eval/topics.yaml` — curated 10-topic set across routing categories
