# Multi-Agent Research & Report Pipeline

A production-ready multi-agent system that automatically researches topics and generates structured reports. Built with **LangGraph** state machines, **CrewAI** role definitions, and served via an async **FastAPI** REST API.

## Architecture

```
POST /research
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Planner   │────▶│  Researcher  │────▶│   Writer    │
│  (LangGraph)│     │  (CrewAI)    │     │  (CrewAI)   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                                         │
      └─────────────────────────────────────────┘
                  Structured Report
```

## Features
- **LangGraph** state machine orchestrates agent transitions
- **CrewAI** role definitions enforce task boundaries and reduce hallucination
- Structured output parsing with automatic retry logic
- Async FastAPI REST API with full OpenAPI docs
- Dockerized for one-command deployment
- GitHub Actions CI/CD pipeline

## Quickstart

```bash
git clone https://github.com/axon011/multi-agent-pipeline
cd multi-agent-pipeline
cp .env.example .env   # add your OPENAI_API_KEY
docker-compose up --build
```

API docs available at: `http://localhost:8000/docs`

## API Usage

```bash
curl -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"topic": "Latest advances in RAG systems", "depth": "detailed"}'
```

## Stack
- Python 3.11, FastAPI, LangGraph, CrewAI, LangChain
- OpenAI GPT-4o (configurable)
- Docker, GitHub Actions
