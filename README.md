# Support Agent

A customer support AI assistant powered by RAG (Retrieval-Augmented Generation) that answers queries using a knowledge base of Help Center articles.

## Features

- **Intelligent Responses** — Uses OpenAI GPT-4o-mini with RAG to provide accurate, sourced answers
- **Action Classification** — Automatically classifies responses as:
  - `CLOSE` — Query resolved, no further action needed
  - `HANDOVER` — Escalate to human support
  - `WAIT` — Needs clarification from user
- **Multi-turn Conversations** — Maintains context across conversation turns
- **Knowledge Base Search** — Semantic search over 200+ Help Center articles using ChromaDB
- **Full Observability** — LangFuse integration for tracing, metrics, and evaluation
- **Web Interface** — Gradio chat UI with action badges and source citations
- **REST API** — FastAPI backend for programmatic access
- **Evaluation Pipeline** — Batch evaluation with metrics and CSV/JSON export

## Architecture

### Query Flow

1. **User submits query** via CLI, API, or Gradio UI
2. **Session context loaded** (for multi-turn conversations)
3. **RAG retrieval** — Query embedded and matched against ChromaDB
4. **Agent reasoning** — CrewAI agent processes query + context
5. **LLM generation** — OpenAI generates response with action classification
6. **Response returned** with action, message, sources, and confidence

### Action Types

| Action | Description | Next Step |
|--------|-------------|-----------|
| `CLOSE` | Query fully resolved | Conversation ends |
| `HANDOVER` | Needs human support | Escalate to agent |
| `WAIT` | Needs more info | Ask clarifying question |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Package Manager | uv |
| Agent Framework | CrewAI |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB |
| Frontend | Gradio |
| API | FastAPI |
| Observability | LangFuse |
| Testing | pytest (457 tests) |

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/support-agent-demo-project.git
cd support-agent-demo-project

# Install dependencies with uv
uv sync
```

### 2. Configure Environment

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-...
MODEL_LLM = "gpt-4o-mini"
MODEL_EMBEDDING = "text-embedding-3-small"
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_ENABLED=true

# optionnal:
LANGFUSE_HOST=https://cloud.langfuse.com
CHROMA_PERSIST_DIR=./data/db/chroma_db
LOG_LEVEL=INFO
CREWAI_TRACING_ENABLED=true
```

### 3. Index the Knowledge Base

```bash
uv run python -m cli.main index
```

### 4. Start the Server

```bash
uv run python -m cli.main serve
```

Open http://localhost:7860 for the Gradio UI, or http://localhost:8000/docs for the API.

## Usage

### CLI Commands

```bash
# Index knowledge base
uv run python -m cli.main index              # Index articles
uv run python -m cli.main index --force      # Clear and rebuild

# Chat
uv run python -m cli.main chat "How do I reset my password?"
uv run python -m cli.main chat               # Interactive mode

# Server
uv run python -m cli.main serve              # Start API + UI
uv run python -m cli.main serve --api-only   # API only

# Evaluation
uv run python -m cli.main evaluate --sample 100
uv run python -m cli.main evaluate --all --output results/eval.json

# Stats
uv run python -m cli.main stats              # Knowledge base statistics
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Single-turn query |
| POST | `/sessions` | Create conversation session |
| GET | `/sessions/{id}` | Get session history |
| POST | `/sessions/{id}/messages` | Send message (multi-turn) |
| DELETE | `/sessions/{id}` | End session |

### Example API Call

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I integrate with Slack?"}'
```

## Project Structure

```
src/support_agent/
├── config.py          # Pydantic settings
├── logging.py         # Structlog configuration
├── exceptions.py      # Custom exceptions
├── clients/           # OpenAI & LangFuse clients
├── indexing/          # CSV parsing, chunking, ChromaDB indexing
├── rag/               # Retriever, CrewAI tools
├── agent/             # CrewAI crew/tasks, schemas
├── memory/            # Conversation session management
├── evaluation/        # Batch evaluation, metrics
├── api/               # FastAPI backend
└── ui/                # Gradio interface

cli/
└── main.py            # Typer CLI

data/
├── help.csv           # Knowledge base (~200 articles)
├── questions.csv      # Evaluation queries (1165)
└── conversations/     # Multi-turn test scenarios
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src/support_agent

# Specific test file
uv run pytest tests/unit/test_agent.py -v
```

### Linting

```bash
uv run ruff check src tests cli
uv run ruff format src tests cli
```

### Test Markers

```python
@pytest.mark.slow         # Tests hitting real APIs
@pytest.mark.integration  # Integration tests
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `LANGFUSE_PUBLIC_KEY` | required | LangFuse public key |
| `LANGFUSE_SECRET_KEY` | required | LangFuse secret key |
| `LANGFUSE_ENABLED` | `true` | Enable/disable tracing |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | LangFuse host |
| `CHROMA_PERSIST_DIR` | `data/db` | ChromaDB storage path |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `console` | `console` or `json` |
