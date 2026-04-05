# Robust Package Design — agentic-doc-search v1.0

**Date:** 2026-04-05
**Approach:** API-First Redesign — design the full public API upfront, then implement against that contract.

## Goals

Transform agentic-doc-search from a working prototype into a production-ready Python package that developers can install, configure, and integrate into their own applications. The package should be async-first, highly configurable, and support multiple document sources.

## Core Public API

Async-first. Three primary exports: `Agent`, `Corpus`, `SearchResult`.

```python
from agentic_doc_search import Agent, Corpus, SearchResult

corpus = Corpus("./my-docs")
agent = Agent(
    model="claude-sonnet-4-20250514",

    # Reasoning
    extended_thinking=False,     # Chain-of-thought before acting
    reasoning_trace="full",      # "full" | "summary" | "none"

    # Extensibility
    extra_tools=[...],           # Custom ToolDefinitions alongside built-ins
    on_tool_call=my_logger,      # Hook: after each tool call
    on_query_rewrite=my_cb,      # Hook: after query rewrite
    on_iteration=my_step_cb,     # Hook: after each ReAct iteration

    # LLM backend
    llm_config=LLMConfig(...),   # Easy path (litellm under the hood)
    llm_backend=MyBackend(),     # Escape hatch (full control, overrides llm_config)
)

result = await agent.search(
    corpus,
    "What is the refund policy?",

    # Per-query search pipeline config
    rewrite_query=True,          # LLM rewrites query into better search terms
    bm25_rerank=True,            # Re-rank search results by BM25 relevance
    max_results=50,              # Cap results per tool call
    max_iterations=20,           # Max ReAct loop iterations
    token_budget=100_000,        # Hard cap on tokens (stop and give best answer)
    bypass_budget=False,         # Override token budget
    context_compression=True,    # Summarize older tool results to free context
)
```

### SearchResult

```python
result.answer              # str
result.sources             # list[SourceReference] — parsed from agent output
result.reasoning           # list[ToolCall] — controlled by reasoning_trace
result.rewritten_query     # str | None — the rewritten query if enabled
result.token_usage         # TokenUsage
```

### TokenUsage (expanded)

```python
result.token_usage.prompt_tokens
result.token_usage.completion_tokens
result.token_usage.total_tokens
result.token_usage.estimated_cost_usd    # Based on model pricing
result.token_usage.per_iteration         # list of per-step token counts
```

### Agent-level vs Search-level Config

| Concern | Where | Why |
|---------|-------|-----|
| Model, thinking mode, reasoning trace | `Agent(...)` | Engine config, set once |
| Custom tools, hooks, LLM backend | `Agent(...)` | Structural, doesn't change per query |
| Query rewrite, BM25, max results, iterations | `agent.search(...)` | May vary per query |
| Token budget, compression, budget bypass | `agent.search(...)` | Per-query cost control |

## Document Parsing — Docling Integration

Docling is a Corpus concern, not an Agent concern. The agent auto-detects whether the corpus has Docling-parsed documents and adjusts available tools accordingly.

```python
# Text-only (no extra deps)
corpus = Corpus("./my-docs")

# Docling-enabled (requires pip install agentic-doc-search[docling])
corpus = Corpus("./my-docs", use_docling=True)
```

### When Docling is enabled

- Binary files (PDF, DOCX, PPTX, images with OCR) are parsed into a section hierarchy
- The agent gets additional tools:
  - `list_sections(path)` — show heading/section hierarchy
  - `read_section(path, section_id)` — read a specific section
  - `search_in_section(path, section_id, pattern)` — search within a section
- Text files still use existing file-based tools unchanged
- If `use_docling=True` but docling isn't installed, raise a clear error pointing to `pip install agentic-doc-search[docling]`

### Document Summaries

```python
corpus = Corpus(
    "./my-docs",
    use_docling=True,
    generate_summaries=True,   # LLM creates abstract per document on ingest
)
```

- Summaries are cached in `.ads_cache/` directory (next to corpus root)
- Agent reads summaries first to decide which docs are worth reading in full
- Saves tokens and speeds up search — like reading abstracts before papers

## Corpus Backends (Connectors)

All connectors implement the `CorpusBackend` protocol:

```python
class CorpusBackend(Protocol):
    async def discover_files(self) -> list[str]: ...
    async def read_file(self, path: str, start_line: int, end_line: int | None) -> str: ...
    async def search(self, pattern: str, max_results: int, filters: dict | None) -> str: ...
    async def file_info(self, path: str) -> str: ...
    async def tree(self, max_depth: int) -> str: ...
    def metadata(self, path: str) -> dict: ...
```

### Available Connectors

**Local:**
- `Corpus` — local filesystem (current, refactored to implement protocol)

**Cloud storage:**
- `S3Corpus` — AWS S3
- `GCSCorpus` — Google Cloud Storage
- `AzureBlobCorpus` — Azure Blob Storage

**Databases:**
- `PostgresCorpus` — PostgreSQL
- `MySQLCorpus` — MySQL
- `SQLiteCorpus` — SQLite
- `MongoCorpus` — MongoDB
- `RedisCorpus` — Redis
- `ElasticsearchCorpus` / `OpenSearchCorpus` — Elasticsearch / OpenSearch

**SaaS / Document platforms:**
- `NotionCorpus` — Notion
- `ConfluenceCorpus` — Confluence
- `GoogleDriveCorpus` — Google Drive
- `SharePointCorpus` — SharePoint / OneDrive
- `AirtableCorpus` — Airtable

**Data platforms:**
- `SnowflakeCorpus` — Snowflake
- `BigQueryCorpus` — BigQuery
- `PineconCorpus` — Pinecone

**File protocols:**
- `SFTPCorpus` — SFTP / FTP
- `HTTPCorpus` — HTTP / WebDAV (crawl a URL)

### Connector Usage

```python
from agentic_doc_search.connectors import S3Corpus, PostgresCorpus

corpus = S3Corpus(
    bucket="my-docs-bucket",
    prefix="knowledge-base/",
    use_docling=True,
)

corpus = PostgresCorpus(
    connection_string="postgresql://...",
    table="documents",
    content_column="body",
    metadata_columns=["author", "created_at", "category"],
)
```

### MultiCorpus — Combined Search

```python
from agentic_doc_search import MultiCorpus

corpus = MultiCorpus([
    Corpus("./local-docs", use_docling=True),
    S3Corpus(bucket="my-bucket", prefix="docs/"),
    PostgresCorpus(connection_string="postgresql://...", table="articles"),
])

agent = Agent(model="claude-sonnet-4-20250514")
result = await agent.search(corpus, "What is the refund policy?")
```

- `discover_files` returns results from all backends, prefixed by source (e.g. `local:docs/policy.md`, `s3:knowledge-base/faq.pdf`, `postgres:articles/42`)
- Search fans out to all backends concurrently, merges and deduplicates
- BM25 reranking happens on the merged result set
- Metadata includes a `source` field so the agent can filter by backend

## Metadata & Filtering

Tools (`grep_corpus`, `list_files`) accept optional filters:

```python
grep_corpus(
    pattern="refund",
    filters={
        "extension": [".md", ".pdf"],
        "min_size": 100,
        "modified_after": "2024-01-01",
        "text_contains": "policy",
    }
)
```

Metadata sources:
- **Built-in (always):** extension, size, modified date, line count
- **Docling (when enabled):** section count, has tables, document type, page count
- **Custom:** user-provided via `metadata_provider` callable on Corpus

```python
corpus = Corpus(
    "./my-docs",
    metadata_provider=my_metadata_fn,  # (file_path) -> dict
)
```

## Extensibility

### Custom Tools

```python
from agentic_doc_search import Agent, ToolDefinition

db_tool = ToolDefinition(
    name="search_database",
    description="Search an external database for related records",
    parameters={...},
    execute=search_database,
)

agent = Agent(extra_tools=[db_tool])
```

### Hooks

```python
agent = Agent(
    on_tool_call=my_logger,           # After each tool call
    on_query_rewrite=my_rewrite_cb,   # After query rewrite
    on_iteration=my_step_cb,          # After each ReAct iteration
)
```

### Pluggable LLM Backend

```python
from agentic_doc_search import LLMBackend

class MyBackend(LLMBackend):
    async def chat_completion(self, messages, tools=None):
        return response_message, token_usage

agent = Agent(llm_backend=MyBackend())
```

`LLMConfig` is the easy path (litellm). `LLMBackend` is the escape hatch. `llm_backend` takes precedence over `model`/`llm_config`.

## Performance

### Parallel Tool Execution

When the LLM returns multiple tool calls in one response, execute them concurrently via `asyncio.gather` instead of sequentially.

### Model Fallback

Start with a cheaper/faster model for initial corpus exploration, escalate to a more capable model when the agent needs to synthesize or reason deeply. Configurable:

```python
agent = Agent(
    model="claude-sonnet-4-20250514",           # Primary
    fallback_model="claude-haiku-4-5-20251001",  # Cheaper model for exploration
)
```

### Context Compression

When `context_compression=True`, older tool results in the conversation history are summarized to free context window space. Recent results stay verbatim.

## Token Cost Management

### Tracking

Every search returns detailed token usage:

```python
result.token_usage.prompt_tokens
result.token_usage.completion_tokens
result.token_usage.total_tokens
result.token_usage.estimated_cost_usd
result.token_usage.per_iteration
```

### Budget

```python
result = await agent.search(
    corpus, "query",
    token_budget=100_000,     # Hard cap
    bypass_budget=False,      # Set True to ignore budget
)
```

When budget is reached, agent stops and gives best answer from current findings.

## Testing Strategy

| Layer | What | How |
|-------|------|-----|
| Unit | Corpus, tools, BM25, query rewrite, metadata filtering, source parsing | Real file fixtures, no LLM calls |
| Connector | Each backend (S3, Postgres, Mongo, etc.) | Mocked clients (moto for S3, testcontainers or SQLite for DB) |
| Agent loop | ReAct iteration, tool dispatch, max iterations, reasoning trace | Mocked LLM returning scripted tool calls |
| Integration | Full end-to-end search with real LLM | Marked slow, skipped in CI by default |
| MultiCorpus | Fan-out, merge, dedup, source prefixing | Mocked backends returning known results |
| Docling | Section hierarchy navigation, binary parsing | Fixture PDFs/DOCX with known structure |

## Packaging

```toml
[project.optional-dependencies]
docling = ["docling>=2.0"]
s3 = ["boto3>=1.28"]
gcs = ["google-cloud-storage>=2.14"]
azure = ["azure-storage-blob>=12.19"]
postgres = ["asyncpg>=0.29"]
mysql = ["aiomysql>=0.2"]
mongo = ["motor>=3.3"]
redis = ["redis[hiredis]>=5.0"]
elasticsearch = ["elasticsearch[async]>=8.12"]
notion = ["notion-client>=2.2"]
confluence = ["atlassian-python-api>=3.41"]
gdrive = ["google-api-python-client>=2.118", "google-auth>=2.28"]
sharepoint = ["msal>=1.26", "office365-rest-python-client>=2.5"]
airtable = ["pyairtable>=2.3"]
snowflake = ["snowflake-connector-python>=3.7"]
bigquery = ["google-cloud-bigquery>=3.17"]
pinecone = ["pinecone-client>=3.1"]
sftp = ["asyncssh>=2.14"]
all = ["agentic-doc-search[docling,s3,gcs,azure,postgres,mysql,mongo,redis,elasticsearch,notion,confluence,gdrive,sharepoint,airtable,snowflake,bigquery,pinecone,sftp]"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4", "mypy>=1.10", "moto>=5.0"]
```

Core dependencies (always installed):
- `litellm` — LLM abstraction
- `click` — CLI
- `rich` — CLI formatting
- `rank-bm25` — BM25 reranking

## Project Structure

```
src/agentic_doc_search/
├── __init__.py              # Public API exports
├── agent.py                 # Async ReAct loop
├── corpus.py                # Local CorpusBackend + Corpus
├── backend.py               # CorpusBackend protocol
├── multi_corpus.py          # MultiCorpus fan-out
├── tools.py                 # Built-in + custom tool registration
├── llm.py                   # LLMConfig + default litellm backend
├── llm_backend.py           # LLMBackend protocol
├── models.py                # SearchResult, ToolCall, SourceReference, TokenUsage
├── bm25.py                  # BM25 reranking
├── rewriter.py              # Query rewrite logic
├── parsing.py               # Source reference parsing from agent output
├── compression.py           # Context compression
├── cost.py                  # Token cost estimation by model
├── cli.py                   # CLI (stays sync, wraps async with asyncio.run)
└── connectors/
    ├── __init__.py
    ├── s3.py
    ├── gcs.py
    ├── azure_blob.py
    ├── postgres.py
    ├── mysql.py
    ├── sqlite.py
    ├── mongo.py
    ├── redis.py
    ├── elasticsearch.py
    ├── notion.py
    ├── confluence.py
    ├── gdrive.py
    ├── sharepoint.py
    ├── airtable.py
    ├── snowflake.py
    ├── bigquery.py
    ├── pinecone.py
    ├── sftp.py
    └── http.py
```
