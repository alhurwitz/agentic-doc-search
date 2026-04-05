# agentic-doc-search

Vectorless agentic document search — an LLM agent that navigates and reasons through document corpora without embeddings or indexes.

## What is this?

Most RAG tools require you to embed your documents into a vector database before you can search them. `agentic-doc-search` takes a different approach: it gives an LLM agent a set of file-exploration tools and lets it navigate your documents like a human researcher would — scanning directory structures, grepping for keywords, reading relevant files, and following cross-references.

**No vectors. No embeddings. No indexing step. Just point it at a folder and ask a question.**

## Install

```bash
pip install agentic-doc-search
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add agentic-doc-search
```

## Quick start

### Python API

```python
from agentic_doc_search import Agent, Corpus

corpus = Corpus("./my-docs")
agent = Agent(model="claude-sonnet-4-20250514")

result = agent.search(corpus, "What is the refund policy?")
print(result.answer)
print(result.sources)
```

### CLI

```bash
# Search a document corpus
ads search ./my-docs "What is the refund policy?"

# Use a different model
ads search ./my-docs "Summarize the architecture" -m gpt-4o

# Show the corpus structure
ads tree ./my-docs

# List all document files
ads files ./my-docs
```

## How it works

The agent runs a ReAct (Reason + Act) loop:

1. **Observe** the corpus structure (`corpus_tree`, `list_files`)
2. **Search** broadly for relevant keywords (`grep_corpus`)
3. **Read** promising files in detail (`read_file`)
4. **Refine** by searching within specific files (`search_in_file`)
5. **Answer** with citations when it has enough information

The agent decides at each step what to do next — it might grep first, or browse the directory tree, or go straight to a file if the name is informative. This dynamic navigation is what makes it "agentic" rather than a fixed pipeline.

## Supported models

Any model that [litellm supports](https://docs.litellm.ai/docs/providers), including:

- Anthropic Claude (`claude-sonnet-4-20250514`, `claude-opus-4-20250514`)
- OpenAI (`gpt-4o`, `gpt-4o-mini`)
- Local models via Ollama (`ollama/llama3`)
- Azure, AWS Bedrock, Google Vertex, and more

## Supported file formats

Currently text-based formats: Markdown, plain text, HTML, reStructuredText, CSV, JSON, YAML, TOML, XML, and common programming languages.

PDF, DOCX, and other binary formats are planned.

## License

MIT
