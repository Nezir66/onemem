# OneMem

OneMem V1 is a small, local operational memory layer for AI agents.

It keeps the canonical memory in readable Markdown files and uses a rebuildable SQLite sidecar for full-text search, lightweight vector search, and graph edges.

## Local Start

```bash
python3 --version
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

onemem init
onemem capture "Nora prefers concise technical summaries for the OneMem project." --source manual --session onemem
onemem flush
onemem retrieve "What does Nora prefer for OneMem?"
onemem invalidate fact_123 --reason "superseded by newer evidence"
```

Without installing the package, use:

```bash
python3 --version  # must be Python 3.12+
PYTHONPATH=src python3 -m onemem init
PYTHONPATH=src python3 -m onemem capture "OneMem stores canonical memory in Markdown files."
PYTHONPATH=src python3 -m onemem flush
PYTHONPATH=src python3 -m onemem retrieve "canonical memory"
```

## Use As A Package

Install OneMem into another agent or chatbot project:

```bash
pip install -e /Users/nezir/Documents/Projects/onemem
```

Then import the stable V1 runtime:

```python
from onemem import MemoryRuntime

memory = MemoryRuntime("memory")
memory.init()

context = memory.retrieve(user_message, limit=8).context()
answer = call_your_llm(user_message, context)

memory.capture(
    f"User asked: {user_message}\nAssistant answered: {answer}",
    source="my-agent",
    session="chat",
)
memory.flush()
```

See `docs/INTEGRATION.md` for a fuller adapter example.

## What V1 Implements

- Episodic buffer: `capture` writes raw observations as append-style episode files.
- Consolidation: `flush` extracts atomic facts, concept anchors, explicit concept edges, and a current summary.
- Canonical store: Markdown files under `memory/` are the source of truth.
- Sidecar index: SQLite FTS, graph edges, aliases, metadata, and deterministic local vectors live under `memory/.sidecar/`.
- Retrieval: hybrid lexical/vector candidates, mild graph expansion, temporal/status filtering, and type-budgeted context packing.
- Maintenance: `maintain` applies simple decay/archive rules and rebuilds the sidecar.
- Invalidation: `invalidate` marks canonical nodes as `deprecated` with `valid_to` instead of deleting or overwriting them.

## Chatbot Test

The chatbot uses the OpenAI Responses API. It retrieves relevant OneMem context before each answer and captures the turn afterwards.

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-5.2"

PYTHONPATH=src python3 -m onemem --root memory chat
```

Inside the chat:

```text
> Remember that I prefer short technical answers.
> What do you know about my answer style?
> /exit
```

Each turn is stored as an episode. By default the chatbot also runs `flush` after each answer, so new information becomes retrievable in the next turns.

## Eval

Run the built-in golden eval suite:

```bash
onemem eval run evals/basic_memory.json
```

The eval creates temporary memory stores, captures episodes, runs `flush`, tests retrieval expectations, checks sidecar rebuilds, and verifies explicit invalidation where configured.

JSON output:

```bash
onemem eval run evals/basic_memory.json --json
```

Convert a LongMemEval Oracle file into OneMem's eval format:

```bash
onemem eval import-longmemeval benchmarks/longmemeval/longmemeval_oracle.json \
  --out evals/longmemeval_oracle_sample.json \
  --limit 25

onemem eval run evals/longmemeval_oracle_sample.json
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Repository Layout

```text
src/onemem/
  __init__.py         public package exports
  cli.py              command-line entry points
  runtime.py          high-level capture/flush/retrieve API
  chatbot.py          optional chatbot adapter
  eval.py             golden eval runner
  markdown_store.py   canonical Markdown read/write
  consolidator.py     deterministic V1 consolidation
  index.py            rebuildable SQLite sidecar
  retrieval.py        retrieval orchestration and context packing
  maintenance.py      decay/archive worker
tests/
  test_memory_flow.py end-to-end V1 checks
  test_eval.py        eval runner checks
```
