# OneMem

OneMem V1 is a small, local operational memory layer for AI agents.

It keeps the canonical memory in readable Markdown files and uses a rebuildable SQLite sidecar for full-text search, lightweight vector search, and graph edges.

## Local Start

```bash
python3 --version
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

cp .env.example .env
# edit .env if you want to use OpenAI chat or Gemini embeddings

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
- Chat memory write policy: the chatbot stores explicit memory requests, preferences, corrections, and substantial durable statements while ignoring obvious low-information turns.
- Embedding providers: hash embeddings remain the offline default; Gemini embeddings can be enabled with environment variables.

## Embedding Providers

Default offline mode:

```bash
ONEMEM_EMBEDDING_PROVIDER=hash onemem rebuild-index
```

Gemini embedding mode:

```bash
export ONEMEM_EMBEDDING_PROVIDER=gemini
export GEMINI_API_KEY="your_key"
export ONEMEM_GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
export ONEMEM_EMBEDDING_DIMENSIONS=768

onemem rebuild-index
```

Before changing Gemini model IDs, check the current official Google Gemini API or Vertex AI embedding docs. V2 keeps `hash` as the default so local tests and rebuilds do not require network access.

### Verifying Gemini Embeddings

A small opt-in smoke test confirms the provider returns the configured
dimension, unit-normed vectors, and non-degenerate cosine similarities:

```bash
export GEMINI_API_KEY=...
python3 scripts/gemini_smoke_test.py
```

Switching the default sidecar from `hash` to `gemini` measurably improves
retrieval quality on paraphrased queries but adds a per-text HTTP round-trip
on capture and rebuild. The embedding cache (see the `embedding_cache` table
in the sidecar) amortizes repeated texts; budget ~100–300 ms per unique text
for a warm cache miss against `generativelanguage.googleapis.com`.

## Environment File

The CLI automatically loads `.env` from the current working directory. Existing shell environment variables win over values in `.env`.

```bash
cp .env.example .env
```

Example:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.4-mini
ONEMEM_EMBEDDING_PROVIDER=hash
```

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

## Temporal Retrieval

OneMem extracts event dates from captured observations and reranks retrieval
for temporal queries (`first`, `last`, `most recent`, `before`, `after`,
`how many days`, `when did`). Episodes can be captured with an explicit event
date; retrieval takes an optional reference date for relative questions.

```python
memory.capture(
    "I bought a new car on February 10th, 2023.",
    source="manual",
    session="car_timeline",
    event_date="2023-02-10",
)
memory.retrieve("What was my first car event?", reference_date="2023-06-01")
```

From the CLI:

```bash
onemem retrieve "What was the most recent event with my car?" \
    --reference-date "2024-01-01" --debug
```

The LongMemEval importer parses each `haystack_dates` entry and the
`question_date`, and emits `episode["event_date"]` and `case["reference_date"]`
so the runner can drive temporal ranking. See `evals/temporal_basic.json` for a
small deterministic regression and `src/onemem/temporal.py` for the date and
intent helpers. V2 intentionally improves **retrieval of the right evidence**,
not final-answer reasoning; a reasoning reader is scheduled for V3.

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

### Evaluation Metrics

`EvalReport.summary()` (and the formatted CLI output) now separates answer-text
and evidence-session scoring:

- `answer_recall` — fraction of queries where every `must_contain` fragment
  appears in the top-k context.
- `answer_mrr` — mean reciprocal rank of the first matching answer.
- `evidence_recall_at_5` / `evidence_recall_at_10` — fraction of queries with
  `expected_source_refs` whose first evidence hit lands in the top-5 / top-10.
- `evidence_mrr` — MRR restricted to evidence queries.
- `context_pollution` — for evidence queries, `1 - (hits_in_top_k / top_k)`,
  averaged across queries. Lower is better.
- `mrr` — unchanged, kept for backwards-compatible dashboards.

### Temporal Eval

A small deterministic temporal regression:

```bash
onemem eval run evals/temporal_basic.json
```

### Write-Filter Eval

Score `MemoryWritePolicy` against a labeled message set:

```bash
onemem eval write-filter evals/write_filter_basic.json
```

Exits 0 only when every message is classified correctly; otherwise prints each
mistake so the dataset can grow over time.

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
