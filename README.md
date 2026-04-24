# OneMem

OneMem is a small, local operational memory layer for AI agents.

It keeps canonical memory in readable Markdown files and uses a rebuildable SQLite sidecar for full-text search, lightweight vector search, graph edges, and an operations audit log. Canonical files are truth — sidecars are derived views; writes go through a validator.

## Local Start

```bash
python3 --version
python3 -m venv .venv
. .venv/bin/activate
pip install -e .

cp .env.example .env
# edit .env if you want to use Gemini embeddings

onemem init
onemem capture "Nora prefers concise technical summaries for the OneMem project." --source manual --session onemem
onemem flush
onemem retrieve "What does Nora prefer for OneMem?"
onemem answer "What does Nora prefer for OneMem?"
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

Install OneMem into another agent project:

```bash
pip install -e /Users/nezir/Documents/Projects/onemem
```

Then import the runtime:

```python
from onemem import MemoryRuntime, MemoryOperation

memory = MemoryRuntime("memory")
memory.init()

# retrieve context for an LLM prompt
context = memory.retrieve(user_message, limit=8).context()
answer = call_your_llm(user_message, context)

# or get a grounded, citation-bearing answer without an LLM
grounded = memory.answer(user_message)
if not grounded.abstained:
    print(grounded.answer, grounded.memory_ids)

memory.capture(
    f"User asked: {user_message}\nAssistant answered: {answer}",
    source="my-agent",
    session="chat",
)
memory.flush()
```

See `docs/INTEGRATION.md` for a fuller adapter example.

## Features

**Core storage and retrieval**

- Episodic buffer: `capture` writes raw observations as append-style episode files.
- Consolidation: `flush` extracts atomic facts, concept anchors, explicit concept edges, and a current summary.
- Canonical store: Markdown files under `memory/` are the source of truth.
- Sidecar index: SQLite FTS, graph edges, aliases, embeddings, and metadata live under `memory/.sidecar/`.
- Retrieval: hybrid lexical/vector candidates, typed-relation graph expansion with a personalised-PageRank-style boost, temporal/status filtering, and type-budgeted context packing.
- Maintenance: `maintain` applies simple decay/archive rules and rebuilds the sidecar.
- Embedding providers: hash embeddings remain the offline default; Gemini embeddings can be enabled with environment variables.

**Controlled writes**

- Operation validator: `MemoryOperationManager` exposes `ADD` / `UPDATE` / `INVALIDATE` / `LINK` / `MERGE` / `PROMOTE` / `DEMOTE`. Proposals are validated (provenance, valid IDs, confidence/status/ISO ranges) before apply. `UPDATE` cannot silently deprecate — use `INVALIDATE`.
- Invalidation: marks canonical nodes as `deprecated` with `valid_to` instead of deleting or overwriting them.
- Audit log: every applied operation is appended to `memory/.sidecar/operations.jsonl`.
- Write policy: `MemoryWritePolicy` classifies turns as explicit memory requests, preferences, corrections, or durable statements — and ignores obvious low-information turns. Consumers call it from their own agent/chat layer.

**Reasoning and abstention**

- Reader: `runtime.answer()` returns a grounded answer with `memory_ids` + `source_refs`, supporting temporal comparisons, counting, multi-hop chains via graph neighbors, and extractive answers.
- Abstention / OOD: reader abstains when there is no relevant memory, evidence is weak, evidence conflicts on a shared concept, or the question needs reasoning beyond retrieved context.

**Dedupe and merge**

- `onemem merge candidates` lists likely duplicate facts scored on embedding cosine, token overlap, and shared concepts. Candidates are persisted in the sidecar so IDs stay stable.
- `onemem merge approve <id>` applies a `MERGE` through the validator, preserves provenance, writes a `merged_into` relation + alias from deprecated to target.

**Summary layers**

- `summaries refresh` rebuilds session, topic, profile, and project summaries. Summaries cite their source facts / episodes and never become canonical truth.

**Feedback**

- `onemem feedback <node_id> <signal>` — `used` / `confirmed` / `corrected` / `wrong` / `failed` / `pin` / `unpin`. Feedback is routed through the operation manager (auditable).

**Inspector**

- CLI: `onemem inspect <id>`, `onemem list --kind ...`, `onemem graph neighbors <id>`.
- Local read-only UI: `onemem serve` (http://127.0.0.1:7070).

## CLI Reference

```bash
onemem init                                 # create memory/ and .sidecar/
onemem capture "..." --event-date 2023-02-10
onemem flush                                # consolidate episodes + rebuild summaries
onemem retrieve "query" --debug             # hybrid retrieval with ranking components
onemem answer "query" --reference-date 2024-01-01
onemem inspect <node_id> [--json]
onemem list --kind fact [--archive]
onemem graph neighbors <node_id>
onemem merge candidates [--json]
onemem merge approve <candidate_id>
onemem feedback <node_id> confirmed|wrong|pin|...
onemem summaries refresh
onemem invalidate <node_id> --reason "..."
onemem maintain --episode-ttl-days 30 --hypothesis-ttl-days 14
onemem rebuild-index
onemem serve --port 7070
onemem eval run evals/basic_memory.json
onemem eval write-filter evals/write_filter_basic.json
onemem eval import-longmemeval <path> --out evals/lme.json --limit 25
```

## Embedding Providers

`hash` is the default and needs no configuration — works offline, deterministic, fine for tests and local dev.

### As a library

```python
from onemem import MemoryRuntime, GeminiEmbeddingProvider

provider = GeminiEmbeddingProvider(
    api_key=get_secret("GEMINI_API_KEY"),
    model="gemini-embedding-001",
    dimensions=768,
)
memory = MemoryRuntime("memory", embedding_provider=provider)
```

OneMem's core never touches `os.environ`. If you don't pass a provider, you get
a stumm `HashEmbeddingProvider()` regardless of any `ONEMEM_*` env vars.

### From the CLI (env-driven)

The CLI loads `.env` and reads `ONEMEM_EMBEDDING_PROVIDER`:

```bash
ONEMEM_EMBEDDING_PROVIDER=hash onemem rebuild-index

export ONEMEM_EMBEDDING_PROVIDER=gemini
export GEMINI_API_KEY="your_key"
export ONEMEM_GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
export ONEMEM_EMBEDDING_DIMENSIONS=768
onemem rebuild-index
```

### Opt-in env path from your own code

If you explicitly want the env behavior in a script:

```python
from onemem import MemoryRuntime, provider_from_env, load_default_env

load_default_env()
memory = MemoryRuntime("memory", embedding_provider=provider_from_env())
```

Before changing Gemini model IDs, check the current official Google Gemini API or Vertex AI embedding docs.

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
ONEMEM_EMBEDDING_PROVIDER=hash
GEMINI_API_KEY=...
```

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
memory.answer("What was my first car event?", reference_date="2023-06-01")
```

From the CLI:

```bash
onemem retrieve "What was the most recent event with my car?" \
    --reference-date "2024-01-01" --debug
onemem answer "What was my first car event?" --reference-date 2023-06-01
```

The LongMemEval importer parses each `haystack_dates` entry and the
`question_date`, and emits `episode["event_date"]` and `case["reference_date"]`
so the runner can drive temporal ranking. See `evals/temporal_basic.json` for a
small deterministic regression and `src/onemem/temporal.py` for the date and
intent helpers.

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

`EvalReport.summary()` (and the formatted CLI output) separates answer-text
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
  runtime.py          high-level capture/flush/retrieve/answer API
  markdown_store.py   canonical Markdown read/write
  models.py           MemoryNode / MemoryOperation / Relation
  consolidator.py     deterministic episode → fact consolidation
  index.py            rebuildable SQLite sidecar (FTS, vectors, edges, aliases, merge_candidates)
  retrieval.py        hybrid retrieval with typed-relation + PPR boost
  reader.py           grounded answers, abstention, multi-hop, OOD
  operations.py       validated memory operations + audit log
  dedupe.py           embedding + lexical + concept dedupe with persistent candidates
  summaries.py        session / topic / profile / project summary layers
  write_policy.py     classify chat turns worth capturing
  maintenance.py      decay/archive worker
  temporal.py         event date + temporal intent helpers
  server.py           local read-only inspector UI
  embeddings.py       hash embeddings + cosine helpers
  embedding_providers.py  pluggable embedding providers (hash, gemini)
  eval.py             golden eval runner + LongMemEval importer
tests/
  test_memory_flow.py end-to-end storage / retrieval / write-policy checks
  test_v3.py          operation validator, reader, dedupe, feedback, audit
  test_eval.py        eval runner checks
```
