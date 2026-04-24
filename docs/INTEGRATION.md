# Integration Guide

OneMem is packaged as a Python library first and a CLI second.

## Install Locally

From another Python project:

```bash
pip install -e /Users/YOUR_USER/Documents/Projects/onemem
```

From a Git repository later:

```bash
pip install git+https://github.com/YOUR_USER/onemem.git
```

## Configuration

OneMem's core never reads `os.environ`. Library consumers pass config explicitly:

```python
from onemem import MemoryRuntime, HashEmbeddingProvider, GeminiEmbeddingProvider

# Default — offline hash embeddings, zero config
memory = MemoryRuntime("memory")

# Explicit Gemini provider (recommended for app/library code)
provider = GeminiEmbeddingProvider(
    api_key=get_secret("GEMINI_API_KEY"),
    model="gemini-embedding-001",
    dimensions=768,
)
memory = MemoryRuntime("memory", embedding_provider=provider)
```

If you want to honor env variables (e.g. in a CLI or dev script), use the
explicit factory — no magic in library code:

```python
from onemem import MemoryRuntime, provider_from_env, load_default_env

load_default_env()   # optional: load .env from CWD
memory = MemoryRuntime("memory", embedding_provider=provider_from_env())
```

Env variables read by `provider_from_env()`:

| Variable | Purpose | Default |
|---|---|---|
| `ONEMEM_EMBEDDING_PROVIDER` | `hash` or `gemini` | `hash` |
| `ONEMEM_EMBEDDING_DIMENSIONS` | vector length | 64 (hash) / 768 (gemini) |
| `ONEMEM_GEMINI_EMBEDDING_MODEL` | Gemini model ID | `gemini-embedding-001` |
| `ONEMEM_GEMINI_BASE_URL` | Gemini endpoint | Google default |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | API key | — |

The `onemem` CLI calls `load_default_env()` + `provider_from_env()` for you, so
`.env` works out-of-the-box on the command line.

## Core Concepts

- **Canonical files are truth.** `memory/*.md` holds every node. The SQLite sidecar under `memory/.sidecar/` is rebuildable from the files.
- **Writes are controlled.** All mutations (except raw `capture` and `flush`) go through `MemoryOperationManager`, which validates provenance, IDs, ranges, and status transitions before applying. Every applied operation is appended to `memory/.sidecar/operations.jsonl`.
- **Retrieval is hybrid.** Lexical (FTS) + vector (embeddings) + typed-relation graph expansion + temporal reranking.
- **Answers are grounded.** `runtime.answer()` returns either an extractive/temporal/counting/multi-hop answer with citations, or an abstention with a reason.

## Pattern 1 — Retrieval-augmented agent (your LLM answers)

Use when you want the LLM to be the final answerer and just need relevant memory in the prompt.

```python
from onemem import MemoryRuntime, MemoryWritePolicy

memory = MemoryRuntime("memory")
memory.init()
policy = MemoryWritePolicy()

def ask(user_message: str) -> str:
    context = memory.retrieve(user_message, limit=8).context()
    prompt = f"Memory context:\n{context}\n\nUser: {user_message}"
    answer = call_your_llm(prompt)

    decision = policy.evaluate(user_message, answer)
    if decision.capture:
        memory.capture(
            f"User asked: {user_message}\nAssistant answered: {answer}",
            source="my-agent",
            session="chat",
            salience=decision.salience,
        )
        memory.flush()
    return answer
```

## Pattern 2 — Grounded reader first, LLM as fallback

Use when you want zero hallucination on factual/temporal questions.

```python
from onemem import MemoryRuntime

memory = MemoryRuntime("memory")
memory.init()

def ask(user_message: str) -> str:
    grounded = memory.answer(user_message)
    if not grounded.abstained and grounded.confidence >= 0.6:
        return f"{grounded.answer}\n(memory: {', '.join(grounded.memory_ids)})"

    # fall back to LLM with retrieval context
    context = memory.retrieve(user_message, limit=8).context()
    return call_your_llm(f"Memory:\n{context}\n\nUser: {user_message}")
```

## Pattern 3 — LLM-proposed operations, deterministic apply

Use when the LLM should propose consolidations (merge duplicates, promote stable facts, invalidate contradicted ones). The LLM never writes canonical memory directly.

```python
from onemem import MemoryRuntime, MemoryOperation, MemoryNode

memory = MemoryRuntime("memory")
memory.init()

proposals = call_your_llm_to_propose_ops(recent_turns, memory.retrieve(...).context())
# e.g. [{"op":"ADD","node":{...},"source_refs":[...]}, {"op":"INVALIDATE","node_id":"fact_123","reason":"user corrected"}]

operations = [MemoryOperation(**proposal) for proposal in proposals]
applied = memory.apply_operations(operations)  # validator + audit log
for result in applied:
    print(result.message, result.changed_ids)
```

Validation rejects invalid confidence, missing provenance for facts, silent deprecation via UPDATE, invalid ISO dates, or non-existent targets. Each apply appends a line to `.sidecar/operations.jsonl`.

## Feedback Loop

When the user confirms or corrects memory, route it through the runtime (auditable):

```python
memory.record_feedback("fact_123", "confirmed")  # or "wrong", "pin", ...
```

`confirmed` / `used` bumps salience + confidence; `wrong` / `corrected` lowers both and may demote `candidate` to `hypothesis`; `pin` / `unpin` toggle the pinned flag.

## Dedupe Hygiene

Run periodically (e.g. after N captures, or nightly):

```python
candidates = memory.merge_candidates(limit=25)
for candidate in candidates:
    if candidate.score >= 0.95:
        memory.approve_merge(candidate.id)          # auto-apply near-duplicates
    else:
        present_to_user_for_approval(candidate)     # surface in your UI
```

Candidates are persisted in the sidecar, so IDs stay stable between list and approve.

## Inspector

For local debugging, OneMem ships a read-only web UI:

```bash
onemem --root memory serve --port 7070
```

Or inspect programmatically:

```python
memory.inspect("fact_123")
memory.list_nodes(kind="fact")
memory.graph_neighbors("fact_123")
```

## Full API Surface

```python
from onemem import MemoryRuntime, MemoryOperation, MemoryNode

memory = MemoryRuntime("memory")
memory.init()

# capture + consolidate
memory.capture("Observation text", source="agent", session="chat", event_date="2024-03-01")
memory.flush()

# retrieve
result = memory.retrieve("question", limit=8, reference_date="2024-06-01")
context = result.context(include_debug=False)

# grounded answer
answer = memory.answer("question", reference_date="2024-06-01")
# answer.abstained, answer.memory_ids, answer.source_refs, answer.confidence, answer.reason

# controlled writes
memory.apply_operations([MemoryOperation(op="INVALIDATE", node_id="fact_id", reason="superseded")])
memory.invalidate("fact_id", reason="superseded")          # shorthand

# dedupe + merge
memory.merge_candidates(limit=25)
memory.approve_merge("merge_abc123")

# feedback
memory.record_feedback("fact_id", "confirmed")

# summaries
memory.refresh_summaries()

# inspection
memory.inspect("node_id")
memory.list_nodes(kind="fact")
memory.graph_neighbors("node_id")

# index maintenance
memory.rebuild_index()
```

## Adapter Rule

Keep framework-specific code outside the core.

Good:

```text
onemem.adapters.langchain
onemem.adapters.fastapi
your_project/onemem_adapter.py
```

Avoid:

```text
runtime.py importing LangChain, FastAPI, CrewAI, or a specific LLM SDK
```

This keeps storage, retrieval, consolidation, and maintenance reusable across agents and chatbots.

## Optional Extras

The package declares optional dependency groups:

```bash
pip install "onemem[google]"
pip install "onemem[server]"
pip install "onemem[dev]"
```

The current core does not require these extras. They are reserved for clean adapters and packaging workflows.
