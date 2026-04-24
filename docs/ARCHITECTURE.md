# Architecture Notes

## Source of Truth

Markdown files in `memory/` are canonical. The SQLite database under `memory/.sidecar/` is derived and can be deleted and rebuilt from files.

## V1 Data Flow

```text
capture -> episode Markdown -> flush -> facts/concepts/summary Markdown -> rebuild sidecar -> retrieve
                                                         -> invalidate -> deprecated node with valid_to
chat -> retrieve context -> LLM response -> capture turn -> optional flush
eval -> temp memory -> capture/flush/retrieve -> rebuild/invalidate checks -> score report
```

Retrieval combines FTS, deterministic local vectors, graph expansion through explicit edges, and a small additive reranker.

## Deliberate V1 Decisions

- The consolidator is deterministic and local. `GOAL.md` recommends LLM-assisted consolidation, but V1 keeps the first runnable flow offline, inspectable, and testable. The consolidator is the one place to replace with an LLM-backed implementation later.
- Vector search uses a deterministic hashing vector stored in SQLite instead of `sqlite-vec`. This preserves the V1 role of vectors as a rebuildable candidate generator without requiring native extensions.
- Graph memory is an edge table, not a graph database. This matches the V1 constraint that graph is model structure, not early infrastructure.
- Facts are not overwritten silently. Repeated flushes upsert deterministic fact IDs and union evidence through source references.
- Invalidation is explicit. Deprecated nodes remain in Markdown for audit, but retrieval filters them out.

## Boundaries

- `onemem.__init__` exposes the stable V1 library API.
- `MarkdownStore` owns canonical file IO.
- `SidecarIndex` owns derived SQLite state.
- `SimpleConsolidator` turns episodes into long-term nodes.
- `RetrievalOrchestrator` reads only the sidecar and builds working context.
- `MaintenanceWorker` applies decay/archive policy and never defines truth independently of files.
- `onemem.adapters` is reserved for thin framework integrations. Core modules should not import agent frameworks or LLM SDKs.
- `EvalRunner` creates isolated temporary memories and measures the lifecycle behavior expected from V1.
- `LongMemEvalImporter` converts benchmark evidence sessions into OneMem eval cases; it does not alter core memory behavior.
