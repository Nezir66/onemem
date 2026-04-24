from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embedding_providers import provider_from_env
from .env import load_default_env
from .eval import EvalRunner, LongMemEvalImporter, format_report, run_write_filter_eval
from .maintenance import MaintenanceWorker
from .runtime import MemoryRuntime
from .server import serve as serve_inspector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onemem", description="Operational file-backed memory V1")
    parser.add_argument("--root", default="memory", help="memory root directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create memory directories and sidecar database")

    capture = sub.add_parser("capture", help="append an observation to the episodic buffer")
    capture.add_argument("observation")
    capture.add_argument("--source", default="manual")
    capture.add_argument("--session", default="default")
    capture.add_argument("--salience", type=float, default=0.5)
    capture.add_argument("--event-date", default=None)

    sub.add_parser("flush", help="consolidate episodes into facts, concepts, and summary")
    sub.add_parser("rebuild-index", help="rebuild sidecar indexes from canonical files")

    retrieve = sub.add_parser("retrieve", help="retrieve memory context for a query")
    retrieve.add_argument("query")
    retrieve.add_argument("--limit", type=int, default=8)
    retrieve.add_argument("--json", action="store_true")
    retrieve.add_argument("--debug", action="store_true", help="include ranking component details")
    retrieve.add_argument(
        "--reference-date",
        default=None,
        help="reference timestamp for temporal queries (ISO or LongMemEval-style)",
    )

    answer = sub.add_parser("answer", help="answer a question from retrieved evidence and cite memory IDs")
    answer.add_argument("query")
    answer.add_argument("--limit", type=int, default=8)
    answer.add_argument("--json", action="store_true")
    answer.add_argument("--reference-date", default=None)

    inspect = sub.add_parser("inspect", help="inspect a canonical memory node")
    inspect.add_argument("node_id")
    inspect.add_argument("--json", action="store_true")

    list_nodes = sub.add_parser("list", help="list canonical memory nodes")
    list_nodes.add_argument("--kind", choices=["episode", "fact", "concept", "summary"], default=None)
    list_nodes.add_argument("--archive", action="store_true")
    list_nodes.add_argument("--limit", type=int, default=50)
    list_nodes.add_argument("--json", action="store_true")

    graph = sub.add_parser("graph", help="inspect graph relations")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)
    neighbors = graph_sub.add_parser("neighbors", help="list graph neighbors for a node")
    neighbors.add_argument("node_id")
    neighbors.add_argument("--limit", type=int, default=20)
    neighbors.add_argument("--json", action="store_true")

    merge = sub.add_parser("merge", help="review or approve duplicate fact merges")
    merge_sub = merge.add_subparsers(dest="merge_command", required=True)
    merge_candidates = merge_sub.add_parser("candidates", help="list likely duplicate facts")
    merge_candidates.add_argument("--limit", type=int, default=25)
    merge_candidates.add_argument("--json", action="store_true")
    merge_approve = merge_sub.add_parser("approve", help="apply a merge candidate")
    merge_approve.add_argument("candidate_id")

    feedback = sub.add_parser("feedback", help="record explicit memory feedback")
    feedback.add_argument("node_id")
    feedback.add_argument("signal", choices=["used", "confirmed", "corrected", "wrong", "failed", "pin", "unpin"])

    summaries = sub.add_parser("summaries", help="manage derived summary layers")
    summaries_sub = summaries.add_subparsers(dest="summaries_command", required=True)
    summaries_sub.add_parser("refresh", help="refresh layered summaries from canonical nodes")

    maintain = sub.add_parser("maintain", help="run basic decay/archive maintenance")
    maintain.add_argument("--episode-ttl-days", type=int, default=30)
    maintain.add_argument("--hypothesis-ttl-days", type=int, default=14)

    invalidate = sub.add_parser("invalidate", help="deprecate a canonical memory node")
    invalidate.add_argument("node_id")
    invalidate.add_argument("--reason", default="manual invalidation")

    serve = sub.add_parser("serve", help="start the local read-only memory inspector UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7070)

    eval_parser = sub.add_parser("eval", help="run memory quality evals")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run", help="run a JSON eval suite")
    eval_run.add_argument("path")
    eval_run.add_argument("--json", action="store_true")
    eval_run.add_argument("--include-context", action="store_true")
    eval_run.add_argument("--keep-tmp", action="store_true")
    eval_import_lme = eval_sub.add_parser("import-longmemeval", help="convert LongMemEval JSON to OneMem eval JSON")
    eval_import_lme.add_argument("path")
    eval_import_lme.add_argument("--out", required=True)
    eval_import_lme.add_argument("--limit", type=int, default=None)
    eval_import_lme.add_argument("--top-k", type=int, default=10)
    eval_wf = eval_sub.add_parser("write-filter", help="score MemoryWritePolicy against a labeled dataset")
    eval_wf.add_argument("path")
    eval_wf.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_default_env()
    args = build_parser().parse_args(argv)
    runtime = MemoryRuntime(Path(args.root), embedding_provider=provider_from_env())

    if args.command == "init":
        runtime.init()
        print(f"initialized memory at {Path(args.root).resolve()}")
        return 0

    if args.command == "capture":
        runtime.init()
        node = runtime.capture(
            args.observation,
            source=args.source,
            session=args.session,
            salience=args.salience,
            event_date=args.event_date,
        )
        print(node.id)
        return 0

    if args.command == "flush":
        runtime.init()
        print(json.dumps(runtime.flush(), indent=2, sort_keys=True))
        return 0

    if args.command == "rebuild-index":
        runtime.init()
        runtime.rebuild_index()
        print("rebuilt sidecar index")
        return 0

    if args.command == "retrieve":
        runtime.init()
        result = runtime.retrieve(args.query, limit=args.limit, reference_date=args.reference_date)
        if args.json:
            print(json.dumps([memory.__dict__ for memory in result.memories], indent=2, sort_keys=True))
        else:
            print(result.context(include_debug=args.debug))
        return 0

    if args.command == "answer":
        runtime.init()
        answer_result = runtime.answer(args.query, limit=args.limit, reference_date=args.reference_date)
        if args.json:
            print(json.dumps(answer_result.to_dict(), indent=2, sort_keys=True))
        else:
            print(answer_result.answer)
            if answer_result.memory_ids:
                print(f"memory_ids: {', '.join(answer_result.memory_ids)}")
            if answer_result.source_refs:
                print(f"source_refs: {', '.join(answer_result.source_refs)}")
            print(f"confidence: {answer_result.confidence:.4f}")
        return 0

    if args.command == "inspect":
        runtime.init()
        node = runtime.inspect(args.node_id)
        payload = node.metadata()
        payload["body"] = node.body
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"{node.id} [{node.kind}:{node.status}] {node.title}")
            print(f"confidence={node.confidence:.3f} salience={node.salience:.3f} pinned={node.pinned}")
            print(f"source_refs={', '.join(node.source_refs) or 'none'}")
            print(f"concept_refs={', '.join(node.concept_refs) or 'none'}")
            print()
            print(node.body)
        return 0

    if args.command == "list":
        runtime.init()
        nodes = runtime.list_nodes(kind=args.kind, include_archive=args.archive)[: args.limit]
        if args.json:
            payload = [node.metadata() | {"body": node.body} for node in nodes]
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for node in nodes:
                print(f"{node.id}\t{node.kind}\t{node.status}\t{node.title}")
        return 0

    if args.command == "graph" and args.graph_command == "neighbors":
        runtime.init()
        rows = runtime.graph_neighbors(args.node_id, limit=args.limit)
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for row in rows:
                print(
                    f"{row['id']}\t{row['kind']}\t{row['status']}\t"
                    f"{row['relation']}:{row['direction']}:{row['weight']}\t{row['title']}"
                )
        return 0

    if args.command == "merge" and args.merge_command == "candidates":
        runtime.init()
        candidates = runtime.merge_candidates(limit=args.limit)
        if args.json:
            print(json.dumps([candidate.__dict__ for candidate in candidates], indent=2, sort_keys=True))
        else:
            for candidate in candidates:
                print(
                    f"{candidate.id}\tscore={candidate.score:.4f}\t"
                    f"{candidate.source_id} -> {candidate.target_id}\t{candidate.reason}"
                )
        return 0

    if args.command == "merge" and args.merge_command == "approve":
        runtime.init()
        applied = runtime.approve_merge(args.candidate_id)
        print(applied.message)
        return 0

    if args.command == "feedback":
        runtime.init()
        node = runtime.record_feedback(args.node_id, args.signal)
        print(f"updated {node.id}: status={node.status} confidence={node.confidence:.3f} salience={node.salience:.3f}")
        return 0

    if args.command == "summaries" and args.summaries_command == "refresh":
        runtime.init()
        nodes = runtime.refresh_summaries()
        print(json.dumps({"summaries": len(nodes), "ids": [node.id for node in nodes]}, indent=2, sort_keys=True))
        return 0

    if args.command == "maintain":
        runtime.init()
        outcome = MaintenanceWorker(runtime.store).run(
            episode_ttl_days=args.episode_ttl_days,
            hypothesis_ttl_days=args.hypothesis_ttl_days,
        )
        runtime.rebuild_index()
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return 0

    if args.command == "invalidate":
        runtime.init()
        node = runtime.invalidate(args.node_id, reason=args.reason)
        print(f"deprecated {node.id}")
        return 0

    if args.command == "serve":
        runtime.init()
        serve_inspector(root=Path(args.root), host=args.host, port=args.port)
        return 0

    if args.command == "eval" and args.eval_command == "run":
        report = EvalRunner().run_file(Path(args.path), keep_tmp=args.keep_tmp)
        if args.json:
            print(json.dumps(report.to_dict(include_context=args.include_context), indent=2, sort_keys=True))
        else:
            print(format_report(report))
        return 0 if report.passed else 1

    if args.command == "eval" and args.eval_command == "import-longmemeval":
        result = LongMemEvalImporter().import_file(
            Path(args.path),
            Path(args.out),
            limit=args.limit,
            top_k=args.top_k,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "eval" and args.eval_command == "write-filter":
        report = run_write_filter_eval(Path(args.path))
        if args.json:
            print(json.dumps({"summary": report.summary(), "mistakes": report.mistakes}, indent=2))
        else:
            summary = report.summary()
            print(
                f"write-filter: {summary['correct']}/{summary['total']} "
                f"accuracy={summary['accuracy']:.4f} mistakes={summary['mistakes']}"
            )
            for mistake in report.mistakes:
                print(
                    f"  expected={mistake['expected']} decision={mistake['decision']} "
                    f"reason={mistake['reason']!r} text={mistake['text']!r}"
                )
        return 0 if report.correct == report.total else 1

    return 1
