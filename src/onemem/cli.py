from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chatbot import MemoryChatbot, OpenAIResponsesClient
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

    maintain = sub.add_parser("maintain", help="run basic decay/archive maintenance")
    maintain.add_argument("--episode-ttl-days", type=int, default=30)
    maintain.add_argument("--hypothesis-ttl-days", type=int, default=14)

    invalidate = sub.add_parser("invalidate", help="deprecate a canonical memory node")
    invalidate.add_argument("node_id")
    invalidate.add_argument("--reason", default="manual invalidation")

    chat = sub.add_parser("chat", help="start a memory-connected chatbot")
    chat.add_argument("--model", default=None, help="OpenAI model; defaults to OPENAI_MODEL or gpt-5.2")
    chat.add_argument("--memory-limit", type=int, default=8)
    chat.add_argument("--no-auto-flush", action="store_true", help="capture turns without consolidating after each answer")

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
    runtime = MemoryRuntime(Path(args.root))

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

    if args.command == "chat":
        runtime.init()
        client = OpenAIResponsesClient.from_env(model=args.model)
        bot = MemoryChatbot(
            runtime,
            client,
            memory_limit=args.memory_limit,
            auto_flush=not args.no_auto_flush,
        )
        print("OneMem chat started. Type /exit to stop.")
        while True:
            try:
                user_message = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if user_message in {"/exit", "/quit"}:
                return 0
            if not user_message:
                continue
            print(bot.ask(user_message))

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
