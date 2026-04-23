from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtime import MemoryRuntime


@dataclass(slots=True)
class QueryEvalResult:
    query: str
    passed: bool
    rank: int | None
    top_k: int
    must_contain: list[str]
    missing: list[str]
    context: str
    score_type: str = "answer"
    expected_source_refs: list[str] = field(default_factory=list)
    evidence_rank: int | None = None
    evidence_passed: bool | None = None

    def scoring_rank(self) -> int | None:
        if self.score_type == "evidence":
            return self.evidence_rank
        return self.rank


@dataclass(slots=True)
class CaseEvalResult:
    name: str
    passed: bool
    queries: list[QueryEvalResult]
    rebuild_passed: bool | None = None
    invalidation_passed: bool | None = None


@dataclass(slots=True)
class EvalReport:
    path: str
    passed: bool
    cases: list[CaseEvalResult]

    def summary(self) -> dict[str, Any]:
        total_queries = sum(len(case.queries) for case in self.cases)
        passed_queries = sum(1 for case in self.cases for query in case.queries if query.passed)
        ranks = [
            query.scoring_rank()
            for case in self.cases
            for query in case.queries
            if query.scoring_rank() is not None
        ]
        mrr = sum(1 / rank for rank in ranks) / total_queries if total_queries else 0.0
        return {
            "path": self.path,
            "passed": self.passed,
            "cases": len(self.cases),
            "passed_cases": sum(1 for case in self.cases if case.passed),
            "queries": total_queries,
            "passed_queries": passed_queries,
            "mrr": round(mrr, 4),
        }

    def to_dict(self, include_context: bool = False) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "cases": [
                {
                    "name": case.name,
                    "passed": case.passed,
                    "rebuild_passed": case.rebuild_passed,
                    "invalidation_passed": case.invalidation_passed,
                    "queries": [
                        {
                            "query": query.query,
                            "passed": query.passed,
                            "rank": query.rank,
                            "top_k": query.top_k,
                            "must_contain": query.must_contain,
                            "missing": query.missing,
                            "score_type": query.score_type,
                            "expected_source_refs": query.expected_source_refs,
                            "evidence_rank": query.evidence_rank,
                            "evidence_passed": query.evidence_passed,
                            **({"context": query.context} if include_context else {}),
                        }
                        for query in case.queries
                    ],
                }
                for case in self.cases
            ],
        }


class EvalRunner:
    def run_file(self, path: Path | str, *, keep_tmp: bool = False) -> EvalReport:
        eval_path = Path(path)
        spec = json.loads(eval_path.read_text(encoding="utf-8"))
        cases: list[CaseEvalResult] = []

        if keep_tmp:
            root = Path(tempfile.mkdtemp(prefix="onemem-eval-")) / "memory"
            try:
                cases = [self._run_case(case, root / case["name"]) for case in spec.get("cases", [])]
            finally:
                print(f"kept eval data under {root.parent}")
        else:
            with tempfile.TemporaryDirectory(prefix="onemem-eval-") as tmp:
                base = Path(tmp) / "memory"
                cases = [self._run_case(case, base / case["name"]) for case in spec.get("cases", [])]

        return EvalReport(
            path=str(eval_path),
            passed=all(case.passed for case in cases),
            cases=cases,
        )

    def _run_case(self, case: dict[str, Any], root: Path) -> CaseEvalResult:
        runtime = MemoryRuntime(root)
        runtime.init()
        for episode in case.get("episodes", []):
            runtime.capture(
                episode["text"] if isinstance(episode, dict) else str(episode),
                source=episode.get("source", "eval") if isinstance(episode, dict) else "eval",
                session=episode.get("session", case["name"]) if isinstance(episode, dict) else case["name"],
                salience=float(episode.get("salience", 0.7)) if isinstance(episode, dict) else 0.7,
            )
        runtime.flush()

        query_results = [self._run_query(runtime, query) for query in case.get("queries", [])]
        rebuild_passed = self._check_rebuild(runtime, root, case) if case.get("check_rebuild", True) else None
        invalidation_passed = (
            self._check_invalidation(runtime, case["invalidation"]) if "invalidation" in case else None
        )
        passed = all(query.passed for query in query_results)
        if rebuild_passed is not None:
            passed = passed and rebuild_passed
        if invalidation_passed is not None:
            passed = passed and invalidation_passed
        return CaseEvalResult(
            name=case["name"],
            passed=passed,
            queries=query_results,
            rebuild_passed=rebuild_passed,
            invalidation_passed=invalidation_passed,
        )

    def _run_query(self, runtime: MemoryRuntime, query: dict[str, Any]) -> QueryEvalResult:
        top_k = int(query.get("top_k", 5))
        result = runtime.retrieve(str(query["query"]), limit=top_k)
        context = result.context()
        must_contain = [str(item) for item in query.get("must_contain", [])]
        lower_bodies = [memory.body.lower() for memory in result.memories]
        rank = None
        for index, body in enumerate(lower_bodies, start=1):
            if all(expected.lower() in body for expected in must_contain):
                rank = index
                break
        missing = [expected for expected in must_contain if expected.lower() not in context.lower()]
        expected_source_refs = [str(item) for item in query.get("expected_source_refs", [])]
        evidence_rank = self._evidence_rank(runtime, result.memories, expected_source_refs)
        evidence_passed = evidence_rank is not None if expected_source_refs else None
        answer_passed = not missing and rank is not None
        score_type = str(query.get("score", "evidence" if expected_source_refs else "answer"))
        if score_type == "evidence":
            passed = bool(evidence_passed)
        elif score_type == "both":
            passed = answer_passed and bool(evidence_passed)
        else:
            passed = answer_passed
        return QueryEvalResult(
            query=str(query["query"]),
            passed=passed,
            rank=rank,
            top_k=top_k,
            must_contain=must_contain,
            missing=missing,
            context=context,
            score_type=score_type,
            expected_source_refs=expected_source_refs,
            evidence_rank=evidence_rank,
            evidence_passed=evidence_passed,
        )

    def _evidence_rank(self, runtime: MemoryRuntime, memories: list[Any], expected_refs: list[str]) -> int | None:
        if not expected_refs:
            return None
        expected = set(expected_refs)
        for index, memory in enumerate(memories, start=1):
            if self._memory_evidence_refs(runtime, memory) & expected:
                return index
        return None

    def _memory_evidence_refs(self, runtime: MemoryRuntime, memory: Any) -> set[str]:
        refs = {memory.id, *memory.source_refs}
        for source_ref in memory.source_refs:
            if source_ref.startswith("episode_"):
                episode = runtime.store.get(source_ref)
                if episode:
                    refs.add(episode.id)
                    refs.update(episode.source_refs)
        return refs

    def _check_rebuild(self, runtime: MemoryRuntime, root: Path, case: dict[str, Any]) -> bool:
        sidecar = root / ".sidecar" / "index.sqlite3"
        if sidecar.exists():
            sidecar.unlink()
        runtime.rebuild_index()
        return all(self._run_query(runtime, query).passed for query in case.get("queries", []))

    def _check_invalidation(self, runtime: MemoryRuntime, invalidation: dict[str, Any]) -> bool:
        query = str(invalidation["query"])
        must_contain = [str(item) for item in invalidation.get("must_contain", [])]
        result = runtime.retrieve(query, limit=int(invalidation.get("top_k", 5)))
        target = None
        for memory in result.memories:
            if all(expected.lower() in memory.body.lower() for expected in must_contain):
                target = memory
                break
        if target is None:
            return False
        runtime.invalidate(target.id, reason=str(invalidation.get("reason", "eval invalidation")))
        after = runtime.retrieve(query, limit=int(invalidation.get("top_k", 5)))
        return all(target.id != memory.id for memory in after.memories)


class LongMemEvalImporter:
    def import_file(
        self,
        source: Path | str,
        out: Path | str,
        *,
        limit: int | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        source_path = Path(source)
        out_path = Path(out)
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("LongMemEval input must be a JSON list")

        selected = data[:limit] if limit is not None else data
        cases = [self._convert_item(item, top_k=top_k) for item in selected]
        suite = {
            "name": f"longmemeval_import_{source_path.stem}",
            "source": str(source_path),
            "cases": cases,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "source": str(source_path),
            "out": str(out_path),
            "cases": len(cases),
            "top_k": top_k,
        }

    def _convert_item(self, item: dict[str, Any], *, top_k: int) -> dict[str, Any]:
        question_id = str(item.get("question_id", "unknown"))
        question = str(item["question"])
        answer_text = self._answer_text(item.get("answer", ""))
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        session_ids = item.get("haystack_session_ids", [])
        episodes = []

        for index, session in enumerate(sessions):
            session_id = str(session_ids[index]) if index < len(session_ids) else f"{question_id}_session_{index + 1}"
            date = str(dates[index]) if index < len(dates) else ""
            episodes.append(
                {
                    "text": self._session_text(session, date=date, session_id=session_id),
                    "source": f"longmemeval:{question_id}:{session_id}",
                    "session": question_id,
                    "salience": 0.8,
                }
            )

        return {
            "name": question_id,
            "metadata": {
                "benchmark": "LongMemEval",
                "question_type": item.get("question_type"),
                "question_date": item.get("question_date"),
                "answer_session_ids": item.get("answer_session_ids", []),
            },
            "episodes": episodes,
            "queries": [
                {
                    "query": question,
                    "must_contain": [answer_text],
                    "expected_source_refs": [
                        f"longmemeval:{question_id}:{session_id}"
                        for session_id in item.get("answer_session_ids", [])
                    ],
                    "score": "evidence",
                    "top_k": top_k,
                }
            ],
            "check_rebuild": True,
        }

    def _session_text(self, session: Any, *, date: str, session_id: str) -> str:
        lines = [f"LongMemEval session {session_id}", f"Date: {date}".strip()]
        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict):
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    lines.append(f"{role}: {content}")
                else:
                    lines.append(str(turn))
        else:
            lines.append(str(session))
        return "\n".join(line for line in lines if line)

    def _answer_text(self, answer: Any) -> str:
        if isinstance(answer, list):
            return " ".join(str(item) for item in answer)
        if isinstance(answer, dict):
            return json.dumps(answer, ensure_ascii=False, sort_keys=True)
        return str(answer)


def format_report(report: EvalReport) -> str:
    lines = []
    summary = report.summary()
    status = "PASS" if report.passed else "FAIL"
    lines.append(
        f"{status} cases={summary['passed_cases']}/{summary['cases']} "
        f"queries={summary['passed_queries']}/{summary['queries']} mrr={summary['mrr']:.4f}"
    )
    for case in report.cases:
        case_status = "PASS" if case.passed else "FAIL"
        extras = []
        if case.rebuild_passed is not None:
            extras.append(f"rebuild={'PASS' if case.rebuild_passed else 'FAIL'}")
        if case.invalidation_passed is not None:
            extras.append(f"invalidation={'PASS' if case.invalidation_passed else 'FAIL'}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {case_status} {case.name}{suffix}")
        for query in case.queries:
            query_status = "PASS" if query.passed else "FAIL"
            rank_value = query.scoring_rank()
            rank = rank_value if rank_value is not None else "-"
            missing = f" missing={query.missing}" if query.missing else ""
            evidence = ""
            if query.expected_source_refs:
                evidence_status = "PASS" if query.evidence_passed else "FAIL"
                evidence = f" evidence={evidence_status}"
            lines.append(
                f"  - {query_status} {query.score_type}_rank={rank}@{query.top_k}"
                f"{evidence} query={query.query!r}{missing}"
            )
    return "\n".join(lines)
