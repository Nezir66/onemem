from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onemem.models import MemoryNode, MemoryOperation, Relation, utc_now
from onemem.runtime import MemoryRuntime


class V3OperationTest(unittest.TestCase):
    def test_operation_apply_validates_and_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture("Nora prefers concise summaries.", source="test", session="profile")
            runtime.flush()
            fact_id = next(node.id for node in runtime.store.nodes_by_kind("fact"))

            applied = runtime.apply_operations(
                [MemoryOperation(op="INVALIDATE", node_id=fact_id, reason="superseded")]
            )
            node = runtime.inspect(fact_id)

            self.assertEqual(applied[0].changed_ids, [fact_id])
            self.assertEqual(node.status, "deprecated")
            self.assertIn("superseded", node.body)

    def test_operation_update_refuses_deprecated_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture("The project color is green.", source="test")
            runtime.flush()
            fact_id = next(node.id for node in runtime.store.nodes_by_kind("fact"))
            runtime.invalidate(fact_id, reason="changed")

            with self.assertRaises(ValueError):
                runtime.apply_operations(
                    [MemoryOperation(op="UPDATE", node_id=fact_id, updates={"body": "The project color is blue."})]
                )

    def test_merge_candidate_preserves_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            left = MemoryNode(
                id="fact_left",
                kind="fact",
                title="Nora prefers concise summaries",
                body="Nora prefers concise summaries.",
                status="stable",
                confidence=0.8,
                salience=0.6,
                created_at=now,
                updated_at=now,
                source_refs=["episode_left"],
                concept_refs=["answer_style"],
            )
            right = MemoryNode(
                id="fact_right",
                kind="fact",
                title="Nora prefers concise summaries",
                body="Nora prefers concise summaries.",
                status="candidate",
                confidence=0.7,
                salience=0.5,
                created_at=now,
                updated_at=now,
                source_refs=["episode_right"],
                concept_refs=["answer_style"],
            )
            runtime.store.write(left)
            runtime.store.write(right)
            runtime.rebuild_index()

            candidates = runtime.merge_candidates()
            runtime.approve_merge(candidates[0].id)

            target = runtime.inspect(candidates[0].target_id)
            source = runtime.inspect(candidates[0].source_id)
            self.assertEqual(source.status, "deprecated")
            self.assertIn("episode_left", target.source_refs)
            self.assertIn("episode_right", target.source_refs)


class V3ReaderAndSummaryTest(unittest.TestCase):
    def test_reader_answers_temporal_question_with_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture(
                "I bought a new car on February 10th, 2023.",
                source="test:first",
                session="car_timeline",
                event_date="2023-02-10",
                salience=0.8,
            )
            runtime.capture(
                "I got the car detailed on May 4th, 2023.",
                source="test:last",
                session="car_timeline",
                event_date="2023-05-04",
                salience=0.8,
            )

            answer = runtime.answer("What was the first event with my car?", reference_date="2023-06-01")

            self.assertFalse(answer.abstained)
            self.assertIn("February", answer.answer)
            self.assertTrue(answer.memory_ids)
            self.assertTrue(answer.source_refs)

    def test_reader_abstains_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()

            answer = runtime.answer("What is my favorite color?")

            self.assertTrue(answer.abstained)
            self.assertIn("reliable memory", answer.answer)

    def test_layered_summaries_cite_source_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            runtime.capture("Nora prefers concise technical summaries.", source="test", session="profile")
            runtime.capture("Nora prefers short implementation notes.", source="test", session="profile")
            runtime.flush()

            summary_ids = {node.id for node in runtime.store.nodes_by_kind("summary")}
            profile = runtime.inspect("summary_profile_user")

            self.assertIn("summary_profile_user", summary_ids)
            self.assertIn("Facts and episodes remain the truth layer", profile.body)
            self.assertTrue(profile.source_refs)

    def test_graph_neighbors_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            concept = MemoryNode(
                id="concept_answer",
                kind="concept",
                title="answer_style",
                body="Concept anchor.",
                status="stable",
                created_at=now,
                updated_at=now,
                concept_refs=["answer_style"],
            )
            fact = MemoryNode(
                id="fact_answer",
                kind="fact",
                title="Answer style",
                body="Nora prefers concise answers.",
                status="candidate",
                created_at=now,
                updated_at=now,
                source_refs=["episode_answer"],
                concept_refs=["answer_style"],
                relations=[Relation(target_id="concept_answer", type="mentions_concept", weight=0.8)],
            )
            runtime.store.write(concept)
            runtime.store.write(fact)
            runtime.rebuild_index()

            neighbors = runtime.graph_neighbors("fact_answer")
            updated = runtime.record_feedback("fact_answer", "confirmed")

            self.assertEqual(neighbors[0]["id"], "concept_answer")
            self.assertGreater(updated.salience, fact.salience)
            self.assertGreater(updated.confidence, fact.confidence)


class V3HardenedOperationTest(unittest.TestCase):
    def _seed_fact(self, runtime: MemoryRuntime) -> str:
        runtime.capture("Nora prefers concise technical summaries.", source="test", session="profile")
        runtime.flush()
        return next(node.id for node in runtime.store.nodes_by_kind("fact"))

    def test_update_rejects_status_deprecated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            fact_id = self._seed_fact(runtime)

            with self.assertRaisesRegex(ValueError, "INVALIDATE"):
                runtime.apply_operations(
                    [MemoryOperation(op="UPDATE", node_id=fact_id, updates={"status": "deprecated"})]
                )

    def test_update_validates_confidence_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            fact_id = self._seed_fact(runtime)

            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                runtime.apply_operations(
                    [MemoryOperation(op="UPDATE", node_id=fact_id, updates={"confidence": 1.5})]
                )

    def test_update_validates_iso_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            fact_id = self._seed_fact(runtime)

            with self.assertRaisesRegex(ValueError, "ISO"):
                runtime.apply_operations(
                    [MemoryOperation(op="UPDATE", node_id=fact_id, updates={"valid_from": "yesterday"})]
                )

    def test_update_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            fact_id = self._seed_fact(runtime)

            runtime.apply_operations(
                [
                    MemoryOperation(
                        op="UPDATE",
                        node_id=fact_id,
                        updates={"title": "Short summary style", "salience": 0.9},
                    )
                ]
            )
            node = runtime.inspect(fact_id)
            self.assertEqual(node.title, "Short summary style")
            self.assertAlmostEqual(node.salience, 0.9)

    def test_add_merges_operation_source_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            node = MemoryNode(
                id="fact_new",
                kind="fact",
                title="New fact",
                body="Nora prefers short emails.",
                status="candidate",
                created_at=now,
                updated_at=now,
                source_refs=["episode_existing"],
                concept_refs=["answer_style"],
            )

            runtime.apply_operations(
                [
                    MemoryOperation(
                        op="ADD",
                        node=node,
                        source_refs=["episode_extra"],
                    )
                ]
            )
            stored = runtime.inspect("fact_new")
            self.assertIn("episode_existing", stored.source_refs)
            self.assertIn("episode_extra", stored.source_refs)

    def test_promote_and_demote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            fact_id = self._seed_fact(runtime)
            original = runtime.inspect(fact_id).status

            runtime.apply_operations([MemoryOperation(op="PROMOTE", node_id=fact_id)])
            after_promote = runtime.inspect(fact_id).status
            runtime.apply_operations([MemoryOperation(op="DEMOTE", node_id=fact_id)])
            after_demote = runtime.inspect(fact_id).status

            self.assertNotEqual(original, after_promote)
            self.assertEqual(after_demote, original)

    def test_link_adds_relation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            concept = MemoryNode(
                id="concept_style",
                kind="concept",
                title="style",
                body="Concept anchor.",
                status="stable",
                created_at=now,
                updated_at=now,
            )
            runtime.store.write(concept)
            fact_id = self._seed_fact(runtime)
            runtime.rebuild_index()

            runtime.apply_operations(
                [
                    MemoryOperation(
                        op="LINK",
                        node_id=fact_id,
                        target_id="concept_style",
                        relation_type="mentions_concept",
                        relation_weight=0.9,
                    )
                ]
            )
            node = runtime.inspect(fact_id)
            targets = [rel.target_id for rel in node.relations if rel.type == "mentions_concept"]
            self.assertIn("concept_style", targets)

    def test_merge_writes_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            for suffix in ("left", "right"):
                runtime.store.write(
                    MemoryNode(
                        id=f"fact_{suffix}",
                        kind="fact",
                        title="Same fact",
                        body="Nora prefers concise summaries.",
                        status="stable" if suffix == "left" else "candidate",
                        confidence=0.8 if suffix == "left" else 0.7,
                        salience=0.6,
                        created_at=now,
                        updated_at=now,
                        source_refs=[f"episode_{suffix}"],
                        concept_refs=["answer_style"],
                    )
                )
            runtime.rebuild_index()
            candidates = runtime.merge_candidates()
            runtime.approve_merge(candidates[0].id)

            deprecated = candidates[0].source_id
            target = candidates[0].target_id
            self.assertEqual(runtime.index.resolve_alias(deprecated), target)

    def test_audit_log_records_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            fact_id = self._seed_fact(runtime)
            runtime.apply_operations([MemoryOperation(op="INVALIDATE", node_id=fact_id, reason="test")])

            lines = runtime.audit_log.read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(lines)
            last = json.loads(lines[-1])
            self.assertEqual(last["op"], "INVALIDATE")
            self.assertEqual(last["node_id"], fact_id)
            self.assertIn(fact_id, last["changed_ids"])


class V3DedupeTest(unittest.TestCase):
    def test_persistent_merge_candidate_survives_rescore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            for suffix, confidence in (("a", 0.8), ("b", 0.7)):
                runtime.store.write(
                    MemoryNode(
                        id=f"fact_{suffix}",
                        kind="fact",
                        title="Same fact",
                        body="Nora prefers concise summaries.",
                        status="stable",
                        confidence=confidence,
                        salience=0.6,
                        created_at=now,
                        updated_at=now,
                        source_refs=[f"episode_{suffix}"],
                        concept_refs=["answer_style"],
                    )
                )
            runtime.rebuild_index()

            candidates = runtime.merge_candidates()
            self.assertTrue(candidates)
            candidate_id = candidates[0].id
            row = runtime.index.load_merge_candidate(candidate_id)
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "open")

            runtime.approve_merge(candidate_id)
            row_after = runtime.index.load_merge_candidate(candidate_id)
            self.assertEqual(row_after["status"], "applied")


class V3ReaderEdgeCaseTest(unittest.TestCase):
    def test_conflict_abstention_requires_shared_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            runtime.store.write(
                MemoryNode(
                    id="fact_coffee_yes",
                    kind="fact",
                    title="Coffee",
                    body="Nora drinks coffee every morning.",
                    status="stable",
                    confidence=0.8,
                    salience=0.7,
                    created_at=now,
                    updated_at=now,
                    source_refs=["episode_coffee"],
                    concept_refs=["beverages"],
                )
            )
            runtime.store.write(
                MemoryNode(
                    id="fact_coffee_no",
                    kind="fact",
                    title="Coffee",
                    body="Nora does not drink coffee anymore.",
                    status="stable",
                    confidence=0.8,
                    salience=0.7,
                    created_at=now,
                    updated_at=now,
                    source_refs=["episode_coffee_stop"],
                    concept_refs=["beverages"],
                )
            )
            runtime.rebuild_index()
            answer = runtime.answer("Does Nora drink coffee?")
            self.assertTrue(answer.abstained)
            self.assertIn("conflicting", answer.reason)

    def test_conflict_does_not_trigger_on_unrelated_negation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            runtime.store.write(
                MemoryNode(
                    id="fact_tea",
                    kind="fact",
                    title="Tea",
                    body="Nora likes green tea.",
                    status="stable",
                    confidence=0.8,
                    salience=0.7,
                    created_at=now,
                    updated_at=now,
                    source_refs=["episode_tea"],
                    concept_refs=["beverages"],
                )
            )
            runtime.store.write(
                MemoryNode(
                    id="fact_city",
                    kind="fact",
                    title="City",
                    body="Nora has never been to Paris.",
                    status="stable",
                    confidence=0.8,
                    salience=0.7,
                    created_at=now,
                    updated_at=now,
                    source_refs=["episode_city"],
                    concept_refs=["travel"],
                )
            )
            runtime.rebuild_index()
            answer = runtime.answer("Nora tea preference?")
            self.assertFalse(answer.abstained)

    def test_ood_abstention_for_open_reasoning_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            runtime.store.write(
                MemoryNode(
                    id="fact_pizza",
                    kind="fact",
                    title="Pizza",
                    body="Nora likes pizza on Fridays.",
                    status="stable",
                    confidence=0.9,
                    salience=0.9,
                    created_at=now,
                    updated_at=now,
                    source_refs=["episode_pizza"],
                    concept_refs=["food"],
                )
            )
            runtime.rebuild_index()
            answer = runtime.answer("Why does photosynthesis matter?")
            self.assertTrue(answer.abstained)


class V3FeedbackAuditTest(unittest.TestCase):
    def test_feedback_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = MemoryRuntime(Path(tmp) / "memory")
            runtime.init()
            now = utc_now()
            fact = MemoryNode(
                id="fact_audit",
                kind="fact",
                title="Audited",
                body="Nora prefers audited writes.",
                status="candidate",
                confidence=0.6,
                salience=0.5,
                created_at=now,
                updated_at=now,
                source_refs=["episode_audit"],
            )
            runtime.store.write(fact)
            runtime.rebuild_index()

            runtime.record_feedback("fact_audit", "confirmed")
            entries = [
                json.loads(line)
                for line in runtime.audit_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(any(entry["reason"] == "feedback:confirmed" for entry in entries))


if __name__ == "__main__":
    unittest.main()
