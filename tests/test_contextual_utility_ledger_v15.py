from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from memory_demo.associations.plasticity import ContextualPlasticity
from memory_demo.database import Database, SCHEMA_VERSION, utc_now
from memory_demo.repositories.association import AssociationRepository
from memory_demo.types import (
    ContextualRecallCandidate,
    ContextualUtilityLedgerObservation,
    ContextualUtilityObservation,
    PlasticityEvent,
)


def _opaque(namespace: str, value: str) -> str:
    return (
        f"{namespace}:sha256:"
        f"{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    )


class ContextualUtilityLedgerV15Tests(unittest.TestCase):
    """Focused local SQLite tests; no model, network, or user knowledge DB."""

    def _database(self, directory: str) -> Database:
        database = Database(Path(directory) / "utility-ledger.db")
        database.initialize()
        return database

    @staticmethod
    def _repository(database: Database) -> AssociationRepository:
        return AssociationRepository(database)

    @staticmethod
    def _seed_edge(
        database: Database,
        *,
        status: str = "ready",
        verification_status: str = "source_bound",
        create_receipt: bool = True,
        second_ready_receipt: bool = False,
        marker: str = "edge",
    ) -> tuple[int, int | None, int | None]:
        """Seed only opaque local rows needed by the ledger boundary."""

        now = "2026-09-06T00:00:00+00:00"
        with database.transaction() as connection:
            context_cue_id = int(
                connection.execute(
                    """
                    INSERT INTO association_cue_prototype(
                        domain, cue_kind, model_id, dimension, dtype, vector_blob,
                        text_hash, embedding_space_id, display_text,
                        source_request_hash, created_at
                    ) VALUES(?, 'context', 'ledger-test', 3, 'float32', ?, ?,
                             'ledger-space', '', ?, ?)
                    """,
                    ("knowledge", b"\x00" * 12, f"context-{marker}", "opaque", now),
                ).lastrowid
            )
            need_cue_id = int(
                connection.execute(
                    """
                    INSERT INTO association_cue_prototype(
                        domain, cue_kind, model_id, dimension, dtype, vector_blob,
                        text_hash, embedding_space_id, display_text,
                        source_request_hash, created_at
                    ) VALUES(?, 'need', 'ledger-test', 3, 'float32', ?, ?,
                             'ledger-space', '', ?, ?)
                    """,
                    ("knowledge", b"\x01" * 12, f"need-{marker}", "opaque", now),
                ).lastrowid
            )
            association_id = int(
                connection.execute(
                    """
                    INSERT INTO association(
                        from_type, from_id, to_type, to_id, relation_type,
                        relation_key, relation_text, association_mode,
                        context_cue_id, need_cue_id, utility_weight,
                        utility_successes, utility_noops, utility_harms,
                        distinct_query_count, lifecycle_state, source_request_hash,
                        utility_query_hashes, created_at, updated_at
                    ) VALUES(
                        'episode', 1, 'episode', 2, 'contextual_recall', ?, '',
                        'contextual_recall', ?, ?, 0.37, 4, 5, 6, 7, 'probation',
                        'legacy-source-hash', '["legacy-q"]', ?, ?
                    )
                    """,
                    (f"ledger-{marker}", context_cue_id, need_cue_id, now, now),
                ).lastrowid
            )
            if not create_receipt:
                return association_id, None, None

            def insert_receipt(request_marker: str, receipt_status: str) -> int:
                ready = receipt_status == "ready"
                return int(
                    connection.execute(
                        """
                        INSERT INTO contextual_creation_receipt(
                            creation_request_id, creation_request_hash,
                            candidate_fingerprint, association_id, context_cue_id,
                            need_cue_id, domain, model_id, embedding_space_id,
                            dimension, dtype, source_facts_json,
                            verification_refs_json, verification_status,
                            anchor_provenance_json, target_provenance_json, status,
                            ready_index_epoch, durable_artifact_hash, created_at,
                            ready_at
                        ) VALUES(?, ?, ?, ?, ?, ?, 'knowledge', 'ledger-test',
                                 'ledger-space', 3, 'float32', '[]', '[]', ?, ?, ?,
                                 ?, ?, ?, ?, ?)
                        """,
                        (
                            f"receipt-{marker}-{request_marker}",
                            _opaque("request", f"{marker}-{request_marker}"),
                            _opaque("candidate", f"{marker}-{request_marker}"),
                            association_id,
                            context_cue_id,
                            need_cue_id,
                            verification_status,
                            json.dumps([_opaque("anchor", marker)]),
                            json.dumps([_opaque("target", marker)]),
                            receipt_status,
                            1 if ready else None,
                            _opaque("receipt", f"{marker}-{request_marker}"),
                            now,
                            now if ready else None,
                        ),
                    ).lastrowid
                )

            first_receipt_id = insert_receipt("one", status)
            second_receipt_id = (
                insert_receipt("two", "ready") if second_ready_receipt else None
            )
        return association_id, first_receipt_id, second_receipt_id

    @staticmethod
    def _observation(
        association_id: int,
        receipt_id: int,
        *,
        marker: str = "one",
        factual_support_verified: bool = True,
        is_shadow: bool = False,
        treatment_gain_count: int = 1,
        treatment_loss_count: int = 0,
        single_edge_gain_count: int = 1,
        single_edge_loss_count: int = 0,
        leave_one_out_gain_count: int = 1,
        leave_one_out_loss_count: int = 0,
        recall_gain: int = 1,
        work_metric: str = "",
        treatment_work: int | None = None,
        masked_work: int | None = None,
        work_saved: int | None = None,
        provider_receipt_refs: tuple[str, ...] = (),
        harm: bool = False,
        outcome: str = "recall_gain",
    ) -> ContextualUtilityLedgerObservation:
        return ContextualUtilityLedgerObservation(
            observation_id=_opaque("utility-observation", marker),
            family_id=_opaque("utility-family", marker),
            association_id=association_id,
            creation_receipt_id=receipt_id,
            evaluation_as_of="2026-09-06T00:00:00+00:00",
            candidate_universe_fingerprint=_opaque("universe", marker),
            requirements_fingerprint=_opaque("requirements", marker),
            budget_fingerprint=_opaque("budget", marker),
            input_fingerprint=_opaque("input", marker),
            treatment_fingerprint=_opaque("treatment", marker),
            masked_fingerprint=_opaque("masked", marker),
            single_edge_fingerprint=_opaque("single", marker),
            leave_one_out_fingerprint=_opaque("loo", marker),
            factual_support_verified=factual_support_verified,
            is_shadow=is_shadow,
            treatment_episode_count=2,
            masked_episode_count=1,
            treatment_required_count=2,
            masked_required_count=1,
            treatment_gain_count=treatment_gain_count,
            treatment_loss_count=treatment_loss_count,
            single_edge_gain_count=single_edge_gain_count,
            single_edge_loss_count=single_edge_loss_count,
            leave_one_out_gain_count=leave_one_out_gain_count,
            leave_one_out_loss_count=leave_one_out_loss_count,
            recall_gain=recall_gain,
            work_metric=work_metric,  # type: ignore[arg-type]
            treatment_work=treatment_work,
            masked_work=masked_work,
            work_saved=work_saved,
            provider_receipt_refs=provider_receipt_refs,
            harm=harm,
            outcome=outcome,  # type: ignore[arg-type]
        )

    @staticmethod
    def _raw_ledger_insert(
        connection,
        repository: AssociationRepository,
        observation: ContextualUtilityLedgerObservation,
    ) -> None:
        """Exercise database guards independently of repository validation."""

        payload = repository._utility_ledger_payload(observation)
        fields = tuple(payload)
        connection.execute(
            f"""
            INSERT INTO contextual_utility_ledger({', '.join(fields)}, created_at)
            VALUES({', '.join('?' for _ in fields)}, ?)
            """,
            [*(payload[field_name] for field_name in fields), utc_now()],
        )

    def test_typed_append_is_idempotent_and_does_not_mutate_association(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(
                database, contextual_promotion_enabled=True
            )
            association_id, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            observation = self._observation(association_id, receipt_id)
            with database.connection() as connection:
                before = tuple(
                    connection.execute(
                        """
                        SELECT utility_weight, utility_successes, utility_noops,
                               utility_harms, distinct_query_count, lifecycle_state,
                               source_request_hash, utility_query_hashes
                        FROM association WHERE id = ?
                        """,
                        (association_id,),
                    ).fetchone()
                )

            first_counts = repository.record_utility([observation])
            retry = repository.record_contextual_utility_ledger([observation])
            self.assertEqual(1, first_counts["ledger_inserted"])
            self.assertEqual(0, first_counts["legacy_updated"])
            self.assertTrue(retry[0].idempotent)
            self.assertTrue(retry[0].promotion_eligible)
            with database.connection() as connection:
                after = tuple(
                    connection.execute(
                        """
                        SELECT utility_weight, utility_successes, utility_noops,
                               utility_harms, distinct_query_count, lifecycle_state,
                               source_request_hash, utility_query_hashes
                        FROM association WHERE id = ?
                        """,
                        (association_id,),
                    ).fetchone()
                )
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_utility_ledger"
                    ).fetchone()[0]
                )
            self.assertEqual(before, after)
            self.assertEqual(1, count)

    def test_observation_and_family_collisions_fail_closed_and_ledger_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            observation = self._observation(association_id, receipt_id)
            repository.record_contextual_utility_ledger([observation])

            with self.assertRaisesRegex(ValueError, "observation id"):
                repository.record_contextual_utility_ledger(
                    [
                        replace(
                            observation,
                            treatment_fingerprint=_opaque("treatment", "changed"),
                        )
                    ]
                )
            with self.assertRaisesRegex(ValueError, "family"):
                repository.record_contextual_utility_ledger(
                    [replace(observation, observation_id=_opaque("utility-observation", "other"))]
                )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        "UPDATE contextual_utility_ledger SET recall_gain = 0 WHERE id = 1"
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute("DELETE FROM contextual_utility_ledger WHERE id = 1")

    def test_more_than_128_families_keep_earliest_replay_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            observations = [
                self._observation(
                    association_id,
                    receipt_id,
                    marker=f"family-{index}",
                    treatment_gain_count=0,
                    single_edge_gain_count=0,
                    leave_one_out_gain_count=0,
                    recall_gain=0,
                    outcome="no_op",
                )
                for index in range(130)
            ]
            receipts = repository.record_contextual_utility_ledger(observations)
            retry = repository.record_contextual_utility_ledger([observations[0]])
            self.assertEqual(130, len(receipts))
            self.assertTrue(retry[0].idempotent)
            with database.connection() as connection:
                ledger_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_utility_ledger"
                    ).fetchone()[0]
                )
                old_hashes = str(
                    connection.execute(
                        "SELECT utility_query_hashes FROM association WHERE id = ?",
                        (association_id,),
                    ).fetchone()[0]
                )
            self.assertEqual(130, ledger_count)
            self.assertEqual('["legacy-q"]', old_hashes)

    def test_same_process_concurrent_retries_create_exactly_one_ledger_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            observation = self._observation(association_id, receipt_id, marker="race")
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _index: repository.record_contextual_utility_ledger(
                            [observation]
                        )[0],
                        range(16),
                    )
                )
            self.assertEqual(1, sum(not result.idempotent for result in results))
            with database.connection() as connection:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_utility_ledger"
                    ).fetchone()[0]
                )
            self.assertEqual(1, count)

    def test_v14_to_v15_migration_preserves_old_fields_and_creates_no_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            association_id, _receipt_id, _ = self._seed_edge(database)
            with database.connection() as connection:
                before = tuple(
                    connection.execute(
                        """
                        SELECT utility_weight, utility_successes, utility_noops,
                               utility_harms, distinct_query_count, lifecycle_state,
                               source_request_hash, utility_query_hashes
                        FROM association WHERE id = ?
                        """,
                        (association_id,),
                    ).fetchone()
                )
            with database.transaction() as connection:
                for trigger in (
                    "contextual_utility_ledger_source_guard",
                    "contextual_utility_ledger_no_update",
                    "contextual_utility_ledger_no_delete",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                connection.execute("DROP TABLE contextual_utility_ledger")
                connection.execute("UPDATE schema_meta SET schema_version = 14")

            migrated = Database(Path(directory) / "utility-ledger.db")
            migrated.initialize()
            with migrated.connection() as connection:
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
                after = tuple(
                    connection.execute(
                        """
                        SELECT utility_weight, utility_successes, utility_noops,
                               utility_harms, distinct_query_count, lifecycle_state,
                               source_request_hash, utility_query_hashes
                        FROM association WHERE id = ?
                        """,
                        (association_id,),
                    ).fetchone()
                )
                ledger_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_utility_ledger"
                    ).fetchone()[0]
                )
            self.assertEqual(SCHEMA_VERSION, version)
            self.assertEqual(before, after)
            self.assertEqual(0, ledger_count)

    def test_legacy_diagnostics_remain_non_promotable_even_if_old_flag_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(
                database, contextual_promotion_enabled=True
            )
            association_id, _receipt_id, _ = self._seed_edge(database)
            result = repository.record_utility(
                [
                    ContextualUtilityObservation(
                        association_id=association_id,
                        query_hash="legacy-query-1",
                        outcome="sufficient",
                    ),
                    ContextualUtilityObservation(
                        association_id=association_id,
                        query_hash="legacy-query-2",
                        outcome="sufficient",
                    ),
                ]
            )
            # A strict ready V3 creation receipt now fences off the legacy
            # bounded-hash path entirely: legacy diagnostics cannot rewrite
            # its source request hash or mutable utility fields.
            self.assertEqual(0, result["legacy_updated"])
            self.assertEqual([], repository.list_promotable_contextual_utility())
            with database.connection() as connection:
                lifecycle = str(
                    connection.execute(
                        "SELECT lifecycle_state FROM association WHERE id = ?",
                        (association_id,),
                    ).fetchone()[0]
                )
                ledger_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_utility_ledger"
                    ).fetchone()[0]
                )
            self.assertEqual("probation", lifecycle)
            self.assertEqual(0, ledger_count)

    def test_missing_pending_legacy_and_noncanonical_receipts_are_rejected(self) -> None:
        cases = (
            ("no-receipt", "ready", "source_bound", False),
            ("pending", "committed_pending_index", "source_bound", True),
            ("legacy", "legacy_pending_verification", "legacy_compatibility", True),
        )
        for marker, status, verification, create_receipt in cases:
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                database = self._database(directory)
                repository = self._repository(database)
                association_id, receipt_id, _ = self._seed_edge(
                    database,
                    status=status,
                    verification_status=verification,
                    create_receipt=create_receipt,
                    marker=marker,
                )
                observation = self._observation(
                    association_id,
                    int(receipt_id or 999),
                    marker=marker,
                )
                with self.assertRaisesRegex(ValueError, "canonical ready"):
                    repository.record_contextual_utility_ledger([observation])

        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, first_receipt, second_receipt = self._seed_edge(
                database, second_ready_receipt=True, marker="two-ready"
            )
            assert first_receipt is not None and second_receipt is not None
            noncanonical = self._observation(
                association_id, second_receipt, marker="direct-noncanonical"
            )
            with self.assertRaisesRegex(ValueError, "not canonical"):
                repository.record_contextual_utility_ledger(
                    [noncanonical]
                )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError, "canonical ready source-bound"
            ):
                with database.transaction() as connection:
                    self._raw_ledger_insert(connection, repository, noncanonical)
            repository.record_contextual_utility_ledger(
                [self._observation(association_id, first_receipt, marker="first-ready")]
            )

    def test_shadow_harm_and_relevance_only_cannot_become_promotable_gain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            shadow = self._observation(
                association_id, receipt_id, marker="shadow", is_shadow=True
            )
            harm = self._observation(
                association_id,
                receipt_id,
                marker="mixed-harm",
                treatment_gain_count=1,
                treatment_loss_count=1,
                single_edge_gain_count=1,
                single_edge_loss_count=1,
                leave_one_out_gain_count=1,
                leave_one_out_loss_count=0,
                recall_gain=0,
                harm=True,
                outcome="harmful",
            )
            relevance_only_noop = self._observation(
                association_id,
                receipt_id,
                marker="relevance-only-noop",
                factual_support_verified=False,
                treatment_gain_count=0,
                single_edge_gain_count=0,
                leave_one_out_gain_count=0,
                recall_gain=0,
                outcome="no_op",
            )
            shadow_receipt, harm_receipt, _noop_receipt = repository.record_contextual_utility_ledger(
                [shadow, harm, relevance_only_noop]
            )
            self.assertFalse(shadow_receipt.promotion_eligible)
            self.assertFalse(harm_receipt.promotion_eligible)
            self.assertEqual([], repository.list_promotable_contextual_utility())
            with database.connection() as connection:
                factual_flag = int(
                    connection.execute(
                        """
                        SELECT factual_support_verified
                        FROM contextual_utility_ledger WHERE observation_id = ?
                        """,
                        (relevance_only_noop.observation_id,),
                    ).fetchone()[0]
                )
            self.assertEqual(0, factual_flag)
            with self.assertRaisesRegex(ValueError, "relevance-only"):
                self._observation(
                    association_id,
                    receipt_id,
                    marker="relevance-only",
                    factual_support_verified=False,
                )
            with self.assertRaisesRegex(ValueError, "harmful"):
                replace(harm, recall_gain=1)

    def test_work_is_unknown_without_receipts_and_valid_work_is_not_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            with self.assertRaisesRegex(ValueError, "unmeasured work"):
                self._observation(
                    association_id,
                    receipt_id,
                    marker="unknown-work",
                    treatment_gain_count=0,
                    single_edge_gain_count=0,
                    leave_one_out_gain_count=0,
                    recall_gain=0,
                    work_saved=1,
                    outcome="equal_quality_faster",
                )
            measured = self._observation(
                association_id,
                receipt_id,
                marker="measured-work",
                treatment_gain_count=0,
                single_edge_gain_count=0,
                leave_one_out_gain_count=0,
                recall_gain=0,
                work_metric="provider_receipt_delta",
                treatment_work=2,
                masked_work=5,
                work_saved=3,
                provider_receipt_refs=(_opaque("provider-receipt", "measured"),),
                outcome="equal_quality_faster",
            )
            receipt = repository.record_contextual_utility_ledger([measured])[0]
            self.assertFalse(receipt.promotion_eligible)
            self.assertEqual([], repository.list_promotable_contextual_utility())

    def test_plasticity_accepts_only_typed_ledger_and_ignores_creation_round(self) -> None:
        class Repository:
            def __init__(self) -> None:
                self.ledger_calls: list[tuple[ContextualUtilityLedgerObservation, ...]] = []

            def record_contextual_utility_ledger(self, observations):
                self.ledger_calls.append(tuple(observations))
                return ()

            def record_utility(self, _observations):
                return {"updated": 0}

        observation = self._observation(1, 1, marker="plasticity")
        candidate = ContextualRecallCandidate(
            anchor_type="episode",
            anchor_id=1,
            target_episode_id=2,
            context_query_id="context",
            need_query_id="need",
        )
        repository = Repository()
        plasticity = ContextualPlasticity(repository)
        creation = PlasticityEvent(
            request_hash="creation-round",
            domain="knowledge",
            candidates=(candidate,),
            ledger_observations=(observation,),
        )
        result = plasticity.apply_event(creation)
        self.assertEqual([], repository.ledger_calls)
        self.assertEqual(1, result["creation_ledger_observations_ignored"])

        later = PlasticityEvent(
            request_hash="later-round",
            domain="knowledge",
            ledger_observations=(observation,),
        )
        plasticity.apply_event(later)
        self.assertEqual([(observation,)], repository.ledger_calls)

    def test_ledger_serialization_contains_no_source_text_or_source_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = self._repository(database)
            association_id, receipt_id, _ = self._seed_edge(database, marker="privacy")
            assert receipt_id is not None
            repository.record_contextual_utility_ledger(
                [self._observation(association_id, receipt_id, marker="privacy")]
            )
            with database.connection() as connection:
                row = dict(
                    connection.execute(
                        "SELECT * FROM contextual_utility_ledger"
                    ).fetchone()
                )
                columns = {
                    item["name"]
                    for item in connection.execute(
                        "PRAGMA table_info(contextual_utility_ledger)"
                    )
                }
            serialized = json.dumps(row, ensure_ascii=True, sort_keys=True)
            self.assertNotIn("private/source.json", serialized)
            self.assertNotIn("unredacted source sentence", serialized)
            self.assertNotIn("source_key", columns)
            self.assertNotIn("raw_text", columns)


if __name__ == "__main__":
    unittest.main()
