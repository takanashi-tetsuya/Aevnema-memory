from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest

from memory_demo.database import Database, SCHEMA_VERSION, utc_now
from memory_demo.repositories.association import AssociationRepository
from memory_demo.types import (
    ContextualRevisitContract,
    ContextualRevisitContractLookup,
    ContextualRevisitSlotNeedBinding,
    ContextualUtilityLedgerObservation,
    ContextualUtilityObservation,
)


def _opaque(namespace: str, value: str) -> str:
    return f"{namespace}:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


class ContextualRevisitContractV16Tests(unittest.TestCase):
    """Focused local SQLite coverage for v16; no model/network/real DB."""

    @staticmethod
    def _database(directory: str) -> Database:
        database = Database(Path(directory) / "revisit-contract.db")
        database.initialize()
        return database

    @staticmethod
    def _seed_edge(
        database: Database,
        *,
        marker: str = "edge",
        status: str = "ready",
        verification_status: str = "source_bound",
        create_receipt: bool = True,
        second_ready_receipt: bool = False,
    ) -> tuple[int, int | None, int | None]:
        """Seed only redacted durable rows required by the repository boundary."""

        now = "2026-09-06T00:00:00+00:00"
        with database.transaction() as connection:
            context_cue_id = int(
                connection.execute(
                    """
                    INSERT INTO association_cue_prototype(
                        domain, cue_kind, model_id, dimension, dtype, vector_blob,
                        text_hash, embedding_space_id, display_text,
                        source_request_hash, created_at
                    ) VALUES('knowledge', 'context', 'contract-test', 3, 'float32',
                             ?, ?, 'contract-space', '', ?, ?)
                    """,
                    (
                        b"\x00" * 12,
                        _opaque("cue", f"context-{marker}"),
                        _opaque("request", marker),
                        now,
                    ),
                ).lastrowid
            )
            need_cue_id = int(
                connection.execute(
                    """
                    INSERT INTO association_cue_prototype(
                        domain, cue_kind, model_id, dimension, dtype, vector_blob,
                        text_hash, embedding_space_id, display_text,
                        source_request_hash, created_at
                    ) VALUES('knowledge', 'need', 'contract-test', 3, 'float32',
                             ?, ?, 'contract-space', '', ?, ?)
                    """,
                    (
                        b"\x01" * 12,
                        _opaque("cue", f"need-{marker}"),
                        _opaque("request", marker),
                        now,
                    ),
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
                    (f"contract-{marker}", context_cue_id, need_cue_id, now, now),
                ).lastrowid
            )
            if not create_receipt:
                return association_id, None, None
            connection.execute(
                """
                UPDATE contextual_index_publication
                SET index_epoch = 1, embedding_space_id = 'contract-space',
                    context_cue_count = 1, need_cue_count = 1, published_at = ?
                WHERE singleton = 1
                """,
                (now,),
            )

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
                            anchor_provenance_json, target_provenance_json,
                            source_request_hash, context_vector_ref, need_vector_ref,
                            anchor_vector_ref, anchor_contribution_id, status,
                            ready_index_epoch, durable_artifact_hash, created_at,
                            ready_at
                        ) VALUES(?, ?, ?, ?, ?, ?, 'knowledge', 'contract-test',
                                 'contract-space', 3, 'float32', '[]', '[]', ?, ?, ?,
                                 ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            _opaque("source-request", marker),
                            _opaque("context-vector", marker),
                            _opaque("need-vector", marker),
                            _opaque("anchor-vector", marker),
                            _opaque("contribution", marker),
                            receipt_status,
                            1 if ready else None,
                            _opaque("artifact", f"{marker}-{request_marker}"),
                            now,
                            now if ready else None,
                        ),
                    ).lastrowid
                )

            first = insert_receipt("one", status)
            second = insert_receipt("two", "ready") if second_ready_receipt else None
        return association_id, first, second

    @staticmethod
    def _contract(
        repository: AssociationRepository,
        receipt_id: int,
        *,
        marker: str = "one",
    ) -> ContextualRevisitContract:
        receipt = repository.get_contextual_creation_receipt(receipt_id)
        assert receipt is not None
        return ContextualRevisitContract(
            creation_receipt_id=receipt_id,
            association_id=int(receipt["association_id"]),
            context_cue_id=int(receipt["context_cue_id"]),
            need_cue_id=int(receipt["need_cue_id"]),
            domain=str(receipt["domain"]),
            model_id=str(receipt["model_id"]),
            embedding_space_id=str(receipt["embedding_space_id"]),
            dimension=int(receipt["dimension"]),
            dtype=str(receipt["dtype"]),
            context_hash=_opaque("context", marker),
            slot_need_bindings=(
                ContextualRevisitSlotNeedBinding(
                    slot_id=_opaque("slot", marker),
                    need_query_id=_opaque("need-query", marker),
                    need_hash=_opaque("need", marker),
                ),
            ),
            requirements_fingerprint=_opaque("requirements", marker),
            source_closure_fingerprint=_opaque("source-closure", marker),
            retrieval_policy_fingerprint=_opaque("retrieval-policy", marker),
            budget_fingerprint=_opaque("budget", marker),
            anchor_manifest_fingerprint=_opaque("anchor-manifest", marker),
            source_fact_roles_fingerprint=_opaque("fact-roles", marker),
            source_fact_refs_fingerprint=_opaque("fact-refs", marker),
            ready_index_epoch=int(receipt["ready_index_epoch"]),
            ready_publication_fingerprint=(
                repository.contextual_revisit_ready_publication_fingerprint(receipt)
            ),
        )

    @staticmethod
    def _utility(
        association_id: int,
        receipt_id: int,
        *,
        marker: str,
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
            factual_support_verified=True,
            is_shadow=False,
            treatment_gain_count=1,
            single_edge_gain_count=1,
            leave_one_out_gain_count=1,
            recall_gain=1,
            outcome="recall_gain",
        )

    @staticmethod
    def _lookup(contract: ContextualRevisitContract) -> ContextualRevisitContractLookup:
        return ContextualRevisitContractLookup(
            domain=contract.domain,
            model_id=contract.model_id,
            embedding_space_id=contract.embedding_space_id,
            dimension=contract.dimension,
            dtype=contract.dtype,
            context_hash=contract.context_hash,
            slot_need_bindings=contract.slot_need_bindings,
            requirements_fingerprint=contract.requirements_fingerprint,
            source_closure_fingerprint=contract.source_closure_fingerprint,
            retrieval_policy_fingerprint=contract.retrieval_policy_fingerprint,
            budget_fingerprint=contract.budget_fingerprint,
            anchor_manifest_fingerprint=contract.anchor_manifest_fingerprint,
            contract_version=contract.contract_version,
        )

    @staticmethod
    def _raw_contract_insert(
        connection,
        contract: ContextualRevisitContract,
        *,
        explicit_id: int | None = None,
    ) -> None:
        payload = contract.storage_payload()
        fields = tuple(payload)
        if explicit_id is not None:
            fields = ("id", *fields)
            values: list[object] = [explicit_id]
        else:
            values = []
        values.extend(payload[field_name] for field_name in payload)
        connection.execute(
            f"""
            INSERT OR REPLACE INTO contextual_revisit_contract(
                {', '.join(fields)}, created_at
            ) VALUES({', '.join('?' for _ in fields)}, ?)
            """,
            [*values, utc_now()],
        )

    def test_create_load_idempotent_and_redacted_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            _, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            contract = self._contract(repository, receipt_id)

            first = repository.create_contextual_revisit_contract(contract)
            retry = repository.create_contextual_revisit_contract(contract)
            restored = repository.load_contextual_revisit_contract(receipt_id)
            self.assertFalse(first.idempotent)
            self.assertTrue(retry.idempotent)
            self.assertEqual(contract, restored)
            self.assertEqual(contract.contract_fingerprint, first.contract_fingerprint)
            self.assertEqual(
                contract,
                repository.find_contextual_revisit_contract(self._lookup(contract)),
            )

            with database.connection() as connection:
                row = connection.execute(
                    "SELECT * FROM contextual_revisit_contract WHERE id = ?",
                    (first.contract_id,),
                ).fetchone()
                assert row is not None
                serialized = json.dumps(dict(row), ensure_ascii=True, sort_keys=True)
                columns = {
                    str(item["name"])
                    for item in connection.execute(
                        "PRAGMA table_info(contextual_revisit_contract)"
                    )
                }
            self.assertEqual(
                contract.source_fact_roles_fingerprint,
                row["source_fact_roles_fingerprint"],
            )
            self.assertEqual(
                contract.source_fact_refs_fingerprint,
                row["source_fact_refs_fingerprint"],
            )
            self.assertNotIn("question prose", serialized)
            self.assertNotIn("source-key", serialized)
            self.assertNotIn("answer prose", serialized)
            self.assertNotIn("vector_blob", columns)
            self.assertNotIn("query_text", columns)
            self.assertNotIn("answer_text", columns)
            self.assertNotIn("source_text", columns)

    def test_historical_pending_legacy_and_noncanonical_receipts_are_misses_or_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            _, ready_receipt, _ = self._seed_edge(database, marker="historical")
            assert ready_receipt is not None
            self.assertIsNone(repository.load_contextual_revisit_contract(ready_receipt))

            _, pending_receipt, _ = self._seed_edge(
                database, marker="pending", status="committed_pending_index"
            )
            assert pending_receipt is not None
            pending = ContextualRevisitContract(
                **{
                    **self._contract(repository, ready_receipt, marker="pending-copy").canonical_payload(),
                    "creation_receipt_id": pending_receipt,
                    "association_id": repository.get_contextual_creation_receipt(pending_receipt)["association_id"],  # type: ignore[index]
                    "context_cue_id": repository.get_contextual_creation_receipt(pending_receipt)["context_cue_id"],  # type: ignore[index]
                    "need_cue_id": repository.get_contextual_creation_receipt(pending_receipt)["need_cue_id"],  # type: ignore[index]
                    "ready_index_epoch": 1,
                    "ready_publication_fingerprint": _opaque("publication", "pending"),
                    "slot_need_bindings": self._contract(
                        repository, ready_receipt, marker="pending-copy"
                    ).slot_need_bindings,
                }
            )
            with self.assertRaisesRegex(ValueError, "canonical ready"):
                repository.create_contextual_revisit_contract(pending)

            _, legacy_receipt, _ = self._seed_edge(
                database, marker="legacy", verification_status="legacy_pending"
            )
            assert legacy_receipt is not None
            legacy = self._contract(repository, legacy_receipt, marker="legacy")
            with self.assertRaisesRegex(ValueError, "canonical ready"):
                repository.create_contextual_revisit_contract(legacy)

            _, first_receipt, second_receipt = self._seed_edge(
                database, marker="noncanonical", second_ready_receipt=True
            )
            assert first_receipt is not None and second_receipt is not None
            with self.assertRaisesRegex(ValueError, "canonical"):
                repository.create_contextual_revisit_contract(
                    self._contract(repository, second_receipt, marker="second")
                )

    def test_contract_conflicts_immutability_and_replace_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            _, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            contract = self._contract(repository, receipt_id)
            receipt = repository.create_contextual_revisit_contract(contract)

            with self.assertRaisesRegex(ValueError, "different revisit contract"):
                repository.create_contextual_revisit_contract(
                    replace(contract, requirements_fingerprint=_opaque("requirements", "changed"))
                )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        "UPDATE contextual_revisit_contract SET domain = 'other' WHERE id = ?",
                        (receipt.contract_id,),
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        "DELETE FROM contextual_revisit_contract WHERE id = ?",
                        (receipt.contract_id,),
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    self._raw_contract_insert(
                        connection,
                        replace(contract, requirements_fingerprint=_opaque("requirements", "id")),
                        explicit_id=receipt.contract_id,
                    )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    self._raw_contract_insert(
                        connection,
                        replace(contract, requirements_fingerprint=_opaque("requirements", "receipt")),
                    )

            _, other_receipt, _ = self._seed_edge(database, marker="fingerprint")
            assert other_receipt is not None
            other = self._contract(repository, other_receipt, marker="other")
            forged_payload = other.storage_payload()
            forged_payload["contract_fingerprint"] = contract.contract_fingerprint
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    fields = tuple(forged_payload)
                    connection.execute(
                        f"""
                        INSERT OR REPLACE INTO contextual_revisit_contract(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [
                            *(forged_payload[field_name] for field_name in fields),
                            utc_now(),
                        ],
                    )

    def test_contract_type_and_database_redaction_guards_reject_prose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            _, receipt_id, _ = self._seed_edge(database)
            assert receipt_id is not None
            contract = self._contract(repository, receipt_id)
            with self.assertRaisesRegex(ValueError, "context_hash"):
                replace(contract, context_hash="the original question prose")

            payload = contract.storage_payload()
            payload["context_hash"] = "the original question prose"
            fields = tuple(payload)
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        f"""
                        INSERT INTO contextual_revisit_contract(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [*(payload[field_name] for field_name in fields), utc_now()],
                    )

            payload = contract.storage_payload()
            payload["slot_need_bindings_json"] = json.dumps(
                [{"slot_id": "raw slot", "need_query_id": "raw need", "need_hash": "raw"}]
            )
            fields = tuple(payload)
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        f"""
                        INSERT INTO contextual_revisit_contract(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [*(payload[field_name] for field_name in fields), utc_now()],
                    )

    def test_same_process_contract_retries_insert_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            _, receipt_id, _ = self._seed_edge(database, marker="race")
            assert receipt_id is not None
            contract = self._contract(repository, receipt_id, marker="race")
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _index: repository.create_contextual_revisit_contract(contract),
                        range(16),
                    )
                )
            self.assertEqual(1, sum(not result.idempotent for result in results))
            with database.connection() as connection:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_contract"
                    ).fetchone()[0]
                )
            self.assertEqual(1, count)

    def test_load_fails_closed_when_bound_cue_metadata_no_longer_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            association_id, receipt_id, _ = self._seed_edge(
                database, marker="mismatch"
            )
            assert receipt_id is not None
            contract = self._contract(repository, receipt_id, marker="mismatch")
            repository.create_contextual_revisit_contract(contract)
            with database.transaction() as connection:
                connection.execute(
                    """
                    UPDATE association SET context_cue_id = need_cue_id
                    WHERE id = ?
                    """,
                    (association_id,),
                )
            self.assertIsNone(repository.load_contextual_revisit_contract(receipt_id))
            self.assertIsNone(
                repository.find_contextual_revisit_contract(self._lookup(contract))
            )

    def test_exact_candidate_lookup_is_ambiguity_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            _, first_receipt, _ = self._seed_edge(database, marker="lookup-one")
            assert first_receipt is not None
            first = self._contract(repository, first_receipt, marker="lookup")
            repository.create_contextual_revisit_contract(first)
            self.assertEqual(
                first,
                repository.find_contextual_revisit_contract(self._lookup(first)),
            )

            # Same exact lookup key on a distinct source-bound edge is not
            # permission to choose arbitrarily; it becomes a safe miss.
            _, second_receipt, _ = self._seed_edge(database, marker="lookup-two")
            assert second_receipt is not None
            second = self._contract(repository, second_receipt, marker="lookup")
            repository.create_contextual_revisit_contract(second)
            self.assertIsNone(
                repository.find_contextual_revisit_contract(self._lookup(first))
            )

    def test_ledger_replace_factual_legacy_and_canonical_stability_hardening(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            repository = AssociationRepository(database)
            association_id, pending_receipt, ready_receipt = self._seed_edge(
                database,
                marker="ledger",
                status="committed_pending_index",
                second_ready_receipt=True,
            )
            assert pending_receipt is not None and ready_receipt is not None
            observation = self._utility(association_id, ready_receipt, marker="one")
            inserted = repository.record_contextual_utility_ledger([observation])[0]

            # An older receipt becoming ready cannot invalidate the original
            # exact retry or move the durable canonical origin.
            with database.transaction() as connection:
                connection.execute(
                    """
                    UPDATE contextual_creation_receipt
                    SET status = 'ready', ready_index_epoch = 2, ready_at = ?
                    WHERE id = ?
                    """,
                    ("2026-09-06T00:01:00+00:00", pending_receipt),
                )
                connection.execute(
                    """
                    UPDATE contextual_index_publication
                    SET index_epoch = 2, published_at = ? WHERE singleton = 1
                    """,
                    ("2026-09-06T00:01:00+00:00",),
                )
            self.assertTrue(
                repository.record_contextual_utility_ledger([observation])[0].idempotent
            )
            self.assertFalse(
                repository.record_contextual_utility_ledger(
                    [self._utility(association_id, ready_receipt, marker="two")]
                )[0].idempotent
            )

            payload = repository._utility_ledger_payload(
                self._utility(association_id, ready_receipt, marker="replace-id")
            )
            fields = ("id", *tuple(payload))
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        f"""
                        INSERT OR REPLACE INTO contextual_utility_ledger(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [
                            inserted.ledger_id,
                            *(payload[field_name] for field_name in payload),
                            utc_now(),
                        ],
                    )
            observation_payload = repository._utility_ledger_payload(observation)
            fields = tuple(observation_payload)
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        f"""
                        INSERT OR REPLACE INTO contextual_utility_ledger(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [
                            *(observation_payload[field_name] for field_name in fields),
                            utc_now(),
                        ],
                    )
            family_payload = repository._utility_ledger_payload(
                replace(
                    observation,
                    observation_id=_opaque("utility-observation", "family-collision"),
                )
            )
            fields = tuple(family_payload)
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        f"""
                        INSERT OR REPLACE INTO contextual_utility_ledger(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [*(family_payload[field_name] for field_name in fields), utc_now()],
                    )

            relevance_payload = repository._utility_ledger_payload(
                self._utility(association_id, ready_receipt, marker="relevance")
            )
            relevance_payload["factual_support_verified"] = 0
            fields = tuple(relevance_payload)
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        f"""
                        INSERT INTO contextual_utility_ledger(
                            {', '.join(fields)}, created_at
                        ) VALUES({', '.join('?' for _ in fields)}, ?)
                        """,
                        [
                            *(relevance_payload[field_name] for field_name in fields),
                            utc_now(),
                        ],
                    )
            with self.assertRaisesRegex(ValueError, "relevance-only"):
                replace(
                    observation,
                    factual_support_verified=False,
                )

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
            counts = repository.record_utility(
                [
                    ContextualUtilityObservation(
                        association_id=association_id,
                        query_hash="legacy-query",
                        outcome="sufficient",
                    )
                ]
            )
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
            self.assertEqual(0, counts["legacy_updated"])
            self.assertEqual(before, after)
            self.assertTrue(repository.list_promotable_contextual_utility())

    def test_v15_to_v16_migration_is_empty_and_adds_unknown_factual_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "early-v15.db"
            schema_path = Path(__file__).resolve().parents[1] / "src" / "memory_demo" / "schema.sql"
            schema = schema_path.read_text(encoding="utf-8")
            v16_start = schema.index("-- V16 stores the redacted")
            fts_start = schema.index("-- FTS is a retrieval index", v16_start)
            schema = schema[:v16_start] + schema[fts_start:]
            schema = re.sub(
                r"\n    factual_support_verified INTEGER NOT NULL\n        CHECK\(factual_support_verified IN \(0, 1\)\),",
                "",
                schema,
            )
            schema = re.sub(
                r"\n    CHECK\(\n        factual_support_verified = 1\n        OR \(\n            recall_gain = 0\n            AND treatment_gain_count = 0\n            AND single_edge_gain_count = 0\n            AND leave_one_out_gain_count = 0\n        \)\n    \),",
                "",
                schema,
            )
            schema = re.sub(
                r"\nDROP TRIGGER IF EXISTS contextual_utility_ledger_factual_guard;\nCREATE TRIGGER contextual_utility_ledger_factual_guard.*?END;\n",
                "\n",
                schema,
                flags=re.DOTALL,
            )
            connection = sqlite3.connect(database_path)
            connection.create_function(
                "memory_bigram_tokens", 1, lambda value: "", deterministic=True
            )
            try:
                connection.executescript(schema)
                connection.execute(
                    "INSERT INTO schema_meta(schema_version, created_at) VALUES(15, 'legacy')"
                )
                connection.execute(
                    """
                    INSERT INTO association(
                        from_type, from_id, to_type, to_id, relation_type,
                        relation_key, relation_text, association_mode,
                        utility_weight, utility_successes, utility_noops,
                        utility_harms, distinct_query_count, lifecycle_state,
                        source_request_hash, utility_query_hashes, created_at, updated_at
                    ) VALUES('episode', 1, 'episode', 2, 'contextual_recall',
                             'old-edge', '', 'contextual_recall', 0.37, 4, 5, 6,
                             7, 'probation', 'old-source-hash', '["old"]', 'old', 'old')
                    """
                )
                context_cue_id = int(
                    connection.execute(
                        """
                        INSERT INTO association_cue_prototype(
                            domain, cue_kind, model_id, dimension, dtype, vector_blob,
                            text_hash, embedding_space_id, display_text,
                            source_request_hash, created_at
                        ) VALUES('knowledge', 'context', 'contract-test', 3, 'float32',
                                 ?, ?, 'contract-space', '', ?, 'old')
                        """,
                        (
                            b"\x00" * 12,
                            _opaque("cue", "old-context"),
                            _opaque("request", "old"),
                        ),
                    ).lastrowid
                )
                need_cue_id = int(
                    connection.execute(
                        """
                        INSERT INTO association_cue_prototype(
                            domain, cue_kind, model_id, dimension, dtype, vector_blob,
                            text_hash, embedding_space_id, display_text,
                            source_request_hash, created_at
                        ) VALUES('knowledge', 'need', 'contract-test', 3, 'float32',
                                 ?, ?, 'contract-space', '', ?, 'old')
                        """,
                        (
                            b"\x01" * 12,
                            _opaque("cue", "old-need"),
                            _opaque("request", "old"),
                        ),
                    ).lastrowid
                )
                ledger_association_id = int(
                    connection.execute(
                        """
                        INSERT INTO association(
                            from_type, from_id, to_type, to_id, relation_type,
                            relation_key, relation_text, association_mode,
                            context_cue_id, need_cue_id, utility_weight,
                            utility_successes, utility_noops, utility_harms,
                            distinct_query_count, lifecycle_state, source_request_hash,
                            utility_query_hashes, created_at, updated_at
                        ) VALUES('episode', 3, 'episode', 4, 'contextual_recall',
                                 'old-ledger-edge', '', 'contextual_recall', ?, ?,
                                 0.2, 0, 0, 0, 0, 'probation', '', '[]', 'old', 'old')
                        """,
                        (context_cue_id, need_cue_id),
                    ).lastrowid
                )
                connection.execute(
                    """
                    UPDATE contextual_index_publication
                    SET index_epoch = 1, embedding_space_id = 'contract-space',
                        context_cue_count = 1, need_cue_count = 1, published_at = 'old'
                    WHERE singleton = 1
                    """
                )
                receipt_id = int(
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
                        ) VALUES(?, ?, ?, ?, ?, ?, 'knowledge', 'contract-test',
                                 'contract-space', 3, 'float32', '[]', '[]',
                                 'source_bound', '[]', '[]', 'ready', 1, ?, 'old', 'old')
                        """,
                        (
                            "old-ledger-receipt",
                            _opaque("request", "old-ledger"),
                            _opaque("candidate", "old-ledger"),
                            ledger_association_id,
                            context_cue_id,
                            need_cue_id,
                            _opaque("artifact", "old-ledger"),
                        ),
                    ).lastrowid
                )
                connection.execute(
                    """
                    INSERT INTO contextual_utility_ledger(
                        observation_id, family_id, association_id, creation_receipt_id,
                        evaluation_as_of, candidate_universe_fingerprint,
                        requirements_fingerprint, budget_fingerprint, input_fingerprint,
                        treatment_fingerprint, masked_fingerprint,
                        single_edge_fingerprint, leave_one_out_fingerprint,
                        treatment_episode_count, masked_episode_count,
                        treatment_required_count, masked_required_count,
                        treatment_gain_count, treatment_loss_count,
                        single_edge_gain_count, single_edge_loss_count,
                        leave_one_out_gain_count, leave_one_out_loss_count,
                        recall_gain, work_metric, provider_receipt_refs_json,
                        harm, is_shadow, outcome, created_at
                    ) VALUES(?, ?, ?, ?, 'old', ?, ?, ?, ?, ?, ?, ?, ?,
                             1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, '', '[]',
                             0, 0, 'recall_gain', 'old')
                    """,
                    (
                        _opaque("utility-observation", "old"),
                        _opaque("utility-family", "old"),
                        ledger_association_id,
                        receipt_id,
                        _opaque("universe", "old"),
                        _opaque("requirements", "old"),
                        _opaque("budget", "old"),
                        _opaque("input", "old"),
                        _opaque("treatment", "old"),
                        _opaque("masked", "old"),
                        _opaque("single", "old"),
                        _opaque("loo", "old"),
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            migrated = Database(database_path)
            migrated.initialize()
            with migrated.connection() as connection:
                version = int(
                    connection.execute("SELECT schema_version FROM schema_meta").fetchone()[0]
                )
                columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(contextual_utility_ledger)"
                    )
                }
                contracts = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM contextual_revisit_contract"
                    ).fetchone()[0]
                )
                old_edge = tuple(
                    connection.execute(
                        """
                        SELECT utility_weight, utility_successes, utility_noops,
                               utility_harms, distinct_query_count, lifecycle_state,
                               source_request_hash, utility_query_hashes
                        FROM association WHERE relation_key = 'old-edge'
                        """
                    ).fetchone()
                )
                old_ledger = tuple(
                    connection.execute(
                        """
                        SELECT factual_support_verified, recall_gain
                        FROM contextual_utility_ledger
                        WHERE association_id = ?
                        """,
                        (ledger_association_id,),
                    ).fetchone()
                )
            self.assertEqual(SCHEMA_VERSION, version)
            self.assertIn("factual_support_verified", columns)
            self.assertEqual(0, contracts)
            self.assertEqual((0, 1), old_ledger)
            self.assertEqual(
                (0.37, 4, 5, 6, 7, "probation", "old-source-hash", '["old"]'),
                old_edge,
            )
