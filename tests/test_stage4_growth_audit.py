from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from benchmarks.support.growth_audit import audit_growth


class Stage4GrowthAuditTests(unittest.TestCase):
    def test_dual_accepted_historical_context_is_evidence_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "evaluation-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "question_order": "original",
                        "modes": {
                            "graph_growing": [
                                {
                                    "id": "q1",
                                    "status": "completed",
                                    "result": {
                                        "new_association_ids": [1],
                                        "reinforced_association_ids": [],
                                        "association_ids": [1],
                                    },
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            connection = sqlite3.connect(root / "graph_growing.db")
            try:
                connection.execute(
                    """
                    CREATE TABLE association(
                        id INTEGER PRIMARY KEY,
                        audit_status TEXT NOT NULL,
                        claim_level TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        audit_json TEXT NOT NULL,
                        relation_key TEXT NOT NULL,
                        relation_text TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO association VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        1,
                        "dual_accepted",
                        "historical_context",
                        json.dumps([{"id": 10}, {"id": 20}]),
                        json.dumps(
                            [
                                {
                                    "primary_accept": True,
                                    "adversarial_accept": True,
                                }
                            ]
                        ),
                        "historical_support_context",
                        "Only a historical link; no causal mechanism is claimed.",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            result = audit_growth(report_path)

        self.assertTrue(result["passed"])
        self.assertEqual(result["edge_audits"][0]["issues"], [])


if __name__ == "__main__":
    unittest.main()
