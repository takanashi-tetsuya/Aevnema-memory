from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.run_q2_baseline_comparison import (
    BASELINE_SCHEMA,
    _ordinary_app,
    run_baseline_comparison,
)
from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig, ModelConfig


class Q2BaselineComparisonTests(unittest.TestCase):
    def test_ordinary_config_disables_association_and_exact_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / ".env"
            env_file.write_text("\n", encoding="utf-8")
            database = root / "ordinary.sqlite"
            app, config = _ordinary_app(
                env_file=env_file,
                database=database,
                log_dir=root / "logs",
            )

            self.assertFalse(config.retrieval.contextual_association_enabled)
            self.assertFalse(config.retrieval.contextual_association_shadow)
            self.assertFalse(config.retrieval.contextual_promotion_enabled)
            self.assertEqual(database, app.config.database_path)

    def test_preserves_preedge_and_writes_new_ordinary_arm_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sqlite"
            source_app = MemoryApplication(
                AppConfig(
                    database_path=source,
                    log_dir=root / "source-logs",
                    model=ModelConfig(embedding_dimension=3),
                )
            )
            manifest = root / "case.json"
            question = "Which organization did the witness say she supported?"
            scope = "baseline-compare-scope"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "aevnema.v3.q1_q2_diagnostic_source_slice.v1",
                        "formal_scoring_eligible": False,
                        "promotion_prohibited": True,
                        "pilot_input": {
                            "q1_text": question,
                            "q2_text": question,
                            "contextual_domain": "knowledge",
                            "contextual_revisit_scope": scope,
                        },
                    }
                ),
                encoding="utf-8",
            )
            n11d = root / "n11d.full_local.json"
            source_hash = sha256(source.read_bytes()).hexdigest()
            n11d.write_text(
                json.dumps(
                    {
                        "schema": "aevnema.v3.q1_q2_diagnostic_pilot.v2",
                        "status": "diagnostic_complete",
                        "source_database": {"sha256": source_hash},
                        "case": {"q2_text": question},
                        "arms": {
                            "before_commit_probe": {
                                "arm": "before_commit_probe",
                                "summary": {
                                    "run_status": "exact_revisit_miss",
                                    "provider_counts": {"http_attempts": 0},
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            env_file = root / ".env"
            env_file.write_text("\n", encoding="utf-8")
            output = root / "output"

            def ordinary_arm(**kwargs):
                name = kwargs["name"]
                return {
                    "arm": name,
                    "status": "terminal",
                    "summary": {
                        "run_status": "completed",
                        "delivered_episode_ids": [7],
                        "evidence_acquisition_ms": 12.5,
                    },
                    "provider_counts": {"http_attempts": 3},
                }

            with patch(
                "benchmarks.run_q2_baseline_comparison._run_ordinary_arm",
                side_effect=ordinary_arm,
            ):
                result = run_baseline_comparison(
                    source_database=source_app.config.database_path,
                    case_manifest=manifest,
                    n11d_full_local=n11d,
                    output_dir=output,
                    env_file=env_file,
                )

            self.assertEqual("terminal", result["ordinary_status"])
            aggregate = json.loads(
                (output / "q2_baseline_comparison.full_local.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(BASELINE_SCHEMA, aggregate["schema"])
            self.assertEqual(
                "observed_in_preserved_n11d_trace",
                json.loads(
                    (output / "pre_edge_exact_reuse.full_local.json").read_text(
                        encoding="utf-8"
                    )
                )["status"],
            )
            self.assertEqual(
                "terminal",
                json.loads(
                    (output / "ordinary_search.full_local.json").read_text(
                        encoding="utf-8"
                    )
                )["status"],
            )
            self.assertEqual(
                "terminal",
                json.loads(
                    (output / "ordinary_query_embedding_cache.full_local.json").read_text(
                        encoding="utf-8"
                    )
                )["status"],
            )


if __name__ == "__main__":
    unittest.main()
