from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from threading import Barrier, Event, Lock, get_ident
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import requests

from memory_demo.config import AppConfig, ModelConfig
from memory_demo.database import Database
from memory_demo.ingestion.extractor import (
    EmptyEpisodeAudit,
    EmptyEpisodeExtraction,
    MemoryExtractor,
)
from memory_demo.llm.client import (
    HTTPSConnection,
    ModelClient,
    ModelClientError,
    ModelTransportUnavailable,
    _SerializedHTTPSConnection,
    _SerializedHTTPSConnectionPool,
)
from memory_demo.llm.validation import (
    parse_concept_batch_text,
    parse_concept_text,
    parse_episode_text,
    parse_source_scoped_episode_text,
)
from memory_demo.llm.prompts import (
    ANSWER_AUDIT_SYSTEM,
    ANSWER_SYSTEM,
    GROWTH_AUDIT_SYSTEM,
    GROWTH_SYSTEM,
    HOP_QUERY_SYSTEM,
    answer_audit_prompt,
)
from memory_demo.retrieval.context import source_excerpt, verified_evidence_views
from memory_demo.ingestion.pipeline import ImportPipeline
from memory_demo.repositories.extraction import ExtractionRepository
from memory_demo.repositories.source import SourceRepository
from memory_demo.types import AssociationDraft, ConceptDraft, EpisodeDraft


class FlakyEmbeddingClient(ModelClient):
    def __init__(self):
        super().__init__(
            ModelConfig(
                api_key="fake",
                embedding_model="fixed-embedding-model",
                fallback_model="reasoning-fallback-must-not-be-used",
                embedding_dimension=4,
                max_retries=1,
            )
        )
        self.calls = 0
        self.models: list[str] = []

    def _post(self, endpoint, payload):
        self.calls += 1
        self.models.append(payload["model"])
        if self.calls == 1:
            raise ModelClientError("temporary")
        return {
            "data": [
                {"index": index, "embedding": [1.0, 0.0, 0.0, 0.0]}
                for index, _text in enumerate(payload["input"])
            ]
        }


class ModelAndEvidenceTests(unittest.TestCase):
    def test_source_scoped_plain_text_is_assembled_without_model_metadata(self):
        episodes, errors = parse_source_scoped_episode_text(
            "1. Morgan提交了申请。\n\n2. Riley批准了申请。",
            timeline_scope="main",
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [episode.text for episode in episodes],
            ["Morgan提交了申请。", "Riley批准了申请。"],
        )
        self.assertTrue(all(episode.participants == [] for episode in episodes))
        self.assertTrue(all(episode.evidence_spans == [] for episode in episodes))

    def test_empty_episode_audit_accepts_only_independent_exact_safe_skip(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class AuditModel:
            config = Config()

            def __init__(self):
                self.kwargs = None
                self.prompt = ""

            def chat_json(self, _system, prompt, **kwargs):
                self.kwargs = kwargs
                self.prompt = prompt
                return {
                    "contract_version": "empty_episode_adversarial_audit_v1",
                    "verdict": "safe_skip",
                    "source_kind": "control_only",
                    "reason": "only a rendered control command remains",
                    "line_reviews": [
                        {
                            "start_line": 3,
                            "end_line": 3,
                            "kind": "control",
                            "quote": "zh-CN: #clearST",
                            "reason": "engine display-state command",
                        },
                        {
                            "start_line": 4,
                            "end_line": 4,
                            "kind": "control",
                            "quote": "en: #clearST",
                            "reason": "engine display-state command",
                        },
                    ],
                    "required_ranges": [],
                }

        source = (
            "[source_key: control.json]\n"
            "[record: 1]\n"
            "zh-CN: #clearST\n"
            "en: #clearST"
        )
        model = AuditModel()
        audit = MemoryExtractor(model).review_empty_episode(source)

        self.assertEqual(audit.verdict, "safe_skip")
        self.assertEqual(audit.required_ranges, [])
        self.assertEqual(
            model.kwargs,
            {
                "model": "independent-reviewer",
                "allow_fallback": False,
                "max_retries": 0,
            },
        )
        self.assertNotIn("primary-extractor response", model.prompt)

    def test_empty_episode_audit_fails_closed_for_forged_quote(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class ForgedAuditModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                return {
                    "contract_version": "empty_episode_adversarial_audit_v1",
                    "verdict": "safe_skip",
                    "source_kind": "control_only",
                    "reason": "claimed control",
                    "line_reviews": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "kind": "control",
                            "quote": "zh-CN: forged",
                            "reason": "forged quote",
                        }
                    ],
                    "required_ranges": [],
                }

        audit = MemoryExtractor(ForgedAuditModel()).review_empty_episode(
            "[record: 1]\nzh-CN: #clearST"
        )

        self.assertEqual(audit.verdict, "uncertain")
        self.assertTrue(audit.validation_errors)
        self.assertIn("does not exactly match Source", audit.validation_errors[0])

    def test_empty_episode_audit_safe_skip_must_cover_script_raw_too(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class IncompleteAuditModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                return {
                    "contract_version": "empty_episode_adversarial_audit_v1",
                    "verdict": "safe_skip",
                    "source_kind": "control_only",
                    "reason": "reviewed translated command only",
                    "line_reviews": [
                        {
                            "start_line": 3,
                            "end_line": 3,
                            "kind": "control",
                            "quote": "zh-CN: #clearST",
                            "reason": "display command",
                        }
                    ],
                    "required_ranges": [],
                }

        audit = MemoryExtractor(IncompleteAuditModel()).review_empty_episode(
            "[record: 1]\n[script_raw: #all;hide]\nzh-CN: #clearST"
        )

        self.assertEqual(audit.verdict, "uncertain")
        self.assertTrue(
            any("2" in error and "cover" in error for error in audit.validation_errors or [])
        )

    def test_empty_episode_audit_safe_skip_rejects_multiline_review(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class MultilineAuditModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                return {
                    "contract_version": "empty_episode_adversarial_audit_v1",
                    "verdict": "safe_skip",
                    "source_kind": "control_only",
                    "reason": "both controls were reviewed together",
                    "line_reviews": [
                        {
                            "start_line": 2,
                            "end_line": 3,
                            "kind": "control",
                            "quote": "zh-CN: #clearST\nen: #clearST",
                            "reason": "both are display-state commands",
                        }
                    ],
                    "required_ranges": [],
                }

        audit = MemoryExtractor(MultilineAuditModel()).review_empty_episode(
            "[record: 1]\nzh-CN: #clearST\nen: #clearST"
        )

        self.assertEqual(audit.verdict, "uncertain")
        self.assertTrue(audit.validation_errors)

    def test_empty_episode_audit_safe_skip_must_cover_language_continuations(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class IncompleteContinuationAuditModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                return {
                    "contract_version": "empty_episode_adversarial_audit_v1",
                    "verdict": "safe_skip",
                    "source_kind": "control_only",
                    "reason": "reviewed only the first physical language line",
                    "line_reviews": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "kind": "control",
                            "quote": "zh-CN: #clearST",
                            "reason": "display-state command",
                        }
                    ],
                    "required_ranges": [],
                }

        audit = MemoryExtractor(IncompleteContinuationAuditModel()).review_empty_episode(
            "[record: 1]\nzh-CN: #clearST\n#all;hide"
        )

        self.assertEqual(audit.verdict, "uncertain")
        self.assertTrue(
            any("3" in error and "cover" in error for error in audit.validation_errors or [])
        )

    def test_empty_episode_audit_reraises_transport_unavailable(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class TransportAuditModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                raise ModelTransportUnavailable("test transport circuit is open")

        with self.assertRaises(ModelTransportUnavailable):
            MemoryExtractor(TransportAuditModel()).review_empty_episode(
                "[record: 1]\nzh-CN: #clearST"
            )

    def test_empty_episode_rescue_reraises_transport_unavailable(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class TransportRescueModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                raise ModelTransportUnavailable("test transport circuit is open")

        source = "[record: 1]\nunknown: Morgan submitted the request."
        audit = EmptyEpisodeAudit(
            verdict="episode_required",
            source_kind="eventful",
            reason="a concrete action is present",
            line_reviews=[
                {
                    "start_line": 2,
                    "end_line": 2,
                    "kind": "event",
                    "quote": "unknown: Morgan submitted the request.",
                    "reason": "submission is an action",
                }
            ],
            required_ranges=[(2, 2)],
            primary_model="primary-extractor",
            reviewer_model="independent-reviewer",
            source_sha256="test-source",
        )

        with self.assertRaises(ModelTransportUnavailable):
            MemoryExtractor(TransportRescueModel()).extract_after_empty_episode_review(
                source, "main", audit
            )

    def test_empty_episode_audit_rejects_reviewer_already_used_for_extraction(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "unused-fallback"

        class ReusedCandidateAuditModel:
            config = Config()

            def __init__(self):
                self.calls = 0

            def chat_json(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("a previously used extraction model must not audit")

        model = ReusedCandidateAuditModel()
        extractor = MemoryExtractor(
            model,
            empty_episode_audit_model="Independent-Reviewer",
        )
        audit = extractor.review_empty_episode(
            "[record: 1]\nzh-CN: #clearST",
            attempted_models=("independent-reviewer",),
        )

        self.assertEqual(audit.verdict, "uncertain")
        self.assertEqual(model.calls, 0)
        self.assertIn("independent-reviewer", audit.attempted_models)

    def test_source_scoped_clean_empty_records_every_attempted_model(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "fallback-extractor"

        class EmptySourceScopedModel:
            config = Config()
            semantic_output_format = "natural_text"

            def chat_text(self, *_args, **_kwargs):
                return "no extractable episodes"

        with self.assertRaises(EmptyEpisodeExtraction) as raised:
            MemoryExtractor(
                EmptySourceScopedModel(),
                episode_extraction_profile="source_scoped_plain",
            ).extract_episodes("[record: 1]\nunknown: #clearST", "main")

        self.assertEqual(
            raised.exception.attempted_models,
            ("primary-extractor", "fallback-extractor"),
        )

    def test_single_pass_clean_empty_records_every_attempted_model(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "fallback-extractor"

        class EmptySinglePassModel:
            config = Config()

            def chat_json(self, *_args, **_kwargs):
                return {"episodes": []}

        with self.assertRaises(EmptyEpisodeExtraction) as raised:
            MemoryExtractor(
                EmptySinglePassModel(),
                episode_extraction_profile="single_pass_evidence",
            ).extract_episodes("[record: 1]\nunknown: #clearST", "main")

        self.assertEqual(
            raised.exception.attempted_models,
            ("primary-extractor", "fallback-extractor"),
        )

    def test_empty_episode_audit_requires_one_independent_evidence_bound_rescue(self):
        class Config:
            reasoning_model = "primary-extractor"
            fallback_model = "independent-reviewer"

        class RescueModel:
            config = Config()

            def __init__(self):
                self.calls: list[tuple[str, dict]] = []

            def chat_json(self, system, _prompt, **kwargs):
                self.calls.append((system, kwargs))
                if "空 Episode 对抗审核器" in system:
                    return {
                        "contract_version": "empty_episode_adversarial_audit_v1",
                        "verdict": "episode_required",
                        "source_kind": "eventful",
                        "reason": "a concrete action is present",
                        "line_reviews": [
                            {
                                "start_line": 2,
                                "end_line": 2,
                                "kind": "event",
                                "quote": "unknown: Morgan submitted the request.",
                                "reason": "submission is an action",
                            }
                        ],
                        "required_ranges": [[2, 2]],
                    }
                if "忠实事件提取器" in system:
                    return {
                        "episodes": [
                            {
                                "text": "Morgan submitted the request.",
                                "participants": ["Morgan"],
                                "event_type": "submission",
                                "confidence": 0.9,
                                "evidence_spans": [[2, 2]],
                            }
                        ]
                    }
                raise AssertionError(system)

        source = "[record: 1]\nunknown: Morgan submitted the request."
        model = RescueModel()
        extractor = MemoryExtractor(model)
        audit = extractor.review_empty_episode(source)
        episodes, errors = extractor.extract_after_empty_episode_review(
            source, "main", audit
        )

        self.assertEqual(audit.verdict, "episode_required")
        self.assertEqual(errors, [])
        self.assertEqual([episode.text for episode in episodes], ["Morgan submitted the request."])
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(
            all(
                kwargs
                == {
                    "model": "independent-reviewer",
                    "allow_fallback": False,
                    "max_retries": 0,
                }
                for _system, kwargs in model.calls
            )
        )

    def test_source_scoped_plain_text_ignores_format_only_separators(self):
        episodes, errors = parse_source_scoped_episode_text(
            "Morgan提交了申请。\n\n---\n\nRiley批准了申请。",
            timeline_scope="main",
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [episode.text for episode in episodes],
            ["Morgan提交了申请。", "Riley批准了申请。"],
        )

    def test_source_scoped_profile_gives_model_only_prose_task(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            def chat_text(self, system, user, **_kwargs):
                self.calls.append((system, user))
                return "アユム提醒リン注意风险。\n\nリン决定继续准备。"

        model = NaturalTextModel()
        source = (
            "[record: 1]\n"
            "[speaker_raw: アユム]\n"
            "ja: リン先輩、注意してください。\n\n"
            "[record: 2]\n"
            "[speaker_raw: リン]\n"
            "ja: 分かりました。準備を続けます。"
        )

        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="source_scoped_plain",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(len(model.calls), 1)
        system, user = model.calls[0]
        self.assertLess(len(system), 500)
        self.assertNotIn("[record:", user)
        self.assertNotIn("[speaker_raw:", user)
        self.assertNotIn("[L000", user)
        self.assertNotIn("SPEAKER_", user)
        self.assertIn("アユム", user)
        self.assertIn("リン", user)
        self.assertEqual(episodes[0].participants, ["アユム", "リン"])
        self.assertEqual(episodes[1].participants, ["リン"])
        self.assertEqual(episodes[0].evidence_quotes, [])
        self.assertEqual(episodes[0].evidence_spans, [])
        self.assertTrue(
            all(episode.epistemic_status == "reported" for episode in episodes)
        )
        self.assertTrue(all(episode.generation == 0 for episode in episodes))

    def test_source_scoped_profile_omits_speaker_instruction_without_speakers(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            def __init__(self):
                self.calls: list[tuple[str, str]] = []

            def chat_text(self, system, user, **_kwargs):
                self.calls.append((system, user))
                return "两个备选回答彼此互斥，实际选择未记录。"

        model = NaturalTextModel()
        source = (
            "[record: 1]\n"
            "zh-CN: 以下两个备选回答属于同一选择组，实际选择未记录。"
        )

        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="source_scoped_plain",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes), 1)
        system, user = model.calls[0]
        self.assertNotIn("SPEAKER_", system)
        self.assertNotIn("SPEAKER_", user)
        self.assertEqual(
            episodes[0].text,
            "两个备选回答彼此互斥，实际选择未记录。",
        )
        self.assertEqual(episodes[0].evidence_origin, "source")
        self.assertEqual(episodes[0].epistemic_status, "speculative")
        self.assertEqual(episodes[0].generation, 0)
        self.assertIn("互斥备选", episodes[0].epistemic_note)

    def test_unresolved_alternative_floor_requires_both_explicit_signals(self):
        self.assertTrue(
            MemoryExtractor._source_declares_unresolved_alternatives(
                "The answers are mutually exclusive; the actual choice was not recorded."
            )
        )
        self.assertFalse(
            MemoryExtractor._source_declares_unresolved_alternatives(
                "The answers are mutually exclusive."
            )
        )
        self.assertFalse(
            MemoryExtractor._source_declares_unresolved_alternatives(
                "The actual choice was not recorded."
            )
        )

    def test_source_scoped_profile_falls_back_only_after_program_rejection(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            class Config:
                fallback_model = "fallback"

            config = Config()

            def __init__(self):
                self.models: list[str | None] = []

            def chat_text(self, _system, user, **kwargs):
                self.models.append(kwargs.get("model"))
                if kwargs.get("model") is None:
                    return "”的行动被打断。"
                return "アユム提醒リン注意风险。"

        model = NaturalTextModel()
        source = "[record: 1]\n[speaker_raw: アユム]\nja: リン先輩、注意してください。"

        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="source_scoped_plain",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.models, [None, "fallback"])
        self.assertEqual(episodes[0].text, "アユム提醒リン注意风险。")

    def test_source_scoped_rejects_only_unbound_leading_pronouns(self):
        errors = MemoryExtractor._source_scoped_self_containment_errors(
            [
                EpisodeDraft(text="她认为计划仍有风险。"),
                EpisodeDraft(
                    text="她认为计划仍有风险。",
                    participants=["Morgan"],
                ),
                EpisodeDraft(text="老师认为计划仍有风险。"),
            ]
        )

        self.assertEqual(
            errors,
            ["episode 0: unbound leading pronoun is not self-contained"],
        )

    def test_source_scoped_profile_splits_on_program_speaker_boundaries(self):
        drafts = [
            EpisodeDraft(
                text=(
                    "SPEAKER_001提出计划。SPEAKER_001解释原因。"
                    "SPEAKER_002表示反对。SPEAKER_002说明风险。"
                    "SPEAKER_003给出折中方案。"
                )
            )
        ]

        split = MemoryExtractor._split_source_scoped_episode_blocks(drafts)

        self.assertEqual(
            [draft.text for draft in split],
            [
                "SPEAKER_001提出计划。SPEAKER_001解释原因。",
                "SPEAKER_002表示反对。SPEAKER_002说明风险。",
                "SPEAKER_003给出折中方案。",
            ],
        )
        self.assertIsNot(split[0].participants, split[1].participants)

    def test_plain_episode_text_is_assembled_by_program(self):
        episodes, errors = parse_episode_text(
            "Morgan提交了申请。[证据：L0002-L0004]",
            timeline_scope="main",
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].text, "Morgan提交了申请。")
        self.assertEqual(episodes[0].participants, [])
        self.assertEqual(episodes[0].event_type, "")
        self.assertEqual(episodes[0].evidence_spans, [(2, 4)])
        self.assertEqual(episodes[0].timeline_scope, "main")

    def test_real_style_model_uses_plain_text_not_chat_json(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            def __init__(self):
                self.text_calls = 0

            def chat_text(self, _system, _user, **_kwargs):
                self.text_calls += 1
                return "SPEAKER_001提交了申请。[证据：L0001-L0001]"

            def chat_json(self, *_args, **_kwargs):
                raise AssertionError("natural extraction must not request JSON")

        model = NaturalTextModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes("Morgan: 提交了申请。", "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.text_calls, 1)
        self.assertEqual(episodes[0].text, "Morgan提交了申请。")
        self.assertEqual(episodes[0].participants, ["Morgan"])

    def test_natural_episode_retries_when_model_translates_a_speaker_marker(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            def __init__(self):
                self.calls = 0

            def chat_text(self, _system, _user, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return "阿梓提醒凛注意风险。[证据：L0001-L0003]"
                return "SPEAKER_001提醒凛注意风险。[证据：L0001-L0003]"

        model = NaturalTextModel()
        source = (
            "[record: 1]\n"
            "[speaker_raw: アユム]\n"
            "unknown: リン先輩、自分の立場が危うくなるかもしれません。"
        )

        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.calls, 2)
        self.assertEqual(episodes[0].text, "アユム提醒凛注意风险。")
        self.assertEqual(episodes[0].participants, ["アユム"])

    def test_natural_episode_uses_program_owned_speaker_tokens(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            def __init__(self):
                self.prompt = ""

            def chat_text(self, _system, prompt, **_kwargs):
                self.prompt = prompt
                return "SPEAKER_001はSPEAKER_002に報告した。[证据：L0001-L0006]"

        model = NaturalTextModel()
        source = (
            "[record: 1]\n"
            "[speaker_raw: アユム]\n"
            "unknown: リン先輩、報告があります。\n"
            "[record: 2]\n"
            "[speaker_raw: リン]\n"
            "unknown: 分かりました。"
        )

        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertNotIn("アユム", model.prompt)
        self.assertNotIn("リン先輩", model.prompt)
        self.assertIn("SPEAKER_001", model.prompt)
        self.assertIn("SPEAKER_002", model.prompt)
        self.assertEqual(episodes[0].text, "アユムはリンに報告した。")
        self.assertEqual(episodes[0].participants, ["アユム", "リン"])

    def test_adjacent_speaker_record_is_added_to_evidence_by_program(self):
        source_lines = [
            "[record: 1]",
            "[speaker_raw: SPEAKER_001]",
            "unknown: 決定しました。",
            "[record: 2]",
            "[speaker_raw: SPEAKER_002]",
            "unknown: あなたの決定には理由が必要です。",
        ]
        episode = EpisodeDraft(
            text="SPEAKER_002はSPEAKER_001に理由を求めた。",
            evidence_spans=[(4, 6)],
        )
        tokens = {"SPEAKER_001": "リン", "SPEAKER_002": "アオイ"}

        expanded = MemoryExtractor._expand_adjacent_speaker_evidence(
            source_lines, [episode], tokens
        )

        self.assertEqual(expanded, 1)
        self.assertEqual(episode.evidence_spans, [(1, 6)])
        self.assertEqual(
            MemoryExtractor._single_pass_speaker_token_errors(
                source_lines, [episode], tokens
            ),
            [],
        )

    def test_content_led_actor_is_programmatically_marked_reported(self):
        episode = EpisodeDraft(
            text="SPEAKER_002指示SPEAKER_001准备会议。",
            participants=[],
            evidence_quotes=[
                "[record: 1]\n"
                "[speaker_raw: アユム]\n"
                "unknown: リン先輩……。アユム、会议の準備を。"
            ],
            epistemic_status="asserted",
            confidence=0.9,
        )
        tokens = {"SPEAKER_001": "アユム", "SPEAKER_002": "リン"}

        MemoryExtractor._complete_participants_from_speaker_tokens([episode], tokens)
        MemoryExtractor._restore_speaker_tokens([episode], tokens)
        MemoryExtractor._apply_speaker_attribution_provenance([episode])

        self.assertEqual(episode.participants, ["リン", "アユム"])
        self.assertEqual(episode.epistemic_status, "reported")
        self.assertEqual(episode.generation, 0)
        self.assertEqual(episode.confidence, 0.8)
        self.assertIn("不是证据记录的声明发言者", episode.epistemic_note)

    def test_plain_text_extraction_drops_structural_front_matter(self):
        class NaturalTextModel:
            semantic_output_format = "natural_text"

            def chat_text(self, _system, _user, **_kwargs):
                return (
                    "本文整理自旧资料。[证据：L0001-L0004]\n\n"
                    "SPEAKER_001提交了申请。[证据：L0005-L0007]"
                )

        source = (
            "[record: 0]\n"
            "[document_role: front_matter]\n"
            "unknown: 本文整理自旧资料。\n\n"
            "[record: 1]\n"
            "[speaker_raw: Morgan]\n"
            "unknown: Morgan提交了申请。"
        )
        episodes, errors = MemoryExtractor(
            NaturalTextModel(),
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual([episode.text for episode in episodes], ["Morgan提交了申请。"])

    def test_curated_reference_records_are_persisted_verbatim_without_model(self):
        class NoModelCalls:
            semantic_output_format = "natural_text"

            def chat_text(self, *_args, **_kwargs):
                raise AssertionError("curated reference prose must not be summarised")

            def chat_json(self, *_args, **_kwargs):
                raise AssertionError("curated reference prose must not request JSON")

        source = (
            "[source_key: facts.txt]\n"
            "[segment_index: 0]\n\n"
            "[record: 2]\n"
            "[document_style: reference]\n"
            "unknown: 【官方剧情事实】\n"
            "委员会由 Morgan 负责。Dana 不是委员会成员。\n\n"
            "[record: 3]\n"
            "[evidence_origin: importer]\n"
            "[epistemic_status: speculative]\n"
            "[evidence_generation: 1]\n"
            "[epistemic_note: 导入文档标记为推测]\n"
            "[document_style: reference]\n"
            "unknown: 【推测（非官方定论）】\n"
            "Morgan 可能在此前见过 Dana。"
        )

        episodes, errors = MemoryExtractor(
            NoModelCalls(),
            episode_extraction_profile="adaptive_anchor_map",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(
            [episode.text for episode in episodes],
            [
                "委员会由 Morgan 负责。Dana 不是委员会成员。",
                "Morgan 可能在此前见过 Dana。",
            ],
        )
        self.assertEqual(episodes[0].epistemic_status, "asserted")
        self.assertEqual(episodes[1].evidence_origin, "importer")
        self.assertEqual(episodes[1].epistemic_status, "speculative")
        self.assertEqual(episodes[1].generation, 1)
        self.assertTrue(all(episode.evidence_quotes for episode in episodes))

    def test_plain_concept_text_is_assembled_by_program(self):
        concepts, errors = parse_concept_text(
            "阿罗娜：模型写出的额外解释。\n什亭之箱：模型写出的额外解释。",
            evidence_text=(
                "什亭之箱是老师持有的特殊终端。阿罗娜是什亭之箱的系统管理者。"
            ),
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [concept.canonical_name for concept in concepts],
            ["阿罗娜", "什亭之箱"],
        )
        self.assertEqual(concepts[0].aliases, [])
        self.assertEqual(concepts[0].description, "阿罗娜是什亭之箱的系统管理者。")
        self.assertNotIn("额外解释", concepts[0].embedding_text)

    def test_plain_concept_batch_uses_readable_episode_headings(self):
        groups, group_errors, errors = parse_concept_batch_text(
            "Episode 0\n阿罗娜：系统管理者。\n\nEpisode 1\n无",
            2,
        )

        self.assertEqual(errors, [])
        self.assertEqual(group_errors, {})
        self.assertEqual([item.canonical_name for item in groups[0]], ["阿罗娜"])
        self.assertEqual(groups[1], [])

    def test_program_attaches_only_literal_alias_legend_entries(self):
        concept = ConceptDraft(
            canonical_name="日奈",
            description="日奈参与当前事件。",
            embedding_text="日奈。日奈参与当前事件。",
        )

        MemoryExtractor._attach_literal_source_aliases(
            [concept],
            "히나: zh-CN=日奈 | en=Hina",
        )

        self.assertEqual(concept.aliases, [("히나", "unknown"), ("Hina", "en")])

    def test_program_derives_alias_legend_from_literal_speaker_label(self):
        context = MemoryExtractor._speaker_alias_context(
            "[record: 1]\n[speaker_raw: 先生（老师 / Sensei / 선생님）]\nunknown: ……？"
        )
        concept = ConceptDraft(
            canonical_name="先生",
            description="先生参与当前事件。",
            embedding_text="先生。先生参与当前事件。",
        )

        MemoryExtractor._attach_literal_source_aliases([concept], context)

        self.assertEqual(
            concept.aliases,
            [
                ("老师", "unknown"),
                ("Sensei", "unknown"),
                ("선생님", "unknown"),
            ],
        )

    def test_episode_draft_validates_transient_evidence_fields(self):
        draft = EpisodeDraft.from_dict(
            {
                "text": "Morgan accepted the request.",
                "evidence_quotes": [" Morgan accepted the request. "],
                "evidence_spans": [[1, 2]],
            }
        )

        self.assertEqual(draft.evidence_quotes, ["Morgan accepted the request."])
        self.assertEqual(draft.evidence_spans, [(1, 2)])
        flat_span = EpisodeDraft.from_dict(
            {
                "text": "Morgan accepted the request.",
                "evidence_spans": [1, 2],
            }
        )
        self.assertEqual(flat_span.evidence_spans, [(1, 2)])
        prefixed_span = EpisodeDraft.from_dict(
            {
                "text": "Morgan accepted the request.",
                "evidence_spans": [["L0001", "L0002"]],
            }
        )
        self.assertEqual(prefixed_span.evidence_spans, [(1, 2)])
        singleton_spans = EpisodeDraft.from_dict(
            {
                "text": "Morgan accepted the request.",
                "evidence_spans": [["L0009"], [10], [12]],
            }
        )
        self.assertEqual(singleton_spans.evidence_spans, [(9, 10), (12, 12)])
        with self.assertRaisesRegex(
            ValueError, "evidence_quotes must be a list of strings"
        ):
            EpisodeDraft.from_dict({"text": "invalid", "evidence_quotes": "not-a-list"})
        with self.assertRaisesRegex(
            ValueError, "evidence_spans must be a list of integer"
        ):
            EpisodeDraft.from_dict({"text": "invalid", "evidence_spans": [[1, "2"]]})

    def test_single_pass_accepts_literal_evidence_without_quality_audit(self):
        class EvidenceModel:
            def __init__(self):
                self.calls: list[str] = []

            def chat_json(self, system, user, **kwargs):
                self.calls.append(system)
                self.assert_no_legacy_audit(system)
                return {
                    "episodes": [
                        {
                            "text": "Morgan accepted the request.",
                            "participants": ["Morgan"],
                            "event_type": "acceptance",
                            "confidence": 0.9,
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

            @staticmethod
            def assert_no_legacy_audit(system):
                if "审计器" in system:
                    raise AssertionError("legacy audit must not run")

        model = EvidenceModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_audit_mode="combined",
            episode_audit_always=True,
            episode_factual_audit_mode="always",
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes("Morgan accepted the request.", "main")

        self.assertEqual(errors, [])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            episodes[0].evidence_quotes,
            ["Morgan accepted the request."],
        )

    def test_single_pass_retries_invalid_span_with_full_replacement(self):
        class RepairingEvidenceModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                span = [99, 99] if self.calls == 1 else [1, 1]
                if self.calls == 2:
                    self.last_retry_prompt = user
                return {
                    "episodes": [
                        {
                            "text": "Morgan accepted the request.",
                            "participants": ["Morgan"],
                            "event_type": "acceptance",
                            "evidence_spans": [span],
                        }
                    ]
                }

        model = RepairingEvidenceModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes("Morgan accepted the request.", "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.calls, 2)
        self.assertIn("没有通过确定性验证", model.last_retry_prompt)
        self.assertEqual(
            episodes[0].evidence_quotes,
            ["Morgan accepted the request."],
        )

    def test_single_pass_supplements_gap_without_replacing_good_episodes(self):
        source = "\n".join(f"角色甲: 关键事实{index}" for index in range(1, 11))

        class GapSupplementModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, _system, user, **_kwargs):
                self.calls += 1
                if "局部缺口补抽" in user:
                    return {
                        "episodes": [
                            {
                                "text": "角色甲说明中段事实。",
                                "participants": ["角色甲"],
                                "event_type": "middle",
                                "evidence_spans": [[3, 8]],
                            }
                        ]
                    }
                return {
                    "episodes": [
                        {
                            "text": "角色甲说明开端事实。",
                            "participants": ["角色甲"],
                            "event_type": "start",
                            "evidence_spans": [[1, 2]],
                        },
                        {
                            "text": "角色甲说明结尾事实。",
                            "participants": ["角色甲"],
                            "event_type": "end",
                            "evidence_spans": [[9, 10]],
                        },
                    ]
                }

        model = GapSupplementModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.calls, 2)
        self.assertEqual(
            [draft.event_type for draft in episodes],
            ["start", "middle", "end"],
        )

    def test_single_pass_removes_unreferenced_participant_metadata_noise(self):
        class OverInclusiveModel:
            def chat_json(self, _system, _user, **_kwargs):
                return {
                    "episodes": [
                        {
                            "text": "角色甲报告事件。",
                            "participants": ["角色甲", "未参与者"],
                            "event_type": "report",
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

        episodes, errors = MemoryExtractor(
            OverInclusiveModel(),
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes("角色甲: 事件已经发生。", "main")

        self.assertEqual(errors, [])
        self.assertEqual(episodes[0].participants, ["角色甲"])

    def test_single_pass_rejects_unrepairable_invalid_evidence_span(self):
        class InvalidEvidenceModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                return {
                    "episodes": [
                        {
                            "text": "Invented event.",
                            "participants": [],
                            "event_type": "invention",
                            "evidence_spans": [[99, 99]],
                        }
                    ]
                }

        model = InvalidEvidenceModel()
        with self.assertRaisesRegex(
            ValueError, "single-pass Episode extraction failed"
        ):
            MemoryExtractor(
                model,
                episode_extraction_profile="single_pass_evidence",
            ).extract_episodes("Morgan accepted the request.", "main")

        self.assertEqual(model.calls, 2)

    def test_single_pass_clamps_tiny_end_of_source_span_overflow(self):
        class EndOverflowModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                return {
                    "episodes": [
                        {
                            "text": "Fubuki accepted the assignment.",
                            "participants": ["Fubuki"],
                            "event_type": "acceptance",
                            "evidence_spans": [[2, 4]],
                        }
                    ]
                }

        model = EndOverflowModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(
            "Coordinator: assignment ready.\nFubuki: accepted.",
            "main",
        )

        self.assertEqual(errors, [])
        self.assertEqual(model.calls, 1)
        self.assertEqual(episodes[0].evidence_spans, [(2, 2)])
        self.assertEqual(episodes[0].evidence_quotes, ["Fubuki: accepted."])

    def test_single_pass_aligns_evidence_to_complete_normalized_record(self):
        source = """[record: 7]
[speaker_raw: Morgan]
unknown: Morgan paused.
The board then approved the request.
[record: 8]
[speaker_raw: River]
unknown: River acknowledged the decision."""

        class MidRecordSpanModel:
            def chat_json(self, _system, _user, **_kwargs):
                return {
                    "episodes": [
                        {
                            "text": "Morgan said the board approved the request.",
                            "participants": ["Morgan"],
                            "event_type": "approval",
                            "evidence_spans": [[4, 4]],
                        },
                        {
                            "text": "River acknowledged the decision.",
                            "participants": ["River"],
                            "event_type": "acknowledgement",
                            "evidence_spans": [[7, 7]],
                        },
                    ]
                }

        episodes, errors = MemoryExtractor(
            MidRecordSpanModel(),
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(episodes[0].evidence_spans, [(1, 4)])
        self.assertIn("[speaker_raw: Morgan]", episodes[0].evidence_quotes[0])

    def test_single_pass_accepts_several_short_evidence_spans(self):
        source = "\n".join(f"line {index} has evidence" for index in range(13))

        class ManyShortQuotesModel:
            def chat_json(self, system, user, **kwargs):
                return {
                    "episodes": [
                        {
                            "text": "The source records several related facts.",
                            "participants": [],
                            "event_type": "record",
                            "evidence_spans": [[index, index] for index in range(1, 9)],
                        }
                    ]
                }

        episodes, errors = MemoryExtractor(
            ManyShortQuotesModel(),
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes[0].evidence_quotes), 8)
        self.assertEqual(episodes[0].evidence_spans[-1], (8, 8))

    def test_single_pass_rejects_too_many_evidence_spans(self):
        source = "\n".join(f"line {index} has evidence" for index in range(17))

        class SourceCopyingModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                return {
                    "episodes": [
                        {
                            "text": "The source was copied instead of cited.",
                            "participants": [],
                            "event_type": "record",
                            "evidence_spans": [
                                [index, index] for index in range(1, 10)
                            ],
                        }
                    ]
                }

        model = SourceCopyingModel()
        with self.assertRaisesRegex(ValueError, "exceeds 8 ranges"):
            MemoryExtractor(
                model,
                episode_extraction_profile="single_pass_evidence",
            ).extract_episodes(source, "main")
        self.assertEqual(model.calls, 2)

    def test_single_pass_rejects_one_whole_document_span(self):
        source = "\n".join(f"line {index} has evidence" for index in range(1, 80))

        class WholeDocumentSpanModel:
            def chat_json(self, system, user, **kwargs):
                return {
                    "episodes": [
                        {
                            "text": "One summary claims the whole document.",
                            "participants": [],
                            "event_type": "oversized",
                            "evidence_spans": [[1, 79]],
                        }
                    ]
                }

        with self.assertRaisesRegex(ValueError, "exceeds 64 lines"):
            MemoryExtractor(
                WholeDocumentSpanModel(),
                episode_extraction_profile="single_pass_evidence",
            ).extract_episodes(source, "main")

    def test_document_map_builds_one_transient_context_per_segment(self):
        class MapModel:
            def chat_json(self, system, user, **kwargs):
                indexes = sorted(
                    {
                        int(value)
                        for value in re.findall(r'"segment_index"\s*:\s*(\d+)', user)
                    }
                )
                return {
                    "overview": "The document moves from setup to response.",
                    "segment_contexts": [
                        {
                            "segment_index": index,
                            "role_in_document": f"stage {index}",
                            "event_stages": [
                                {
                                    "stage_index": 0,
                                    "start_line": 1,
                                    "end_line": 1,
                                    "hint": f"event {index}",
                                    "time_mode": "current",
                                    "unresolved": [],
                                }
                            ],
                            "participants": ["Morgan"],
                            "timeline_notes": ["current"],
                            "unresolved": ["motive remains unresolved"],
                        }
                        for index in indexes
                    ],
                }

        contexts = MemoryExtractor(
            MapModel(),
            episode_extraction_profile="document_map_assisted",
        ).build_document_map(
            "story.txt",
            [(0, "Morgan arrived."), (1, "Morgan answered.")],
        )

        self.assertEqual(set(contexts), {0, 1})
        self.assertIn("setup to response", contexts[0])
        self.assertIn("motive remains unresolved", contexts[1])

    def test_document_map_rejects_missing_segment_contexts(self):
        class IncompleteMapModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                return {
                    "overview": "Incomplete map.",
                    "segment_contexts": [
                        {
                            "segment_index": 0,
                            "role_in_document": "only first stage",
                            "event_stages": [
                                {
                                    "stage_index": 0,
                                    "start_line": 1,
                                    "end_line": 1,
                                    "hint": "first event",
                                    "time_mode": "current",
                                    "unresolved": [],
                                }
                            ],
                            "participants": [],
                            "timeline_notes": [],
                            "unresolved": [],
                        }
                    ],
                }

        model = IncompleteMapModel()
        with self.assertRaisesRegex(ValueError, "document map failed"):
            MemoryExtractor(
                model,
                episode_extraction_profile="document_map_assisted",
            ).build_document_map(
                "story.txt",
                [(0, "First."), (1, "Second.")],
            )
        self.assertEqual(model.calls, 2)

    def test_adaptive_anchor_map_contains_only_literal_navigation(self):
        class AnchorModel:
            def chat_json(self, system, _user, **_kwargs):
                self.system = system
                return {
                    "segment_anchors": [
                        {
                            "segment_index": 0,
                            "role_in_document": "setup",
                            "time_mode": "current",
                            "anchor_terms": ["Morgan arrived"],
                            "participants": ["Morgan"],
                        },
                        {
                            "segment_index": 1,
                            "role_in_document": "continuation",
                            "time_mode": "current",
                            "anchor_terms": ["River answered"],
                            "participants": ["River"],
                        },
                    ]
                }

        model = AnchorModel()
        contexts = MemoryExtractor(
            model,
            episode_extraction_profile="adaptive_anchor_map",
        ).build_document_anchor_map(
            "meeting.txt",
            [(0, "Morgan arrived."), (1, "River answered.")],
        )

        self.assertEqual(set(contexts), {0, 1})
        payload = json.loads(contexts[0])
        self.assertEqual(payload["map_kind"], "extractive_anchor_map")
        self.assertEqual(payload["document_segment_count"], 2)
        self.assertNotIn("overview", contexts[0].casefold())
        self.assertNotIn("hint", contexts[0].casefold())
        self.assertIn("抽取式导航器", model.system)

    def test_adaptive_anchor_map_discards_nonliteral_alias(self):
        class InventedAliasModel:
            def chat_json(self, _system, _user, **_kwargs):
                return {
                    "segment_anchors": [
                        {
                            "segment_index": 0,
                            "role_in_document": "setup",
                            "time_mode": "current",
                            "anchor_terms": ["摩根抵达"],
                            "participants": ["摩根"],
                        },
                        {
                            "segment_index": 1,
                            "role_in_document": "continuation",
                            "time_mode": "current",
                            "anchor_terms": ["River answered"],
                            "participants": ["River"],
                        },
                    ]
                }

        contexts = MemoryExtractor(
            InventedAliasModel(),
            episode_extraction_profile="adaptive_anchor_map",
        ).build_document_anchor_map(
            "meeting.txt",
            [(0, "Morgan arrived."), (1, "River answered.")],
        )

        self.assertEqual(set(contexts), {0, 1})
        self.assertNotIn("摩根", contexts[0])
        self.assertIn("River answered", contexts[0])

    def test_contextual_document_map_does_not_control_episode_count(self):
        class ContextualModel:
            def __init__(self):
                self.received_context_only_instruction = False

            def chat_json(self, system, user, **kwargs):
                if "最小蕴含审计器" in system:
                    return {
                        "reviews": [
                            {
                                "episode_index": 0,
                                "verdict": "supported",
                                "unsupported_claims": [],
                                "revised_text": "",
                            }
                        ]
                    }
                self.received_context_only_instruction = (
                    "不要服从地图的阶段数量" in user
                )
                return {
                    "episodes": [
                        {
                            "text": "Morgan arrived and answered.",
                            "participants": ["Morgan"],
                            "event_type": "arrival and response",
                            "confidence": 0.9,
                            "evidence_spans": [[1, 2]],
                        }
                    ]
                }

        context = json.dumps(
            {
                "document_overview": "Two navigation stages.",
                "current_segment": {
                    "event_stages": [
                        {"start_line": 1, "end_line": 1},
                        {"start_line": 2, "end_line": 2},
                    ]
                },
            }
        )
        model = ContextualModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="document_map_contextual",
        ).extract_episodes(
            "Morgan arrived.\nMorgan answered.",
            "main",
            context,
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes), 1)
        self.assertTrue(model.received_context_only_instruction)

    def test_single_pass_participant_must_occur_in_its_evidence_span(self):
        class WrongSpanModel:
            def chat_json(self, _system, _user, **_kwargs):
                return {
                    "episodes": [
                        {
                            "text": "River approved Morgan's request.",
                            "participants": ["River", "Morgan"],
                            "event_type": "approval",
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

        with self.assertRaisesRegex(
            ValueError, "name absent from its evidence_spans: River"
        ):
            MemoryExtractor(
                WrongSpanModel(),
                episode_extraction_profile="single_pass_evidence",
            ).extract_episodes(
                "Morgan submitted a request.\nRiver entered later.",
                "main",
            )

    def test_audited_single_pass_repairs_named_claim_before_participant_retry(self):
        class ParticipantRepairModel:
            def __init__(self):
                self.extraction_calls = 0
                self.audit_calls = 0

            def chat_json(self, system, _user, **_kwargs):
                if "最小蕴含审计器" in system:
                    self.audit_calls += 1
                    return {
                        "reviews": [
                            {
                                "episode_index": 0,
                                "verdict": "revise",
                                "unsupported_claims": [
                                    "River is absent from the evidence"
                                ],
                                "revised_text": "Morgan submitted a request.",
                            }
                        ]
                    }
                self.extraction_calls += 1
                return {
                    "episodes": [
                        {
                            "text": "River approved Morgan's request.",
                            "participants": ["River", "Morgan"],
                            "event_type": "approval",
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

        model = ParticipantRepairModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_audited",
        ).extract_episodes("Morgan submitted a request.", "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.extraction_calls, 1)
        self.assertEqual(model.audit_calls, 1)
        self.assertEqual(episodes[0].text, "Morgan submitted a request.")
        self.assertEqual(episodes[0].participants, ["Morgan"])

    def test_audited_coverage_supplement_repairs_named_claim_without_full_retry(self):
        class SupplementRepairModel:
            def __init__(self):
                self.extraction_calls = 0
                self.supplement_calls = 0
                self.audit_calls = 0

            def chat_json(self, system, user, **_kwargs):
                if "最小蕴含审计器" in system:
                    self.audit_calls += 1
                    return {
                        "reviews": [
                            {
                                "episode_index": 0,
                                "verdict": "supported",
                                "unsupported_claims": [],
                                "revised_text": "",
                            },
                            {
                                "episode_index": 1,
                                "verdict": "revise",
                                "unsupported_claims": [
                                    "River is absent from the evidence"
                                ],
                                "revised_text": "Morgan recorded the later details.",
                            },
                        ]
                    }
                if "局部缺口补抽" in user:
                    self.supplement_calls += 1
                    return {
                        "episodes": [
                            {
                                "text": "River recorded Morgan's later details.",
                                "participants": ["River", "Morgan"],
                                "event_type": "record",
                                "evidence_spans": [[2, 7]],
                            }
                        ]
                    }
                self.extraction_calls += 1
                return {
                    "episodes": [
                        {
                            "text": "Morgan started the record.",
                            "participants": ["Morgan"],
                            "event_type": "record",
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

        model = SupplementRepairModel()
        source = "\n".join(
            ["Morgan started the record."]
            + [f"Morgan recorded later detail {index}." for index in range(2, 8)]
        )
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_audited",
        ).extract_episodes(source, "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.extraction_calls, 1)
        self.assertEqual(model.supplement_calls, 1)
        self.assertEqual(model.audit_calls, 1)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[1].text, "Morgan recorded the later details.")
        self.assertEqual(episodes[1].participants, ["Morgan"])

    def test_entailment_audit_receives_participants_for_alias_review(self):
        class AliasAuditModel:
            def __init__(self):
                self.audit_received_participants = False

            def chat_json(self, system, user, **_kwargs):
                if "最小蕴含审计器" in system:
                    self.audit_received_participants = (
                        '"participants": ["Ayumu"]' in user
                    )
                    return {
                        "reviews": [
                            {
                                "episode_index": 0,
                                "verdict": "revise",
                                "unsupported_claims": [
                                    "Aru is not the evidenced participant"
                                ],
                                "revised_text": "Ayumu agreed to help.",
                            }
                        ]
                    }
                return {
                    "episodes": [
                        {
                            "text": "Aru agreed to help.",
                            "participants": ["Ayumu"],
                            "event_type": "agreement",
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

        model = AliasAuditModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_audited",
        ).extract_episodes("Ayumu agreed to help.", "main")

        self.assertEqual(errors, [])
        self.assertTrue(model.audit_received_participants)
        self.assertEqual(episodes[0].text, "Ayumu agreed to help.")

    def test_single_pass_entailment_audit_weakens_unsupported_claim(self):
        class AuditedModel:
            def __init__(self):
                self.audit_calls = 0

            def chat_json(self, system, user, **kwargs):
                if "最小蕴含审计器" in system:
                    self.audit_calls += 1
                    return {
                        "reviews": [
                            {
                                "episode_index": 0,
                                "verdict": "revise",
                                "unsupported_claims": [
                                    "The evidence does not say Morgan accepted."
                                ],
                                "revised_text": (
                                    "Morgan said there was no other option."
                                ),
                            }
                        ]
                    }
                return {
                    "episodes": [
                        {
                            "text": "Morgan accepted the proposal.",
                            "participants": ["Morgan"],
                            "event_type": "decision",
                            "confidence": 0.95,
                            "evidence_spans": [[1, 1]],
                        }
                    ]
                }

        model = AuditedModel()
        episodes, errors = MemoryExtractor(
            model,
            episode_extraction_profile="single_pass_audited",
        ).extract_episodes("Morgan said there was no other option.", "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.audit_calls, 1)
        self.assertEqual(
            episodes[0].text,
            "Morgan said there was no other option.",
        )
        self.assertEqual(episodes[0].confidence, 0.85)

    def test_explicit_importer_theory_cannot_be_upgraded_by_model(self):
        class UpgradingModel:
            def chat_json(self, system, user, **kwargs):
                return {
                    "episodes": [
                        {
                            "text": "时间已经发生循环。",
                            "participants": [],
                            "event_type": "理论",
                            "confidence": 0.9,
                            "evidence_origin": "source",
                            "epistemic_status": "asserted",
                            "generation": 0,
                        }
                    ]
                }

        source = (
            "[record: 0]\n"
            "[evidence_origin: importer]\n"
            "[epistemic_status: speculative]\n"
            "[evidence_generation: 1]\n"
            "[epistemic_note: 导入文档标记为推测]\n"
            "unknown: 有人推测时间发生循环。"
        )

        episodes, errors = MemoryExtractor(
            UpgradingModel(), episode_audit_mode="off"
        ).extract_episodes(source, "theory")

        self.assertEqual(errors, [])
        self.assertEqual(episodes[0].evidence_origin, "importer")
        self.assertEqual(episodes[0].epistemic_status, "speculative")
        self.assertEqual(episodes[0].generation, 1)
        self.assertEqual(episodes[0].epistemic_note, "导入文档标记为推测")

    def test_mixed_reference_only_downgrades_the_inferential_record(self):
        class UpgradingModel:
            def chat_json(self, system, user, **kwargs):
                return {
                    "episodes": [
                        {
                            "text": "角色明确来到这里。",
                            "participants": ["角色"],
                            "event_type": "事实",
                            "evidence_origin": "source",
                            "epistemic_status": "asserted",
                        },
                        {
                            "text": "时间已经发生循环。",
                            "participants": [],
                            "event_type": "理论",
                            "evidence_origin": "source",
                            "epistemic_status": "asserted",
                        },
                    ]
                }

        source = (
            "[record: 0]\nunknown: [资料类型：官方剧情事实]\n角色明确来到这里。\n\n"
            "[record: 1]\n[evidence_origin: importer]\n"
            "[epistemic_status: speculative]\n[evidence_generation: 1]\n"
            "unknown: [资料类型：推测]\n有人推测时间发生循环。"
        )

        episodes, errors = MemoryExtractor(
            UpgradingModel(), episode_audit_mode="off"
        ).extract_episodes(source, "mixed-reference")

        self.assertEqual(errors, [])
        self.assertEqual(episodes[0].epistemic_status, "asserted")
        self.assertEqual(episodes[0].generation, 0)
        self.assertEqual(episodes[1].evidence_origin, "importer")
        self.assertEqual(episodes[1].epistemic_status, "speculative")
        self.assertEqual(episodes[1].generation, 1)

    def test_reference_document_coverage_is_repaired_before_acceptance(self):
        class CoverageModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                count = 2 if "覆盖不足" in user else 1
                return {
                    "episodes": [
                        {
                            "text": f"角色{index}的独立事实。",
                            "participants": [f"角色{index}"],
                            "event_type": "角色事实",
                            "confidence": 0.9,
                        }
                        for index in range(count)
                    ]
                }

        source = (
            "[record: 0]\nunknown: [资料类型：角色百科 | 实体：角色0]\n事实0\n\n"
            "[record: 1]\nunknown: [资料类型：角色百科 | 实体：角色1]\n事实1"
        )
        model = CoverageModel()
        extractor = MemoryExtractor(model, episode_audit_mode="off")

        episodes, errors = extractor.extract_episodes(source, "reference")

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes), 2)
        self.assertEqual(model.calls, 2)

    def test_reference_document_title_summary_is_not_persisted_as_episode(self):
        class TitleSummaryModel:
            def chat_json(self, system, user, **kwargs):
                return {
                    "episodes": [
                        {
                            "text": "角色百科文档发布。",
                            "participants": ["未知文档作者"],
                            "event_type": "文档发布",
                            "confidence": 1.0,
                        },
                        {
                            "text": "角色甲的独立事实。",
                            "participants": ["角色甲"],
                            "event_type": "角色事实",
                            "confidence": 0.9,
                        },
                        {
                            "text": "角色乙的独立事实。",
                            "participants": ["角色乙"],
                            "event_type": "角色事实",
                            "confidence": 0.9,
                        },
                    ]
                }

        source = (
            "[record: 0]\nunknown: 角色百科标题\n\n"
            "[record: 1]\nunknown: [资料类型：角色百科 | 实体：角色甲]\n事实甲\n\n"
            "[record: 2]\nunknown: [资料类型：角色百科 | 实体：角色乙]\n事实乙"
        )
        episodes, errors = MemoryExtractor(
            TitleSummaryModel(), episode_audit_mode="off"
        ).extract_episodes(source, "reference")

        self.assertEqual(errors, [])
        self.assertEqual(
            [episode.text for episode in episodes],
            [
                "角色甲的独立事实。",
                "角色乙的独立事实。",
            ],
        )

    def test_reference_coverage_retry_cannot_reintroduce_title_episode(self):
        class CoverageRetryModel:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "episodes": [{"text": "事实甲。", "event_type": "角色事实"}]
                    }
                return {
                    "episodes": [
                        {
                            "text": "《角色百科》作为作品标题被正式确认。",
                            "event_type": "作品标题确认",
                        },
                        {"text": "事实甲。", "event_type": "角色事实"},
                        {"text": "事实乙。", "event_type": "角色事实"},
                    ]
                }

        source = (
            "[record: 0]\nunknown: 角色百科标题\n\n"
            "[record: 1]\nunknown: [资料类型：角色百科]\n事实甲\n\n"
            "[record: 2]\nunknown: [资料类型：角色百科]\n事实乙"
        )
        model = CoverageRetryModel()
        episodes, errors = MemoryExtractor(
            model, episode_audit_mode="off"
        ).extract_episodes(source, "reference")

        self.assertEqual(errors, [])
        self.assertEqual(model.calls, 2)
        self.assertEqual(
            [episode.text for episode in episodes],
            [
                "事实甲。",
                "事实乙。",
            ],
        )

    def test_single_pass_reference_coverage_ignores_unlabelled_preamble(self):
        source = (
            "[record: 0]\nunknown: This document was整理自 an earlier file.\n\n"
            "[record: 1]\nunknown: [资料类型：角色百科 | 实体：角色甲]\n事实甲\n\n"
            "[record: 2]\nunknown: [资料类型：角色百科 | 实体：角色乙]\n事实乙"
        )

        class ReferenceSinglePassModel:
            def chat_json(self, _system, _user, **_kwargs):
                return {
                    "episodes": [
                        {
                            "text": "文档整理来源说明。",
                            "event_type": "文档声明与说明",
                            "evidence_spans": [[1, 2]],
                        },
                        {
                            "text": "角色甲的事实。",
                            "participants": ["角色甲"],
                            "event_type": "角色事实",
                            "evidence_spans": [[3, 5]],
                        },
                        {
                            "text": "角色乙的事实。",
                            "participants": ["角色乙"],
                            "event_type": "角色事实",
                            "evidence_spans": [[6, 8]],
                        },
                    ]
                }

        episodes, errors = MemoryExtractor(
            ReferenceSinglePassModel(),
            episode_extraction_profile="single_pass_evidence",
        ).extract_episodes(source, "reference")

        self.assertEqual(errors, [])
        self.assertEqual(
            [episode.text for episode in episodes],
            ["角色甲的事实。", "角色乙的事实。"],
        )

    def test_chat_json_uses_fallback_after_unrepairable_primary_output(self):
        class InvalidPrimaryClient(ModelClient):
            def __init__(self):
                super().__init__(
                    ModelConfig(
                        api_key="test",
                        reasoning_model="primary",
                        fallback_model="fallback",
                        max_retries=0,
                    )
                )
                self.calls: list[str] = []

            def _chat_once(self, system, user, model, temperature=0.1):
                self.calls.append(model)
                if model == "primary":
                    return "not valid json"
                return '{"status":"ok"}'

        client = InvalidPrimaryClient()

        self.assertEqual({"status": "ok"}, client.chat_json("system", "user"))
        self.assertEqual(["primary", "primary", "fallback"], client.calls)

    def test_chat_json_can_disable_fallback_for_invalid_output(self):
        class InvalidClient(ModelClient):
            def __init__(self):
                super().__init__(
                    ModelConfig(
                        api_key="test",
                        reasoning_model="primary",
                        fallback_model="fallback",
                        max_retries=0,
                    )
                )
                self.calls: list[str] = []

            def _chat_once(self, system, user, model, temperature=0.1):
                self.calls.append(model)
                return "not valid json"

        client = InvalidClient()

        with self.assertRaisesRegex(ModelClientError, "invalid JSON after repair"):
            client.chat_json("system", "user", allow_fallback=False)
        self.assertEqual(["primary", "primary"], client.calls)

    def test_chat_json_repairs_bare_line_ids_locally(self):
        class BareLineIdClient(ModelClient):
            def __init__(self):
                super().__init__(
                    ModelConfig(
                        api_key="test",
                        reasoning_model="primary",
                        fallback_model="fallback",
                        max_retries=0,
                    )
                )
                self.calls = 0

            def _chat_once(self, system, user, model, temperature=0.1):
                self.calls += 1
                return (
                    'prefix {"note":"keep [L0009] as text",'
                    '"evidence_spans":[[L0005,L0008]]} suffix'
                )

        client = BareLineIdClient()

        self.assertEqual(
            {
                "note": "keep [L0009] as text",
                "evidence_spans": [["L0005", "L0008"]],
            },
            client.chat_json("system", "user", allow_fallback=False),
        )
        self.assertEqual(1, client.calls)

    def test_dedicated_reranker_preserves_provider_order_and_scores(self):
        class CapturingClient(ModelClient):
            def _post(self, endpoint, payload):
                self.endpoint = endpoint
                self.payload = payload
                return {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.42},
                    ]
                }

        client = CapturingClient(
            ModelConfig(
                api_key="test",
                reranker_model="Pro/BAAI/bge-reranker-v2-m3",
            )
        )
        ranked = client.rerank("谁发动政变？", ["渚开会", "未花发动政变"])

        self.assertEqual("rerank", client.endpoint)
        self.assertEqual(
            "Pro/BAAI/bge-reranker-v2-m3",
            client.payload["model"],
        )
        self.assertFalse(client.payload["return_documents"])
        self.assertEqual(
            [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
            ],
            ranked,
        )

    def test_reranker_does_not_repeat_a_nonretryable_provider_error(self):
        class RejectingClient(ModelClient):
            def __init__(self):
                super().__init__(ModelConfig(api_key="test", max_retries=3))
                self.calls = 0

            def _post(self, endpoint, payload):
                self.calls += 1
                raise ModelClientError("invalid request", retryable=False)

        client = RejectingClient()

        with self.assertRaisesRegex(ModelClientError, "invalid request"):
            client.rerank("问题", ["候选"])
        self.assertEqual(client.calls, 1)

    def test_chat_request_can_bound_output_and_disable_thinking(self):
        class CapturingClient(ModelClient):
            def _post(self, endpoint, payload):
                self.endpoint = endpoint
                self.payload = payload
                return {"choices": [{"message": {"content": "ok"}}]}

        client = CapturingClient(
            ModelConfig(
                api_key="test",
                reasoning_max_tokens=2400,
                reasoning_enable_thinking=False,
            )
        )
        self.assertEqual("ok", client.chat_text("system", "user"))
        self.assertEqual("chat/completions", client.endpoint)
        self.assertEqual(2400, client.payload["max_tokens"])
        self.assertFalse(client.payload["enable_thinking"])

    def test_truncated_chat_skips_same_model_retries_and_uses_reasoning_fallback(self):
        class TruncatedPrimaryClient(ModelClient):
            def __init__(self):
                super().__init__(
                    ModelConfig(
                        api_key="test",
                        reasoning_model="primary",
                        fallback_model="fallback",
                        max_retries=3,
                    )
                )
                self.models: list[str] = []

            def _post(self, endpoint, payload):
                self.models.append(payload["model"])
                if payload["model"] == "primary":
                    return {
                        "choices": [
                            {
                                "message": {"content": '{"episodes": ['},
                                "finish_reason": "length",
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "message": {"content": '{"status":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ]
                }

        client = TruncatedPrimaryClient()

        self.assertEqual({"status": "ok"}, client.chat_json("system", "user"))
        self.assertEqual(["primary", "fallback"], client.models)

    def test_chat_retry_limit_can_be_reduced_for_oversized_batch_calls(self):
        class TimingOutClient(ModelClient):
            def __init__(self):
                super().__init__(
                    ModelConfig(
                        api_key="test",
                        reasoning_model="primary",
                        fallback_model="primary",
                        max_retries=3,
                    )
                )
                self.calls = 0

            def _chat_once(self, system, user, model, temperature=0.1):
                self.calls += 1
                raise ModelClientError("timed out")

        client = TimingOutClient()

        with self.assertRaisesRegex(ModelClientError, "timed out"):
            client.chat_json(
                "system",
                "user",
                allow_fallback=False,
                max_retries=0,
            )
        self.assertEqual(client.calls, 1)

    def test_optimization_profile_can_restore_historical_strict_path(self):
        config = AppConfig()
        self.assertEqual(config.optimization_profile, "balanced")
        self.assertEqual(config.ingestion.episode_audit_mode, "combined")
        self.assertEqual(config.retrieval.rerank_review_mode, "adaptive")

        config.apply_optimization_profile("strict")

        self.assertEqual(config.ingestion.episode_audit_mode, "split")
        self.assertEqual(config.ingestion.second_pass_context_mode, "full")
        self.assertFalse(config.concept_extraction.promotion_prefilter_enabled)
        self.assertEqual(config.retrieval.rerank_review_mode, "strict")

    def test_import_concurrency_and_relation_batch_can_be_overridden(self):
        with patch.dict(
            "os.environ",
            {
                "MEMORY_IMPORT_PREPARE_WORKERS": "8",
                "MEMORY_IMPORT_RELATION_WORKERS": "4",
                "MEMORY_IMPORT_RELATION_BATCH_SIZE": "12",
            },
            clear=True,
        ):
            config = AppConfig.from_env("missing-test.env")

        self.assertEqual(config.ingestion.prepare_workers, 8)
        self.assertEqual(config.ingestion.relation_workers, 4)
        self.assertEqual(config.ingestion.relation_batch_size, 12)

    def test_model_timeout_bounds_stalled_import_requests(self):
        self.assertEqual(ModelConfig().timeout_seconds, 90.0)
        self.assertEqual(ModelConfig().max_concurrent_requests, 8)
        with patch.dict(
            "os.environ",
            {
                "MEMORY_MODEL_TIMEOUT_SECONDS": "45",
                "MEMORY_MODEL_MAX_CONCURRENT_REQUESTS": "6",
            },
            clear=True,
        ):
            config = AppConfig.from_env("missing-test.env")

        self.assertEqual(config.model.timeout_seconds, 45.0)
        self.assertEqual(config.model.max_concurrent_requests, 6)

    def test_reasoning_and_fallback_models_can_be_overridden_from_environment(self):
        with patch.dict(
            "os.environ",
            {
                "MEMORY_REASONING_MODEL": "zai-org/GLM-4.5-Air",
                "MEMORY_FALLBACK_MODEL": "zai-org/GLM-4.5-Air",
            },
            clear=True,
        ):
            config = AppConfig.from_env("missing-test.env")

        self.assertEqual(config.model.reasoning_model, "zai-org/GLM-4.5-Air")
        self.assertEqual(config.model.fallback_model, "zai-org/GLM-4.5-Air")

    def test_source_segment_budget_can_be_overridden_safely(self):
        with patch.dict(
            "os.environ",
            {
                "MEMORY_SEGMENT_TARGET_CHARS": "4000",
                "MEMORY_SEGMENT_MAX_CHARS": "5000",
                "MEMORY_SEGMENT_OVERLAP_CHARS": "400",
            },
            clear=True,
        ):
            config = AppConfig.from_env("missing-test.env")

        self.assertEqual(config.segment.target_chars, 4000)
        self.assertEqual(config.segment.max_chars, 5000)
        self.assertEqual(config.segment.overlap_chars, 400)

        with patch.dict(
            "os.environ",
            {
                "MEMORY_SEGMENT_TARGET_CHARS": "6000",
                "MEMORY_SEGMENT_MAX_CHARS": "5000",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                AppConfig.from_env("missing-test.env")

    def test_empty_episode_audit_is_configurable_from_environment(self):
        with patch.dict(
            "os.environ",
            {
                "MEMORY_IMPORT_EMPTY_EPISODE_AUDIT_MODE": "off",
                "MEMORY_IMPORT_EMPTY_EPISODE_AUDIT_MODEL": "independent-reviewer",
            },
            clear=True,
        ):
            config = AppConfig.from_env("missing-test.env")

        self.assertEqual(config.ingestion.empty_episode_audit_mode, "off")
        self.assertEqual(
            config.ingestion.empty_episode_audit_model, "independent-reviewer"
        )
        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EMPTY_EPISODE_AUDIT_MODE": "invalid"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "off or adversarial"):
                AppConfig.from_env("missing-test.env")

    def test_episode_extraction_profile_is_reversible_from_environment(self):
        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EPISODE_PROFILE": "single_pass_evidence"},
            clear=True,
        ):
            experimental = AppConfig.from_env("missing-test.env")

        self.assertEqual(
            experimental.ingestion.episode_extraction_profile,
            "single_pass_evidence",
        )
        self.assertEqual(experimental.prompt_version, "v4.1_single_pass_line_spans")

        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EPISODE_PROFILE": "legacy"},
            clear=True,
        ):
            legacy = AppConfig.from_env("missing-test.env")

        self.assertEqual(legacy.ingestion.episode_extraction_profile, "legacy")
        self.assertEqual(legacy.prompt_version, "v3.45_structural_grounded_roles")

        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EPISODE_PROFILE": "document_map_assisted"},
            clear=True,
        ):
            mapped = AppConfig.from_env("missing-test.env")

        self.assertEqual(
            mapped.ingestion.episode_extraction_profile,
            "document_map_assisted",
        )
        self.assertEqual(mapped.prompt_version, "v4.5_document_map_planned_audited")

        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EPISODE_PROFILE": "document_map_contextual"},
            clear=True,
        ):
            contextual = AppConfig.from_env("missing-test.env")

        self.assertEqual(
            contextual.ingestion.episode_extraction_profile,
            "document_map_contextual",
        )
        self.assertEqual(contextual.prompt_version, "v4.6_document_map_context_audited")

        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EPISODE_PROFILE": "adaptive_anchor_map"},
            clear=True,
        ):
            anchored = AppConfig.from_env("missing-test.env")

        self.assertEqual(
            anchored.ingestion.episode_extraction_profile,
            "adaptive_anchor_map",
        )
        self.assertEqual(
            anchored.prompt_version,
            "v4.7_adaptive_literal_anchor_audited",
        )

        with patch.dict(
            "os.environ",
            {"MEMORY_IMPORT_EPISODE_PROFILE": "single_pass_audited"},
            clear=True,
        ):
            audited = AppConfig.from_env("missing-test.env")

        self.assertEqual(
            audited.ingestion.episode_extraction_profile,
            "single_pass_audited",
        )
        self.assertEqual(audited.prompt_version, "v4.4_single_pass_entailment_roles")

    def test_ready_relation_batches_are_judged_concurrently_and_written_once(self):
        barrier = Barrier(2, timeout=2.0)
        lock = Lock()
        judge_thread_ids: set[int] = set()
        writer_thread_ids: set[int] = set()
        stored: list[AssociationDraft] = []
        caller_thread_id = get_ident()

        class Builder:
            def judge_new_concept_batches(self, jobs):
                with lock:
                    judge_thread_ids.add(get_ident())
                barrier.wait()
                concept_id = jobs[0][0]
                return [
                    AssociationDraft(
                        "concept",
                        concept_id,
                        "concept",
                        concept_id + 100,
                        "semantic",
                        "related_to",
                        "并发判断结果",
                    )
                ]

            def judge_episode_batches(self, _groups):
                raise AssertionError("episode path is not used by this test")

            @staticmethod
            def relation_draft_sort_key(draft):
                return draft.from_id

            def store_relation_drafts(self, drafts):
                writer_thread_ids.add(get_ident())
                stored.extend(drafts)

        pipeline = object.__new__(ImportPipeline)
        pipeline.build_inference_relations = True
        pipeline.relation_batch_size = 2
        pipeline.relation_workers = 2
        pipeline.builder = Builder()
        pipeline.logger = None
        concept_jobs = [
            (
                index,
                ConceptDraft(
                    canonical_name=f"概念{index}",
                    description="测试",
                    embedding_text=f"概念{index}",
                    aliases=[],
                    confidence=0.9,
                ),
                [],
            )
            for index in range(4)
        ]

        pipeline._flush_relation_batches(concept_jobs, [], force=False)

        self.assertEqual(concept_jobs, [])
        self.assertEqual(len(judge_thread_ids), 2)
        self.assertNotIn(caller_thread_id, judge_thread_ids)
        self.assertEqual(writer_thread_ids, {caller_thread_id})
        self.assertEqual([draft.from_id for draft in stored], [0, 2])

    def test_persistent_relation_pool_overlaps_import_until_forced_drain(self):
        release = Event()
        stored: list[AssociationDraft] = []
        writer_thread_ids: set[int] = set()
        caller_thread_id = get_ident()

        class Builder:
            def judge_new_concept_batches(self, jobs):
                if not release.wait(timeout=2.0):
                    raise TimeoutError("test relation worker was not released")
                concept_id = jobs[0][0]
                return [
                    AssociationDraft(
                        "concept",
                        concept_id,
                        "concept",
                        concept_id + 100,
                        "semantic",
                        "related_to",
                        "流水化判断结果",
                    )
                ]

            def judge_episode_batches(self, _groups):
                raise AssertionError("episode path is not used by this test")

            @staticmethod
            def relation_draft_sort_key(draft):
                return draft.from_id

            def store_relation_drafts(self, drafts):
                writer_thread_ids.add(get_ident())
                stored.extend(drafts)

        pipeline = object.__new__(ImportPipeline)
        pipeline.build_inference_relations = True
        pipeline.relation_batch_size = 2
        pipeline.relation_workers = 2
        pipeline.builder = Builder()
        pipeline.logger = None
        concept_jobs = [
            (
                index,
                ConceptDraft(
                    canonical_name=f"概念{index}",
                    description="测试",
                    embedding_text=f"概念{index}",
                    aliases=[],
                    confidence=0.9,
                ),
                [],
            )
            for index in range(4)
        ]
        pending = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            pipeline._flush_relation_batches(
                concept_jobs,
                [],
                force=False,
                executor=executor,
                pending_futures=pending,
            )
            self.assertEqual(concept_jobs, [])
            self.assertEqual(len(pending), 2)
            self.assertEqual(stored, [])

            release.set()
            pipeline._flush_relation_batches(
                [],
                [],
                force=True,
                executor=executor,
                pending_futures=pending,
            )

        self.assertEqual(pending, [])
        self.assertEqual([draft.from_id for draft in stored], [0, 2])
        self.assertEqual(writer_thread_ids, {caller_thread_id})

    def test_remote_disconnect_is_wrapped_for_normal_retry_handling(self):
        client = ModelClient(ModelConfig(api_key="test-key"))

        with patch(
            "memory_demo.llm.client.requests.Session.post",
            side_effect=requests.ConnectionError("remote closed"),
        ):
            with self.assertRaisesRegex(ModelClientError, "remote closed"):
                client._post("chat/completions", {"model": "test"})

    def test_model_client_limits_active_http_requests(self):
        client = ModelClient(
            ModelConfig(api_key="test-key", max_concurrent_requests=2)
        )
        active = 0
        peak_active = 0
        active_lock = Lock()
        two_requests_started = Event()
        release_requests = Event()

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {}

            def close(self):
                return None

        def delayed_post(*_args, **_kwargs):
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
                if active == 2:
                    two_requests_started.set()
            try:
                self.assertTrue(release_requests.wait(timeout=2))
                return FakeResponse()
            finally:
                with active_lock:
                    active -= 1

        with patch(
            "memory_demo.llm.client.requests.Session.post", side_effect=delayed_post
        ):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(client._post, "chat/completions", {"model": "test"})
                    for _ in range(4)
                ]
                self.assertTrue(two_requests_started.wait(timeout=1))
                self.assertEqual(peak_active, 2)
                release_requests.set()
                for future in futures:
                    self.assertEqual(future.result(timeout=2), {})

    def test_new_https_connections_are_established_one_at_a_time(self):
        client = ModelClient(ModelConfig(api_key="test-key"))
        self.assertIs(
            client._http_adapter.poolmanager.pool_classes_by_scheme["https"],
            _SerializedHTTPSConnectionPool,
        )
        active = 0
        peak_active = 0
        active_lock = Lock()
        first_connection_started = Event()
        release_connections = Event()

        def delayed_connect(_connection):
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
                first_connection_started.set()
            try:
                self.assertTrue(release_connections.wait(timeout=2))
            finally:
                with active_lock:
                    active -= 1

        with patch.object(HTTPSConnection, "connect", new=delayed_connect):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        _SerializedHTTPSConnection(host="example.test").connect
                    )
                    for _ in range(2)
                ]
                self.assertTrue(first_connection_started.wait(timeout=1))
                self.assertEqual(peak_active, 1)
                release_connections.set()
                for future in futures:
                    self.assertIsNone(future.result(timeout=2))

    def test_socket_access_denial_opens_one_shared_transport_circuit(self):
        client = ModelClient(ModelConfig(api_key="test-key", max_retries=6))
        error = requests.ConnectionError("[WinError 10013] socket access denied")
        with patch(
            "memory_demo.llm.client.requests.Session.post", side_effect=error
        ) as post:
            with self.assertRaises(ModelTransportUnavailable):
                client.chat_text("system", "user", allow_fallback=True)
            # The primary retry loop and fallback model must not create more
            # sockets after the local transport circuit opens.
            self.assertEqual(post.call_count, 1)
            with self.assertRaises(ModelTransportUnavailable):
                client.chat_text("system", "user", allow_fallback=True)
            self.assertEqual(post.call_count, 1)

    def test_retryable_run_cleanup_removes_only_its_source_and_metadata(self):
        with TemporaryDirectory() as directory:
            database = Database(Path(directory) / "memory.db")
            database.initialize()
            extractions = ExtractionRepository(database)
            sources = SourceRepository(database)
            run_id = extractions.start_run({}, {}, "test.log")
            source_id = sources.insert("temporary source")
            task_id = extractions.start_task(
                run_id,
                "favor/test.json",
                0,
                "pass1",
                "test-model",
                "test-prompt",
            )
            extractions.set_source(task_id, source_id)

            removed = extractions.discard_incomplete_run(run_id)

            self.assertEqual(removed["deleted_run"], 1)
            self.assertEqual(removed["deleted_tasks"], 1)
            self.assertEqual(removed["deleted_sources"], 1)
            self.assertEqual(sources.count(), 0)
            with database.connection() as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM extraction_run").fetchone()[0],
                    0,
                )
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    @staticmethod
    def _episode(text: str, event_type: str = "主要事件") -> EpisodeDraft:
        return EpisodeDraft(
            text=text,
            participants=["角色"],
            event_type=event_type,
            location_text="",
            story_time_text="",
            timeline_scope="main",
            confidence=0.9,
        )

    def test_granularity_audit_trigger_detects_fragmentation_and_identity_guess(self):
        extractor = MemoryExtractor(model=None)
        fragmented = [
            self._episode(f"角色第{index}次点头。", "点头反应") for index in range(10)
        ]
        self.assertTrue(extractor._needs_granularity_audit(fragmented))
        self.assertTrue(
            extractor._needs_granularity_audit(
                [self._episode("未标注发言者（可能是老师）提出建议。")]
            )
        )
        self.assertFalse(
            extractor._needs_granularity_audit(
                [self._episode("角色们围绕同一目标完成了一段连续行动。")]
            )
        )

    def test_combined_episode_audit_handles_temporal_and_granularity_in_one_call(self):
        class AuditModel:
            def __init__(self):
                self.audit_calls = 0

            def chat_json(self, system, _user, **_kwargs):
                if "事实提取器" in system:
                    return {
                        "episodes": [
                            {
                                "text": "未标注发言者（可能是老师）提到过去发生过冲突。",
                                "participants": ["未标注发言者（可能是老师）"],
                                "event_type": "回忆",
                                "story_time_text": "",
                                "confidence": 0.8,
                            }
                        ]
                    }
                if "边界与证据审计器" in system:
                    self.audit_calls += 1
                    return {
                        "episodes": [
                            {
                                "text": "据未标注发言者说法，过去曾发生冲突。",
                                "participants": ["未标注发言者"],
                                "event_type": "过去冲突",
                                "story_time_text": "过去，据未标注发言者说法",
                                "confidence": 0.8,
                            }
                        ]
                    }
                raise AssertionError(system)

        model = AuditModel()
        extractor = MemoryExtractor(model, episode_audit_mode="combined")
        episodes, errors = extractor.extract_episodes("过去发生过冲突。", "main")

        self.assertEqual(errors, [])
        self.assertEqual(model.audit_calls, 1)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].participants, ["未标注发言者"])

    def test_unknown_speaker_identity_guess_is_removed_deterministically(self):
        extractor = MemoryExtractor(model=None)
        draft = EpisodeDraft(
            text="未标注发言者（根据上下文，应为日富美）表示仍然生气。",
            participants=["未标注发言者（可能是日富美）", "纱织"],
            event_type="争论",
        )
        self.assertTrue(extractor._needs_granularity_audit([draft]))
        sanitized = MemoryExtractor.sanitize_episode_draft(draft)
        self.assertEqual(sanitized.text, "未标注发言者表示仍然生气。")
        self.assertEqual(sanitized.participants, ["未标注发言者", "纱织"])

    def test_generic_current_story_time_is_removed_deterministically(self):
        current = self._episode("角色在当前现场讨论行动。")
        current.story_time_text = "当前故事时间"
        past = self._episode("据角色回忆，过去发生过冲突。")
        past.story_time_text = "过去，据角色回忆"

        self.assertEqual(
            MemoryExtractor.sanitize_episode_draft(current).story_time_text,
            "",
        )
        self.assertEqual(
            MemoryExtractor.sanitize_episode_draft(past).story_time_text,
            "过去，据角色回忆",
        )

    def test_source_grounded_foreign_name_typos_are_repaired(self):
        source = """[record: 1]
[speaker_raw: ???]
unknown: --- title ---
フランシス: 宣告开始。
カイザーの特殊部隊员A: 报告位置。
"""
        episodes = [
            EpisodeDraft(
                text="フランシ斯发表宣告，カイザ尔部队报告位置。",
                participants=["フランシス"],
                location_text="カイザ尔据点",
            )
        ]

        MemoryExtractor._repair_source_name_typos(source, episodes)

        self.assertEqual(
            episodes[0].text,
            "フランシス发表宣告，カイザー部队报告位置。",
        )
        self.assertEqual(episodes[0].location_text, "カイザー据点")

    def test_source_name_repair_does_not_guess_chinese_translation(self):
        source = """[record: 1]
クズノハ: 原文中的名字。
"""
        episodes = [EpisodeDraft(text="黑野提出建议。", participants=["黑野"])]

        MemoryExtractor._repair_source_name_typos(source, episodes)

        self.assertEqual(episodes[0].text, "黑野提出建议。")
        self.assertEqual(episodes[0].participants, ["黑野"])

    def test_source_name_repair_does_not_fuzzy_replace_same_script_word(self):
        source = """[record: 1]
アロナ: 原文中的名字。
"""
        episodes = [EpisodeDraft(text="角色提到了アロハ衬衫。")]

        MemoryExtractor._repair_source_name_typos(source, episodes)

        self.assertEqual(episodes[0].text, "角色提到了アロハ衬衫。")

    def test_source_name_repair_handles_kana_hangul_substitution(self):
        source = """[record: 1]
ノノミ: 原文中的名字。
"""
        episodes = [EpisodeDraft(text="ノノ미参与了讨论。")]

        MemoryExtractor._repair_source_name_typos(source, episodes)

        self.assertEqual(episodes[0].text, "ノノミ参与了讨论。")

    def test_source_name_repair_does_not_cross_prose_token_boundaries(self):
        source = """ミカ: 作战に参加する。\nウミカ: 別の場所で待機する。\n"""
        episodes = [
            EpisodeDraft(
                text="ナギサ征求ミカ的同意，ヒマリ解释了情况。",
                participants=["ミカ", "ヒマリ"],
            )
        ]

        MemoryExtractor._repair_source_name_typos(source, episodes)

        self.assertEqual(
            episodes[0].text,
            "ナギサ征求ミカ的同意，ヒマリ解释了情况。",
        )

    def test_single_pass_repairs_mixed_script_speaker_copy_in_own_evidence(self):
        draft = EpisodeDraft(
            text="ツバ키负责避难。",
            participants=["ツバ키(椿/Tsubaki/츠바키)"],
            evidence_quotes=["ツバキ(椿/Tsubaki/츠바키): 避难を担当する。"],
            evidence_spans=[(1, 1)],
        )

        MemoryExtractor._repair_single_pass_evidence_name_typos([draft])

        self.assertEqual(draft.text, "ツバキ负责避难。")
        self.assertEqual(
            draft.participants,
            ["ツバキ(椿/Tsubaki/츠바키)"],
        )
        self.assertEqual(
            MemoryExtractor._single_pass_participant_evidence_errors([draft]),
            [],
        )

    def test_source_name_repair_reads_normalized_speaker_and_body_tokens(self):
        source = """[record: 4]
[speaker_raw: アユム]
unknown: アユムとモモカちゃんが復旧作業に参加します。
"""
        draft = EpisodeDraft(
            text="アユ姆与モモ卡将参与复旧作业。",
            participants=["アユ姆"],
            evidence_quotes=[source],
            evidence_spans=[(1, 3)],
        )

        MemoryExtractor._repair_single_pass_evidence_name_typos([draft])

        self.assertEqual(draft.text, "アユム与モモカ将参与复旧作业。")
        self.assertEqual(draft.participants, ["アユム"])

    def test_single_pass_coverage_rejects_sustained_uncovered_dialogue(self):
        source_lines = [f"角色: 关键事实{index}" for index in range(1, 11)]
        drafts = [
            EpisodeDraft(
                text="开端事实",
                evidence_spans=[(1, 2)],
            ),
            EpisodeDraft(
                text="结尾事实",
                evidence_spans=[(9, 10)],
            ),
        ]

        errors = MemoryExtractor._single_pass_coverage_errors(source_lines, drafts)

        self.assertEqual(len(errors), 1)
        self.assertIn("3-8", errors[0])

    def test_single_pass_coverage_allows_short_reactions_and_metadata(self):
        source_lines = [
            "[record: 1]",
            "角色甲: 主要事实已经发生。",
            "角色乙: ……",
            "角色乙: 好的",
            "[record: 2]",
            "角色甲: 下一阶段事实。",
        ]
        drafts = [
            EpisodeDraft(text="第一阶段", evidence_spans=[(2, 2)]),
            EpisodeDraft(text="第二阶段", evidence_spans=[(6, 6)]),
        ]

        self.assertEqual(
            MemoryExtractor._single_pass_coverage_errors(source_lines, drafts),
            [],
        )

    def test_single_pass_coverage_rejects_one_uncovered_high_information_line(self):
        source_lines = [
            "角色甲: 开始行动。",
            "角色乙: 学校资产的表决权属于学生会，因此只有学生会能够合法完成这项土地交易。",
            "角色甲: 行动结束。",
        ]
        drafts = [
            EpisodeDraft(text="开始行动", evidence_spans=[(1, 1)]),
            EpisodeDraft(text="行动结束", evidence_spans=[(3, 3)]),
        ]

        errors = MemoryExtractor._single_pass_coverage_errors(source_lines, drafts)

        self.assertEqual(len(errors), 1)
        self.assertIn("2-2", errors[0])

    def test_record_coverage_ignores_transcript_preamble_but_keeps_long_turn(self):
        source_lines = [
            "[source_key: transcript.txt]",
            "[record: 0]",
            "unknown: Document title and import notes that are not story events.",
            "[record: 1]",
            "unknown: More provenance explaining how the transcript was prepared.",
            "[record: 2]",
            "[speaker_raw: Morgan]",
            "unknown: Morgan opened the meeting.",
            "[record: 3]",
            "[speaker_raw: River]",
            "unknown: The board retained voting authority and rejected the transfer request.",
            "[record: 4]",
            "[speaker_raw: Morgan]",
            "unknown: Morgan closed the meeting.",
        ]
        drafts = [
            EpisodeDraft(text="opening", evidence_spans=[(6, 8)]),
            EpisodeDraft(text="closing", evidence_spans=[(12, 14)]),
        ]

        errors = MemoryExtractor._single_pass_coverage_errors(source_lines, drafts)

        self.assertEqual(len(errors), 1)
        self.assertIn("9-11", errors[0])
        self.assertNotIn("2-5", errors[0])

    def test_coverage_excerpt_retains_original_line_numbers(self):
        indexed = "\n".join(f"[L{index:04d}] line {index}" for index in range(1, 101))

        excerpt = MemoryExtractor._indexed_coverage_excerpt(
            indexed, [(50, 52)], context_lines=2
        )

        self.assertIn("[L0048]", excerpt)
        self.assertIn("[L0054]", excerpt)
        self.assertNotIn("[L0001]", excerpt)
        self.assertNotIn("[L0100]", excerpt)

    def test_unsupported_translated_aliases_fall_back_to_source_names(self):
        source = """[record: 1]
セイア: クズノハと会った。
アロナ(阿罗娜/Arona/아로나): シャーレへ行った。
"""
        episodes = [
            EpisodeDraft(
                text=(
                    "赛亚（セイア）与黑野（クズノハ）相遇，"
                    "アロナ（阿罗娜）随后去了シャーレ（Shale）。"
                ),
                participants=[
                    "赛亚 / セイア / Seia",
                    "黑野 / クズノハ / Kuzunoha",
                    "阿罗娜 / アロナ / Arona",
                ],
            )
        ]

        MemoryExtractor._remove_unsupported_name_aliases(source, episodes)

        self.assertEqual(
            episodes[0].text,
            "セイア与クズノハ相遇，アロナ（阿罗娜）随后去了シャーレ。",
        )
        self.assertEqual(
            episodes[0].participants,
            ["セイア", "クズノハ", "阿罗娜 / アロナ / Arona"],
        )

    def test_alias_cleanup_does_not_treat_chinese_verb_phrase_as_name(self):
        source = """[record: 1]
アロナ(阿罗娜/Arona/아로나): 先生は大丈夫ですか？
先生(老师/Sensei/선생님): 大丈夫。
"""
        episodes = [
            EpisodeDraft(
                text="アロナ（阿罗娜）关心老师（Sensei）的状况。",
                participants=[
                    "アロナ / 阿罗娜 / Arona / 아로나",
                    "老师 / Sensei / 선생님",
                ],
            )
        ]

        MemoryExtractor._remove_unsupported_name_aliases(source, episodes)

        self.assertEqual(
            episodes[0].text,
            "アロナ（阿罗娜）关心老师（Sensei）的状况。",
        )

    def test_adaptive_factual_audit_uses_independent_model_and_preserves_boundary(self):
        class CaptureLogger:
            def __init__(self):
                self.events = []

            def emit(self, event, **payload):
                self.events.append((event, payload))

        class Config:
            fallback_model = "independent-verifier"

        class FactualAuditModel:
            config = Config()

            def __init__(self):
                self.factual_kwargs = None

            def chat_json(self, system, _prompt, **kwargs):
                if "事实提取器" in system:
                    return {
                        "episodes": [
                            {
                                "text": "Morgan 是评审委员会。",
                                "participants": ["Morgan", "评审委员会"],
                                "event_type": "任命说明",
                                "confidence": 0.8,
                            }
                        ]
                    }
                if "原子命题证据审计器" in system:
                    self.factual_kwargs = kwargs
                    return {
                        "reviews": [
                            {
                                "episode_index": 0,
                                "status": "corrected",
                                "issue_types": ["predicate_collapse"],
                                "evidence_frames": [
                                    {
                                        "quote": "Alex: The review board appointed Morgan as investigator.",
                                        "subject_span": "review board",
                                        "predicate_span": "appointed",
                                        "object_span": "Morgan",
                                        "semantic_predicate": "任命",
                                        "attribution": "Alex",
                                        "qualifiers": [],
                                    }
                                ],
                                "reason": "appointment was collapsed into identity",
                                "corrected_episode": {
                                    "text": "据 Alex 陈述，review board 任命 Morgan 为调查员。",
                                    "participants": ["Morgan", "review board", "Alex"],
                                    "event_type": "任命说明",
                                    "confidence": 0.8,
                                },
                            }
                        ]
                    }
                raise AssertionError(system)

        model = FactualAuditModel()
        logger = CaptureLogger()
        extractor = MemoryExtractor(
            model,
            logger,
            episode_audit_mode="off",
            episode_factual_audit_mode="adaptive",
            episode_factual_audit_model="independent-verifier",
        )

        episodes, errors = extractor.extract_episodes(
            "Alex: The review board appointed Morgan as investigator.\n"
            "Dana: I acknowledge the appointment.",
            "main",
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes), 1)
        completed = [
            payload
            for event, payload in logger.events
            if event == "episode_factual_audit_completed"
        ]
        self.assertEqual(
            completed[0]["accepted_episode_indexes"],
            [0],
        )
        self.assertEqual(
            episodes[0].text, "据 Alex 陈述，review board 任命 Morgan 为调查员。"
        )
        self.assertEqual(
            model.factual_kwargs,
            {"model": "independent-verifier", "allow_fallback": False},
        )

    def test_natural_factual_audit_uses_plain_text_and_program_merge(self):
        class NaturalAuditModel:
            semantic_output_format = "natural_text"

            def __init__(self):
                self.kwargs = None

            def chat_text(self, _system, _prompt, **kwargs):
                self.kwargs = kwargs
                return (
                    "Episode 0：需修改\n"
                    "问题：施事与对象方向相反。\n"
                    "修改为：Alice 催促 Morgan 离开。"
                )

            def chat_json(self, *_args, **_kwargs):
                raise AssertionError("natural factual audit must not request JSON")

        model = NaturalAuditModel()
        extractor = MemoryExtractor(
            model,
            episode_factual_audit_mode="always",
            episode_factual_audit_model="independent-verifier",
        )
        episodes = [
            EpisodeDraft(
                text="Morgan 催促 Alice 离开。",
                evidence_quotes=["Alice: Morgan，尽快离开。"],
                evidence_spans=[(1, 1)],
            )
        ]

        audited, errors = extractor._audit_episode_facts(
            "Alice: Morgan，尽快离开。",
            "main",
            episodes,
        )

        self.assertEqual(errors, [])
        self.assertEqual(audited[0].text, "Alice 催促 Morgan 离开。")
        self.assertEqual(
            model.kwargs,
            {
                "allow_fallback": False,
                "max_retries": 0,
                "model": "independent-verifier",
            },
        )

    def test_fact_review_evidence_requires_literal_predicate_arguments(self):
        source = "The review board appointed Morgan as investigator."
        valid = {
            0: {
                "evidence_frames": [
                    {
                        "quote": source,
                        "subject_span": "review board",
                        "predicate_span": "appointed",
                        "object_span": "Morgan",
                    }
                ]
            }
        }
        invalid = {
            0: {
                "evidence_frames": [
                    {
                        "quote": source,
                        "subject_span": "review board",
                        "predicate_span": "is identical to",
                        "object_span": "Morgan",
                    }
                ]
            }
        }

        self.assertEqual(
            MemoryExtractor._validate_fact_review_evidence(source, valid), []
        )
        self.assertIn(
            "no frame grounds literal quote and predicate",
            MemoryExtractor._validate_fact_review_evidence(source, invalid)[0],
        )

    def test_adaptive_factual_audit_skips_plain_action_segment(self):
        extractor = MemoryExtractor(
            model=None,
            episode_audit_mode="off",
            episode_factual_audit_mode="adaptive",
        )
        self.assertFalse(
            extractor._should_audit_episode_facts(
                "角色们在操场完成训练。",
                [EpisodeDraft(text="角色们在操场完成训练。")],
            )
        )
        self.assertTrue(
            extractor._should_audit_episode_facts(
                "Alice: 我看见 Morgan 打开了门。\nBob: 我只听见了声音。",
                [EpisodeDraft(text="Alice 报告 Morgan 打开了门。")],
            )
        )

    def test_factual_audit_reviews_entire_attributed_source(self):
        source = "Alice: I saw Morgan open the door.\nBob: I heard a sound."
        episodes = [
            EpisodeDraft(text="众人完成普通训练。"),
            EpisodeDraft(text="梦中某位访客向 Morgan 递交文件。"),
            EpisodeDraft(text="委员会任命 Morgan 为调查员。"),
            EpisodeDraft(text="之后众人离开。"),
        ]

        self.assertEqual(
            MemoryExtractor._factual_audit_candidate_indexes(source, episodes),
            [0, 1, 2, 3],
        )

    def test_combined_episode_audit_can_be_forced_for_every_source(self):
        class AlwaysAuditModel:
            def __init__(self):
                self.audit_calls = 0

            def chat_json(self, system, _user, **_kwargs):
                if "事实提取器" in system:
                    return {
                        "episodes": [
                            {
                                "text": "角色们围绕同一目标完成连续行动。",
                                "participants": ["角色"],
                                "event_type": "连续行动",
                                "confidence": 0.9,
                            }
                        ]
                    }
                if "边界与证据审计器" in system:
                    self.audit_calls += 1
                    return {
                        "episodes": [
                            {
                                "text": "角色们围绕同一目标完成连续行动。",
                                "participants": ["角色"],
                                "event_type": "连续行动",
                                "confidence": 0.9,
                            }
                        ]
                    }
                raise AssertionError(system)

        model = AlwaysAuditModel()
        extractor = MemoryExtractor(
            model,
            episode_audit_mode="combined",
            episode_audit_always=True,
        )
        episodes, errors = extractor.extract_episodes("角色们连续行动。", "main")

        self.assertEqual(errors, [])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(model.audit_calls, 1)

    def test_participant_absent_from_source_is_quarantined_generically(self):
        draft = EpisodeDraft(
            text="InventedName approved Morgan's request.",
            participants=["InventedName", "Morgan"],
        )

        errors = MemoryExtractor._ground_episode_participants(
            "Alex: Morgan submitted a request.", [draft]
        )

        self.assertEqual(draft.participants, ["???", "Morgan"])
        self.assertEqual(draft.text, "??? approved Morgan's request.")
        self.assertIn("absent from Source", errors[0])

    def test_source_speaker_named_in_episode_is_completed_without_guessing(self):
        draft = EpisodeDraft(
            text="Morgan asked River to inspect the archive.",
            participants=["Morgan"],
        )

        MemoryExtractor._complete_episode_participants_from_source(
            "Morgan: River should inspect the archive.\nRiver: Understood.\n",
            [draft],
        )

        self.assertEqual(draft.participants, ["Morgan", "River"])

    def test_single_pass_finalization_uses_episode_evidence_not_whole_source(self):
        source = (
            "ヒマリ(日鞠/Himari): 守护者について説明する。\n"
            "マリナ(真里奈/Marina): 別の場所で待機する。"
        )
        draft = EpisodeDraft(
            text="ヒマリナ释了守护者。",
            participants=["ヒマリ"],
            evidence_quotes=["ヒマリ(日鞠/Himari): 守护者について説明する。"],
        )

        errors = MemoryExtractor.finalize_episode_drafts(source, [draft])

        self.assertEqual(errors, [])
        self.assertEqual(draft.text, "ヒマリ释了守护者。")
        self.assertEqual(draft.participants, ["ヒマリ"])

    def test_participant_completion_rejects_katakana_substring_collision(self):
        draft = EpisodeDraft(
            text="ヒマリナ释了情况。",
            participants=["ヒマリ"],
        )

        MemoryExtractor._complete_episode_participants_from_source(
            "ヒマリ: 説明する。\nマリナ: 待機する。",
            [draft],
        )

        self.assertEqual(draft.participants, ["ヒマリ"])

    def test_finalization_quarantines_unsupported_name_without_failing_import(self):
        class Model:
            def chat_json(self, _system, _prompt, **_kwargs):
                return {
                    "episodes": [
                        {
                            "text": "InventedName approved Morgan's request.",
                            "participants": ["InventedName", "Morgan"],
                            "event_type": "approval",
                            "confidence": 0.9,
                        }
                    ]
                }

        episodes, errors = MemoryExtractor(Model()).extract_episodes(
            "Alex: Morgan submitted a request.", "meeting"
        )

        self.assertEqual(errors, [])
        self.assertEqual(episodes[0].participants, ["???", "Morgan"])
        self.assertIn("???", episodes[0].text)

    def test_identity_guess_before_unknown_marker_is_removed_deterministically(self):
        draft = EpisodeDraft(
            text=(
                "爱丽丝向老师（???）报告任务完成。"
                "老师（???）建议爱丽丝先干擦，日富美([USERNAME])表示同意。"
            ),
            participants=["老师（???）", "日富美([USERNAME])", "爱丽丝"],
            event_type="建议",
        )
        sanitized = MemoryExtractor.sanitize_episode_draft(draft)
        self.assertEqual(
            sanitized.text,
            "爱丽丝向???报告任务完成。???建议爱丽丝先干擦，[USERNAME]表示同意。",
        )
        self.assertEqual(sanitized.participants, ["???", "[USERNAME]", "爱丽丝"])

    def test_combined_audit_rejects_only_extreme_episode_expansion(self):
        self.assertFalse(MemoryExtractor.episode_audit_expanded_too_far(5, 12))
        self.assertFalse(MemoryExtractor.episode_audit_expanded_too_far(9, 17))
        self.assertTrue(MemoryExtractor.episode_audit_expanded_too_far(5, 16))

        class ExplodingAuditModel:
            def chat_json(self, system, _prompt, **_kwargs):
                self.assert_system = system
                return {
                    "episodes": [
                        {
                            "text": f"对白碎片 {index}",
                            "participants": ["人物"],
                            "event_type": "微小反应",
                            "timeline_scope": "main",
                            "confidence": 0.8,
                        }
                        for index in range(16)
                    ]
                }

        original = [
            EpisodeDraft(text=f"事件阶段 {index}", timeline_scope="main")
            for index in range(5)
        ]
        extractor = MemoryExtractor(ExplodingAuditModel())
        audited, errors = extractor._audit_episode_quality(
            "SOURCE", "main", original, temporal_risk=True, granularity_risk=True
        )
        self.assertEqual(errors, [])
        self.assertEqual(audited, original)

    def test_reasoning_view_keeps_alias_legend_and_primary_translations(self):
        source = """[source_key: main/test.json]
[segment_index: 0]
[speaker_alias_legend]
히나: zh-CN=日奈 | ja=ヒナ | en=Hina | ko=히나

[record: 1]
[speaker_raw: 히나]
[script_raw: 3;히나;00;한국어 대사]
zh-CN: 中文对白
ja: 日本語の台詞
en: English line

[record: 2]
[speaker_raw: ???]
[script_raw: #na;???;대사]
ja: 中文缺失时使用日文
en: English fallback
"""
        compact = MemoryExtractor.compact_source_for_reasoning(source)
        self.assertIn("speaker_alias_legend", compact)
        self.assertIn("zh-CN: 中文对白", compact)
        self.assertIn("日本語の台詞", compact)
        self.assertIn("English line", compact)
        self.assertIn("ja: 中文缺失时使用日文", compact)
        self.assertIn("ko: 한국어 대사", compact)
        self.assertNotIn("[record: 2]\n[speaker_raw: ???]\n[speaker_raw: ???]", compact)
        self.assertNotIn("script_raw", compact)

    def test_reasoning_view_keeps_multiline_text_record_content(self):
        source = """[source_key: private/conversation.txt]
[segment_index: 0]

[record: 0]
[evidence_origin: source]
[epistemic_status: unknown]
unknown: conversation_record:
user: 请记住暗号是蓝莓雨伞。
meaning: 心情很糟时需要安静陪伴。
"""
        compact = MemoryExtractor.compact_source_for_reasoning(source)
        self.assertIn("user: 请记住暗号是蓝莓雨伞。", compact)
        self.assertIn("meaning: 心情很糟时需要安静陪伴。", compact)

    @patch("memory_demo.llm.client.time.sleep")
    def test_embedding_retries_same_model(self, _sleep):
        client = FlakyEmbeddingClient()
        matrix = client.embed(["hello"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(
            ["fixed-embedding-model", "fixed-embedding-model"], client.models
        )
        self.assertEqual(matrix.dtype.name, "float32")
        self.assertEqual(matrix.shape, (1, 4))

    @patch("memory_demo.llm.client.random.uniform", return_value=2.5)
    def test_rate_limit_retry_uses_long_jittered_backoff(self, _uniform):
        exc = ModelClientError("rate limited", status_code=429)
        self.assertEqual(ModelClient._retry_delay(exc, 0), 10.5)
        self.assertEqual(ModelClient._retry_delay(exc, 2), 34.5)

    def test_retry_after_header_takes_precedence_and_is_capped(self):
        exc = ModelClientError("rate limited", status_code=429, retry_after=180.0)
        self.assertEqual(ModelClient._retry_delay(exc, 0), 120.0)

    def test_growth_prompt_allows_evidence_bound_semantic_bridges(self):
        self.assertIn("解释性", GROWTH_SYSTEM)
        self.assertIn("查询综合推论", GROWTH_SYSTEM)
        self.assertIn("问题措辞不是证据", GROWTH_SYSTEM)
        self.assertIn("不表示情绪正负", GROWTH_SYSTEM)
        self.assertIn("属性", GROWTH_SYSTEM)
        self.assertIn("身份", GROWTH_AUDIT_SYSTEM)

    def test_answer_prompt_distinguishes_source_records_from_episodes(self):
        self.assertIn("Source record N", ANSWER_SYSTEM)
        self.assertIn("不能称作 Episode", ANSWER_SYSTEM)
        self.assertIn("不同 timeline_scope", ANSWER_SYSTEM)
        self.assertIn("不能覆盖或改写端点 Episode", ANSWER_SYSTEM)
        self.assertIn("不能自动证明身份", ANSWER_SYSTEM)
        self.assertIn("跨\nsource_key", ANSWER_AUDIT_SYSTEM)
        self.assertIn("多个 Episode", ANSWER_AUDIT_SYSTEM)
        self.assertIn("必须出现在问题或节点原文", HOP_QUERY_SYSTEM)
        self.assertIn("尚无直接证据", HOP_QUERY_SYSTEM)
        self.assertIn("精确 source_key", ANSWER_SYSTEM)
        self.assertIn("source_evidence_delivery", ANSWER_SYSTEM)
        self.assertIn("source_excerpt", ANSWER_AUDIT_SYSTEM)
        self.assertIn("source_evidence_delivery", ANSWER_AUDIT_SYSTEM)
        self.assertIn("问题中的提示只用于检索", ANSWER_AUDIT_SYSTEM)

    def test_source_excerpt_keeps_relevant_complete_record(self):
        raw = "\n\n".join(
            [
                "[source_key: main/a.json]",
                *[f"[record: {index}]\nzh-CN: 普通对话{index}" for index in range(20)],
                "[record: 20]\n[speaker_raw: 阿洛娜]\nzh-CN: 阿洛娜遇见老师",
            ]
        )
        excerpt = source_excerpt(raw, "阿洛娜第一次遇见老师", ["阿洛娜", "老师"], 300)
        self.assertLessEqual(len(excerpt), 300)
        self.assertIn("阿洛娜遇见老师", excerpt)
        self.assertIn("[source_key: main/a.json]", excerpt)

    def test_source_excerpt_prefers_persisted_episode_evidence(self):
        direct = "[record: 99]\n[speaker_raw: Morgan]\nzh-CN: Morgan 明确批准了请求。"
        raw = "\n\n".join(
            [
                "[source_key: main/a.json]\n[segment_index: 4]",
                *[
                    f"[record: {index}]\nzh-CN: Morgan 正在讨论其他事情。"
                    for index in range(1, 20)
                ],
                direct,
            ]
        )
        excerpt = source_excerpt(
            raw,
            "Morgan 批准了请求。",
            ["Morgan"],
            300,
            evidence_quotes=[direct],
        )
        self.assertLessEqual(len(excerpt), 300)
        self.assertIn(direct, excerpt)
        self.assertIn("[source_key: main/a.json]", excerpt)

    def test_source_excerpt_delivers_raw_record_for_verified_reasoning_view_quote(self):
        raw_record = (
            "[record: 100]\n"
            "[speaker_raw: Mika]\n"
            "[script_raw: 3;Mika;01;raw engine text]\n"
            "zh-CN: 我一直在暗中支援阿里乌斯。\n"
            "en: I have been secretly supporting Arius.\n"
            "zh-TW: 我一直暗中支援奧利斯。"
        )
        raw = "[source_key: main/a.json]\n[segment_index: 4]\n\n" + raw_record
        projected_quote = (
            "[record: 100]\n"
            "[speaker_raw: Mika]\n"
            "zh-CN: 我一直在暗中支援阿里乌斯。\n"
            "en: I have been secretly supporting Arius.\n"
            "ko: 아리우스를 몰래 지원해 왔어."
        )

        views = verified_evidence_views(raw, [projected_quote])
        excerpt = source_excerpt(
            raw,
            "Mika 一直暗中支援阿里乌斯。",
            ["Mika"],
            1_000,
            evidence_quotes=[projected_quote],
        )

        self.assertEqual([raw_record], views)
        self.assertIn(raw_record, excerpt)
        self.assertIn("[script_raw: 3;Mika;01;raw engine text]", excerpt)

    def test_projected_evidence_record_with_mismatching_shared_translation_is_rejected(self):
        raw = (
            "[source_key: main/a.json]\n\n"
            "[record: 100]\n[speaker_raw: Mika]\n"
            "zh-CN: 我一直在暗中支援阿里乌斯。\n"
            "en: I have been secretly supporting Arius."
        )
        forged_projection = (
            "[record: 100]\n[speaker_raw: Mika]\n"
            "zh-CN: 我从未支援阿里乌斯。\n"
            "en: I have been secretly supporting Arius."
        )

        self.assertEqual([], verified_evidence_views(raw, [forged_projection]))

    def test_projected_evidence_rejects_duplicate_language_fields(self):
        raw = (
            "[source_key: main/a.json]\n\n"
            "[record: 100]\n[speaker_raw: Mika]\n"
            "[script_raw: raw metadata]\n"
            "zh-CN: Mika 支援阿里乌斯。"
        )
        ambiguous_projection = (
            "[record: 100]\n[speaker_raw: Mika]\n"
            "zh-CN: Mika 支援错误组织。\n"
            "zh-CN: Mika 支援阿里乌斯。"
        )

        self.assertEqual([], verified_evidence_views(raw, [ambiguous_projection]))

    def test_projected_evidence_rejects_multiline_translation_field(self):
        raw = (
            "[source_key: main/a.json]\n\n"
            "[record: 100]\n[speaker_raw: Mika]\n"
            "[script_raw: raw metadata]\n"
            "zh-CN: Mika 支援阿里乌斯。"
        )
        ambiguous_projection = (
            "[record: 100]\n[speaker_raw: Mika]\n"
            "zh-CN: Mika 支援阿里乌斯。\n"
            "这行续文未受格式化约束。"
        )

        self.assertEqual([], verified_evidence_views(raw, [ambiguous_projection]))

    def test_answer_audit_prompt_exposes_source_delivery_status(self):
        prompt = answer_audit_prompt(
            "谁批准了请求？",
            {},
            "Morgan 批准了请求。",
            [
                {
                    "id": 7,
                    "text": "Morgan 批准了请求。",
                    "participants": ["Morgan"],
                    "source_key": "main/a.json",
                    "source_evidence_delivery": "source_bound",
                    "source_evidence_quote_count": 1,
                    "source_text": "[record: 99]\\nzh-CN: Morgan 明确批准了请求。",
                }
            ],
        )
        self.assertIn('"source_evidence_delivery": "source_bound"', prompt)
        self.assertIn("Morgan 明确批准了请求", prompt)

    def test_answer_audit_prompt_receives_the_full_delivered_source_excerpt(self):
        source_text = (
            "[record: 1]\\nzh-CN: 前置证据。\\n"
            + "x" * 2_400
            + "\\n[record: 2]\\n[speaker_raw: Morgan]\\n"
            "zh-CN: Morgan 明确批准了请求。"
        )
        prompt = answer_audit_prompt(
            "谁批准了请求？",
            {},
            "Morgan 批准了请求。",
            [{"id": 7, "text": "摘要", "source_text": source_text}],
        )

        self.assertIn("[record: 2]", prompt)
        self.assertIn("[speaker_raw: Morgan]", prompt)


if __name__ == "__main__":
    unittest.main()
