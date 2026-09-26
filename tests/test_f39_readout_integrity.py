"""The local guard corrects only exact, unique, visible citation mistakes."""
import importlib.util
from pathlib import Path

MODULE = (Path(__file__).resolve().parents[1] /
          "validation/recall-f39-semantic-and-connectivity-20260923/readout_integrity.py")
spec = importlib.util.spec_from_file_location("f39_readout_integrity", MODULE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_repairs_unique_quote_with_wrong_source_number():
    registry = {(59, 48): {"text": "另一段原文", "speaker": "渚"},
                (60, 48): {"text": "圣三一和格黑娜长期以来的敌对关系给彼此带来了巨大的负担。", "speaker": "渚"}}
    result = guard.validate_or_repair_citation({"source_id": 59, "record": 48,
        "speaker": "渚", "quote": "圣三一和格黑娜长期以来的敌对关系"}, registry)
    assert result["status"] == "repaired_unique_exact_quote"
    assert result["citation"]["source_id"] == 60


def test_missing_or_ambiguous_quote_is_quarantined():
    registry = {(1, 1): {"text": "同一句話", "speaker": "甲"},
                (2, 2): {"text": "同一句話", "speaker": "甲"}}
    assert guard.validate_or_repair_citation({"source_id": 1, "record": 1}, registry)["status"] == "quarantined_quote_missing"
    assert guard.validate_or_repair_citation({"source_id": 3, "record": 3, "quote": "同一句話"}, registry)["status"] == "quarantined_ambiguous"


def test_inferred_actor_requires_separate_attributed_evidence():
    registry = {(14, 10): {"text": "我們是新的伊甸條約機構", "speaker": None}}
    card = {"event_key": "later_eto", "phase": "廢墟後續", "location": "古聖堂廢墟",
            "action": "宣告", "actor": "老師", "actor_basis": "inference",
            "citations": [{"source_id": 14, "record": 10, "quote": "我們是新的伊甸條約機構"}]}
    assert guard.validate_event_cards([card], registry)[0]["status"] == "quarantined_unattributed_inference"


def test_direct_actor_cannot_be_supplied_by_unknown_speaker():
    registry = {(15, 379): {"text": "我们是，新的伊甸条约机构。", "speaker": None}}
    card = {"event_key": "later_eto", "phase": "廢墟後續", "location": "古聖堂廢墟",
            "action": "宣告", "actor": "老師", "actor_basis": "speaker_tag",
            "citations": [{"source_id": 15, "record": 379, "quote": "新的伊甸条约机构"}]}
    assert guard.validate_event_cards([card], registry)[0]["status"] == "quarantined_actor_tag_conflict"
