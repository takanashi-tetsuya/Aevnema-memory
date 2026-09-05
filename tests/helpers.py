from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from memory_demo.config import AppConfig, ModelConfig, SegmentConfig


def test_config(root: Path, dimension: int = 8) -> AppConfig:
    return AppConfig(
        database_path=root / "memory.db",
        log_dir=root / "logs",
        model=ModelConfig(
            api_key="fake",
            embedding_dimension=dimension,
            max_retries=0,
        ),
        segment=SegmentConfig(
            target_chars=200,
            max_chars=350,
            overlap_chars=60,
            minimum_blocks=1,
        ),
    )


test_config.__test__ = False


class FakeModel:
    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.logger = None
        self.growth_calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = np.frombuffer(digest, dtype=np.uint8)[: self.dimension].astype(np.float32)
            values = values - np.float32(127.5)
            if not np.any(values):
                values[0] = 1.0
            rows.append(values)
        return np.stack(rows)

    def chat_json(self, system: str, user: str, **_):
        if "事实提取器" in system:
            return {
                "episodes": [
                    {
                        "text": "阿洛娜第一次见到[USERNAME]（老师），并开始交流。",
                        "participants": ["阿洛娜", "老师", "[USERNAME]"],
                        "event_type": "初次相遇",
                        "location_text": "什亭之匣",
                        "story_time_text": "故事开始时",
                        "timeline_scope": "main",
                        "confidence": 0.9,
                    }
                ]
            }
        if "Concept 提取器" in system:
            return {
                "concepts": [
                    {
                        "canonical_name": "阿洛娜",
                        "description": "什亭之匣中的人工智能少女",
                        "embedding_text": "阿洛娜，什亭之匣中的人工智能少女",
                        "aliases": [
                            {"alias": "Arona", "language": "en"},
                            {"alias": "アロナ", "language": "ja"},
                        ],
                        "confidence": 0.9,
                    }
                ]
            }
        if "查询解析器" in system:
            return {
                "language": "zh",
                "target_entities": ["阿洛娜", "老师"],
                "search_queries": ["阿洛娜和老师最早在哪里建立联系"],
                "requested_relation": "第一次见面",
                "temporal_constraint": "earliest_actual_event",
                "causal_constraint": "",
                "answer_shape": "encounter_stages",
                "uncertainty_required": True,
            }
        if "多跳检索规划器" in system:
            return {"followup_queries": []}
        if "覆盖缺口审计器" in system:
            return {"coverage": [], "missing_aspects": []}
        if "证据重排器" in system:
            return {
                "selected_episode_ids": [1],
                "coverage": [],
                "missing_aspects": [],
            }
        if "证据重排结果的独立复核器" in system:
            return {
                "valid": True,
                "final_episode_ids": [1],
                "replacements": [],
                "missing_aspects": [],
            }
        if "答案证据审计器" in system:
            return {
                "valid": True,
                "issues": [],
                "correction_instructions": [],
            }
        if "增长关系证据审计器" in system:
            return {
                "reviews": [
                    {"index": 0, "accept": True, "reason": "端点证据支持"}
                ]
            }
        if "自主生长" in system:
            self.growth_calls += 1
            return {
                "relationships": [
                    {
                        "from_type": "episode",
                        "from_id": 1,
                        "to_type": "concept",
                        "to_id": 1,
                        "relation_type": "interpersonal",
                        "relation_key": "first_contact_context",
                        "relation_text": "该 Episode 描述阿洛娜与老师开始建立联系",
                        "polarity": 1,
                        "llm_score": 0.9,
                        "confidence": 0.8,
                    }
                ]
            }
        if "关系判断器" in system:
            return {"relationships": []}
        if "二次理解器" in system:
            return {
                "episode_id": 1,
                "episode": {
                    "text": "阿洛娜第一次见到[USERNAME]（老师），并开始交流。",
                    "participants": ["阿洛娜", "[USERNAME]", "老师"],
                    "event_type": "初次相遇",
                    "location_text": "什亭之匣",
                    "story_time_text": "故事开始时",
                    "timeline_scope": "main",
                    "confidence": 0.9,
                },
            }
        raise AssertionError(f"unexpected system prompt: {system[:80]}")

    def chat_text(self, system: str, user: str, **_) -> str:
        if "证据约束" not in system:
            raise AssertionError(f"unexpected text prompt: {system[:80]}")
        return "根据已检索的 Episode，阿洛娜在故事开始时于什亭之匣与老师建立了最初联系。"
