from memory_demo.retrieval.cue_index import (
    LexicalAssociationCueIndex,
    association_cue_text,
    cue_features,
)


ROWS = [
    {
        "id": 1,
        "relation_text": (
            "渚以成绩与退学机制为补课部的表面安排，把潜在内鬼集中起来排查；"
            "梓的阿里乌斯出身构成政治筛查风险。"
        ),
    },
    {
        "id": 2,
        "relation_text": (
            "伊甸园条约袭击由阿里乌斯执行，未花与阿里乌斯的合作构成"
            "茶会内部政治选择与外部袭击的前置联系。"
        ),
    },
    {
        "id": 3,
        "relation_text": (
            "星野独自牺牲保护后辈的倾向，与梦前辈死亡后的创伤和"
            "独自承担责任模式相连。"
        ),
    },
]


def test_features_include_cjk_bigrams_and_latin_words():
    features = cue_features("伊甸园 treaty attack")

    assert "c:伊" in features
    assert "n2:伊甸" in features
    assert "n2:甸园" in features
    assert "w:treaty" in features


def test_evidence_bridge_indexes_query_and_slots_not_long_observations():
    row = {
        "relation_key": "evidence_bridge",
        "from_participants_json": '["未花", "阿里乌斯"]',
        "to_participants_json": '["阿里乌斯", "渚"]',
        "relation_text": (
            "查询综合推论：检索证据桥：问题“谁袭击条约现场？”中的槽"
            "“袭击者”由 Episode 1 直接支持，其观察为：很长的端点原文；"
            "槽“合作背景”由 Episode 2 直接支持。"
        ),
    }

    text = association_cue_text(row)

    assert text == "谁袭击条约现场？ 袭击者 合作背景 未花 阿里乌斯 渚"
    assert "很长的端点原文" not in text


def test_evidence_bridge_participants_improve_relation_paraphrase_not_entity_chat():
    row = {
        "id": 4,
        "relation_key": "evidence_bridge",
        "from_participants_json": '["未花", "阿里乌斯"]',
        "to_participants_json": '["阿里乌斯"]',
        "relation_text": (
            "检索证据桥：问题“谁袭击了条约现场，内部是谁协助的？”"
            "中的槽“袭击执行者”由 Episode 1 支持，其观察为：长文；"
            "槽“内部协助者”由 Episode 2 支持。"
        ),
    }
    index = LexicalAssociationCueIndex([row])

    positive = index.search("条约危机的执行者是谁，未花和这个势力有什么联系？", 1)
    negative = index.search("未花喜欢什么甜点？", 1)

    assert positive[0][1] >= 0.18
    assert negative[0][1] < 0.18


def test_lexical_cue_index_recovers_paraphrased_relation():
    ranked = LexicalAssociationCueIndex(ROWS).search(
        "为什么说补课部不只是帮助差生，梓的阿里乌斯背景和渚寻找内鬼有什么联系？",
        3,
    )

    assert ranked[0][0]["id"] == 1
    assert ranked[0][1] >= 0.18
    assert ranked[0][1] - ranked[1][1] >= 0.04


def test_lexical_cue_index_keeps_entity_only_question_below_fast_threshold():
    ranked = LexicalAssociationCueIndex(ROWS).search(
        "梓使用什么武器，她平时喜欢什么？",
        3,
    )

    assert ranked[0][1] < 0.18
