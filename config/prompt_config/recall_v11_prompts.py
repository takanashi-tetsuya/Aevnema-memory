"""D1 adds historical gap attention only when an earlier NEED left gaps."""
from config.prompt_config.recall_v4_prompts import MAP

GAP_MAP = MAP + '''
附加的need_feedback仅是上一轮已提交NEED的历史搜索诊断，不是故事事实、证据或新的回答条件。优先检查当前records能否填补items中的具体答案类型、关系或限定，并可提出当前原文中有助于这些缺口的followup_cues；没有合适原文时保留缺口，不要为填空制造事实。previous_value/previous_reason可能错误、过严或过期，也可能带有历史F/R别名；这些别名不指向本轮records。以原始question和当前原文为准，不因旧判断缩改问题。不得把need_feedback的文字作为record、事实引用、关联证明或原文cue；所有facts/links/cues仍遵守上面的当前原文准入规则。'''
