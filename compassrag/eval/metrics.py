"""评测指标：SQuAD 风格 EM/F1（多跳论文标准口径）+ 支撑段落召回率。"""

import re
import string
from collections import Counter


def squad_normalize(text: str) -> str:
    """小写、去标点、去冠词、压缩空白（SQuAD 官方口径）。"""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_score(prediction: str, gold: str) -> float:
    return float(squad_normalize(prediction) == squad_normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = squad_normalize(prediction).split()
    gold_tokens = squad_normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(prediction: str, golds: list[str], metric) -> float:
    """多个参考答案时取最大值（MuSiQue 的 answer_aliases）。"""
    return max((metric(prediction, g) for g in golds), default=0.0)


def recall_at_k(retrieved_titles: list[str], gold_titles: set[str], k: int) -> float:
    """top-k 检索结果覆盖的支撑标题比例（支撑标题去重后计）。"""
    if not gold_titles:
        return 0.0
    top = set(retrieved_titles[:k])
    return len(top & gold_titles) / len(gold_titles)


def full_hit_at_k(retrieved_titles: list[str], gold_titles: set[str], k: int) -> float:
    """全部支撑标题都进 top-k 才算命中（多跳答题的硬前提）。"""
    if not gold_titles:
        return 0.0
    return float(gold_titles <= set(retrieved_titles[:k]))
