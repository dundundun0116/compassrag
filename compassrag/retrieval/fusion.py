"""多路检索结果融合：RRF（倒数排名融合，免调参）+ chunk_id 精确去重。

近重语义去重（embedding 余弦）需要向量通路，接入后补（S2）。
"""

from collections import defaultdict

RRF_K_DEFAULT = 60


def rrf_fuse(routes: list[list[str]], k: int = RRF_K_DEFAULT, top_n: int | None = None) -> list[tuple[str, float]]:
    """routes = 各路检索的 id 有序排名列表；返回 [(id, rrf_score)] 按融合分降序。

    得分 = Σ 1/(k + rank)，rank 从 1 计；同分按首见顺序稳定排序。
    """
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for route in routes:
        for rank, cid in enumerate(route, start=1):
            scores[cid] += 1.0 / (k + rank)
            first_seen.setdefault(cid, len(first_seen))
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return ordered[:top_n] if top_n is not None else ordered


def dedup_by_id(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out
