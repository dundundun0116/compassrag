import numpy as np

from compassrag.retrieval.dense import embed_shards, load_embeddings, plan_shards


class FakeEmbedder:
    """确定性假编码器：按文本内容取全局下标 i → [i, i+0.5, -i, 1.0]（可跨片断言）。"""

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            i = int(t[1:].split("\n")[0])  # "T12\nt12" → 12
            rows.append([float(i), i + 0.5, -float(i), 1.0])
        return np.array(rows, dtype=np.float32)


def _chunks(n):
    return [{"chunk_id": f"c{i}", "title": f"T{i}", "text": f"t{i}"} for i in range(n)]


def test_plan_shards_covers_all_with_short_tail():
    assert plan_shards(12000, 5000) == [(0, 0, 5000), (1, 5000, 10000), (2, 10000, 12000)]
    assert plan_shards(3, 5000) == [(0, 0, 3)]
    assert plan_shards(0, 100) == []


def test_embed_shards_writes_all_and_resumes(tmp_path):
    emb = FakeEmbedder()
    chunks = _chunks(7)
    done = embed_shards(chunks, tmp_path, emb, shard_size=3, log=lambda *a: None)
    assert done == 3
    assert len(emb.calls) == 3
    # 再跑一遍：全部已存在，零新计算
    emb2 = FakeEmbedder()
    done2 = embed_shards(chunks, tmp_path, emb2, shard_size=3, log=lambda *a: None)
    assert done2 == 0 and emb2.calls == []
    # 删掉中间一片：只补那一片
    (tmp_path / "shard_001.npy").unlink()
    emb3 = FakeEmbedder()
    done3 = embed_shards(chunks, tmp_path, emb3, shard_size=3, log=lambda *a: None)
    assert done3 == 1 and len(emb3.calls) == 1 and emb3.calls[0] == ["T3\nt3", "T4\nt4", "T5\nt5"]


def test_load_embeddings_stitches_in_order(tmp_path):
    chunks = _chunks(7)
    embed_shards(chunks, tmp_path, FakeEmbedder(), shard_size=3, log=lambda *a: None)
    mat = load_embeddings(tmp_path, n_items=7, shard_size=3)
    assert mat.shape == (7, 4)
    assert mat[0].tolist() == [0.0, 0.5, 0.0, 1.0]
    assert mat[6].tolist() == [6.0, 6.5, -6.0, 1.0]
