"""本地 BGE-M3 稠密向量（离线、无 API）。

分片落盘 + 断点续跑：每片 shard_XXX.npy 覆盖 chunks.jsonl 的固定行区间，
片数与行数在重跑时校验，缺片/坏片自动补算。
"""

from pathlib import Path

import numpy as np

MODEL_NAME_DEFAULT = "BAAI/bge-m3"
SHARD_SIZE_DEFAULT = 5000


def plan_shards(n_items: int, shard_size: int) -> list[tuple[int, int, int]]:
    """[(shard_idx, start, end)]，末片可短；0 条时为空。"""
    return [(i, i * shard_size, min(n_items, (i + 1) * shard_size))
            for i in range((n_items + shard_size - 1) // shard_size)]


class LocalDenseEmbedder:
    """sentence-transformers 惰性加载封装；默认归一化（余弦 = 点积）。"""

    def __init__(self, model_name: str = MODEL_NAME_DEFAULT, device: str | None = None,
                 batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_length
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure()
        return np.asarray(model.encode(list(texts), batch_size=self.batch_size,
                                       normalize_embeddings=True, convert_to_numpy=True,
                                       show_progress_bar=False), dtype=np.float32)


def _chunk_text(c: dict) -> str:
    return f"{c['title']}\n{c['text']}"


def embed_shards(chunks: list[dict], out_dir: Path, embedder,
                 shard_size: int = SHARD_SIZE_DEFAULT, log=print) -> int:
    """逐片向量化并落盘；已存在的完整片跳过。返回本次新完成的片数。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for idx, start, end in plan_shards(len(chunks), shard_size):
        f = out_dir / f"shard_{idx:03d}.npy"
        if f.exists():
            try:
                if np.load(f, mmap_mode="r").shape[0] == end - start:
                    continue
            except Exception:  # noqa: BLE001 坏片重算
                pass
        vecs = embedder.encode([_chunk_text(c) for c in chunks[start:end]])
        np.save(f, np.asarray(vecs, dtype=np.float32))
        done += 1
        log(f"shard {idx:03d}: 行 {start}-{end} 完成 -> {f.name}")
    return done


def load_embeddings(out_dir: Path, n_items: int, shard_size: int = SHARD_SIZE_DEFAULT) -> np.ndarray:
    """按片拼回完整 (n_items, dim) 矩阵。"""
    parts = []
    for idx, start, end in plan_shards(n_items, shard_size):
        arr = np.load(Path(out_dir) / f"shard_{idx:03d}.npy")
        assert arr.shape[0] == end - start, f"shard_{idx:03d} 行数不符：{arr.shape[0]} != {end - start}"
        parts.append(arr)
    return np.vstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)
