#!/usr/bin/env python3
"""检索召回率评测（无 LLM，纯本地）：抽样题上的支撑段落召回。

模式：bm25 / dense / hybrid（两路 RRF）/ wiki（三路检索）/ hybrid_wiki（三路 RRF）。
产出 runs/retrieval_{mode}/{bench}.jsonl（逐题，可断点续跑）+ 汇总表。
dense/hybrid/wiki 需要向量化产物（scripts/embed_corpus.py）；wiki/hybrid_wiki 还需
wiki 索引（scripts/build_wiki.py）。
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compassrag.corpus.benchmarks import BENCHMARKS, load_dev
from compassrag.eval.metrics import full_hit_at_k, recall_at_k
from compassrag.eval.runner import run_records
from compassrag.index.wiki import WikiIndex
from compassrag.retrieval.bm25 import BM25Index
from compassrag.retrieval.dense import DenseIndex, LocalDenseEmbedder
from compassrag.retrieval.fusion import rrf_fuse

K_VALUES = (5, 10, 20)


def _build_searcher(mode, bench, chunks, chunk_ids, args, embedder):
    """返回 search(query, k) -> [(chunk_id, score)]。"""
    bm25 = None
    dense = None
    wiki = None
    if mode in ("bm25", "hybrid", "hybrid_wiki"):
        index_dir = Path(args.index_dir) / f"{bench}__full_dev"
        if (index_dir / "chunk_ids.json").exists():
            bm25 = BM25Index.load(index_dir)
        else:
            print(f"[{bench}] 建 BM25 索引（{len(chunks)} 块）…")
            bm25 = BM25Index.build(chunks)
            bm25.save(index_dir)
    if mode in ("dense", "hybrid", "wiki", "hybrid_wiki"):
        dense = DenseIndex.load(Path(args.embeddings_dir) / f"{bench}__full_dev",
                                chunk_ids, embedder=embedder)
    if mode in ("wiki", "hybrid_wiki"):
        wiki = WikiIndex.load(Path(args.wiki_dir) / f"{bench}__full_dev", dense=dense, chunks=chunks)

    def search(query, k):
        if mode == "bm25":
            return bm25.search(query, k=k)
        if mode == "dense":
            return dense.search(query, k=k)
        if mode == "wiki":
            return wiki.search(query, k=k)
        routes = [[cid for cid, _ in bm25.search(query, k=k)],
                  [cid for cid, _ in dense.search(query, k=k)]]
        if mode == "hybrid_wiki":
            routes.append([cid for cid, _ in wiki.search(query, k=k)])
        return rrf_fuse(routes, top_n=k)

    return search


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--modes", default="bm25", help="逗号分隔：bm25,dense,hybrid,wiki,hybrid_wiki")
    ap.add_argument("--k-max", type=int, default=max(K_VALUES))
    ap.add_argument("--raw-dir", default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--samples-dir", default=REPO_ROOT / "data" / "samples")
    ap.add_argument("--corpus-dir", default=REPO_ROOT / "data" / "cache" / "corpus")
    ap.add_argument("--index-dir", default=REPO_ROOT / "data" / "cache" / "bm25")
    ap.add_argument("--embeddings-dir", default=REPO_ROOT / "data" / "cache" / "embeddings")
    ap.add_argument("--wiki-dir", default=REPO_ROOT / "data" / "cache" / "wiki")
    ap.add_argument("--out-dir", default=REPO_ROOT / "runs")
    ap.add_argument("--device", default=None, help="dense 模式的 BGE-M3 设备（mps / cpu）")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    needs_vectors = any(m in modes for m in ("dense", "hybrid", "wiki", "hybrid_wiki"))
    embedder = LocalDenseEmbedder(device=args.device) if needs_vectors else None

    summary = {}
    for bench in [b.strip() for b in args.benchmarks.split(",")]:
        records = load_dev(bench, Path(args.raw_dir))
        sample_file = sorted(Path(args.samples_dir).glob(f"{bench}_dev_n*_seed*.jsonl"))[-1]
        sample_ids = {json.loads(l)["id"] for l in sample_file.read_text().splitlines() if l.strip()}
        sampled = [r for r in records if r["id"] in sample_ids]

        chunks_path = Path(args.corpus_dir) / f"{bench}__full_dev" / "chunks.jsonl"
        chunks = [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]
        chunk_ids = [c["chunk_id"] for c in chunks]
        title_by_id = {c["chunk_id"]: c["title"] for c in chunks}

        for mode in modes:
            search = _build_searcher(mode, bench, chunks, chunk_ids, args, embedder)

            def step_fn(q, search=search):
                hits = search(q["question"], k=args.k_max)
                titles = [title_by_id[cid] for cid, _ in hits]
                gold = {p["title"] for p in q["paragraphs"] if p["is_supporting"]}
                row = {"id": q["id"], "stratum": q["stratum"], "gold_titles": sorted(gold)}
                for k in K_VALUES:
                    row[f"recall@{k}"] = recall_at_k(titles, gold, k)
                    row[f"full_hit@{k}"] = full_hit_at_k(titles, gold, k)
                row[f"retrieved_top{args.k_max}"] = titles[:args.k_max]
                return row

            out = Path(args.out_dir) / f"retrieval_{mode}" / f"{bench}.jsonl"
            rows, skipped = run_records(sampled, step_fn, out)
            all_rows = rows + _load_existing(out, skipped)
            n = len(all_rows)
            metrics = [f"recall@{k}" for k in K_VALUES] + [f"full_hit@{k}" for k in K_VALUES]
            summary[(bench, mode)] = {m: sum(r[m] for r in all_rows) / n for m in metrics} | {"n": n}

    metrics = [f"recall@{k}" for k in K_VALUES] + [f"full_hit@{k}" for k in K_VALUES]
    print(f"\n{'基准':<10} {'模式':<8} {'题数':>5} " + " ".join(f"{m:>12}" for m in metrics))
    for (bench, mode), s in summary.items():
        print(f"{bench:<10} {mode:<8} {s['n']:>5} " + " ".join(f"{s[m]:>12.3f}" for m in metrics))


def _load_existing(out: Path, skipped: int) -> list[dict]:
    """续跑时把已完成行也计入聚合（它们在本轮 rows 里不存在）。"""
    if skipped == 0:
        return []
    with out.open(encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[:skipped]


if __name__ == "__main__":
    main()
