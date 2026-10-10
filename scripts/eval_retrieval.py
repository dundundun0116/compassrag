#!/usr/bin/env python3
"""检索召回率评测（无 LLM，纯本地）：抽样题上的支撑段落召回。

模式：bm25 / dense / hybrid（两路 RRF）/ wiki（三路检索）/ hybrid_wiki（三路等权 RRF）
     ——以上为既有模式；
修订模式（2026-10-10，针对 hybrid_wiki 被稀释的修订实验）：
     hybrid_wwiki  加权 RRF：wiki 路票权 --wiki-weight（默认 0.5）、贡献深度截断 --wiki-depth（0=不截断）
     hybrid_eprior 条目层先验重排：hybrid 宽候选 + --entry-boost × 条目先验分（1/(60+条目名次)）
                   重排；--entry-expand 时从命中条目内部补捞候选（只调座次为主，不与主路竞争）
产出 runs/retrieval_{模式名(含参数)}/{bench}.jsonl（逐题，可断点续跑）+ 汇总表。
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


def _parse_mode(spec: str, args) -> tuple[str, str]:
    """模式规格 -> (实现族, 产物目录名)。修订模式把参数编进目录名，变体互不覆盖。"""
    if spec == "hybrid_wwiki":
        out = f"hybrid_wwiki_w{args.wiki_weight:g}"
        if args.wiki_depth:
            out += f"_d{args.wiki_depth}"
        return "hybrid_wwiki", out
    if spec == "hybrid_eprior":
        out = f"hybrid_eprior_e{args.entry_top}b{args.entry_boost:g}" + ("_x" if args.entry_expand else "")
        return "hybrid_eprior", out
    return spec, spec


def _build_searcher(family, bench, chunks, chunk_ids, args, embedder):
    """返回 search(query, k) -> [(chunk_id, score)]。"""
    bm25 = None
    dense = None
    wiki = None
    if family in ("bm25", "hybrid", "hybrid_wiki", "hybrid_wwiki", "hybrid_eprior"):
        index_dir = Path(args.index_dir) / f"{bench}__full_dev"
        if (index_dir / "chunk_ids.json").exists():
            bm25 = BM25Index.load(index_dir)
        else:
            print(f"[{bench}] 建 BM25 索引（{len(chunks)} 块）…")
            bm25 = BM25Index.build(chunks)
            bm25.save(index_dir)
    if family in ("dense", "hybrid", "wiki", "hybrid_wiki", "hybrid_wwiki", "hybrid_eprior"):
        dense = DenseIndex.load(Path(args.embeddings_dir) / f"{bench}__full_dev",
                                chunk_ids, embedder=embedder)
    if family in ("wiki", "hybrid_wiki", "hybrid_wwiki", "hybrid_eprior"):
        wiki = WikiIndex.load(Path(args.wiki_dir) / f"{bench}__full_dev", dense=dense, chunks=chunks)

    def search(query, k):
        if family == "bm25":
            return bm25.search(query, k=k)
        if family == "dense":
            return dense.search(query, k=k)
        if family == "wiki":
            return wiki.search(query, k=k)
        if family == "hybrid_wwiki":
            wiki_route = [cid for cid, _ in wiki.search(query, k=k)]
            if args.wiki_depth:
                wiki_route = wiki_route[:args.wiki_depth]
            routes = [[cid for cid, _ in bm25.search(query, k=k)],
                      [cid for cid, _ in dense.search(query, k=k)], wiki_route]
            return rrf_fuse(routes, top_n=k, weights=[1.0, 1.0, args.wiki_weight])
        if family == "hybrid_eprior":
            wide = args.eprior_wide
            fused = rrf_fuse([[cid for cid, _ in bm25.search(query, k=wide)],
                              [cid for cid, _ in dense.search(query, k=wide)]], top_n=wide)
            prior = wiki.prior_scores(query, top_e=args.entry_top)
            scored = {cid: s + args.entry_boost * prior.get(cid, 0.0) for cid, s in fused}
            if args.entry_expand:
                for cid in wiki.pullback(query, top_e=args.entry_top):
                    scored.setdefault(cid, args.entry_boost * prior.get(cid, 0.0))
            return sorted(scored.items(), key=lambda kv: -kv[1])[:k]
        routes = [[cid for cid, _ in bm25.search(query, k=k)],
                  [cid for cid, _ in dense.search(query, k=k)]]
        if family == "hybrid_wiki":
            routes.append([cid for cid, _ in wiki.search(query, k=k)])
        return rrf_fuse(routes, top_n=k)

    return search


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--modes", default="bm25",
                    help="逗号分隔：bm25,dense,hybrid,wiki,hybrid_wiki,hybrid_wwiki,hybrid_eprior")
    ap.add_argument("--k-max", type=int, default=max(K_VALUES))
    # hybrid_wwiki 参数
    ap.add_argument("--wiki-weight", type=float, default=0.5, help="wiki 路票权（修订一）")
    ap.add_argument("--wiki-depth", type=int, default=0, help="wiki 路贡献深度截断，0=不截断")
    # hybrid_eprior 参数
    ap.add_argument("--entry-top", type=int, default=3, help="条目先验命中的条目数")
    ap.add_argument("--entry-boost", type=float, default=0.5, help="条目先验分权重")
    ap.add_argument("--entry-expand", action="store_true", help="从命中条目内部补捞候选")
    ap.add_argument("--eprior-wide", type=int, default=50, help="先验重排的宽候选池大小")
    ap.add_argument("--raw-dir", default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--samples-dir", default=REPO_ROOT / "data" / "samples")
    ap.add_argument("--corpus-dir", default=REPO_ROOT / "data" / "cache" / "corpus")
    ap.add_argument("--index-dir", default=REPO_ROOT / "data" / "cache" / "bm25")
    ap.add_argument("--embeddings-dir", default=REPO_ROOT / "data" / "cache" / "embeddings")
    ap.add_argument("--wiki-dir", default=REPO_ROOT / "data" / "cache" / "wiki")
    ap.add_argument("--out-dir", default=REPO_ROOT / "runs")
    ap.add_argument("--device", default=None, help="dense 模式的 BGE-M3 设备（mps / cpu）")
    args = ap.parse_args()

    specs = [m.strip() for m in args.modes.split(",")]
    parsed = [_parse_mode(s, args) for s in specs]
    needs_vectors = any(f in ("dense", "hybrid", "wiki", "hybrid_wiki", "hybrid_wwiki", "hybrid_eprior")
                        for f, _ in parsed)
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

        for family, out_name in parsed:
            search = _build_searcher(family, bench, chunks, chunk_ids, args, embedder)

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

            out = Path(args.out_dir) / f"retrieval_{out_name}" / f"{bench}.jsonl"
            rows, skipped = run_records(sampled, step_fn, out)
            all_rows = rows + _load_existing(out, skipped)
            n = len(all_rows)
            metrics = [f"recall@{k}" for k in K_VALUES] + [f"full_hit@{k}" for k in K_VALUES]
            summary[(bench, out_name)] = {m: sum(r[m] for r in all_rows) / n for m in metrics} | {"n": n}

    metrics = [f"recall@{k}" for k in K_VALUES] + [f"full_hit@{k}" for k in K_VALUES]
    print(f"\n{'基准':<10} {'模式':<24} {'题数':>5} " + " ".join(f"{m:>12}" for m in metrics))
    for (bench, mode), s in summary.items():
        print(f"{bench:<10} {mode:<24} {s['n']:>5} " + " ".join(f"{s[m]:>12.3f}" for m in metrics))


def _load_existing(out: Path, skipped: int) -> list[dict]:
    """续跑时把已完成行也计入聚合（它们在本轮 rows 里不存在）。"""
    if skipped == 0:
        return []
    with out.open(encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[:skipped]


if __name__ == "__main__":
    main()
