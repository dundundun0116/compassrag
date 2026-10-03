#!/usr/bin/env python3
"""检索召回率评测（无 LLM，纯本地）：BM25 在抽样题上的支撑段落召回。

产出 runs/retrieval_bm25/{bench}.jsonl（逐题，可断点续跑）+ 汇总表。
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
from compassrag.retrieval.bm25 import BM25Index

K_VALUES = (5, 10, 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--k-max", type=int, default=max(K_VALUES))
    ap.add_argument("--raw-dir", default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--samples-dir", default=REPO_ROOT / "data" / "samples")
    ap.add_argument("--corpus-dir", default=REPO_ROOT / "data" / "cache" / "corpus")
    ap.add_argument("--index-dir", default=REPO_ROOT / "data" / "cache" / "bm25")
    ap.add_argument("--out-dir", default=REPO_ROOT / "runs" / "retrieval_bm25")
    args = ap.parse_args()

    summary = {}
    for bench in [b.strip() for b in args.benchmarks.split(",")]:
        records = load_dev(bench, Path(args.raw_dir))
        sample_file = sorted(Path(args.samples_dir).glob(f"{bench}_dev_n*_seed*.jsonl"))[-1]
        sample_ids = {json.loads(l)["id"] for l in sample_file.read_text().splitlines() if l.strip()}
        sampled = [r for r in records if r["id"] in sample_ids]

        chunks_path = Path(args.corpus_dir) / f"{bench}__full_dev" / "chunks.jsonl"
        chunks = [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]
        title_by_id = {c["chunk_id"]: c["title"] for c in chunks}

        index_dir = Path(args.index_dir) / f"{bench}__full_dev"
        if (index_dir / "chunk_ids.json").exists():
            index = BM25Index.load(index_dir)
        else:
            print(f"[{bench}] 建 BM25 索引（{len(chunks)} 块）…")
            index = BM25Index.build(chunks)
            index.save(index_dir)

        def step_fn(q):
            hits = index.search(q["question"], k=args.k_max)
            titles = [title_by_id[cid] for cid, _ in hits]
            gold = {p["title"] for p in q["paragraphs"] if p["is_supporting"]}
            row = {"id": q["id"], "stratum": q["stratum"], "gold_titles": sorted(gold)}
            for k in K_VALUES:
                row[f"recall@{k}"] = recall_at_k(titles, gold, k)
                row[f"full_hit@{k}"] = full_hit_at_k(titles, gold, k)
            row["retrieved_top20"] = titles[:20]
            return row

        out = Path(args.out_dir) / f"{bench}.jsonl"
        rows, skipped = run_records(sampled, step_fn, out)
        n = len(rows) + skipped
        summary[bench] = {
            m: sum(r[m] for r in rows + _load_existing(out, skipped)) / n
            for m in (f"recall@{k}" for k in K_VALUES)
        } | {
            f"full_hit@{k}": sum(r[f"full_hit@{k}"] for r in rows + _load_existing(out, skipped)) / n
            for k in K_VALUES
        }
        summary[bench]["n"] = n
        summary[bench]["skipped"] = skipped

    print(f"\n{'基准':<10} {'题数':>5} " + " ".join(f"{m:>12}" for m in
          [f"recall@{k}" for k in K_VALUES] + [f"full_hit@{k}" for k in K_VALUES]))
    for bench, s in summary.items():
        print(f"{bench:<10} {s['n']:>5} " + " ".join(f"{s[m]:>12.3f}" for m in
              [f"recall@{k}" for k in K_VALUES] + [f"full_hit@{k}" for k in K_VALUES]))


def _load_existing(out: Path, skipped: int) -> list[dict]:
    """续跑时把已完成行也计入聚合（它们在本轮 rows 里不存在）。"""
    if skipped == 0:
        return []
    with out.open(encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[:skipped]


if __name__ == "__main__":
    main()
