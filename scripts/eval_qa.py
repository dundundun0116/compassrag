#!/usr/bin/env python3
"""端到端 QA 评测（消融第 0 级 naive）：scripted agent + 直接生成 → EM/F1。

逐题增量落盘 + 断点续跑；产出 runs/qa/{bench}__{config}.jsonl + 汇总表。
注意：需要可用的 LLM 通道（402/403 会即刻失败，属预期）。
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compassrag.agent import Agent, AgentConfig
from compassrag.corpus.benchmarks import BENCHMARKS, load_dev
from compassrag.eval.metrics import f1_score, em_score, recall_at_k
from compassrag.eval.runner import run_records
from compassrag.llm.client import LLMClient
from compassrag.retrieval.bm25 import BM25Index
from compassrag.retrieval.dense import DenseIndex, LocalDenseEmbedder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--config", default=REPO_ROOT / "configs" / "ablation" / "naive.yaml")
    ap.add_argument("--raw-dir", default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--samples-dir", default=REPO_ROOT / "data" / "samples")
    ap.add_argument("--corpus-dir", default=REPO_ROOT / "data" / "cache" / "corpus")
    ap.add_argument("--index-dir", default=REPO_ROOT / "data" / "cache" / "bm25")
    ap.add_argument("--embeddings-dir", default=REPO_ROOT / "data" / "cache" / "embeddings")
    ap.add_argument("--out-dir", default=REPO_ROOT / "runs" / "qa")
    ap.add_argument("--device", default=None, help="dense 查询编码设备（mps / cpu；后台向量化时用 cpu 避免抢 MPS）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    agent_cfg = AgentConfig(**cfg["agent"])
    llm = LLMClient()
    config_name = Path(args.config).stem
    embedder = LocalDenseEmbedder(device=args.device) if agent_cfg.use_dense else None

    summary = {}
    for bench in [b.strip() for b in args.benchmarks.split(",")]:
        records = load_dev(bench, Path(args.raw_dir))
        sample_file = sorted(Path(args.samples_dir).glob(f"{bench}_dev_n*_seed*.jsonl"))[-1]
        sample_ids = {json.loads(l)["id"] for l in sample_file.read_text().splitlines() if l.strip()}
        sampled = [r for r in records if r["id"] in sample_ids]

        chunks = [json.loads(l) for l in
                  (Path(args.corpus_dir) / f"{bench}__full_dev" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        store = {c["chunk_id"]: c for c in chunks}
        tools = {"bm25": BM25Index.load(Path(args.index_dir) / f"{bench}__full_dev")}
        if agent_cfg.use_dense:
            tools["dense"] = DenseIndex.load(
                Path(args.embeddings_dir) / f"{bench}__full_dev",
                [c["chunk_id"] for c in chunks], embedder=embedder)
        print(f"[{bench}] 语料 {len(chunks)} 块；工具：{list(tools)}")

        agent = Agent(llm=llm, tools=tools, chunk_store=store, config=agent_cfg)

        def step_fn(q, agent=agent):
            r = agent.answer(q["question"])
            gold_titles = {p["title"] for p in q["paragraphs"] if p["is_supporting"]}
            ev_titles = [store[cid]["title"] for cid in r.evidence if cid in store]
            return {
                "id": q["id"], "stratum": q["stratum"],
                "pred": r.answer, "gold": q["answer"],
                "em": em_score(r.answer, q["answer"]), "f1": f1_score(r.answer, q["answer"]),
                "ev_recall": recall_at_k(ev_titles, gold_titles, k=agent_cfg.top_k),
                "rounds": r.rounds,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                # 生成协议诊断字段：重试次数与各次 finish_reason（"length"=撞预算上限）
                "n_calls": r.n_calls, "finish_reasons": r.finish_reasons,
                "answer_empty": not r.answer.strip(),
            }

        out = Path(args.out_dir) / f"{bench}__{config_name}.jsonl"
        rows, skipped = run_records(sampled, step_fn, out)
        all_rows = _load_all(out)
        summary[bench] = _aggregate(all_rows)
        print(f"[{bench}] 完成 {len(rows)} 题（续跑跳过 {skipped}）→ {out}")

    print(f"\n{'基准':<10} {'题数':>5} {'EM':>7} {'F1':>7} {'证据召回':>8} {'空答率':>7} {'重试率':>7} {'均token/题':>10}")
    for bench, s in summary.items():
        print(f"{bench:<10} {s['n']:>5} {s['em']:>7.3f} {s['f1']:>7.3f} {s['ev_recall']:>8.3f} "
              f"{s['empty_rate']:>7.3f} {s['retry_rate']:>7.3f} {s['tokens_per_q']:>10.0f}")


def _load_all(out: Path) -> list[dict]:
    if not out.exists():
        return []
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "em": sum(r["em"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "ev_recall": sum(r["ev_recall"] for r in rows) / n,
        "tokens_per_q": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n,
        # 空答率应趋近 0；非 0 说明生成协议仍被思维链打转拖垮（见 runs/diag/）
        "empty_rate": sum(1 for r in rows if not r["pred"].strip()) / n,
        "retry_rate": sum(1 for r in rows if r.get("n_calls", 1) > 1) / n,
    }


if __name__ == "__main__":
    main()
