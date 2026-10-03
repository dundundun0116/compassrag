#!/usr/bin/env python3
"""构建 distractor 共享检索语料库（默认全量 dev context 池）。"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compassrag.corpus.benchmarks import BENCHMARKS
from compassrag.corpus.build import build_corpus


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--from-samples", action="store_true",
                    help="只用抽样题的 context 池建库（小库模式）")
    ap.add_argument("--raw-dir", default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--samples-dir", default=REPO_ROOT / "data" / "samples")
    ap.add_argument("--out-dir", default=REPO_ROOT / "data" / "cache" / "corpus")
    args = ap.parse_args()

    print(f"{'基准':<10} {'模式':<10} {'题数':>6} {'唯一段落':>8} {'块数':>7} {'支撑块':>7} {'≈token':>10}")
    for bench in [b.strip() for b in args.benchmarks.split(",")]:
        s = build_corpus(bench, args.raw_dir, args.out_dir,
                         from_samples=args.from_samples, samples_dir=args.samples_dir)
        print(f"{s['benchmark']:<10} {s['mode']:<10} {s['n_questions']:>6} "
              f"{s['n_unique_paragraphs']:>8} {s['n_chunks']:>7} {s['n_supporting_chunks']:>7} "
              f"{s['n_tokens_est_total']:>10}")
    print("\n块长（字符）: " + ", ".join(
        f"{bench}: min={s['char_len_min']} 中位={s['char_len_median']} p95={s['char_len_p95']} max={s['char_len_max']}"
        for bench in [b.strip() for b in args.benchmarks.split(",")]
        for s in [__import__("json").loads(
            (Path(args.out_dir) / bench / "stats.json").read_text())]))


if __name__ == "__main__":
    main()
