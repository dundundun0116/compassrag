# CompassRAG

> Adaptive Retrieval Decisions over Structured Indexes for Multi-Hop QA

面向多跳问答的 agentic RAG：以自适应检索决策为大脑，查询改写融合为执行组件，结构化分层索引为地基（进行中，完整设计与计划见 [PROJECT.md](PROJECT.md)）。

**状态**（2026-10-10）：S1 完成；S2 检索层完成（BM25 + BGE-M3 稠密混合，musique hybrid recall@20 = 0.698）；S3 wiki 分层摘要索引建成（musique 211 条目 / 100% 覆盖；检索层实测无净增益，修订候选已试尽、定案不并入主线）；S4 查询侧、S5 决策层实现完成。**消融表已出前两行（100 题分层子集，同题集）**：naive EM 0.250 / F1 0.325 → +② EM 0.290 / F1 0.391。逐单元进展与后续计划见 [PROJECT.md](PROJECT.md) 第 8/11 节。
