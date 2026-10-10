# CompassRAG

> Adaptive Retrieval Decisions over Structured Indexes for Multi-Hop QA

面向多跳问答的 agentic RAG：以自适应检索决策为大脑，查询改写融合为执行组件，结构化分层索引为地基（进行中，完整设计与计划见 [PROJECT.md](PROJECT.md)）。

**状态**（2026-10-10）：S1 完成（含冒烟验收）；S2 检索层完成（BM25 + BGE-M3 稠密混合，musique hybrid recall@20 = 0.698）；S3 wiki 分层摘要索引建成（musique 211 条目 / 100% 覆盖）——检索层实测暂未见增益，设计修订中；S4 查询侧（多跳分解 + HyDE）实现完成。消融各行数字待跑。逐单元进展与后续计划见 [PROJECT.md](PROJECT.md) 第 8/11 节。
