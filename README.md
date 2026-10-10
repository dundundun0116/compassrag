# CompassRAG

> Adaptive Retrieval Decisions over Structured Indexes for Multi-Hop QA

面向多跳问答的 agentic RAG：以自适应检索决策为大脑，查询改写融合为执行组件，结构化分层索引为地基（进行中，完整设计与计划见 [PROJECT.md](PROJECT.md)）。

**状态**（2026-10-10）：S1 完成（含冒烟验收）；S2 检索层完成（BM25 + BGE-M3 稠密混合，musique hybrid recall@20 = 0.698），L0 hybrid 基线 musique 部分出数 150/300；**S3 wiki 分层摘要索引实现完成**（聚类 → LLM 条目 → see-also → 三路检索，冒烟通过），全量构建进行中。逐单元进展与后续计划见 [PROJECT.md](PROJECT.md) 第 8/11 节。
