"""agent 主循环骨架。

S2 阶段跑 scripted（固定流程）模式 = 消融的关闭态：
问题 → 工具检索（多路 RRF 融合）→ 证据上下文 → 直接生成 → 收手。
S3/S4 给 tools 加能力（wiki 索引视图 / 改写）；S5 起 use_decision=True，
控制流（路由 / 预算 / 早停）交给 LLM，循环变为完整的 agentic 迭代。
"""

from dataclasses import dataclass

from .retrieval.fusion import rrf_fuse

ANSWER_PROMPT = """Answer the question based only on the evidence passages below.
Give a short, direct answer (a few words). If the evidence is insufficient, answer "unknown".

Evidence:
{context}

Question: {question}
Answer:"""


@dataclass
class AgentConfig:
    top_k: int = 10
    max_rounds: int = 1          # scripted：固定一轮
    use_rewrite: bool = False    # S4：多跳分解 + HyDE
    use_wiki: bool = False       # S3：wiki 索引视图
    use_decision: bool = False   # S5：LLM 接管控制流
    answer_max_tokens: int = 32


@dataclass
class AgentResult:
    answer: str
    evidence: list[str]          # 送入上下文的 chunk_id（按融合排名）
    rounds: int
    prompt_tokens: int
    completion_tokens: int


class Agent:
    """tools: {name: obj.search(query, k) -> [(chunk_id, score)]}；chunk_store: {chunk_id: chunk}。"""

    def __init__(self, llm, tools: dict, chunk_store: dict, config: AgentConfig | None = None):
        self.llm = llm
        self.tools = tools
        self.store = chunk_store
        self.config = config or AgentConfig()

    def answer(self, question: str) -> AgentResult:
        routes = [tool.search(question, k=self.config.top_k) for tool in self.tools.values()]
        fused = rrf_fuse([[cid for cid, _ in route] for route in routes], top_n=self.config.top_k)
        evidence = [cid for cid, _ in fused]

        resp = self.llm.chat(
            [{"role": "system",
              "content": "You answer multi-hop questions using only the provided evidence."},
             {"role": "user",
              "content": ANSWER_PROMPT.format(context=self._format(evidence), question=question)}],
            tag="answer", max_tokens=self.config.answer_max_tokens)
        return AgentResult(answer=resp.text.strip(), evidence=evidence, rounds=1,
                           prompt_tokens=resp.prompt_tokens,
                           completion_tokens=resp.completion_tokens)

    def _format(self, evidence_ids: list[str]) -> str:
        lines = []
        for cid in evidence_ids:
            c = self.store.get(cid)
            if c is not None:
                lines.append(f"[{len(lines) + 1}] {c['title']}: {c['text']}")
        return "\n".join(lines) if lines else "(no evidence retrieved)"
