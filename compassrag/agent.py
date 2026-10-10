"""agent 主循环骨架。

S2 阶段跑 scripted（固定流程）模式 = 消融的关闭态：
问题 → 工具检索（多路 RRF 融合）→ 证据上下文 → 直接生成 → 收手。
S3/S4 给 tools 加能力（wiki 索引视图 / 改写）；S5 起 use_decision=True，
控制流（路由 / 预算 / 早停）交给 LLM，循环变为完整的 agentic 迭代。
"""

from dataclasses import dataclass, field

from .retrieval.fusion import rrf_fuse

ANSWER_PROMPT = """Answer the question based only on the evidence passages below.
Give a short, direct answer (a few words). If the evidence is insufficient, answer "unknown".

Evidence:
{context}

Question: {question}
Answer:"""

RETRY_NUDGE = ("Conclude now with a few words: the answer if the evidence contains it, "
               "otherwise \"unknown\".")


@dataclass
class AgentConfig:
    top_k: int = 10
    max_rounds: int = 1          # scripted：固定一轮
    use_rewrite: bool = False    # S4：多跳分解 + HyDE
    use_wiki: bool = False       # S3：wiki 索引视图
    use_decision: bool = False   # S5：LLM 接管控制流
    answer_max_tokens: int = 1024        # 含思维链；过小会让推理吃光预算、答案为空
    answer_retry_max_tokens: int = 2048  # 空答案重试的更大预算


@dataclass
class AgentResult:
    answer: str
    evidence: list[str]          # 送入上下文的 chunk_id（按融合排名）
    rounds: int
    prompt_tokens: int
    completion_tokens: int
    finish_reasons: list[str] = field(default_factory=list)
    n_calls: int = 1


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

        messages = [
            {"role": "system",
             "content": "You answer multi-hop questions using only the provided evidence."},
            {"role": "user",
             "content": ANSWER_PROMPT.format(context=self._format(evidence), question=question)},
        ]
        resp = self.llm.chat(messages, tag="answer", max_tokens=self.config.answer_max_tokens)
        calls = [resp]
        # 推理链吃光预算（finish_reason=length）时 content 为空；带收尾提示重试一次可破打转
        if not resp.text.strip():
            resp = self.llm.chat(messages + [{"role": "user", "content": RETRY_NUDGE}],
                                 tag="answer_retry", max_tokens=self.config.answer_retry_max_tokens)
            calls.append(resp)
        return AgentResult(answer=resp.text.strip(), evidence=evidence, rounds=1,
                           prompt_tokens=sum(c.prompt_tokens for c in calls),
                           completion_tokens=sum(c.completion_tokens for c in calls),
                           finish_reasons=[c.finish_reason for c in calls],
                           n_calls=len(calls))

    def _format(self, evidence_ids: list[str]) -> str:
        lines = []
        for cid in evidence_ids:
            c = self.store.get(cid)
            if c is not None:
                lines.append(f"[{len(lines) + 1}] {c['title']}: {c['text']}")
        return "\n".join(lines) if lines else "(no evidence retrieved)"
