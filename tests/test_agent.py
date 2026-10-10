from compassrag.agent import RETRY_NUDGE, Agent, AgentConfig
from compassrag.llm.client import LLMResponse


class FakeTool:
    def __init__(self, ranking):
        self.ranking = ranking
        self.queries = []

    def search(self, query, k=10):
        self.queries.append(query)
        return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(self.ranking[:k])]


class FakeLLM:
    def __init__(self, text="Paris"):
        self.text = text
        self.queue: list[str] = []
        self.calls = []

    def script(self, texts):
        """按序返回给定文本，用完后回落 self.text。"""
        self.queue = list(texts)
        return self

    def chat(self, messages, *, tag="chat", temperature=0.0, max_tokens=None, model=None):
        self.calls.append({"messages": messages, "tag": tag, "max_tokens": max_tokens})
        text = self.queue.pop(0) if self.queue else self.text
        return LLMResponse(text=text, model="fake", prompt_tokens=10, completion_tokens=2,
                           total_tokens=12, latency_ms=1.0,
                           finish_reason="stop" if text.strip() else "length")


STORE = {
    "c1": {"chunk_id": "c1", "title": "Alpha", "text": "Alpha text."},
    "c2": {"chunk_id": "c2", "title": "Beta", "text": "Beta text."},
    "c3": {"chunk_id": "c3", "title": "Gamma", "text": "Gamma text."},
}


def test_scripted_single_round_single_tool():
    llm, tool = FakeLLM(), FakeTool(["c1", "c2", "c3"])
    agent = Agent(llm, {"bm25": tool}, STORE, AgentConfig(top_k=2))
    result = agent.answer("Where is Alpha?")
    assert result.answer == "Paris"
    assert result.rounds == 1
    assert result.evidence == ["c1", "c2"]  # top_k 截断
    assert len(llm.calls) == 1 and llm.calls[0]["tag"] == "answer"
    assert tool.queries == ["Where is Alpha?"]


def test_multi_tool_rrf_fusion_orders_by_consensus():
    a, b = FakeTool(["c1", "c2"]), FakeTool(["c2", "c3"])
    agent = Agent(FakeLLM(), {"bm25": a, "dense": b}, STORE, AgentConfig(top_k=3))
    result = agent.answer("q")
    assert result.evidence[0] == "c2"  # 两路都命中，RRF 最高


def test_prompt_carries_evidence_and_question():
    llm = FakeLLM()
    agent = Agent(llm, {"bm25": FakeTool(["c1"])}, STORE, AgentConfig(top_k=1))
    agent.answer("Who is Beta?")
    user_content = llm.calls[0]["messages"][-1]["content"]
    assert "Alpha" in user_content and "Alpha text." in user_content
    assert "Who is Beta?" in user_content


def test_no_evidence_still_answers_with_placeholder():
    agent = Agent(FakeLLM(), {"bm25": FakeTool([])}, STORE, AgentConfig(top_k=3))
    result = agent.answer("q")
    assert result.evidence == [] and result.answer == "Paris"


def test_empty_answer_triggers_nudge_retry_with_bigger_budget():
    llm = FakeLLM().script(["", "Paris"])  # 首轮撞预算为空 → 重试出答案
    agent = Agent(llm, {"bm25": FakeTool(["c1"])}, STORE, AgentConfig(top_k=1))
    result = agent.answer("q")
    assert result.answer == "Paris" and result.n_calls == 2
    assert [c["tag"] for c in llm.calls] == ["answer", "answer_retry"]
    retry_msgs = llm.calls[1]["messages"]
    assert retry_msgs[-1]["content"] == RETRY_NUDGE
    assert llm.calls[1]["max_tokens"] > llm.calls[0]["max_tokens"]
    assert result.finish_reasons == ["length", "stop"]
    assert result.prompt_tokens == 20  # 两次调用聚合


def test_empty_after_retry_kept_empty_with_flags():
    llm = FakeLLM().script(["", ""])
    agent = Agent(llm, {"bm25": FakeTool(["c1"])}, STORE, AgentConfig(top_k=1))
    result = agent.answer("q")
    assert result.answer == "" and result.n_calls == 2
    assert result.finish_reasons == ["length", "length"]


def test_nonempty_answer_skips_retry():
    llm = FakeLLM()
    agent = Agent(llm, {"bm25": FakeTool(["c1"])}, STORE, AgentConfig(top_k=1))
    result = agent.answer("q")
    assert result.n_calls == 1 and result.finish_reasons == ["stop"]
    assert [c["tag"] for c in llm.calls] == ["answer"]
