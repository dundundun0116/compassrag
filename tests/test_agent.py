from compassrag.agent import Agent, AgentConfig
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
        self.calls = []

    def chat(self, messages, *, tag="chat", temperature=0.0, max_tokens=None, model=None):
        self.calls.append({"messages": messages, "tag": tag, "max_tokens": max_tokens})
        return LLMResponse(text=self.text, model="fake", prompt_tokens=10, completion_tokens=2,
                           total_tokens=12, latency_ms=1.0)


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
