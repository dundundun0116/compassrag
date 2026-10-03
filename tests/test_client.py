import json

import pytest

from compassrag.llm.client import LLMClient, LLMError, LLMResponse, batch_texts


def _make_client(tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://example.invalid/v1\n"
        "LLM_API_KEY=sk-test\n"
        "LLM_MODEL=test-model\n"
        "EMBEDDING_MODEL=test-embed\n"
    )
    return LLMClient(repo_root=tmp_path)


def test_batch_texts_sizes():
    assert batch_texts(list(range(250)), 96) == [list(range(96)), list(range(96, 192)), list(range(192, 250))]
    assert batch_texts(["a"], 8) == [["a"]]


def test_missing_env_raises_actionable_error(tmp_path, monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMError, match=r"\.env"):
        LLMClient(repo_root=tmp_path)


def test_init_reads_config_and_defaults(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LLM_BASE_URL=https://example.invalid/v1\n"
        "LLM_API_KEY=sk-test\n"
        "LLM_MODEL=test-model\n"
    )
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "models.yaml").write_text(
        "request:\n  timeout_s: 5\n  max_retries: 1\n"
        "embedding:\n  batch_size: 7\n"
        "telemetry:\n  path: runs/t.jsonl\n"
    )
    client = LLMClient(repo_root=tmp_path)
    assert client.model == "test-model"
    assert client.embedding_model == "test-model"  # 未单独配置时回退主模型
    assert client._batch_size == 7
    assert client.telemetry.path == tmp_path / "runs" / "t.jsonl"


def test_llm_response_fields():
    r = LLMResponse(text="hi", model="m", prompt_tokens=1, completion_tokens=2, total_tokens=3, latency_ms=4.0)
    assert (r.text, r.model, r.total_tokens) == ("hi", "m", 3)


def test_chat_failure_recorded_then_raised(tmp_path):
    client = _make_client(tmp_path)

    def boom(**kwargs):
        raise RuntimeError("boom: quota exhausted")

    client._client.chat.completions.create = boom
    with pytest.raises(RuntimeError, match="boom"):
        client.chat([{"role": "user", "content": "hi"}], tag="smoke")
    rows = [json.loads(l) for l in (tmp_path / "runs" / "telemetry.jsonl").read_text().splitlines()]
    assert rows[-1]["kind"] == "chat" and rows[-1]["error"].startswith("boom")


def test_embed_failure_recorded_then_raised(tmp_path):
    client = _make_client(tmp_path)

    def boom(**kwargs):
        raise RuntimeError("boom: endpoint down")

    client._client.embeddings.create = boom
    with pytest.raises(RuntimeError, match="boom"):
        client.embed(["a", "b"], tag="smoke")
    rows = [json.loads(l) for l in (tmp_path / "runs" / "telemetry.jsonl").read_text().splitlines()]
    assert rows[-1]["kind"] == "embed" and rows[-1]["error"].startswith("boom")
