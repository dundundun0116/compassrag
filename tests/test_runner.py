import json

from compassrag.eval.runner import run_records


def test_processes_all_and_persists(tmp_path):
    out = tmp_path / "o.jsonl"
    rows, skipped = run_records(
        [{"id": "a"}, {"id": "b"}],
        lambda r: {"id": r["id"], "ok": True},
        out_path=out)
    assert len(rows) == 2 and skipped == 0
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["id"] for l in lines] == ["a", "b"]


def test_resume_skips_already_done_ids(tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_text(json.dumps({"id": "a", "ok": True}) + "\n")
    rows, skipped = run_records(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        lambda r: {"id": r["id"], "ok": True},
        out_path=out)
    assert skipped == 1
    assert [r["id"] for r in rows] == ["b", "c"]
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["id"] for l in lines] == ["a", "b", "c"]  # 原有结果保留在前
