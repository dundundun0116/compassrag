"""评测骨架：按题跑 step_fn，增量落盘、断点续跑（按 id 跳过已完成）。"""

import json
from pathlib import Path


def run_records(records: list[dict], step_fn, out_path: Path) -> tuple[list[dict], int]:
    """对 records 逐条调 step_fn(record) -> dict，追加写入 out_path。

    out_path 已有的 id 直接跳过（断点续跑）；返回 (本轮新跑的行, 跳过数)。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])

    new_rows: list[dict] = []
    skipped = 0
    with out_path.open("a", encoding="utf-8") as f:
        for r in records:
            if r["id"] in done:
                skipped += 1
                continue
            row = step_fn(r)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            new_rows.append(row)
    return new_rows, skipped
