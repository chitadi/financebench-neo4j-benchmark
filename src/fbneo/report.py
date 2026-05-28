from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate_markdown(run_path: Path) -> str:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    results = payload.get("results", [])

    lines: list[str] = []
    lines.append(f"# FinanceBench Neo4j Run {payload.get('run_id', run_path.stem)}")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    for key, value in (payload.get("config") or {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in summary.items():
        lines.append(f"| {key} | {_fmt(value)} |")

    by_type: dict[str, list[dict]] = defaultdict(list)
    by_reasoning: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        by_type[item.get("question_type") or "unknown"].append(item)
        by_reasoning[item.get("question_reasoning") or "unknown"].append(item)

    def add_group(title: str, groups: dict[str, list[dict]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Group | N | Answer | Doc Recall | Page Recall | Avg Latency |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, items in sorted(groups.items()):
            def avg(metric: str) -> float | None:
                values = [
                    item.get("scores", {}).get(metric)
                    for item in items
                    if item.get("scores", {}).get(metric) is not None
                ]
                return sum(values) / len(values) if values else None

            latency = sum(item["timing"]["total_s"] for item in items) / len(items)
            lines.append(
                f"| {name} | {len(items)} | {_fmt(avg('answer_correct'))} | "
                f"{_fmt(avg('doc_recall'))} | {_fmt(avg('page_recall'))} | {_fmt(latency)} |"
            )

    add_group("By Question Type", by_type)
    add_group("By Reasoning", by_reasoning)

    worst = sorted(
        results,
        key=lambda item: (
            item.get("scores", {}).get("answer_correct") or 0,
            item.get("scores", {}).get("doc_recall") or 0,
            item.get("scores", {}).get("page_recall") or 0,
        ),
    )[:10]
    lines.append("")
    lines.append("## Lowest Scoring Questions")
    lines.append("")
    lines.append("| ID | Answer | Doc Recall | Page Recall | Question |")
    lines.append("|---|---:|---:|---:|---|")
    for item in worst:
        q = str(item.get("question", "")).replace("|", "\\|")
        lines.append(
            f"| {item.get('financebench_id')} | "
            f"{_fmt(item.get('scores', {}).get('answer_correct'))} | "
            f"{_fmt(item.get('scores', {}).get('doc_recall'))} | "
            f"{_fmt(item.get('scores', {}).get('page_recall'))} | "
            f"{q[:140]} |"
        )

    return "\n".join(lines) + "\n"


def write_report(run_path: Path) -> Path:
    report_path = run_path.with_suffix(".md")
    report_path.write_text(generate_markdown(run_path), encoding="utf-8")
    return report_path

