from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameplayPoint:
    step: int
    score: float


def _scale(values: list[float], size: float, pad: float) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [pad + size / 2.0 for _ in values]
    return [pad + ((v - lo) / (hi - lo)) * size for v in values]


def _polyline_svg(points: list[GameplayPoint], width: int = 640, height: int = 420) -> str:
    pad = 50.0
    draw_w = width - 2 * pad
    draw_h = height - 2 * pad

    xs = [float(p.step) for p in points]
    ys = [p.score for p in points]

    sx = _scale(xs, draw_w, pad)
    sy = _scale(ys, draw_h, pad)
    sy = [height - y for y in sy]

    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx, sy))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>\n'
        f'  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>\n'
        f'  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>\n'
        f'  <polyline points="{polyline}" fill="none" stroke="seagreen" stroke-width="2"/>\n'
        f'  <text x="{width/2:.1f}" y="25" text-anchor="middle" font-size="16">Gameplay score by checkpoint step</text>\n'
        f'  <text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="12">step</text>\n'
        f'  <text x="18" y="{height/2:.1f}" transform="rotate(-90,18,{height/2:.1f})" text-anchor="middle" font-size="12">score</text>\n'
        "</svg>\n"
    )


def read_gameplay_curve_from_jsonl(path: str | Path) -> list[GameplayPoint]:
    """Read gameplay eval JSONL and return score points sorted by step."""

    points: list[GameplayPoint] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            points.append(GameplayPoint(step=int(row["step"]), score=float(row["score"])))

    points.sort(key=lambda p: p.step)
    return points


def write_gameplay_score_plot_from_jsonl(jsonl_path: str | Path, output_dir: str | Path) -> Path:
    """Generate score-over-time SVG plot from gameplay evaluation JSONL."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = read_gameplay_curve_from_jsonl(jsonl_path)

    out_path = out_dir / "gameplay_score_by_step.svg"
    out_path.write_text(_polyline_svg(points), encoding="utf-8")
    return out_path


def write_gameplay_eval_report_from_jsonl(jsonl_path: str | Path, output_dir: str | Path) -> Path:
    """Write a compact, human-readable gameplay evaluation report."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = read_gameplay_curve_from_jsonl(jsonl_path)
    if not points:
        body = "No gameplay evaluation records found.\n"
    else:
        best = max(points, key=lambda p: p.score)
        body = (
            f"checkpoints: {len(points)}\n"
            f"first_step: {points[0].step}, first_score: {points[0].score:.4f}\n"
            f"last_step: {points[-1].step}, last_score: {points[-1].score:.4f}\n"
            f"best_step: {best.step}, best_score: {best.score:.4f}\n"
        )

    out_path = out_dir / "gameplay_eval_report.txt"
    out_path.write_text(body, encoding="utf-8")
    return out_path
