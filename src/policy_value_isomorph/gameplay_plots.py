from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameplayPoint:
    step: int
    score: float
    win_rate: float
    draw_rate: float
    loss_rate: float


def _scale(values: list[float], size: float, pad: float) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [pad + size / 2.0 for _ in values]
    return [pad + ((v - lo) / (hi - lo)) * size for v in values]


def _polyline_svg(points: list[GameplayPoint], width: int = 760, height: int = 440) -> str:
    pad = 50.0
    draw_w = width - 2 * pad
    draw_h = height - 2 * pad

    xs = [float(p.step) for p in points]
    score_ys = [p.score for p in points]
    win_ys = [p.win_rate for p in points]
    draw_ys = [p.draw_rate for p in points]
    loss_ys = [p.loss_rate for p in points]

    sx = _scale(xs, draw_w, pad)
    all_y = score_ys + win_ys + draw_ys + loss_ys
    if all_y:
        lo = min(all_y)
        hi = max(all_y)
    else:
        lo, hi = -1.0, 1.0

    sy_score = _scale(score_ys, draw_h, pad) if hi != lo else [pad + draw_h / 2.0 for _ in score_ys]
    sy_win = _scale(win_ys, draw_h, pad) if hi != lo else [pad + draw_h / 2.0 for _ in win_ys]
    sy_draw = _scale(draw_ys, draw_h, pad) if hi != lo else [pad + draw_h / 2.0 for _ in draw_ys]
    sy_loss = _scale(loss_ys, draw_h, pad) if hi != lo else [pad + draw_h / 2.0 for _ in loss_ys]
    sy_score = [height - y for y in sy_score]
    sy_win = [height - y for y in sy_win]
    sy_draw = [height - y for y in sy_draw]
    sy_loss = [height - y for y in sy_loss]

    score_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx, sy_score))
    win_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx, sy_win))
    draw_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx, sy_draw))
    loss_polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx, sy_loss))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>\n'
        f'  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>\n'
        f'  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>\n'
        f'  <polyline points="{score_polyline}" fill="none" stroke="seagreen" stroke-width="2"/>\n'
        f'  <polyline points="{win_polyline}" fill="none" stroke="royalblue" stroke-width="2"/>\n'
        f'  <polyline points="{draw_polyline}" fill="none" stroke="darkorange" stroke-width="2"/>\n'
        f'  <polyline points="{loss_polyline}" fill="none" stroke="firebrick" stroke-width="2"/>\n'
        f'  <text x="{width/2:.1f}" y="25" text-anchor="middle" font-size="16">Gameplay score by checkpoint step</text>\n'
        f'  <text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="12">step</text>\n'
        f'  <text x="18" y="{height/2:.1f}" transform="rotate(-90,18,{height/2:.1f})" text-anchor="middle" font-size="12">rate / score</text>\n'
        f'  <text x="{width-160}" y="{pad+15}" font-size="12" fill="seagreen">score</text>\n'
        f'  <text x="{width-160}" y="{pad+32}" font-size="12" fill="royalblue">win_rate</text>\n'
        f'  <text x="{width-160}" y="{pad+49}" font-size="12" fill="darkorange">draw_rate</text>\n'
        f'  <text x="{width-160}" y="{pad+66}" font-size="12" fill="firebrick">loss_rate</text>\n'
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
            points.append(
                GameplayPoint(
                    step=int(row["step"]),
                    score=float(row["score"]),
                    win_rate=float(row.get("win_rate", 0.0)),
                    draw_rate=float(row.get("draw_rate", 0.0)),
                    loss_rate=float(row.get("loss_rate", 0.0)),
                )
            )

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
            f"last_rates: win={points[-1].win_rate:.4f}, draw={points[-1].draw_rate:.4f}, loss={points[-1].loss_rate:.4f}\n"
        )

    out_path = out_dir / "gameplay_eval_report.txt"
    out_path.write_text(body, encoding="utf-8")
    return out_path
