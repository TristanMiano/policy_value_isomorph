from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CurvePoint:
    x: float
    y: float


def _scale(values: list[float], size: float, pad: float) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [pad + size / 2.0 for _ in values]
    return [pad + ((v - lo) / (hi - lo)) * size for v in values]


def _polyline_svg(points: list[CurvePoint], x_label: str, title: str, width: int = 640, height: int = 420) -> str:
    pad = 50.0
    draw_w = width - 2 * pad
    draw_h = height - 2 * pad

    xs = [p.x for p in points]
    ys = [p.y for p in points]

    sx = _scale(xs, draw_w, pad)
    sy = _scale(ys, draw_h, pad)
    sy = [height - y for y in sy]  # SVG origin is top-left.

    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx, sy))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>\n'
        f'  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>\n'
        f'  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>\n'
        f'  <polyline points="{polyline}" fill="none" stroke="royalblue" stroke-width="2"/>\n'
        f'  <text x="{width/2:.1f}" y="25" text-anchor="middle" font-size="16">{title}</text>\n'
        f'  <text x="{width/2:.1f}" y="{height-10}" text-anchor="middle" font-size="12">{x_label}</text>\n'
        f'  <text x="18" y="{height/2:.1f}" transform="rotate(-90,18,{height/2:.1f})" text-anchor="middle" font-size="12">loss</text>\n'
        "</svg>\n"
    )


def read_training_curve_from_csv(path: str | Path, x_field: str) -> list[CurvePoint]:
    """Read telemetry CSV rows into an ordered training curve."""

    if x_field not in {"step", "wall_time_seconds"}:
        raise ValueError("x_field must be 'step' or 'wall_time_seconds'")

    rows: list[CurvePoint] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(CurvePoint(x=float(row[x_field]), y=float(row["loss"])))

    rows.sort(key=lambda p: p.x)
    return rows


def write_loss_plots_from_csv(csv_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Generate loss-vs-step and loss-vs-wall-time SVG plots from telemetry CSV."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_step = read_training_curve_from_csv(csv_path, x_field="step")
    by_time = read_training_curve_from_csv(csv_path, x_field="wall_time_seconds")

    step_svg = out_dir / "loss_by_step.svg"
    time_svg = out_dir / "loss_by_wall_time.svg"

    step_svg.write_text(_polyline_svg(by_step, x_label="step", title="Training loss by step"), encoding="utf-8")
    time_svg.write_text(
        _polyline_svg(by_time, x_label="wall_time_seconds", title="Training loss by wall-clock time"),
        encoding="utf-8",
    )

    return step_svg, time_svg
