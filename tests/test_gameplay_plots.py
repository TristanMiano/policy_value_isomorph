from policy_value_isomorph.gameplay_eval import GameplayEvalRecord
from policy_value_isomorph.gameplay_plots import (
    read_gameplay_curve_from_jsonl,
    write_gameplay_eval_report_from_jsonl,
    write_gameplay_score_plot_from_jsonl,
)
from policy_value_isomorph.telemetry_io import append_gameplay_eval_jsonl, telemetry_run_dir


def test_generate_gameplay_score_plot_and_report(tmp_path) -> None:
    run_dir = telemetry_run_dir(tmp_path, game="connect_four", run_type="policy", seed=17)
    jsonl_path = run_dir / "gameplay_eval.jsonl"

    records = [
        GameplayEvalRecord(step=10, n_games=20, win_rate=0.40, draw_rate=0.10, loss_rate=0.50, score=-0.10),
        GameplayEvalRecord(step=20, n_games=20, win_rate=0.55, draw_rate=0.10, loss_rate=0.35, score=0.20),
        GameplayEvalRecord(step=30, n_games=20, win_rate=0.50, draw_rate=0.15, loss_rate=0.35, score=0.15),
    ]
    append_gameplay_eval_jsonl(jsonl_path, records)

    points = read_gameplay_curve_from_jsonl(jsonl_path)
    assert [p.step for p in points] == [10, 20, 30]
    assert [round(p.score, 2) for p in points] == [-0.10, 0.20, 0.15]

    svg_path = write_gameplay_score_plot_from_jsonl(jsonl_path, run_dir)
    report_path = write_gameplay_eval_report_from_jsonl(jsonl_path, run_dir)

    assert svg_path.exists()
    assert "Gameplay score by checkpoint step" in svg_path.read_text(encoding="utf-8")

    report = report_path.read_text(encoding="utf-8")
    assert "checkpoints: 3" in report
    assert "best_step: 20" in report
