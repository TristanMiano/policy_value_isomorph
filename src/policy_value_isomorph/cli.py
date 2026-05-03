from __future__ import annotations

import argparse

from .evaluation import action_agreement_rate, win_draw_loss_rate
from .gameplay_plots import write_gameplay_eval_report_from_jsonl, write_gameplay_score_plot_from_jsonl
from .policy import heuristic_policy_action
from .policy_mlp import policy_mlp_action, train_policy_mlp
from .rollout_value import generate_value_targets
from .sampling import generate_off_policy_dataset, generate_on_policy_dataset
from .telemetry_io import append_gameplay_eval_jsonl, append_telemetry_csv, append_telemetry_jsonl, telemetry_run_dir
from .training_plots import write_loss_plots_from_csv
from .value_mlp import train_value_mlp, value_mlp_predict
from .crash_logging import CrashLoggerConfig, enable_crash_logging


def _cmd_data(args: argparse.Namespace) -> None:
    if args.mode == "on":
        dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=args.episodes)
    else:
        dataset = generate_off_policy_dataset(n_episodes=args.episodes, seed=args.seed)

    print(f"dataset_mode={args.mode}")
    print(f"episodes={args.episodes}")
    print(f"samples={len(dataset)}")


def _cmd_train(args: argparse.Namespace) -> None:
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=args.episodes)
    trained_policy = train_policy_mlp(
        dataset,
        hidden_dim=args.policy_hidden,
        learning_rate=args.learning_rate,
        epochs=args.policy_epochs,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_n_games=args.eval_games,
        gameplay_samples_path=args.gameplay_samples_path,
        gameplay_samples_per_checkpoint=args.gameplay_samples_per_checkpoint,
        checkpoint_interval=args.policy_checkpoint_interval,
        checkpoint_dir=args.policy_checkpoint_dir,
        resume_checkpoint_path=args.policy_resume_checkpoint,
        resume_latest=args.policy_resume_latest,
    )

    states = [sample.state for sample in dataset]
    value_targets = generate_value_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[args.rollouts])
    trained_value = train_value_mlp(
        value_targets,
        hidden_dim=args.value_hidden,
        learning_rate=args.learning_rate,
        epochs=args.value_epochs,
        seed=args.seed,
        checkpoint_interval=args.value_checkpoint_interval,
        checkpoint_dir=args.value_checkpoint_dir,
        resume_checkpoint_path=args.value_resume_checkpoint,
        resume_latest=args.value_resume_latest,
    )

    if args.telemetry_root_dir:
        policy_run_dir = telemetry_run_dir(args.telemetry_root_dir, game="tic_tac_toe", run_type="policy", seed=args.seed)
        value_run_dir = telemetry_run_dir(args.telemetry_root_dir, game="tic_tac_toe", run_type="value", seed=args.seed)

        append_telemetry_csv(policy_run_dir / "training_telemetry.csv", trained_policy.training_log.telemetry)
        append_telemetry_jsonl(policy_run_dir / "training_telemetry.jsonl", trained_policy.training_log.telemetry)
        if trained_policy.training_log.gameplay_evals:
            append_gameplay_eval_jsonl(policy_run_dir / "gameplay_eval.jsonl", trained_policy.training_log.gameplay_evals)

        append_telemetry_csv(value_run_dir / "training_telemetry.csv", trained_value.training_log.telemetry)
        append_telemetry_jsonl(value_run_dir / "training_telemetry.jsonl", trained_value.training_log.telemetry)

    print(f"policy_loss_initial={trained_policy.training_log.losses[0]:.6f}")
    print(f"policy_loss_final={trained_policy.training_log.losses[-1]:.6f}")
    print(f"value_loss_initial={trained_value.training_log.losses[0]:.6f}")
    print(f"value_loss_final={trained_value.training_log.losses[-1]:.6f}")


def _cmd_eval(args: argparse.Namespace) -> None:
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=args.episodes)
    trained_policy = train_policy_mlp(dataset, hidden_dim=args.policy_hidden, epochs=args.epochs, seed=args.seed)

    states = [sample.state for sample in dataset]
    agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: policy_mlp_action(s, trained_policy.model),
    )
    wdl = win_draw_loss_rate(
        player_policy=lambda s: policy_mlp_action(s, trained_policy.model),
        opponent_policy=heuristic_policy_action,
        n_games=args.games,
    )

    value_targets = generate_value_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[args.rollouts])
    trained_value = train_value_mlp(value_targets, hidden_dim=args.value_hidden, epochs=args.epochs, seed=args.seed)
    first_pred = value_mlp_predict(states[0], trained_value.model)

    print(f"action_agreement={agree:.6f}")
    print(f"wdl={wdl.win_rate:.3f}/{wdl.draw_rate:.3f}/{wdl.loss_rate:.3f}")
    print(f"first_state_value_pred={first_pred:+.6f}")


def _cmd_plots(args: argparse.Namespace) -> None:
    step_svg, time_svg = write_loss_plots_from_csv(args.telemetry_csv, args.output_dir)
    print(f"loss_step_plot={step_svg}")
    print(f"loss_time_plot={time_svg}")

    if args.gameplay_jsonl:
        gameplay_svg = write_gameplay_score_plot_from_jsonl(args.gameplay_jsonl, args.output_dir)
        report_txt = write_gameplay_eval_report_from_jsonl(args.gameplay_jsonl, args.output_dir)
        print(f"gameplay_plot={gameplay_svg}")
        print(f"gameplay_report={report_txt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI entrypoints for tic-tac-toe data, training, and evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Generate and summarize a dataset.")
    p_data.add_argument("--mode", choices=["on", "off"], default="on")
    p_data.add_argument("--episodes", type=int, default=50)
    p_data.add_argument("--seed", type=int, default=0)
    p_data.set_defaults(func=_cmd_data)

    p_train = sub.add_parser("train", help="Train policy and value MLPs and report losses.")
    p_train.add_argument("--episodes", type=int, default=120)
    p_train.add_argument("--policy-hidden", type=int, default=24)
    p_train.add_argument("--value-hidden", type=int, default=24)
    p_train.add_argument("--policy-epochs", type=int, default=60)
    p_train.add_argument("--value-epochs", type=int, default=60)
    p_train.add_argument("--learning-rate", type=float, default=0.03)
    p_train.add_argument("--rollouts", type=int, default=1)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--eval-interval", type=int, default=0)
    p_train.add_argument("--eval-games", type=int, default=20)
    p_train.add_argument("--gameplay-samples-path", type=str, default=None)
    p_train.add_argument("--gameplay-samples-per-checkpoint", type=int, default=2)
    p_train.add_argument("--policy-checkpoint-interval", type=int, default=0)
    p_train.add_argument("--policy-checkpoint-dir", type=str, default=None)
    p_train.add_argument("--policy-resume-checkpoint", type=str, default=None)
    p_train.add_argument("--policy-resume-latest", action="store_true")
    p_train.add_argument("--value-checkpoint-interval", type=int, default=0)
    p_train.add_argument("--value-checkpoint-dir", type=str, default=None)
    p_train.add_argument("--value-resume-checkpoint", type=str, default=None)
    p_train.add_argument("--value-resume-latest", action="store_true")
    p_train.add_argument("--telemetry-root-dir", type=str, default=None)
    p_train.add_argument("--crash-log-enabled", action="store_true")
    p_train.add_argument("--crash-log-path", type=str, default=None)
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="Train a quick policy and report evaluation metrics.")
    p_eval.add_argument("--episodes", type=int, default=90)
    p_eval.add_argument("--policy-hidden", type=int, default=24)
    p_eval.add_argument("--value-hidden", type=int, default=24)
    p_eval.add_argument("--epochs", type=int, default=40)
    p_eval.add_argument("--games", type=int, default=30)
    p_eval.add_argument("--rollouts", type=int, default=1)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.set_defaults(func=_cmd_eval)

    p_plots = sub.add_parser("plots", help="Regenerate training/gameplay plots from saved logs.")
    p_plots.add_argument("--telemetry-csv", required=True)
    p_plots.add_argument("--gameplay-jsonl", default=None)
    p_plots.add_argument("--output-dir", required=True)
    p_plots.set_defaults(func=_cmd_plots)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    enable_crash_logging(
        CrashLoggerConfig(
            enabled=bool(getattr(args, "crash_log_enabled", False)),
            log_path=getattr(args, "crash_log_path", None),
        )
    )

    args.func(args)


if __name__ == "__main__":
    main()
