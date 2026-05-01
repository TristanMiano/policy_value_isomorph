from __future__ import annotations

import argparse

from .evaluation import action_agreement_rate, win_draw_loss_rate
from .policy import heuristic_policy_action
from .policy_mlp import policy_mlp_action, train_policy_mlp
from .rollout_value import generate_value_targets
from .sampling import generate_off_policy_dataset, generate_on_policy_dataset
from .value_mlp import train_value_mlp, value_mlp_predict


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
    )

    states = [sample.state for sample in dataset]
    value_targets = generate_value_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[args.rollouts])
    trained_value = train_value_mlp(
        value_targets,
        hidden_dim=args.value_hidden,
        learning_rate=args.learning_rate,
        epochs=args.value_epochs,
        seed=args.seed,
    )

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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
