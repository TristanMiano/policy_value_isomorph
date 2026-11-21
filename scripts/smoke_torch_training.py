"""Exercise neural policy fitting, frozen-player payoff labels, evaluation and resume."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from policy_value_isomorph.evaluation import win_draw_loss_rate
from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.q_mlp import generate_q_targets, recovered_action_from_q, train_q_mlp
from policy_value_isomorph.rollout_value import frozen_policy_from_mlp, generate_value_targets
from policy_value_isomorph.sampling import StateActionSample, generate_off_policy_dataset
from policy_value_isomorph.telemetry_io import append_telemetry_csv, append_telemetry_jsonl, telemetry_run_dir
from policy_value_isomorph.training_config import TrainingConfig
from policy_value_isomorph.value_mlp import train_value_mlp, value_mlp_predict


def smoke(directory: Path, device: str) -> None:
    seed = 28
    config = TrainingConfig(optimizer="adamw", deterministic=True, device=device)
    checkpoints = directory / "checkpoints"
    states = list(dict.fromkeys(s.state for s in generate_off_policy_dataset(3, seed=seed)))

    def fit(train, data):
        first = train(data, hidden_dim=8, learning_rate=0.003, epochs=1, seed=seed,
                      config=config, checkpoint_dir=checkpoints)
        last = train(data, epochs=2, checkpoint_dir=checkpoints, resume_latest=True)
        for segment in (first, last):
            log = segment.training_log.telemetry
            dest = telemetry_run_dir(directory / "telemetry", log.game, log.run_type, log.seed)
            append_telemetry_csv(dest / "training.csv", log)
            append_telemetry_jsonl(dest / "training.jsonl", log)
        return last.model

    actions = [StateActionSample(s, heuristic_policy_action(s)) for s in states]
    policy = frozen_policy_from_mlp(fit(train_policy_mlp, actions))
    value = fit(train_value_mlp, generate_value_targets(states, policy, root_player=1, rollout_budgets=[1]))
    q = fit(train_q_mlp, generate_q_targets(states, policy, root_player=1, rollout_budgets=[1]))

    def choose_value(s):
        return max(s.legal_moves(), key=lambda a: (s.to_move * value_mlp_predict(s.apply_move(a), value), -a))

    print("The reference opponent takes a win, blocks a win, takes the center, a corner, or any legal cell")
    print("in that order, using the lowest cell number within each priority. The frozen policy chooses")
    print("the highest legal score after training on that opponent's moves. The two payoff networks")
    print("predict X's points from the resulting board or the current board and candidate move.")
    print("X maximizes, O minimizes, and equal scores resolve to the lowest cell number.")
    print("Matches count exact frozen-policy moves on the training boards. Game counts cover both seats.")
    for name, player in (("Frozen policy", policy), ("Resulting board", choose_value),
                         ("Board and move", lambda s: recovered_action_from_q(s, q, root_player=1))):
        matches = sum(player(s) == policy(s) for s in states)
        games = win_draw_loss_rate(player, heuristic_policy_action, n_games=2)
        print(f"{name} matched {matches}/{len(states)} moves, with win/draw/loss counts "
              f"{round(2 * games.win_rate)}/{round(2 * games.draw_rate)}/{round(2 * games.loss_rate)} in 2 games")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, help="Keep the run in a fresh subdirectory under this path")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="smoke-", dir=args.output_dir))
        smoke(directory, args.device)
        print(f"Saved training checkpoints and logs in {directory}")
    else:
        with tempfile.TemporaryDirectory() as directory:
            smoke(Path(directory), args.device)
