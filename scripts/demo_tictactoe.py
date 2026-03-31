from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_value_isomorph.evaluation import (
    action_agreement_rate,
    top_k_agreement_rate,
    value_calibration_curve,
    win_draw_loss_rate,
)
from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import policy_mlp_action, train_policy_mlp
from policy_value_isomorph.rollout_value import estimate_v_pi, generate_value_targets, recovered_action_from_v
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.tictactoe import TicTacToeState, state_from_rows
from policy_value_isomorph.value_mlp import train_value_mlp, value_mlp_predict


def main() -> None:
    root_player = 1  # X
    sample_states = [
        TicTacToeState.initial(),
        state_from_rows(["XX.", "OO.", "..."], to_move=1),
        state_from_rows(["XO.", "X..", ".O."], to_move=-1),
    ]

    for i, state in enumerate(sample_states, start=1):
        print(f"\n=== Sample state {i} ===")
        print(state.as_pretty_string())
        v = estimate_v_pi(state, policy=heuristic_policy_action, root_player=root_player, n_rollouts=1)
        chosen = recovered_action_from_v(
            state,
            policy=heuristic_policy_action,
            root_player=root_player,
            n_rollouts=1,
        )
        print(f"Recovered V^pi(s) from root X perspective: {v:+.2f}")
        print(f"Recovered-policy chosen move index: {chosen}")

    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=90)
    states = [sample.state for sample in dataset[:180]]

    trained_policy = train_policy_mlp(dataset, hidden_dim=24, learning_rate=0.03, epochs=60, seed=7)

    agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: policy_mlp_action(s, trained_policy.model),
    )
    top3 = top_k_agreement_rate(
        states,
        reference_policy=heuristic_policy_action,
        scored_policy=lambda s, mv: 1.0 if mv == policy_mlp_action(s, trained_policy.model) else 0.0,
        k=3,
    )
    wdl = win_draw_loss_rate(
        player_policy=lambda s: policy_mlp_action(s, trained_policy.model),
        opponent_policy=heuristic_policy_action,
        n_games=30,
    )

    value_targets = generate_value_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[1])
    trained_value = train_value_mlp(value_targets, hidden_dim=24, learning_rate=0.03, epochs=60, seed=9)
    calibration = value_calibration_curve(
        value_targets,
        predict_value=lambda s: value_mlp_predict(s, trained_value.model),
        n_bins=6,
    )

    print("\n=== Evaluation metrics ===")
    print(f"Action agreement (policy argmax vs heuristic): {agree:.3f}")
    print(f"Top-3 agreement: {top3:.3f}")
    print(f"Win/draw/loss vs heuristic: {wdl.win_rate:.3f}/{wdl.draw_rate:.3f}/{wdl.loss_rate:.3f}")

    print("\n=== Value calibration bins (for plotting) ===")
    print("bin_range,count,mean_pred,mean_true")
    for b in calibration.bins:
        print(f"[{b.lower:+.2f},{b.upper:+.2f}],{b.count},{b.mean_pred:+.3f},{b.mean_true:+.3f}")


if __name__ == "__main__":
    main()
