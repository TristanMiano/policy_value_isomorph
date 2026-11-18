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
from policy_value_isomorph.q_mlp import generate_q_targets, recovered_action_from_q, train_q_mlp
from policy_value_isomorph.rollout_value import estimate_v_pi, generate_value_targets, recovered_action_from_v
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.tictactoe import Move, TicTacToeState, state_from_rows
from policy_value_isomorph.value_mlp import train_value_mlp, value_mlp_predict


def _recovered_action_from_value_network(state: TicTacToeState, root_player: int, predict_value) -> Move:
    legal = state.legal_moves()
    if not legal:
        raise ValueError("value-network recovery called on terminal state")

    scored = []
    for mv in legal:
        nxt = state.apply_move(mv)
        scored.append((mv, predict_value(nxt)))

    if state.to_move == root_player:
        return max(scored, key=lambda t: (t[1], -t[0]))[0]
    return min(scored, key=lambda t: (t[1], t[0]))[0]


def main() -> None:
    root_player = 1  # X
    print("The original player takes an immediate win, blocks one, takes the center, takes a corner, "
          "or takes any legal cell, in that order. Ties go to the lowest cell number, 0 to 8 across the rows.")
    print("X earns +1 for a win, 0 for a draw and -1 for a loss. For each candidate move, "
          "finish the game with the original rule on both sides. X chooses the largest payoff and O the smallest.")
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
        print(f"X payoff when both sides follow the original rule from this board = {v:+.2f}")
        print(f"Move chosen by comparing the payoffs of the resulting boards = {chosen}")

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

    q_targets = generate_q_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[1])
    trained_q = train_q_mlp(q_targets, hidden_dim=24, learning_rate=0.03, epochs=60, seed=11)

    v_recovered_agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: _recovered_action_from_value_network(
            s,
            root_player=1,
            predict_value=lambda st: value_mlp_predict(st, trained_value.model),
        ),
    )
    q_recovered_agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: recovered_action_from_q(s, trained_q.model, root_player=1),
    )

    print(f"\nMove agreement is the fraction of {len(states)} training decisions, "
          f"including repeated visits to {len(set(states))} distinct boards, that match the original move.")
    print(f"A network trained to copy the original move agreed on {agree:.1%}.")
    print(f"Rank its chosen move first, then all other legal cells by number. "
          f"The original move appeared in the first three on {top3:.1%} of decisions.")
    print(f"Over 30 games against the original player (15 as X and 15 as O), "
          f"the copy won {wdl.win_rate:.1%}, drew {wdl.draw_rate:.1%} and lost {wdl.loss_rate:.1%}.")
    print(f"A network trained to predict X's payoff from the board after each move agreed on {v_recovered_agree:.1%}.")
    print(f"A network trained to predict X's payoff from the current board and move agreed on {q_recovered_agree:.1%}.")

    print("\nGroup the training decisions by the board network's predicted X payoff. "
          "Give the number of decisions and the two average payoffs in each group. Empty groups have no average.")
    print("lower,upper,decisions,mean_predicted_X_payoff,mean_observed_X_payoff")
    for b in calibration.bins:
        averages = f"{b.mean_pred:+.3f},{b.mean_true:+.3f}" if b.count else ","
        print(f"{b.lower:+.2f},{b.upper:+.2f},{b.count},{averages}")


if __name__ == "__main__":
    main()
