from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.rollout_value import estimate_v_pi, recovered_action_from_v
from policy_value_isomorph.tictactoe import TicTacToeState, state_from_rows


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


if __name__ == "__main__":
    main()
