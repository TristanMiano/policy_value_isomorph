from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import policy_mlp_action, train_policy_mlp
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.tictactoe import TicTacToeState


def main() -> None:
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=200)
    trained = train_policy_mlp(dataset, hidden_dim=32, learning_rate=0.03, epochs=80, seed=0)

    print(f"Trained on {len(dataset)} samples")
    print(f"Initial loss: {trained.training_log.losses[0]:.4f}")
    print(f"Final loss:   {trained.training_log.losses[-1]:.4f}")

    start = TicTacToeState.initial()
    pred_move = policy_mlp_action(start, trained.model)
    heuristic_move = heuristic_policy_action(start)
    print(f"Start-state predicted move: {pred_move}")
    print(f"Start-state heuristic move: {heuristic_move}")


if __name__ == "__main__":
    main()
