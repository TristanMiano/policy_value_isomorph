"""Policy-to-value reconstruction demo package (Task 1 tic-tac-toe vertical slice)."""

from .tictactoe import TicTacToeState, check_winner
from .policy import heuristic_policy_action
from .policy_mlp import TinyMLPPolicy, TrainedPolicy, policy_mlp_action, train_policy_mlp
from .rollout_value import estimate_v_pi, recovered_action_from_v
from .sampling import StateActionSample, generate_off_policy_dataset, generate_on_policy_dataset

__all__ = [
    "TicTacToeState",
    "check_winner",
    "heuristic_policy_action",
    "TinyMLPPolicy",
    "TrainedPolicy",
    "train_policy_mlp",
    "policy_mlp_action",
    "estimate_v_pi",
    "recovered_action_from_v",
    "StateActionSample",
    "generate_on_policy_dataset",
    "generate_off_policy_dataset",
]
