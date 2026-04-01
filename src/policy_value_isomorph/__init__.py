"""Policy-to-value reconstruction demo package (Task 1 tic-tac-toe vertical slice)."""

from .tictactoe import TicTacToeState, check_winner
from .policy import heuristic_policy_action
from .policy_mlp import TinyMLPPolicy, TrainedPolicy, policy_mlp_action, train_policy_mlp
from .evaluation import (
    CalibrationBin,
    CalibrationCurve,
    WinDrawLossRate,
    action_agreement_rate,
    top_k_agreement_rate,
    value_calibration_curve,
    win_draw_loss_rate,
)
from .rollout_value import (
    StateValueTarget,
    estimate_v_pi,
    frozen_policy_from_mlp,
    generate_value_targets,
    recovered_action_from_v,
)
from .sampling import StateActionSample, generate_off_policy_dataset, generate_on_policy_dataset
from .sampling import augment_dataset_with_symmetries, reduce_dataset_by_canonical_symmetry
from .symmetry import SYMMETRIES, canonicalize_state, canonicalize_state_action, symmetric_states
from .value_mlp import TinyMLPValue, TrainedValue, train_value_mlp, value_mlp_predict
from .q_mlp import (
    StateActionValueTarget,
    TinyMLPQ,
    TrainedQ,
    estimate_q_pi,
    generate_q_targets,
    q_mlp_predict,
    recovered_action_from_q,
    train_q_mlp,
)

__all__ = [
    "TicTacToeState",
    "check_winner",
    "heuristic_policy_action",
    "TinyMLPPolicy",
    "TrainedPolicy",
    "train_policy_mlp",
    "policy_mlp_action",
    "action_agreement_rate",
    "top_k_agreement_rate",
    "win_draw_loss_rate",
    "WinDrawLossRate",
    "value_calibration_curve",
    "CalibrationBin",
    "CalibrationCurve",
    "estimate_v_pi",
    "StateValueTarget",
    "frozen_policy_from_mlp",
    "generate_value_targets",
    "recovered_action_from_v",
    "StateActionSample",
    "generate_on_policy_dataset",
    "generate_off_policy_dataset",
    "augment_dataset_with_symmetries",
    "reduce_dataset_by_canonical_symmetry",
    "SYMMETRIES",
    "symmetric_states",
    "canonicalize_state",
    "canonicalize_state_action",
    "TinyMLPValue",
    "TrainedValue",
    "train_value_mlp",
    "value_mlp_predict",
    "StateActionValueTarget",
    "TinyMLPQ",
    "TrainedQ",
    "estimate_q_pi",
    "generate_q_targets",
    "train_q_mlp",
    "q_mlp_predict",
    "recovered_action_from_q",
]
