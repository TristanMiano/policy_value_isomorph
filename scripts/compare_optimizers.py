"""Run the opt-in, fixed CPU comparison in experiments/optimizer_comparison."""

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_value_isomorph.pilot_exact import play_match
from policy_value_isomorph.pilot_protocol import GameGraph, board_id, uniform_policy
from policy_value_isomorph.pilot_recovery import Recovery, predict, training_inputs
from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.q_mlp import generate_q_targets, train_q_mlp
from policy_value_isomorph.rollout_value import generate_value_targets
from policy_value_isomorph.sampling import StateActionSample
from policy_value_isomorph.training_config import TrainingConfig
from policy_value_isomorph.value_mlp import train_value_mlp


class SplitUpdate:
    """Apply two native optimizers to disjoint weights in the same training step."""

    def __init__(self, hidden, other):
        self.parts = hidden, other

    def zero_grad(self, **kwargs):
        for part in self.parts:
            part.zero_grad(**kwargs)

    def step(self):
        for part in self.parts:
            part.step()


class MuonComparison(TrainingConfig):
    def make_optimizer(self, model, learning_rate):
        hidden = torch.optim.Muon([model.w1], lr=learning_rate, weight_decay=self.decay,
                                  momentum=SPEC["muon_momentum"], ns_steps=SPEC["muon_steps"],
                                  eps=SPEC["muon_epsilon"], adjust_lr_fn="match_rms_adamw")
        other = torch.optim.AdamW([model.b1, model.w2, model.b2], lr=learning_rate, weight_decay=self.decay,
                                  betas=(self.beta1, self.beta2), eps=self.epsilon)
        return SplitUpdate(hidden, other)


DIR = ROOT / "experiments/optimizer_comparison"
SPEC = json.loads((DIR / "protocol.json").read_text())


def benchmark() -> dict:
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "experiments/tictactoe_pilot/manifest.json").read_text())
    graph = GameGraph.build(manifest["split"])
    decisions, values = training_inputs(graph)
    rng = random.Random(SPEC["data_seed"])
    boards = rng.sample(decisions, SPEC["boards_per_training_draw"])
    v_inputs = rng.sample(values, SPEC["boards_per_training_draw"])
    source = heuristic_policy_action
    datasets = {
        "policy": (train_policy_mlp, [StateActionSample(s, source(s)) for s in boards]),
        "value": (train_value_mlp, generate_value_targets(v_inputs, source, root_player=1, rollout_budgets=[1])),
        "q": (train_q_mlp, generate_q_targets(boards, source, root_player=1, rollout_budgets=[1])),
    }
    test = [s for s in graph.states if graph.children[s] and graph.split[s] == "test"]
    uniform = {s: uniform_policy(s) for s in graph.states if graph.children[s]}
    original = {s: {source(s): 1} for s in uniform}

    def outcomes(player):
        x = play_match(graph, player, uniform)
        o = play_match(graph, uniform, player)[::-1]
        return [(a + b) / 2 for a, b in zip(x, o)]

    settings = {k: SPEC[k] for k in ("beta1", "beta2", "epsilon", "weight_decay")}
    configs = {k: cls(optimizer="adamw", deterministic=True, dtype="float32", **settings)
               for k, cls in (("adamw", TrainingConfig), ("muon", MuonComparison))}
    common = {k: SPEC[k] for k in ("learning_rate", "hidden_dim")}
    for config in configs.values():
        for train, data in datasets.values():
            train(data[:1], epochs=1, seed=0, config=config, **common)
    rows = []
    for i, seed in enumerate(SPEC["fit_seeds"]):
        order = list(configs) if i % 2 == 0 else list(reversed(configs))
        for method in order:
            models, records = {}, {}
            for name, (train, data) in datasets.items():
                start = time.perf_counter()
                fit = train(data, epochs=SPEC["epochs"], seed=seed, config=configs[method], **common)
                records[name] = dict(seconds=time.perf_counter() - start, losses=fit.training_log.losses)
                models[name] = fit.model
            pred = predict(graph, Recovery(**models, stochastic=False, counts={}))
            players = {"policy": pred.imitation, "value": {s: {a: 1} for s, a in pred.v_move.items()},
                       "q": {s: {a: 1} for s, a in pred.q_move.items()}}
            for name, player in players.items():
                rows.append(dict(seed=seed, method=method, network=name, **records[name],
                                 matches=sum(player[s] == original[s] for s in test), win_draw_loss=outcomes(player)))
    summary = {}
    for name in datasets:
        groups = {m: [r for r in rows if (r["network"], r["method"]) == (name, m)] for m in configs}
        def metrics(row):
            return dict(final_loss=row["losses"][-1], seconds=row["seconds"], matches=row["matches"],
                        **dict(zip(("win", "draw", "loss"), row["win_draw_loss"])))
        pairs = [(metrics(a), metrics(b)) for a, b in zip(groups["adamw"], groups["muon"])]
        summary[name] = {}
        for key in pairs[0][0]:
            a = [p[0][key] for p in pairs]
            b = [p[1][key] for p in pairs]
            diffs = [y - x for x, y in zip(a, b)]
            summary[name][key] = dict(adamw=statistics.mean(a), muon=statistics.mean(b),
                                      difference=statistics.mean(diffs),
                                      standard_error=statistics.stdev(diffs) / math.sqrt(len(diffs)))
    return dict(protocol=SPEC, torch=str(torch.__version__), platform=sys.platform, test_boards=len(test),
                training_counts={k: len(d) for k, (_, d) in datasets.items()},
                decision_boards=list(map(board_id, boards)), value_inputs=list(map(board_id, v_inputs)),
                source_win_draw_loss=outcomes(original), fits=rows, paired_summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-muon", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not hasattr(torch.optim, "Muon"):
        parser.error("This benchmark needs a PyTorch build with torch.optim.Muon (2.9 or later).")
    args.output.write_text(json.dumps(benchmark(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
