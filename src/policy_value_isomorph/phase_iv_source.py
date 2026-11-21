"""Train one original player before observing any of its reconstruction results."""

import json

import numpy as np
import torch
from torch.nn import functional as F

from .neural_game_control import adjust_rate, certify_policy, network, optimizer_step
from .phase_iv_game import game, seed


def run(block, dest, tmp):
    b = game()
    order = np.random.default_rng(seed(block, 1)).permutation(9)
    preferred = np.where(b.legal[:, order], b.optimal[:, order], -np.inf).argmax(1)
    target = torch.from_numpy(order[preferred[b.dec]].astype(np.int64))
    torch.manual_seed(6201 + block)
    net = network(10, 9)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    epoch = 0
    while True:
        epoch += 1
        adjust_rate(opt, epoch)
        for ids in torch.randperm(len(target)).split(256):
            optimizer_step(
                F.cross_entropy(net(b.x[ids]).masked_fill(~b.mask[ids], -torch.inf), target[ids]), opt
            )
        if epoch % 10 == 0:
            with torch.no_grad():
                logits = net(b.x)
                moves = logits.masked_fill(~b.mask, -torch.inf).argmax(1)
                if torch.equal(moves, target):
                    break
    cert = certify_policy(b.graph, b.states_dec, moves.tolist())
    cert.update(
        seed=6201 + block,
        passes=epoch,
        priority=order.tolist(),
        own_teacher_matches=int((target == moves).sum()),
        loss_nats=float(F.cross_entropy(logits.masked_fill(~b.mask, -torch.inf), target)),
    )
    (tmp / "source.json").write_text(json.dumps(cert, indent=2))
    torch.save(net.state_dict(), tmp / "source.pt")
    np.savez_compressed(tmp / "source.npz", scores=logits.numpy(), moves=moves.numpy().astype("i1"))


def actors(block, dest):
    b = game()
    with np.load(dest / "source.npz") as data:
        scores = np.zeros((b.n, 9))
        scores[b.dec] = data["scores"]
    original = b.greedy(scores)
    mixed = 0.75 * original + 0.25 * b.uniform
    opponents = {"uniform": b.uniform, "priority": b.priority, "perfect": b.perfect}
    for i in range(5):
        torch.manual_seed(seed(block, 3, 0, i))
        with torch.no_grad():
            net = network(10, 9)
            values = np.zeros_like(scores)
            values[b.dec] = net(b.x).numpy()
        opponents["random" + str(i)] = b.greedy(values)
    return scores, {"fixed": original, "randomized": mixed}, opponents
