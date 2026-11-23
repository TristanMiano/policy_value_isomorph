"""Controlled score replacements for diagnosing fixed-choice recovery."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def classify(scores, legal, original):
    """Separate errors from strict inequalities and errors decided by the legal tie convention."""
    scores = np.where(legal, scores, -np.inf)
    top = scores.max(axis=1)
    favorite_score = scores[np.arange(len(scores)), original]
    chosen = scores.argmax(axis=1)
    return dict(
        match=chosen == original,
        strict=favorite_score < top,
        lost_tie=(favorite_score == top) & (chosen != original),
        multiple_best=((scores == top[:, None]) & legal).sum(1) > 1,
    )


def fit_choice(counts, legal, consequences, initial_seed):
    """Fit a legal-move softmax to observed count fractions, weighting each visited parent equally."""
    counts = counts.astype(float)
    target = counts / counts.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(initial_seed)

    def objective(weights):
        logits = weights[: legal.shape[1]] + consequences @ weights[legal.shape[1] :]
        masked = np.where(legal, logits, -np.inf)
        norm = logsumexp(masked, axis=1)
        loss = np.sum(norm - (target * logits).sum(1))
        residual = np.exp(masked - norm[:, None]) - target
        gradient = np.r_[residual.sum(0), np.einsum("ij,ijk->k", residual, consequences)]
        return loss + 0.005 * (weights @ weights), gradient + 0.01 * weights

    initial = rng.normal(0, 0.012, legal.shape[1] + consequences.shape[2])
    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B", options=dict(maxiter=700, ftol=1e-14, gtol=1e-8)
    )
    if not result.success:
        # A line search can run out of representable loss improvement near the optimum.
        # With only one coefficient per move and at most two other inputs the exact
        # Hessian is cheap: sum the legal-softmax feature covariances plus the penalty.
        identity = np.broadcast_to(np.eye(legal.shape[1]), (*legal.shape, legal.shape[1]))
        features = np.concatenate((identity, consequences), axis=2)

        def hessian(weights):
            p = choice_probabilities(weights, legal, consequences)
            centered = features - np.einsum("ba,bai->bi", p, features)[:, None, :]
            return np.einsum("ba,bai,baj->ij", p, centered, centered) + 0.01 * np.eye(len(weights))

        previous = result
        result = minimize(
            objective, previous.x, jac=True, hess=hessian, method="trust-exact", options=dict(gtol=1e-6)
        )
        result.nit += previous.nit
        if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-5:
            raise ArithmeticError(result.message)
    probabilities = choice_probabilities(result.x, legal, consequences)
    nll = -np.sum(target * np.log(probabilities.clip(1e-300))) / len(target)
    return result.x, dict(
        boards=len(target),
        parameters=len(result.x),
        mean_negative_log_probability=nll,
        objective_per_board=result.fun / len(target),
        updates=int(result.nit),
    )


def choice_probabilities(weights, legal, features):
    logits = weights[: legal.shape[1]] + features @ weights[legal.shape[1] :]
    logits = np.where(legal, logits, -np.inf)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def finite_endpoints(board, continuation):
    """Index the final legal edge after each forced move and a deterministic continuation."""
    last = np.zeros_like(board.child)
    length = np.zeros_like(last)
    edges = np.zeros_like(last)
    edges[board.legal] = np.arange(len(board.after))
    for ids in reversed(board.layers):
        children = board.child[ids]
        terminal = board.ends[children]
        following = continuation[children]
        last[ids] = np.where(terminal, edges[ids], last[children, following])
        length[ids] = np.where(terminal, 1, length[children, following] + 1)
    return last, length


def order_from_observed_ties(outcomes, legal, counts, reverse=False):
    """Recover a move order within equal best guaranteed outcomes on fitting boards."""
    n = legal.shape[1]
    edges = np.zeros((n, n), dtype=bool)
    chosen = counts.argmax(1)
    best = np.where(legal, outcomes, -np.inf).max(1)
    if np.any(counts.sum(1) == 0) or np.any(outcomes[np.arange(len(chosen)), chosen] != best):
        raise ValueError("requires observed fixed choices preserving the best supplied outcome")
    tied = legal & (outcomes == best[:, None])
    i, a = np.where(tied)
    edges[chosen[i], a] = True
    np.fill_diagonal(edges, False)
    remaining = list(range(n))
    priority = np.zeros(n, int)
    for rank in range(n):
        free = [a for a in remaining if not edges[remaining, a].any()]
        if not free:
            raise ValueError("observed equal-outcome preferences contain a cycle")
        next_move = max(free) if reverse else min(free)
        priority[next_move] = n - rank
        remaining.remove(next_move)
    return priority, edges
