# References

PyTorch provides tensor arithmetic, differentiation and weight updates for the move, board and board-and-move networks. The tic-tac-toe and Connect Four rules are implemented locally. NumPy, a library for numerical arrays, is used by the experiments that fit observed move frequencies. [Package requirements](pyproject.toml) · [Training methods and architecture](PYTORCH_MIGRATION_AUDIT.md#replaying-the-frozen-tic-tac-toe-networks)

## Conceptual references
- AlphaZero paper for policy/value framing and notation: https://arxiv.org/abs/1712.01815
- Project inspiration post cited by README: https://thothhermes.substack.com/p/condensed-response-to-hidden-complexity

## Software references

AdamW is the adaptive weight-update rule available to all three trainers. Muon is the matrix update evaluated by the optional optimizer benchmark. [AdamW documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html) · [Muon documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
