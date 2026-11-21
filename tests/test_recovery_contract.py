from dataclasses import replace

import pytest

from policy_value_isomorph.recovery_contract import Continuation, RecoveryContract, ValueArtifact


def test_serialized_contract_carries_the_deployed_rule():
    c = RecoveryContract("tic-tac-toe", "frozen first legal", "behavior", "source actions",
                         "environment", "all decision boards", "all legal", "direct maximum",
                         "first legal", "within board", "frequency", "complete count table",
                         "reachable boards", game=True, opponent_panel="panel.json")
    play = Continuation("signed final move frequency", "legal game rules", "first legal",
                        "last legal", "terminal", "first player", 1, 8)
    sampled = replace(c, decoder="VNM temperature sampling", continuation=play,
                      temperature=0.5, base_weights="equal")
    artifact = ValueArtifact(sampled, "visit score", "counts v1", "8 draws per move", {"scores": [0, 1]})
    assert ValueArtifact.from_json(artifact.to_json()) == artifact
    assert artifact.evaluation_key()["continuation"]["second_seat"] == "last legal"
    assert artifact.artifact_id != replace(artifact, contract=c).artifact_id
    for changes in [dict(decoder="VNM maximum"), dict(semantics="return"),
                    dict(decoder="temperature sampling", temperature=0), dict(opponent_panel="")]:
        with pytest.raises(ValueError):
            replace(c, **changes)
    with pytest.raises(ValueError):
        replace(artifact, payload={"a": float("nan")})
