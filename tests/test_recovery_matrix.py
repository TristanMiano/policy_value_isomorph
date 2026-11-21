import sqlite3

import pytest

from policy_value_isomorph.recovery_contract import RecoveryContract,ValueArtifact
from policy_value_isomorph.recovery_matrix import FACTORS,RunResult,factor_seeds,run_matrix


def test_interruption_resume_matched_seeds_and_aggregate(tmp_path):
    c=RecoveryContract("toy","fair draw","behavior","probabilities","stored scores","one context",
        "both","direct maximum","first","none","score","all","one decision")
    artifact=ValueArtifact(c,"fixed","fixture","exact",{"score":[1,0]})
    seeds=factor_seeds(9,{k:2 if k=="data" else 1 for k in FACTORS})
    assert seeds==factor_seeds(9,{k:len(v) for k,v in seeds.items()})
    calls=[]
    def recover(f):
        calls.append(f.copy())
        if len(calls)==2:
            raise RuntimeError("simulated interruption")
        p=float(f["data"]==seeds["data"][0])
        return RunResult(artifact,{"s":{"a":p,"b":1-p}})
    path=tmp_path/"runs.sqlite"
    with pytest.raises(RuntimeError):
        run_matrix(path,{"source":"v1"},seeds,recover)
    out=run_matrix(path,{"source":"v1"},seeds,recover)
    assert len(calls)==3 and calls[1]==calls[2] and out["pairwise"][0]["mean_tv"]==1
    assert out["mean_law"]["s"]=={"a":0.5,"b":0.5}
    assert out==run_matrix(path,{"source":"v1"},seeds,recover) and len(calls)==3
    with pytest.raises(ValueError):
        run_matrix(path,{"source":"different"},seeds,recover)
    with pytest.raises(ValueError):
        run_matrix(path,{"source":"v1"},{**seeds,"rollouts":seeds["data"]},recover)
