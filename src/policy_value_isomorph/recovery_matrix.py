"""Paired repeat runs. SQLite transactions publish each complete run and its manifest together."""
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from itertools import combinations, product
from math import fsum, isclose, isfinite
from pathlib import Path
from statistics import mean

from .pilot_exact import total_variation
from .recovery_contract import ValueArtifact

FACTORS = ("data","rollouts","initialization","optimizer_order","architecture","candidates")


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def factor_seeds(master: int, levels: dict) -> dict:
    """Identical levels of one factor share a stream across paired runs, distinct streams otherwise."""
    if set(levels)!=set(FACTORS) or any(type(n) is not int or n<1 for n in levels.values()):
        raise ValueError("provide a positive number of levels for each named factor")
    seeds = {}
    for factor in FACTORS:
        seeds[factor] = []
        for i in range(levels[factor]):
            digest = hashlib.sha256(_json([master,factor,i]).encode()).digest()
            seeds[factor].append(int.from_bytes(digest[:8],"big") >> 1)
    return seeds


@dataclass
class RunResult:
    artifact: ValueArtifact
    probabilities: dict  # Common context -> move -> deployed probability, including legal zeros.


def _validate_laws(laws):
    if not isinstance(laws,dict) or not laws:
        raise ValueError("publish a common nonempty evaluation population")
    for context,row in laws.items():
        if (not isinstance(context,str) or not isinstance(row,dict) or not row
                or any(not isinstance(a,str) for a in row)
                or any(p<0 or not isfinite(p) for p in row.values())
                or not isclose(fsum(row.values()),1,rel_tol=1e-12)):
            raise ValueError("invalid evaluation distribution; use named contexts and moves")


def _compare(laws):
    entries = list(laws.values())
    template = entries[0]
    for row in entries[1:]:
        if row.keys()!=template.keys() or any(row[s].keys()!=template[s].keys() for s in template):
            raise ValueError("pair runs on exactly the same contexts and legal menus")
    center = {s:{a:mean(row[s][a] for row in entries) for a in menu} for s,menu in template.items()}
    def distance(left,right):
        return mean(total_variation(left[s],right[s]) for s in template)
    return {"contexts":len(template),"independent_factor_levels":"see per-run seed assignments",
            "pairwise":[{"left":i,"right":j,"mean_tv":distance(laws[i],laws[j])} for i,j in combinations(laws,2)],
            "mean_law":center,"distance_to_mean":[{"run":i,"mean_tv":distance(row,center)} for i,row in laws.items()]}


def run_matrix(path: Path, definition: dict, seeds: dict, recover) -> dict:
    """Resume one fixed experiment. definition identifies source, data, code version and method.
    recover receives six seeds, returning an artifact and evaluation laws. A failed invocation
    rolls back. Another process taking the same job rechecks the committed row under a write lock.
    """
    if not definition or set(seeds)!=set(FACTORS) or any(not seeds[k] for k in FACTORS):
        raise ValueError("need an experiment definition and six nonempty seed factors")
    numbers = [x for k in FACTORS for x in seeds[k]]
    if any(type(x) is not int or not 0<=x<2**63 for x in numbers) or len(set(numbers))!=len(numbers):
        raise ValueError("each distinct factor level requires its own integer seed")
    manifest = _json({"schema":1,"definition":definition,"seeds":seeds,"weights":"equal contexts and runs"})
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(path,timeout=3600)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS experiment (id INTEGER PRIMARY KEY, manifest TEXT, comparisons TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, seeds TEXT, artifact TEXT, laws TEXT)")
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO experiment VALUES (1, ?, NULL)",(manifest,))
            if db.execute("SELECT manifest FROM experiment WHERE id=1").fetchone()[0]!=manifest:
                raise ValueError("stored experiment definition or seed matrix differs")
        for values in product(*(seeds[k] for k in FACTORS)):
            assignment = dict(zip(FACTORS,values))
            encoded = _json(assignment)
            uid = hashlib.sha256(encoded.encode()).hexdigest()
            with db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT 1 FROM runs WHERE id=?",(uid,)).fetchone():
                    continue
                result = recover(assignment.copy())
                _validate_laws(result.probabilities)
                ref = db.execute("SELECT laws FROM runs LIMIT 1").fetchone()
                if ref:
                    _compare({"reference":json.loads(ref[0]),"new":result.probabilities})
                db.execute("INSERT INTO runs VALUES (?,?,?,?)",
                           (uid,encoded,result.artifact.to_json(),_json(result.probabilities)))
        with db:
            db.execute("BEGIN IMMEDIATE")
            laws = {uid:json.loads(data) for uid,data in db.execute("SELECT id,laws FROM runs ORDER BY id")}
            comparisons = _compare(laws)
            db.execute("UPDATE experiment SET comparisons=? WHERE id=1",(_json(comparisons),))
        return comparisons
