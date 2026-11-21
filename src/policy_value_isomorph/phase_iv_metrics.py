"""Matched evaluations on the finite game graph. Exact play sums all possible paths."""
import gzip
import json

import numpy as np
from scipy.sparse import csr_matrix

from .phase_iv_game import game


def dump(path, records):
    with gzip.open(path,"wt",encoding="utf8") as f:
        for row in records:
            f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")


def read(path):
    with gzip.open(path,"rt",encoding="utf8") as f:
        return [json.loads(line) for line in f]


class Evaluation:
    def __init__(self, original, opponents):
        self.b=game()
        b=self.b
        self.source=original
        self.visits=b.visitation(original)
        self.groups=[np.flatnonzero(b.ply==d) for d in range(10)]
        local=np.zeros(b.n,int)
        for ids in self.groups:
            local[ids]=np.arange(len(ids))
        self.transitions=[]
        for d in range(9):
            edge=b.ply[b.parent]==d
            self.transitions.append((edge,local[b.parent[edge]],local[b.after[edge]]))
        self.names=list(opponents)+["source"]
        self.weights=np.array([.05 if k.startswith("random") else .25 for k in opponents])
        self.opponents=[self.matrices(law) for law in [*opponents.values(),original]]

    def matrices(self, law):
        probs=law[self.b.legal]
        matrices=[]
        for d,(edge,parent,child) in enumerate(self.transitions):
            data=probs[edge]
            nz=data>0
            matrices.append(csr_matrix((data[nz],(child[nz],parent[nz])),
                        shape=(len(self.groups[d+1]),len(self.groups[d]))))
        return matrices

    def outcomes(self, law):
        own=self.matrices(law)
        n=len(self.names)
        results=[]
        for seat in (0,1):
            reach=np.ones((1,n))
            total=np.zeros((3,n))
            for d in range(9):
                if d%2==seat:
                    reach=own[d]@reach
                else:
                    reach=np.column_stack([op[d]@reach[:,k] for k,op in enumerate(self.opponents)])
                ids=self.groups[d+1]
                for k,z in enumerate((1,0,-1)):
                    total[k]+=reach[self.b.ends[ids] & (self.b.terminal[ids]==z)].sum(0)
            results.append((total if seat==0 else total[::-1]).T)
        seats=np.stack(results)
        if not np.allclose(seats.sum(2),1,rtol=0,atol=1e-11):
            raise ArithmeticError("lost terminal probability")
        return seats

    def measure(self,law,test,**labels):
        b=self.b
        ids=np.flatnonzero(test & ~b.ends)
        tv=(np.abs(self.source[ids]-law[ids])).sum(1)/2
        match=(self.source[ids]*law[ids]).sum(1)
        visits=self.visits[ids]
        seats=self.outcomes(law)
        wdl=seats.mean(0)[:-1].T@self.weights
        return dict(**labels,boards=len(ids),agreement=float(match.mean()),tv=float(tv.mean()),
                    visit_agreement=float(np.average(match,weights=visits)) if visits.sum() else None,
                    visit_tv=float(np.average(tv,weights=visits)) if visits.sum() else None,
                    selfplay_weight=float(visits.sum()),win=float(wdl[0]),draw=float(wdl[1]),loss=float(wdl[2]),
                    points=float(wdl[0]-wdl[2]),seats=seats.tolist())


def corrected_counts(source,base,recovered,ids):
    best=source[ids].argmax(1)
    before=base[ids].argmax(1)==best
    after=recovered[ids].argmax(1)==best
    return [int(np.sum(i)) for i in (before&after,before&~after,~before&after,~before&~after)]

