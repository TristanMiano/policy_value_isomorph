"""Exact/sampled successor-score constraints and independent checks of board transitions."""
from collections import defaultdict
import heapq
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from policy_value_isomorph.connect_four import ConnectFourState
from policy_value_isomorph.phase_v import OUT,actions,install_source,native,prefixes,seed,spec
from policy_value_isomorph.phase_v_study import scoring

def reference_reversi(s):
    x,o,seat=s
    def placements(side):
        me,them=(x,o) if side==0 else (o,x)
        result={}
        for a in range(16):
            if (x|o)&(1<<a):continue
            flips=0
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==dc==0:continue
                    r,c=a//4+dr,a%4+dc;trail=0
                    while 0<=r<4 and 0<=c<4:
                        bit=1<<(4*r+c)
                        if them&bit:trail|=bit
                        else:
                            if me&bit:flips|=trail
                            break
                        r+=dr;c+=dc
            if flips:
                pair=(me|flips|(1<<a),them&~flips)
                result[a]=(*pair,1-side) if side==0 else (pair[1],pair[0],1-side)
        return result
    found=placements(seat)
    return found if found else {16:(x,o,1-seat)} if placements(1-seat) else {}

def constraints(e,states,preferred):
    edges=[];bychild={};selected=[];source=[];cell=[]
    def number(board):
        key=tuple(board)
        if key not in bychild:bychild[key]=len(bychild)
        return bychild[key]
    for i,(s,fav) in enumerate(zip(states,preferred)):
        moves=actions(e,s);candidate=[number(e.next(s,a)) for a in moves]
        winner=candidate[moves.index(fav)]
        for a,c in zip(moves,candidate):
            selected.append(int(a==fav));source.append(c);cell.append((i,a))
            if a!=fav:edges.append((c,winner,int(a<fav)))
    labels=np.array(selected);source=np.array(source);n=len(bychild)
    count=np.bincount(source,minlength=n);success=np.bincount(source,weights=labels,minlength=n)
    freq=success/count
    aa,bb,strict=np.array(edges).T
    adj=csr_matrix((np.ones(len(edges)),(aa,bb)),shape=(n,n))
    ncomp,comp=connected_components(adj,directed=True,connection="strong")
    conflicts=int(((comp[aa]==comp[bb])&(strict>0)).sum())
    # Longest-path ranking on the graph of strongly connected components; strict inequalities
    # inside a component cannot all hold, but the constructed ranks still give a measured lower bound.
    graph=[{} for _ in range(ncomp)];indegree=np.zeros(ncomp,int)
    for a,c,w in zip(comp[aa],comp[bb],strict):
        if a==c:continue
        if c not in graph[a]:indegree[c]+=1
        graph[a][c]=max(w,graph[a].get(c,0))
    level=np.zeros(ncomp,int);queue=list(np.flatnonzero(indegree==0))
    while queue:
        a=queue.pop()
        for c,w in graph[a].items():
            level[c]=max(level[c],level[a]+w);indegree[c]-=1
            if not indegree[c]:queue.append(c)
    scores=level[comp][source]
    ix=0;correct=0
    for s,fav in zip(states,preferred):
        moves=actions(e,s);correct+=moves[np.argmax(scores[ix:ix+len(moves)])]==fav;ix+=len(moves)
    return dict(boards=len(states),legal_candidates=len(source),successors=n,conflicting_targets=int(((success>0)&(success<count)).sum()),
                minimum_squared_frequency_error=float(np.mean((freq[source]-labels)**2)),
                conflicting_strict_constraints=conflicts,constructed_matches=int(correct),
                upper_matches=len(states)-int(conflicts>0),levels=int(level.max()+1))

def main():
    e=native().Engine(True);allstates=e.all();decisions=[];edges=0;counts=[0,0,0]
    for s in allstates:
        ref=reference_reversi(s)
        assert actions(e,s)==sorted(ref)
        for a,t in ref.items():assert tuple(e.next(s,a))==t
        edges+=len(ref)
        if ref:
            decisions.append(s);counts[2 if 16 in ref else int(len(ref)==1)]+=1
            assert e.solve(s)==max(-e.solve(t) for t in ref.values())
        else:assert e.solve(s)==(1 if s[2]==0 else -1)*int(np.sign(s[0].bit_count()-s[1].bit_count()))
    chosen=[e.teacher(s,1,list(range(17))) for s in decisions]
    r=constraints(e,decisions,chosen)
    c=native().Engine(False);check=prefixes(c,512,np.random.default_rng(745))
    def python_state(s):
        return ConnectFourState(tuple(int(bool(s[0]&(1<<(7*j+5-i))))-int(bool(s[1]&(1<<(7*j+5-i))))
                     for i in range(6) for j in range(7)),1-2*s[2])
    checked=0
    for s in check:
        ref=python_state(s);assert ref.legal_moves()==actions(c,s)
        for a in ref.legal_moves():
            t=c.next(s,a);assert python_state(t)==ref.apply_move(a)
            assert actions(c,t)==python_state(t).legal_moves()
            if not c.legal(t):assert c.outcome(t)==python_state(t).terminal_return(1)
            checked+=1
    cs=[]
    for b in range(24):
        folder=OUT/"connect4"/f"block_{b:02}";install_source(c,folder)
        with np.load(folder/"data_0_0.npz") as d:states=d["states"].tolist()
        cs.append(dict(block=b,**constraints(c,states,scoring(c,states,spec(0)).argmax(1))))
    (OUT/"bounds.json").write_text(json.dumps(dict(reversi4=r,reachable=len(allstates),verified_edges=edges,
          decision_groups=counts,connect4=cs,connect4_checked_candidates=checked),indent=2)+"\n")
if __name__=="__main__":main()
