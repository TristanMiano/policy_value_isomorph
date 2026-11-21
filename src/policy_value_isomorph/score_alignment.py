"""Comparable coordinates for finite scores. A gauge names their origin, scale and meaning."""
from dataclasses import dataclass
from math import fsum, isfinite
from statistics import mean, median, pstdev
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Aligned:
    values: dict
    gauge: str
    cardinal: bool  # Absolute differences in this coordinate have declared units.
    method: str


def finite_row(row: Mapping) -> tuple:
    if not row or not all(isfinite(v) for v in row.values()):
        raise ValueError("align a nonempty finite support; report exclusions separately")
    return tuple(row.values())


def align(row: Mapping, method: str, *, units: str, reference: str, anchor=None,
          temperature: float | None = None, shared: Sequence[tuple[float,float]] = ()) -> Aligned:
    """shared pairs are calibration inputs and reference outputs, separate from the evaluated row.
    reference identifies the shared anchor, row population or calibration scale across fits.
    """
    values = finite_row(row)
    if not units.strip() or not reference.strip():
        raise ValueError("name the units and reference convention")
    cardinal = True
    if method == "raw":
        transform = lambda x:x
    elif method in {"anchor","temperature"}:
        if anchor not in row:
            raise ValueError("shared anchor absent")
        scale = 1
        if method == "temperature":
            if temperature is None or not isfinite(temperature) or temperature <= 0:
                raise ValueError("need a finite positive temperature")
            scale = temperature
        transform = lambda x:(x-row[anchor])/scale
    elif method in {"standard","robust"}:
        center = mean(values) if method=="standard" else median(values)
        scale = pstdev(values) if method=="standard" else median(abs(v-center) for v in values)
        if not scale and min(values)!=max(values):
            raise ValueError("zero robust spread on a nonconstant row")
        transform = lambda x:(x-center)/(scale or 1)
    elif method == "rank":
        ranks = {x:sum(v<x for v in values)+(sum(v==x for v in values)-1)/2 for x in set(values)}
        transform = ranks.__getitem__
        cardinal = False
    elif method in {"affine","isotonic"}:
        if len(shared)<2 or any(not isfinite(v) for pair in shared for v in pair):
            raise ValueError("need finite shared calibration anchors")
        xs,ys = zip(*shared)
        if min(xs)==max(xs):
            raise ValueError("calibration needs distinct source anchors")
        if method == "affine":
            mx,my = mean(xs),mean(ys)
            slope = fsum((x-mx)*(y-my) for x,y in shared)/fsum((x-mx)**2 for x in xs)
            if slope<=0:
                raise ValueError("anchors require a positive scale")
            transform = lambda x:my+slope*(x-mx)
        else:
            # Pool adjacent decreasing groups with their original observation weights.
            from bisect import bisect_left
            levels = sorted(set(xs))
            blocks = []
            for x in levels:
                group = [y for original,y in shared if original==x]
                blocks.append(([x],sum(group),len(group)))
                while len(blocks)>1 and blocks[-2][1]/blocks[-2][2] > blocks[-1][1]/blocks[-1][2]:
                    b, a = blocks.pop(), blocks.pop()
                    blocks.append((a[0]+b[0],a[1]+b[1],a[2]+b[2]))
            fitted = {x:total/n for group,total,n in blocks for x in group}
            def transform(x):
                j = bisect_left(levels,x)
                if j==0 or j==len(levels):
                    return fitted[levels[0 if j==0 else -1]]
                lo,hi=levels[j-1],levels[j]
                return fitted[lo]+(fitted[hi]-fitted[lo])*(x-lo)/(hi-lo)
    else:
        raise ValueError("unknown alignment method")
    # Raw and calibrated predictions in the same externally anchored destination are comparable.
    coordinate = {"raw":"fixed","affine":"fixed","isotonic":"fixed"}.get(method,method)
    gauge = repr((coordinate,units,reference,anchor if method in {"anchor","temperature"} else None))
    return Aligned({k:transform(v) for k,v in row.items()},gauge,cardinal,method)


def cardinal_mse(left: Aligned, right: Aligned) -> float:
    """Mean squared difference per corresponding entry in shared squared units."""
    if (not left.cardinal or not right.cardinal or left.gauge != right.gauge
            or not left.values or left.values.keys()!=right.values.keys()):
        raise ValueError("cardinal comparison needs matching support, meaning, origin and units")
    return fsum((v-right.values[k])**2 for k,v in left.values.items())/len(left.values)
