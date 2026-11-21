import pytest

from policy_value_isomorph.score_alignment import align,cardinal_mse


def test_calibration_and_comparability():
    row = {"a":1,"b":3,"c":5}
    def coords(method,values=row,**kw):
        return align(values,method,units="points",reference="same external scale",**kw)
    truth = coords("raw",{"a":0,"b":1,"c":2})
    assert cardinal_mse(coords("affine",shared=[(1,0),(5,2)]),truth)==0
    fit = coords("isotonic",shared=[(1,0),(3,3),(5,1)])
    assert list(fit.values.values())==[0,2,2]
    assert coords("temperature",anchor="a",temperature=2).values==truth.values
    assert coords("standard",{"a":1,"b":1}).values=={"a":0,"b":0}
    assert coords("robust").values["b"]==0
    with pytest.raises(ValueError):
        cardinal_mse(coords("rank"),coords("rank"))
    with pytest.raises(ValueError):
        cardinal_mse(coords("anchor",anchor="a"),truth)
    with pytest.raises(ValueError):
        cardinal_mse(coords("anchor",anchor="a"),coords("anchor",anchor="b"))
