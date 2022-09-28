import numpy as np
import dask.array as da
import scivar as sv
import pytest


def test_basics():
    var = sv.Variable(['x'], da.arange(4), unit='m')
    assert var.unit == 'm'
    assert var.dims == ('x', )


def test_add():
    var = sv.Variable(['x'], da.arange(4), unit='m')
    out = var + var
    assert isinstance(out.values, da.Array)
    out = out.compute()
    assert out.equals(sv.Variable(['x'], np.arange(4) * 2, unit='m'))
