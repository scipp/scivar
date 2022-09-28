import numpy as np
import scivar as sv
import pytest


def test_basics():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    assert var.unit == 'm'
    assert var.dims == ('x', )


def test_add():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    out = var + var
    assert out.equals(sv.Variable(['x'], np.arange(4) * 2, unit='m'))


def test_multiply():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    out = var * var
    assert out.equals(sv.Variable(['x'], np.arange(4)**2, unit='m**2'))


def test_getitem():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    assert var['x', 1].equals(sv.Variable([], np.array(1), unit='m'))


def test_getitem_raises_with_bad_dim():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    with pytest.raises(ValueError):
        var['y', 1]
