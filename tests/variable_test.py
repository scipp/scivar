import numpy as np
import scivar as sv
import pytest


def test_basics():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    assert var.unit == 'm'
    assert var.dims == ('x',)


def test_add():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    out = var + var
    assert out.equals(sv.Variable(['x'], np.arange(4)*2, unit='m'))


def test_multiply():
    var = sv.Variable(['x'], np.arange(4), unit='m')
    out = var * var
    assert out.equals(sv.Variable(['x'], np.arange(4)**2, unit='m**2'))


def test_add_raises_if_different_vectors():
    vec1 = sx.VectorArray(np.array([1, 2, 3]), ['x', 'y', 'z'])
    vec2 = sx.VectorArray(np.array([3, 4, 5]), ['vx', 'vy', 'vz'])
    with pytest.raises(ValueError):
        vec1 + vec2


