from typing import List
import numpy as np
import numpy.lib.mixins
import scipp as sc

HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for VectorArray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


class Variable(numpy.lib.mixins.NDArrayOperatorsMixin):
    """Array with unit and dimension names"""

    def __init__(self, dims, values: np.ndarray, *, unit):
        self._dims = tuple(dims)
        self._unit = unit if isinstance(unit, sc.Unit) else sc.Unit(unit)
        self._values = values

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims}, shape={self.shape}, "
        f"unit={self.unit}, values={self.values})"

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def dims(self):
        return self._dims

    @property
    def unit(self):
        return self._unit

    @property
    def values(self):
        return self._values

    def _get_by_dim(self, dim, index):
        if not isinstance(index, int):
            raise NotImplementedError("TODO: range index")
        split = self._dims.index(dim)
        if split == 0:
            return Variable(self._dims[1:], self._values[index], unit=self._unit)
        dims = self._dims[:split] + self._dims[split + 1:]
        slc = (slice(None), ) * split + (index, )
        return Variable(dims, self._values[slc], unit=self._unit)

    def __getitem__(self, index):
        if isinstance(index, tuple) and isinstance(index[0], str):
            return self._get_by_dim(*index)
        # TODO remove dim if not range slice
        return Variable(self._dims, self._values[index], self._unit)

    def _to_scipp(self):
        return sc.array(dims=self._dims, values=self._values, unit=self._unit)

    @staticmethod
    def _from_scipp(scipp_var):
        return Variable(scipp_var.dims,
                        values=np.array(scipp_var.values),
                        unit=scipp_var.unit)

    def _is_scipp_compatible(self):
        # Could also handle other things that have "normal" buffer+strides
        return isinstance(self._values, np.ndarray)

    def equals(self, other):
        return sc.identical(self._to_scipp(), other._to_scipp())

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            if all([x._is_scipp_compatible() for x in inputs]):
                arrays = [a._to_scipp() for a in inputs]
                # Temporary hack for testing: Copy to scipp and apply
                if ufunc == np.add:
                    out = sc.add(*arrays)
                elif ufunc == np.multiply:
                    out = sc.multiply(*arrays)
                else:
                    # TODO Fallback to underlying
                    return NotImplemented
                return Variable._from_scipp(out)
            arrays = [a._values for a in inputs]
            # TODO process unit and broadcast/transpose
            return Variable(self.dims, ufunc(*arrays, **kwargs), unit=self._unit)

            #if (out := kwargs.get('out')) is not None:
            #    kwargs['out'] = tuple([self._unwrap_content(v) for v in out])
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, Variable) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def compute(self):
        return Variable(self._dims, self._values.compute(), unit=self._unit)
