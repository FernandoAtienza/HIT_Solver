from __future__ import annotations

import importlib.util

import numpy as np


def array_module(values):
    module = type(values).__module__.split(".")[0]
    if module == "cupy":
        import cupy as cp

        return cp
    return np


def select_array_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "cupy":
        if importlib.util.find_spec("cupy") is None:
            raise RuntimeError("CuPy backend requested, but cupy is not installed.")
        import cupy as cp

        return cp
    if backend == "auto":
        if importlib.util.find_spec("cupy") is not None:
            import cupy as cp

            return cp
        return np
    raise ValueError("backend must be 'auto', 'numpy', or 'cupy'")


def to_numpy(values):
    xp = array_module(values)
    if xp is np:
        return values
    return xp.asnumpy(values)
