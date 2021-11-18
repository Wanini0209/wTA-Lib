# -*- coding: utf-8 -*-
"""Context commonly used by test modules in this package."""

from .._context import TimeSeries


def ts_identical(target: TimeSeries, reference: TimeSeries) -> bool:
    """Determine whether two time-series are identical.

    Two time-sereis are identical if they are equal and has the same name.

    """
    cond_1 = target.equals(reference)
    cond_2 = target.name == reference.name
    return cond_1 and cond_2
