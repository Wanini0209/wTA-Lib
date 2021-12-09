# -*- coding: utf-8 -*-
"""Context commonly used by test modules in this package."""

import datetime
from typing import Dict

from .._context import TimeSeries


def ts_identical(target: TimeSeries, reference: TimeSeries) -> bool:
    """Determine whether two time-series are identical.

    Two time-sereis are identical if they are equal and has the same name.

    """
    cond_1 = target.equals(reference)
    cond_2 = target.name == reference.name
    return cond_1 and cond_2


def dts_identical(target: Dict[datetime.date, TimeSeries],
                  reference: Dict[datetime.date, TimeSeries]) -> bool:
    """Determine whether two dict of time-series are identical.

    Two dict mapping from `datetime.date` to `TimeSeries` are identical if
    they have same keys and identical corresponging time-series.

    """
    if len(target) != len(reference):
        return False
    for each in target:
        if each not in reference:
            return False
        if not ts_identical(target[each], reference[each]):
            return False
    return True
