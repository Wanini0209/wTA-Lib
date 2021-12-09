# -*- coding: utf-8 -*-
"""Context commonly used by test modules in this package."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../../..')))

# pylint: disable=wrong-import-position, unused-import, wrong-import-order
from wtalib._td._array import (  # noqa: E402, F401
    ArithmeticBinaryOperator,
    ArithmeticUnaryFunction,
    ArithmeticUnaryOperator,
    ComparisonOperator,
    LogicalBinaryOperator,
    LogicalUnaryOperator,
    MaskedArray,
    array_equal,
)
from wtalib._td._index import TimeIndex  # noqa: E402, F401
from wtalib._td._series import (  # noqa: E402, F401
    BooleanTimeSeries,
    NumericTimeSeries,
    TimeSeries,
    TimeSeriesSampling,
)
from wtalib._td._unit import TimeUnit  # noqa: E402, F401

# pylint: enable=wrong-import-position, unused-import, wrong-import-order


NP_DATETIME_DTYPES = [
    np.dtype('datetime64[s]'),
    np.dtype('datetime64[m]'),
    np.dtype('datetime64[h]'),
    np.dtype('datetime64[D]'),
    np.dtype('datetime64[M]'),
    np.dtype('datetime64[Y]')]

UNIT_VS_EQUIV_DTYPES = [
    (TimeUnit.SECOND, np.dtype('datetime64[s]')),
    (TimeUnit.MINUTE, np.dtype('datetime64[m]')),
    (TimeUnit.HOUR, np.dtype('datetime64[h]')),
    (TimeUnit.DAY, np.dtype('datetime64[D]')),
    (TimeUnit.MONTH, np.dtype('datetime64[M]')),
    (TimeUnit.YEAR, np.dtype('datetime64[Y]'))]

UNIT_VS_SUB_DTYPES = [
    (TimeUnit.MINUTE, np.dtype('datetime64[s]')),
    (TimeUnit.HOUR, np.dtype('datetime64[s]')),
    (TimeUnit.HOUR, np.dtype('datetime64[m]')),
    (TimeUnit.DAY, np.dtype('datetime64[s]')),
    (TimeUnit.DAY, np.dtype('datetime64[m]')),
    (TimeUnit.DAY, np.dtype('datetime64[h]')),
    (TimeUnit.MONTH, np.dtype('datetime64[s]')),
    (TimeUnit.MONTH, np.dtype('datetime64[m]')),
    (TimeUnit.MONTH, np.dtype('datetime64[h]')),
    (TimeUnit.MONTH, np.dtype('datetime64[D]')),
    (TimeUnit.YEAR, np.dtype('datetime64[s]')),
    (TimeUnit.YEAR, np.dtype('datetime64[m]')),
    (TimeUnit.YEAR, np.dtype('datetime64[h]')),
    (TimeUnit.YEAR, np.dtype('datetime64[D]')),
    (TimeUnit.YEAR, np.dtype('datetime64[M]')),
    (TimeUnit.WEEK, np.dtype('datetime64[s]')),
    (TimeUnit.WEEK, np.dtype('datetime64[m]')),
    (TimeUnit.WEEK, np.dtype('datetime64[h]')),
    (TimeUnit.WEEK, np.dtype('datetime64[D]')),
    (TimeUnit.QUARTER, np.dtype('datetime64[s]')),
    (TimeUnit.QUARTER, np.dtype('datetime64[m]')),
    (TimeUnit.QUARTER, np.dtype('datetime64[h]')),
    (TimeUnit.QUARTER, np.dtype('datetime64[D]')),
    (TimeUnit.QUARTER, np.dtype('datetime64[M]'))]

UNIT_VS_SUPER_DTYPES = [
    (TimeUnit.SECOND, np.dtype('datetime64[m]')),
    (TimeUnit.SECOND, np.dtype('datetime64[h]')),
    (TimeUnit.SECOND, np.dtype('datetime64[D]')),
    (TimeUnit.SECOND, np.dtype('datetime64[M]')),
    (TimeUnit.SECOND, np.dtype('datetime64[Y]')),
    (TimeUnit.MINUTE, np.dtype('datetime64[h]')),
    (TimeUnit.MINUTE, np.dtype('datetime64[D]')),
    (TimeUnit.MINUTE, np.dtype('datetime64[M]')),
    (TimeUnit.MINUTE, np.dtype('datetime64[Y]')),
    (TimeUnit.HOUR, np.dtype('datetime64[D]')),
    (TimeUnit.HOUR, np.dtype('datetime64[M]')),
    (TimeUnit.HOUR, np.dtype('datetime64[Y]')),
    (TimeUnit.DAY, np.dtype('datetime64[M]')),
    (TimeUnit.DAY, np.dtype('datetime64[Y]')),
    (TimeUnit.MONTH, np.dtype('datetime64[Y]')),
    (TimeUnit.WEEK, np.dtype('datetime64[M]')),
    (TimeUnit.WEEK, np.dtype('datetime64[Y]')),
    (TimeUnit.QUARTER, np.dtype('datetime64[Y]'))]
