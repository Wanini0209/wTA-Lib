# -*- coding: utf-8 -*-
"""Time-Indexing Data.

This package provides classes and utilities relative to time-indexing data.

Classes
-------
BooleanTimeSeries : A time-series of boolean data.
NumericTimeSeries : A time-series of numeric data.
TimeSeries : A sequence of data points indexed by time.

"""

# pylint: disable=unused-import
from ._series import BooleanTimeSeries, NumericTimeSeries, TimeSeries  # noqa: F401

# pylint: enable=unused-import

__ALL__ = ['BooleanTimeSeries', 'NumericTimeSeries', 'TimeSeries']
