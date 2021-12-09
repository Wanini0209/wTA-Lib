# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""Time Series.

This module provides the base of time-series and its derivatives.

Classes
-------
BooleanTimeSeries : A time-series of boolean data.
NumericTimeSeries : A time-series of numeric data.
TimeSeries : A sequence of data points indexed by time.
TimeSeriesSampling : Moving samples of time-series.

"""

import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._array import (
    ArithmeticBinaryOperator,
    ComparisonOperator,
    LogicalBinaryOperator,
    MaskedArray,
)
from ._index import TimeIndex
from ._unit import TimeUnit

_MaskedArrayLike = Union[ArrayLike, MaskedArray]
_TimeIndexLike = Union[ArrayLike, TimeIndex]


class TimeSeries:
    """Time Series.

    A sequence of data points indexed by time.

    Parameters
    ----------
    data : array-like or MaskedArray (1-dimensional)
        The data stored in the time-series.
    index : array-like (1-dimensional) or TimeIndex
        The datetime values indexing `data`.
    name : str
        The name given to the time-series.
    sort : bool
        If ``True``, sort the time-serise by `index`; otherwise do not.
        Default is ``True``.

    Notes
    -----
    1. `data` and `index` must have same length.
    2. If `data` is an instance of `MaskedArray`, store it into the time-series
       directly without copy.
    3. If `index` is an instance of `TimeIndex`, set it as the index of the
       time-series directly without copy.
    4. When using `TimeIndex` to construct `TimeSeries`, the constructed
       instance can share the same index with other instances. It can achieve
       better performance in many operations. However, if `sort` is set to
       be `True`, it will generate a new index that is not shared.

    Attributes
    ----------
    name : str
        The name of the time-series.

    Properties
    ----------
    data : MaskedArray
        The data of the time-series.
    index : TimeIndex
        The index of the time-series.
    dtype : numpy dtype object
        Data-type of the data of the time-series.

    Built-in Functions
    ------------------
    len : int
        The length of the time-series. It is the number of data points of the
        time-series.

    Methods
    -------
    rename :
        Return a copy of the time-series with name changed.
    to_pandas : pandas.Series
        Return a copy of the time-series as a pandas series, in which the
        value of N/A elements are replaced with ``numpy.nan``.
    equals : bool
        Determine whether two time-series are equal.
    fillna : TimeSeries
        Fill N/A elements using given value.
    ffill : TimeSeries
        Fill N/A elements using forward-fill method. For each N/A element,
        fill it by the last available element preceding it.
    bfill : TimeSeries
        Fill N/A elements using backward-fill method. For each N/A element,
        fill it by the next available element.
    dropna : TimeSeries
        Remove N/A elements.
    shift : TimeSeries
        Shift index by desired number of periods with given time-unit.
    sampling : TimeSeriesSampling
        Get moving samples along the index by desired step.

    Examples
    --------
    >>> dates = np.arange(3).astype('datetime64[D]')
    >>> data = np.array([1., 2., 3.])
    >>> TimeSeries(data, dates, name='ts1')
    1970-01-01	1.0
    1970-01-02	2.0
    1970-01-03	3.0
    Name: ts1, dtype: float64

    With 'data' containing N/A elements:

    >>> dates = np.arange(4).astype('datetime64[D]')
    >>> data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
    >>> TimeSeries(data, dates, name='ts2')
    1970-01-01	nan
    1970-01-02	2
    1970-01-03	nan
    1970-01-04	4
    Name: ts2, dtype: int32

    With unsorted `index`:

    >>> dates = np.arange(4).astype('datetime64[D]')[::-1]
    >>> data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
    >>> TimeSeries(data, dates, name='ts3')
    1970-01-01	4
    1970-01-02	nan
    1970-01-03	2
    1970-01-04	nan
    Name: ts3, dtype: int32

    With unsorted `index` and `sort=False`:

    >>> dates = np.arange(4).astype('datetime64[D]')[::-1]
    >>> data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
    >>> TimeSeries(data, dates, name='ts4', sort=False)
    1970-01-04	nan
    1970-01-03	2
    1970-01-02	nan
    1970-01-01	4
    Name: ts4, dtype: int32

    """
    def __init__(self, data: _MaskedArrayLike, index: _TimeIndexLike,
                 name: str, sort: bool = True):
        if isinstance(data, MaskedArray):
            self._data = data.copy()
        else:
            self._data = MaskedArray(data)

        if isinstance(index, TimeIndex):
            self._index = index
        else:
            self._index = TimeIndex(index, sort=False)

        if len(self._data) != len(self._index):
            raise ValueError("inconsistent length")

        self._name = name

        if sort:
            self._sort()

    def _sort(self):
        """Sort the time-series by it's index."""
        order = self._index.argsort()
        self._index = self._index[order]
        self._data = self._data[order]

    @property
    def data(self) -> MaskedArray:
        """Data of the time-series.

        Returns
        -------
        MaskedArray

        Notes
        -----
        It return a reference of the data array storing in the time-series not
        a copy.

        See Also
        --------
        MaskedArray

        Examples
        --------
        >>> dates = np.arange(3).astype('datetime64[D]')
        >>> data = np.array([1., 2., 3.])
        >>> TimeSeries(data, dates, name='ts1').data
        array([1., 2., 3.], dtype=float64)

        With 'data' containing N/A elements:

        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        >>> TimeSeries(data, dates, name='ts2').data
        array([nan, 2, nan, 4], dtype=int32)

        """
        return self._data

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the data of the time-series.

        Returns
        -------
        numpy.dtype

        See Also
        --------
        numpy.dtype, numpy.ndarray, numpy.ndarray.dtype,
        MaskedArray.dtype

        Examples
        --------
        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = np.array([1, 2, 3, 4])
        >>> TimeSeries(data, index=dates, name='ts').dtype
        dtype('int32')

        """
        return self._data.dtype

    @property
    def index(self) -> TimeIndex:
        """Index of the time-series.

        Returns
        -------
        TimeIndex

        Notes
        -----
        It return a reference of the time-index of the time-series not a copy.

        Examples
        --------
        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = np.array([1, 2, 3, 4])
        >>> TimeSeries(data, index=dates, name='ts').index
        TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03', '1970-01-04'],
                  dtype='datetime64[D]')

        """
        return self._index

    @property
    def name(self) -> str:
        """Name of the time-series.

        Examples
        --------
        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = np.array([1, 2, 3, 4])
        >>> TimeSeries(data, index=dates, name='ts').name
        'ts'

        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Name of the time-series.

        Examples
        --------
        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = np.array([1, 2, 3, 4])
        >>> tseries = TimeSeries(data, index=dates, name='ts')
        >>> tseries.name = 'ts2'
        >>> tseries
        1970-01-01	1
        1970-01-02	2
        1970-01-03	3
        1970-01-04	4
        Name: ts2, dtype: int32

        """
        self._name = name

    def rename(self, name: str) -> 'TimeSeries':
        """Rename the time-series.

        Parameters
        ----------
        name : str
            The new name for the time-series.

        Returns
        -------
        TimeSeries
            A copy of the time-series with name changed.

        Examples
        --------
        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = np.array([1, 2, 3, 4])
        >>> tseries = TimeSeries(data, index=dates, name='ts')
        >>> tseries.rename('ts1')
        1970-01-01	1
        1970-01-02	2
        1970-01-03	3
        1970-01-04	4
        Name: ts1, dtype: int32

        """
        return self._make(self._data, self._index, name)

    def __len__(self) -> int:
        """Length of the time-series.

        Examples
        --------
        >>> dates = np.arange(4).astype('datetime64[D]')
        >>> data = np.array([1, 2, 3, 4])
        >>> len(TimeSeries(data, index=dates, name='ts'))
        4

        """
        return len(self._data)

    def equals(self, other: 'TimeSeries') -> bool:
        """Determine whether two time-series are equal.

        Two time-series are equal if they have the equal time-index and data
        array.

        Parameters
        ----------
        other : TimeSeries
            The other time-series to compare against.

        Returns
        -------
        bool
            True if `other` is an instance of `TimeSeries` and it has the same
            index and data as the calling object; False otherwise.

        See Also
        --------
        TimeIndex.equals, MaskedArray.equals

        Examples
        --------
        >>> data = [1, 2, 3, 4]
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> ts1 = TimeSeries(data, dates, name='ts1')
        >>> ts2 = TimeSeries(MaskedArray(data), TimeIndex(dates), name='ts2')
        >>> ts1.equals(ts2)
        True

        Different time-index:

        >>> data = [1, 2, 3, 4]
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> ts1 = TimeSeries(data, dates, name='ts1')
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-05']
        >>> ts2 = TimeSeries(MaskedArray(data), dates, name='ts2')
        >>> ts1.equals(ts2)
        False

        Different data-type:

        >>> data = [1, 2, 3, 4]
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> ts1 = TimeSeries(data, dates, name='ts1')
        >>> data = [1., 2., 3., 4.]
        >>> ts2 = TimeSeries(MaskedArray(data), dates, name='ts2')
        >>> ts1.equals(ts2)
        False

        Different data array:

        >>> data = [1, 2, 3, 4]
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> ts1 = TimeSeries(data, dates, name='ts1')
        >>> data = [1, 2, 3, 5]
        >>> ts2 = TimeSeries(MaskedArray(data), dates, name='ts2')
        >>> ts1.equals(ts2)
        False

        Different but equivalent data array:

        >>> data = MaskedArray([1, 2, 3, 4], [False, True, True, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> ts1 = TimeSeries(data, dates, name='ts1')
        >>> data = MaskedArray([1, -2, -3, 4], [False, True, True, False])
        >>> ts2 = TimeSeries(data, dates, name='ts2')
        >>> ts1.equals(ts2)
        True

        Not `TimeSeries` object:

        >>> data = MaskedArray([1, 2, 3, 4], [False, True, True, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> ts = TimeSeries(data, dates, name='ts')
        >>> ts.equals(data)
        TypeError: not support comparison between 'TimeSeries' and 'MaskedArray'

        """
        if not isinstance(other, TimeSeries):
            raise TypeError("not support comparison between '%s' and '%s'"
                            % (self.__class__.__name__,
                               other.__class__.__name__))
        # pylint: disable=protected-access
        if self._index.equals(other._index):
            return self._data.equals(other._data)
        return False
        # pylint: enable=protected-access

    def fillna(self, value: Any) -> 'TimeSeries':
        """Fill N/A elements using the given value.

        Parameters
        ----------
        value : scalar
            Value used to fill N/A elements.

        Returns
        -------
        TimeSeries

        See Also
        --------
        MaskedArray.fillna

        Examples
        --------
        >>> data = MaskedArray([1, 2, 3, 4], [False, True, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').fillna(0)
        2021-11-01	1
        2021-11-02	0
        2021-11-03	3
        2021-11-04	0
        Name: ts, dtype: int32

        Fill float array with integer value:

        >>> data = MaskedArray([1., 2., 3., 4.], [False, True, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').fillna(0)
        2021-11-01	1.0
        2021-11-02	0.0
        2021-11-03	3.0
        2021-11-04	0.0
        Name: ts, dtype: float64

        Fill integer array with float value:

        >>> data = MaskedArray([1, 2, 3, 4], [False, True, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').fillna(0.)
        2021-11-01	1
        2021-11-02	0
        2021-11-03	3
        2021-11-04	0
        Name: ts, dtype: int32

        """
        data = self._data.fillna(value)
        return self._make(data, self._index, self._name)

    def ffill(self) -> 'TimeSeries':
        """Fill N/A elements using forward-fill method.

        Returns
        -------
        TimeSeries

        See Also
        --------
        MaskedArray.ffill

        Examples
        --------
        >>> data = MaskedArray([1, 2, 3], [False, True, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').ffill()
        2021-11-01	1
        2021-11-02	1
        2021-11-03	3
        Name: ts, dtype: int32

        With leading N/A:

        >>> data = MaskedArray([1, 2, 3], [True, False, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').ffill()
        2021-11-01	nan
        2021-11-02	2
        2021-11-03	3
        Name: ts, dtype: int32

        With tailing N/A:

        >>> data = MaskedArray([1, 2, 3], [False, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').ffill()
        2021-11-01	1
        2021-11-02	2
        2021-11-03	2
        Name: ts, dtype: int32

        """
        data = self._data.ffill()
        return self._make(data, self._index, self._name)

    def bfill(self) -> 'TimeSeries':
        """Fill N/A elements using backward-fill method.

        Returns
        -------
        TimeSeries

        See Also
        --------
        MaskedArray.bfill

        Examples
        --------
        >>> data = MaskedArray([1, 2, 3], [False, True, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').bfill()
        2021-11-01	1
        2021-11-02	3
        2021-11-03	3
        Name: ts, dtype: int32

        With leading N/A:

        >>> data = MaskedArray([1, 2, 3], [True, False, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').bfill()
        2021-11-01	2
        2021-11-02	2
        2021-11-03	3
        Name: ts, dtype: int32

        With tailing N/A:

        >>> data = MaskedArray([1, 2, 3], [False, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').bfill()
        2021-11-01	1
        2021-11-02	2
        2021-11-03	nan
        Name: ts, dtype: int32

        """
        data = self._data.bfill()
        return self._make(data, self._index, self._name)

    def dropna(self) -> 'TimeSeries':
        """Remove N/A elements.

        Returns
        -------
        TimeSeries

        Notes
        -----
        If all elements are available, it return a copy of the time-series
        with shared `data` and `index`.

        Examples
        --------
        >>> data = MaskedArray([1, 2, 3, 4], [False, True, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').dropna()
        2021-11-01	1
        2021-11-03	3
        Name: ts, dtype: int32

        Without N/A elements:

        >>> data = MaskedArray([1, 2, 3, 4])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').dropna()
        2021-11-01	1
        2021-11-02	2
        2021-11-03	3
        2021-11-04	4
        Name: ts, dtype: int32

        Without available elements:
        >>> data = MaskedArray([1, 2, 3, 4], [True, True, True, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').dropna()
        TimeSeries([], Name: ts, dtype: int32

        """
        isna = self._data.isna()
        index = self._index
        data = self._data
        # In this case, `isna` is a boolean array, and index is a `TimeIndex`
        # not a `datetime.date`. So, ignore `mypy` for wrong warning below.
        if isna.any():
            index = index[~isna]    # type: ignore
            data = data[~isna]
        return self._make(data, index, self._name)

    def shift(self, period: int, punit: Optional[TimeUnit] = None
              ) -> 'TimeSeries':
        """Shift index by desired number of periods with given time-unit.

        Parameters
        ----------
        period : int
            Number of positions or units, which specified by `punit`, to shift.
            Can be positive or negative but not zero. The actions are described
            as follows:
            - If `period` is set as a positive integr, n, the index are shifted
            backward n positions or n units.
            - If `period` is set as a negative integer, -n, the index are
            shifted forward n positions or n units.
        punit : TimeUnit, optional
            It is optional. It it is specified, it must be an instance of
            `TimeUnit` which is a super-unit of or equivalent to the dtype of
            index.

        Returns
        -------
        TimeSeries
            A copy of the time-series with index shifted by desired periods.

        See Also
        --------
        TimeIndex.shift

        Examples
        --------
        >>> dates = (['2021-11-01', '2021-11-03', '2021-11-06', '2021-11-10',
                      '2021-11-13', '2021-11-15', '2021-11-18', '2021-11-22',
                      '2021-11-25', '2021-11-27', '2021-11-30', '2021-12-03'])
        >>> data = MaskedArray(np.arange(12))
        >>> tseries = TimeSeries(data, dates, 'ts')
        >>> tseries
        2021-11-01	0
        2021-11-03	1
        2021-11-06	2
        2021-11-10	3
        2021-11-13	4
        2021-11-15	5
        2021-11-18	6
        2021-11-22	7
        2021-11-25	8
        2021-11-27	9
        2021-11-30	10
        2021-12-03	11
        Name: ts, dtype: int32

        1A. positive `period` and no specified `punit`:

        >>> tseries.shift(2)
        2021-11-01	nan
        2021-11-03	nan
        2021-11-06	0
        2021-11-10	1
        2021-11-13	2
        2021-11-15	3
        2021-11-18	4
        2021-11-22	5
        2021-11-25	6
        2021-11-27	7
        2021-11-30	8
        2021-12-03	9
        Name: ts.shift(2), dtype: int32

        1B. negative `period` and no specified `punit`:

        >>> tseries.shift(-2)
        2021-11-01	2
        2021-11-03	3
        2021-11-06	4
        2021-11-10	5
        2021-11-13	6
        2021-11-15	7
        2021-11-18	8
        2021-11-22	9
        2021-11-25	10
        2021-11-27	11
        2021-11-30	nan
        2021-12-03	nan
        Name: ts.shift(-2), dtype: int32

        2A. positive `period` and equivalent `punit`:

        >>> tseries.shift(2, TimeUnit.DAY)
        2021-11-01	nan
        2021-11-03	nan
        2021-11-06	0
        2021-11-10	1
        2021-11-13	2
        2021-11-15	3
        2021-11-18	4
        2021-11-22	5
        2021-11-25	6
        2021-11-27	7
        2021-11-30	8
        2021-12-03	9
        Name: ts.shift(2, day), dtype: int32

        2B. negative `period` and equivalent `punit`:

        >>> tseries.shift(-2, TimeUnit.DAY)
        2021-11-01	2
        2021-11-03	3
        2021-11-06	4
        2021-11-10	5
        2021-11-13	6
        2021-11-15	7
        2021-11-18	8
        2021-11-22	9
        2021-11-25	10
        2021-11-27	11
        2021-11-30	nan
        2021-12-03	nan
        Name: ts.shift(-2, day), dtype: int32

        3A. positive `period` and super-unit `punit`:

        >>> tseries.shift(2, TimeUnit.WEEK)
        2021-11-01	nan
        2021-11-03	nan
        2021-11-06	nan
        2021-11-10	nan
        2021-11-13	nan
        2021-11-15	2
        2021-11-18	2
        2021-11-22	4
        2021-11-25	4
        2021-11-27	4
        2021-11-30	6
        2021-12-03	6
        Name: ts.shift(2, week), dtype: int32

        3B. negative `period` and super-unit `punit`:

        >>> tseries.shift(-2, TimeUnit.WEEK)
        2021-11-01	5
        2021-11-03	5
        2021-11-06	5
        2021-11-10	7
        2021-11-13	7
        2021-11-15	10
        2021-11-18	10
        2021-11-22	nan
        2021-11-25	nan
        2021-11-27	nan
        2021-11-30	nan
        2021-12-03	nan
        Name: ts.shift(-2, week), dtype: int32

        4. sub-unit `punit`:

        >>> tseries.shift(1, TimeUnit.HOUR)
        ValueError: not support shift 'TimeUnit.HOUR' on 'datetime64[D]' datetimes

        5. non-integer `period`:

        >>> tseries.shift(1.)
        TypeError: 'period' must be 'int' not 'float'

        6. zero `period`:

        >>> tseries.shift(0)
        ValueError: 'period' can not be zero

        7. undefined `punit`:

        >>> tseries.shift(1, np.dtype('datetime64[D]'))
        TypeError: 'punit' must be 'TimeUnit' not 'dtype[datetime64]'

        >>> tseries.shift(1, 'day')
        TypeError: 'punit' must be 'TimeUnit' not 'str'

        """
        data = self._index.shift(self._data, period, punit)
        if punit is None:
            name = f'{self._name}.shift({period})'
        else:
            name = f'{self._name}.shift({period}, {punit.name})'
        return self._make(data, self._index, name)

    def sampling(self, samples: int, step: int,
                 sunit: Optional[TimeUnit] = None) -> 'TimeSeriesSampling':
        """Get moving samples by desired step along the index .

        Parameters
        ----------
        samples : int
            Number of samples, it must be an integer larger than ``1``.
        step : int
            Number of positions or units, which specified by `sunit`, between
            two samples. Can be positive or negative but not zero. If `step` is
            set as a positive integer, s, it get samples forward per s
            positions or units along the index. If `step` is set as a negative
            integer, -s, it get samples backward per s postions or units along
            the index.
        sunit : TimeUnit, optional
            It is optional. It it is specified, it must be an instance of
            `TimeUnit` which is a super-unit of or equivalent to the dtype of
            index.

        Returns
        -------
        TimeSeriesSampling

        See Also
        --------
        MaskedArray.moving_sampling, TimeIndex.sampling, TimeSeriesSampling.

        Examples
        --------
        >>> dates = (['2021-11-01', '2021-11-03', '2021-11-06', '2021-11-10',
                      '2021-11-13', '2021-11-15', '2021-11-18', '2021-11-22',
                      '2021-11-25', '2021-11-27', '2021-11-30', '2021-12-03'])
        >>> data = MaskedArray(np.arange(12))
        >>> tseries = TimeSeries(data, dates, 'ts')

        1A. positive `step` and no specified `sunit`:

        >>> tseries.sampling(2, 3)
        2021-11-01:	2021-11-01	0
                    2021-11-10	3
        2021-11-03:	2021-11-03	1
                    2021-11-13	4
        2021-11-06:	2021-11-06	2
                    2021-11-15	5
        2021-11-10:	2021-11-10	3
                    2021-11-18	6
        2021-11-13:	2021-11-13	4
                    2021-11-22	7
        2021-11-15:	2021-11-15	5
                    2021-11-25	8
        2021-11-18:	2021-11-18	6
                    2021-11-27	9
        2021-11-22:	2021-11-22	7
                    2021-11-30	10
        2021-11-25:	2021-11-25	8
                    2021-12-03	11
        2021-11-27:	2021-11-27	9
        2021-11-30:	2021-11-30	10
        2021-12-03:	2021-12-03	11
        Name: ts.sampling(2, 3), dtype: int32

        1B. negative `step` and no specified `sunit`:

        >>> tseries.sampling(2, -3)
        2021-11-01:	2021-11-01	0
        2021-11-03:	2021-11-03	1
        2021-11-06:	2021-11-06	2
        2021-11-10:	2021-11-01	0
                    2021-11-10	3
        2021-11-13:	2021-11-03	1
                    2021-11-13	4
        2021-11-15:	2021-11-06	2
                    2021-11-15	5
        2021-11-18:	2021-11-10	3
                    2021-11-18	6
        2021-11-22:	2021-11-13	4
                    2021-11-22	7
        2021-11-25:	2021-11-15	5
                    2021-11-25	8
        2021-11-27:	2021-11-18	6
                    2021-11-27	9
        2021-11-30:	2021-11-22	7
                    2021-11-30	10
        2021-12-03:	2021-11-25	8
                    2021-12-03	11
        Name: ts.sampling(2, -3), dtype: int32

        2A. positive `step` and equivalent `sunit`:

        >>> tseries.sampling(2, 3, TimeUnit.DAY)
        2021-11-01:	2021-11-01	0
                    2021-11-10	3
        2021-11-03:	2021-11-03	1
                    2021-11-13	4
        2021-11-06:	2021-11-06	2
                    2021-11-15	5
        2021-11-10:	2021-11-10	3
                    2021-11-18	6
        2021-11-13:	2021-11-13	4
                    2021-11-22	7
        2021-11-15:	2021-11-15	5
                    2021-11-25	8
        2021-11-18:	2021-11-18	6
                    2021-11-27	9
        2021-11-22:	2021-11-22	7
                    2021-11-30	10
        2021-11-25:	2021-11-25	8
                    2021-12-03	11
        2021-11-27:	2021-11-27	9
        2021-11-30:	2021-11-30	10
        2021-12-03:	2021-12-03	11
        Name: ts.sampling(2, 3, day), dtype: int32

        2B. negative `step` and equivalent `sunit`:

        >>> tseries.sampling(2, -3, TimeUnit.DAY)
        2021-11-01:	2021-11-01	0
        2021-11-03:	2021-11-03	1
        2021-11-06:	2021-11-06	2
        2021-11-10:	2021-11-01	0
                    2021-11-10	3
        2021-11-13:	2021-11-03	1
                    2021-11-13	4
        2021-11-15:	2021-11-06	2
                    2021-11-15	5
        2021-11-18:	2021-11-10	3
                    2021-11-18	6
        2021-11-22:	2021-11-13	4
                    2021-11-22	7
        2021-11-25:	2021-11-15	5
                    2021-11-25	8
        2021-11-27:	2021-11-18	6
                    2021-11-27	9
        2021-11-30:	2021-11-22	7
                    2021-11-30	10
        2021-12-03:	2021-11-25	8
                    2021-12-03	11
        Name: ts.sampling(2, -3, day), dtype: int32

        3A. positive `step` and super-unit `sunit`:

        >>> tseries.sampling(2, 3, TimeUnit.WEEK)
        2021-11-01:	2021-11-01	0
                    2021-11-22	7
        2021-11-03:	2021-11-03	1
                    2021-11-22	7
        2021-11-06:	2021-11-06	2
                    2021-11-22	7
        2021-11-10:	2021-11-10	3
                    2021-11-30	10
        2021-11-13:	2021-11-13	4
                    2021-11-30	10
        2021-11-15:	2021-11-15	5
        2021-11-18:	2021-11-18	6
        2021-11-22:	2021-11-22	7
        2021-11-25:	2021-11-25	8
        2021-11-27:	2021-11-27	9
        2021-11-30:	2021-11-30	10
        2021-12-03:	2021-12-03	11
        Name: ts.sampling(2, 3, week), dtype: int32

        3B. negative `step` and super-unit `sunit`:

        >>> tseries.sampling(2, -3, TimeUnit.WEEK)
        2021-11-01:	2021-11-01	0
        2021-11-03:	2021-11-03	1
        2021-11-06:	2021-11-06	2
        2021-11-10:	2021-11-10	3
        2021-11-13:	2021-11-13	4
        2021-11-15:	2021-11-15	5
        2021-11-18:	2021-11-18	6
        2021-11-22:	2021-11-06	2
                    2021-11-22	7
        2021-11-25:	2021-11-06	2
                    2021-11-25	8
        2021-11-27:	2021-11-06	2
                    2021-11-27	9
        2021-11-30:	2021-11-13	4
                    2021-11-30	10
        2021-12-03:	2021-11-13	4
                    2021-12-03	11
        Name: ts.sampling(2, -3, week), dtype: int32

        4. sub-unit `sunit`:

        >>> tseries.sampling(2, 1, TimeUnit.HOUR)
        ValueError: not support sampling 'TimeUnit.HOUR' on 'datetime64[D]' datetimes

        5A. non-integer `samples`:

        >>> tseries.sampling(2., 1)
        TypeError: 'samples' must be 'int' not 'float'

        5B. invalid `samples`:

        >>> tseries.sampling(1, 1)
        ValueError: 'samples' must be larger than 1

        >>> tseries.sampling(0, 1)
        ValueError: 'samples' must be larger than 1

        >>> tseries.sampling(-1, 1)
        ValueError: 'samples' must be larger than 1

        6A. non-integer `step`:

        >>> tseries.sampling(2, 1.)
        TypeError: 'step' must be 'int' not 'float'

        6B. zero `step`:

        >>> tseries.sampling(2, 0)
        ValueError: 'step' must be non-zero

        7. invalid `sunit`:

        >>> tseries.sampling(2, 1, 'day')
        TypeError: 'sunit' must be 'TimeUnit' not 'str'

        """
        index = self._index
        data = index.sampling(self._data, samples, step, sunit)
        indices = index.sampling(MaskedArray(index.values),
                                 samples, step, sunit)
        if sunit is None:
            name = f'{self._name}.sampling({samples}, {step})'
        else:
            name = f'{self._name}.sampling({samples}, {step}, {sunit.name})'
        return TimeSeriesSampling(data, indices, index, name)

    @classmethod
    def _make(cls, data: MaskedArray, index: TimeIndex, name: str
              ) -> 'TimeSeries':
        """Quick constructor.

        A constructor used by methods to contruct instance quickly.

        """
        return cls(data, index, name=name, sort=False)

    def to_pandas(self) -> pd.Series:
        """Return a copy of the time-series as pandas series.

        Returns
        -------
        pandas.Series
            Return a copy of the time-series as a pandas series, in which the
            value of N/A elements are replaced with ``numpy.nan``.

        See also
        --------
        MaskedArray.to_numpy

        Examples
        --------
        >>> data = MaskedArray([1, 2, 3, 4])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').to_pandas()
        2021-11-01    1
        2021-11-02    2
        2021-11-03    3
        2021-11-04    4
        Name: ts, dtype: int32

        1. integer array with N/As:

        >>> data = MaskedArray([1, 2, 3, 4], [False, True, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').to_pandas()
        2021-11-01    1.0
        2021-11-02    NaN
        2021-11-03    3.0
        2021-11-04    NaN
        Name: ts, dtype: float64

        2. float array with N/As:

        >>> data = MaskedArray([1., 2., 3., 4.], [False, True, False, True])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        >>> TimeSeries(data, dates, 'ts').to_pandas()
        2021-11-01    1.0
        2021-11-02    NaN
        2021-11-03    3.0
        2021-11-04    NaN
        Name: ts, dtype: float64

        3. non-numeric array with N/As:

        >>> data = MaskedArray([False, False, True], [False, True, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').to_pandas()
        2021-11-01    False
        2021-11-02      NaN
        2021-11-03     True
        Name: ts, dtype: object

        >>> data = MaskedArray(['1', '2', '3'], [False, True, False])
        >>> dates = ['2021-11-01', '2021-11-02', '2021-11-03']
        >>> TimeSeries(data, dates, 'ts').to_pandas()
        2021-11-01      1
        2021-11-02    NaN
        2021-11-03      3
        Name: ts, dtype: object

        """
        values = self._data.to_numpy()
        index = self._index.values
        name = self._name
        return pd.Series(values, index=index, name=name)

    def __repr__(self) -> str:
        """String representation for the time-series.

        See Also
        --------
        pandas.Series

        """
        # This code can work but `mypy` raise a warning.
        # So, ignore `mypy` for wrong warning below.
        if len(self) <= 0:
            return f'TimeSeries([], Name: {self._name}, dtype: {self._data.dtype}'
        ret = '\n'.join([f'{d}\t{v}'
                         for d, v in zip(self._index, self._data)])  # type: ignore
        ret = f'{ret}\nName: {self._name}, dtype: {self._data.dtype}'
        return ret


class BooleanTimeSeries(TimeSeries):
    """Time-series of boolean data.

    Support logical operations as follows:
    - Unary : NOT(~)
    - Binary : AND(&), OR(|), XOR(^).

    While NOT operates on one `BooleanTimeSeries`, AND, OR, and XOR operate on
    two `BooleanTimeSeries` which have the equivalent index. These opertaions
    return a `BooleanTimeSeries`. Equal(==) and Not-equal(!=) operations are
    not supported, because they can be realized simply by logical operations.
    For example, `x == y` is equivalent to `x ^ ~y` and `x != y` is equivalent
    to `x ^ y`. To support a binary logical operator between a
    `BooleanTimeSeries` and a boolean scalar is unnecessary, because its
    result are always trivial. For example, let ``bts`` is an arbitrary
    `BooleanTimeSeries`, then `bts & True` is equal to `bts` and `bts & False`
    is equal to a `BooleanTimeSeries` with only ``False``.

    See Also
    --------
    TimeSeries, LogicalBinaryOperator, LogicalUnaryOperator.

    """
    def __init__(self, data: _MaskedArrayLike, index: _TimeIndexLike,
                 name: str, sort: bool = True):
        super().__init__(data, index, name, sort)
        if not np.issubdtype(self.dtype, np.bool_):
            raise ValueError("non-boolean values in 'data'")

    def __invert__(self) -> 'BooleanTimeSeries':
        data = ~self._data
        index = self._index
        name = f'~{self._name}'
        # `_make` is a classmethod inherited from `TimeSeries`, but `mypy`
        # would occur a wrong waring here.
        # So, ignore `mypy`for wrong warning below.
        return self._make(data, index, name)  # type: ignore

    def _logical_op(self, other: 'BooleanTimeSeries',
                    operator: LogicalBinaryOperator) -> 'BooleanTimeSeries':
        if not isinstance(other, BooleanTimeSeries):
            raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'"
                            % (operator.symbol, 'BooleanTimeSeries',
                               other.__class__.__name__))
        # pylint: disable=protected-access
        if not self._index.equals(other._index):
            raise ValueError("inconsistent index")
        data = operator.func(self._data, other._data)
        index = self._index
        name = f'{self._name} {operator.symbol} {other._name}'
        # pylint: enable=protected-access

        # `_make` is a classmethod inherited from `TimeSeries`, but `mypy`
        # would occur a wrong waring here.
        # So, ignore `mypy`for wrong warning below.
        return self._make(data, index, name)  # type: ignore

    def __and__(self, other: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        return self._logical_op(other, LogicalBinaryOperator.AND)

    def __or__(self, other: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        return self._logical_op(other, LogicalBinaryOperator.OR)

    def __xor__(self, other: 'BooleanTimeSeries') -> 'BooleanTimeSeries':
        return self._logical_op(other, LogicalBinaryOperator.XOR)


class NumericTimeSeries(TimeSeries):
    """Time-series of numeric data.

    Support numeric operations as follows:
    - Arithmetic :
        Negative(-), Absolute value(abs), Addition(+), Subtraction(-),
        Multiplication(*), Division(/), Modulus(%), Power(**),
        Floor division(//).
    - Comparison :
        Equal(==), Not equal(!=), Greater than(>), Less than(<),
        Greater than or equal to(>=), Less than or equal to(<=).

    Except 'Negative(-)' and 'Absolute value(abs)' are unary operators, the
    others all are binary operators which could be operated on
    two `NumericTimeSeries` or between a `NumericTimeSeries` and a numeric
    scalar. The result of arithmetic operators is a `NumericTimeSeries` and
    the result of comparison operators is a `BooleanTimeSeries`. Some specified
    rules as follows:
    - Division(/) :
        1. Not support zero division.
        -> raise ZeroDivisionError
    - Modulus(%) :
        1. Not support zero division.
        -> raise ZeroDivisionError
    - Power(**) :
        1. Not support `NumericTimeSeries` to `NumericTimeSeries` powers.
        -> raise TypeError
    - Floor division(//) :
        1. Not support zero division.
        -> raise ZeroDivisionError

    See Also
    --------
    TimeSeries, ArithmeticUnaryOperator, ArithmeticUnaryFunction,
    ArithmeticBinaryOperator, ComparisonOperator

    """
    def __init__(self, data: _MaskedArrayLike, index: _TimeIndexLike,
                 name: str, sort: bool = True):
        super().__init__(data, index, name, sort)
        if not np.issubdtype(self.dtype, np.number):
            raise ValueError("non-numeric values in 'data'")

    def _astype(self, dtype: Union[str, type, np.dtype]
                ) -> 'NumericTimeSeries':
        # `_make` is a classmethod inherited from `TimeSeries`, but `mypy`
        # would occur a wrong waring here.
        # So, ignore `mypy`for wrong warning below.
        ret = self._make(self._data.astype(dtype), self._index, self._name)
        return ret  # type: ignore

    def __neg__(self) -> 'NumericTimeSeries':
        data = -self._data
        index = self._index
        name = f'-{self._name}'
        # `_make` is a classmethod inherited from `TimeSeries`, but `mypy`
        # would occur a wrong waring here.
        # So, ignore `mypy`for wrong warning below.
        return self._make(data, index, name)  # type: ignore

    def __abs__(self) -> 'NumericTimeSeries':
        data = abs(self._data)
        index = self._index
        name = f'abs({self._name})'
        # `_make` is a classmethod inherited from `TimeSeries`, but `mypy`
        # would occur a wrong waring here.
        # So, ignore `mypy`for wrong warning below.
        return self._make(data, index, name)  # type: ignore

    def _arithmetic_op(self, other: Union['NumericTimeSeries', float, int],
                       operator: ArithmeticBinaryOperator
                       ) -> 'NumericTimeSeries':
        if isinstance(other, NumericTimeSeries):
            # pylint: disable=protected-access
            if not self._index.equals(other._index):
                raise ValueError("inconsistent index")
            data = operator.func(self._data, other._data)
            name = f'{self._name} {operator.symbol} {other._name}'
            # pylint: enable=protected-access
        elif np.issubdtype(type(other), np.number):
            data = operator.func(self._data, other)
            name = f'{self._name} {operator.symbol} {other}'
        else:
            raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'"
                            % (operator.symbol, 'NumericTimeSeries',
                               type(other).__name__))
        # `_make` is a classmethod inherited from `TimeSeries`, but `mypy`
        # would occur a wrong waring here.
        # So, ignore `mypy`for wrong warning below.
        return self._make(data, self._index, name)  # type: ignore

    def __add__(self, other: Union['NumericTimeSeries', float, int]
                ) -> 'NumericTimeSeries':
        return self._arithmetic_op(other, ArithmeticBinaryOperator.ADD)

    def __sub__(self, other: Union['NumericTimeSeries', float, int]
                ) -> 'NumericTimeSeries':
        return self._arithmetic_op(other, ArithmeticBinaryOperator.SUB)

    def __mul__(self, other: Union['NumericTimeSeries', float, int]
                ) -> 'NumericTimeSeries':
        return self._arithmetic_op(other, ArithmeticBinaryOperator.MUL)

    def __truediv__(self, other: Union['NumericTimeSeries', float, int]
                    ) -> 'NumericTimeSeries':
        if np.issubdtype(type(other), np.number) and other == 0:
            raise ZeroDivisionError("idivision by zero")
        return self._arithmetic_op(other, ArithmeticBinaryOperator.DIV)

    def __floordiv__(self, other: Union['NumericTimeSeries', float, int]
                     ) -> 'NumericTimeSeries':
        if np.issubdtype(type(other), np.number) and other == 0:
            raise ZeroDivisionError("integer division by zero")
        return self._arithmetic_op(other, ArithmeticBinaryOperator.FDIV)

    def __mod__(self, other: Union['NumericTimeSeries', float, int]
                ) -> 'NumericTimeSeries':
        if np.issubdtype(type(other), np.number) and other == 0:
            raise ZeroDivisionError("modulo by zero")
        return self._arithmetic_op(other, ArithmeticBinaryOperator.MOD)

    def __pow__(self, other: Union[float, int]
                ) -> 'NumericTimeSeries':
        if not np.issubdtype(type(other), np.number):
            raise TypeError("unsupported operand type(s) for **: '%s' and '%s'"
                            % ('NumericTimeSeries', type(other).__name__))
        if np.issubdtype(self.dtype, int) and other < 0:
            ret = self._astype(float)
            return ret._arithmetic_op(other, ArithmeticBinaryOperator.POW)
        return self._arithmetic_op(other, ArithmeticBinaryOperator.POW)

    def _comparison_op(self, other: Union['NumericTimeSeries', float, int],
                       operator: ComparisonOperator) -> BooleanTimeSeries:
        if isinstance(other, NumericTimeSeries):
            # pylint: disable=protected-access
            if not self._index.equals(other._index):
                raise ValueError("inconsistent index")
            data = operator.func(self._data, other._data)
            name = f'{self._name} {operator.symbol} {other._name}'
            # pylint: enable=protected-access
        elif np.issubdtype(type(other), np.number):
            data = operator.func(self._data, other)
            name = f'{self._name} {operator.symbol} {other}'
        else:
            raise TypeError("unsupported operand type(s) for %s: '%s' and '%s'"
                            % (operator.symbol, 'NumericTimeSeries',
                               type(other).__name__))
        return BooleanTimeSeries(data, self._index, name, sort=False)

    # When overriding `__eq__` and `__ne__` methods with specified object and
    # return non-boolean, `mypy` would raise a 'incompatible-override' waring.
    # So we ignore `mypy` above.
    def __eq__(self,  # type: ignore
               other: Union['NumericTimeSeries', float, int]
               ) -> BooleanTimeSeries:
        return self._comparison_op(other, ComparisonOperator.EQ)

    def __ne__(self,  # type: ignore
               other: Union['NumericTimeSeries', float, int]
               ) -> BooleanTimeSeries:
        return self._comparison_op(other, ComparisonOperator.NE)

    def __gt__(self, other: Union['NumericTimeSeries', float, int]
               ) -> BooleanTimeSeries:
        return self._comparison_op(other, ComparisonOperator.GT)

    def __lt__(self, other: Union['NumericTimeSeries', float, int]
               ) -> BooleanTimeSeries:
        return self._comparison_op(other, ComparisonOperator.LT)

    def __ge__(self, other: Union['NumericTimeSeries', float, int]
               ) -> BooleanTimeSeries:
        return self._comparison_op(other, ComparisonOperator.GE)

    def __le__(self, other: Union['NumericTimeSeries', float, int]
               ) -> BooleanTimeSeries:
        return self._comparison_op(other, ComparisonOperator.LE)


class TimeSeriesSampling:
    """Moving Samples of time-series.

    Parameters
    ----------
    data : MaskedArray (2-dimensional)
        The data stored in the object. It must be an instance of `MaskedArray`
        with two dimensions, in which the 1st dimension is equal to the length
        of `index` and the 2nd dimension is the number of samples.
    indices : MaskedArray (2-dimensional)
        The datetimes corresponding to each element in `data`. It has the same
        shape of `data` and its data-type is `numpy.datetime64`.
    index : TimeIndex
        The datetime values indexing `data`.
    name : str
        The name given to the object.

    Notes
    -----
    It is used as the result of `sampling` of `TimeSeries`, so ignore
    unnecessary checks.

    Methods
    -------
    to_pandas : pandas.DataFrame
        Return a copy of the object as a pandas DataFrame, in which the
        value of N/A elements are replaced with ``numpy.nan``.
    to_dict : Dict[datetime.date, TimeSeries]
        Return a copy of the object as a dict from datetime to time-series.

    Examples
    --------
    >>> index = np.array(['1970-01-01', '1970-01-02', '1970-01-03',
                          '1970-01-04', '1970-01-05', '1970-01-06',
                          '1970-01-07', '1970-01-08'], 'datetime64')
    >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> TimeSeries(data, index, 'ts').sampling(3, 2)
    1970-01-01:	1970-01-01	0
                1970-01-03	2
                1970-01-05	4
    1970-01-02:	1970-01-02	1
                1970-01-04	3
                1970-01-06	5
    1970-01-03:	1970-01-03	2
                1970-01-05	4
                1970-01-07	6
    1970-01-04:	1970-01-04	3
                1970-01-06	5
                1970-01-08	7
    1970-01-05:	1970-01-05	4
                1970-01-07	6
    1970-01-06:	1970-01-06	5
                1970-01-08	7
    1970-01-07:	1970-01-07	6
    1970-01-08:	1970-01-08	7
    Name: ts.sampling(3, 2), dtype: int32

    >>> TimeSeries(data, index, 'ts').sampling(3, -2)
    1970-01-01:	1970-01-01	0
    1970-01-02:	1970-01-02	1
    1970-01-03:	1970-01-01	0
                1970-01-03	2
    1970-01-04:	1970-01-02	1
                1970-01-04	3
    1970-01-05:	1970-01-01	0
                1970-01-03	2
                1970-01-05	4
    1970-01-06:	1970-01-02	1
                1970-01-04	3
                1970-01-06	5
    1970-01-07:	1970-01-03	2
                1970-01-05	4
                1970-01-07	6
    1970-01-08:	1970-01-04	3
                1970-01-06	5
                1970-01-08	7
    Name: ts.sampling(3, -2), dtype: int32

    """
    def __init__(self, data: MaskedArray, indices: MaskedArray,
                 index: TimeIndex, name: str):
        self._data = data
        self._indices = indices
        self._index = index
        self._name = name

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the object as pandas dataframe.

        Returns
        -------
        pandas.DataFrame
            A copy of the object as a `pandas.DataFrame`, in which the
            value of N/A elements are replaced with ``numpy.nan``.

        See also
        --------
        MaskedArray.to_numpy

        Examples
        --------
        >>> index = np.array(['1970-01-01', '1970-01-02', '1970-01-03',
                              '1970-01-04', '1970-01-05', '1970-01-06',
                              '1970-01-07', '1970-01-08'], 'datetime64')
        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        >>> TimeSeries(data, index, 'ts').sampling(2, 3).to_pandas()
                    ts.sampling(2, 3)[0]  ts.sampling(2, 3)[1]
        1970-01-01                   0.0                   3.0
        1970-01-02                   1.0                   4.0
        1970-01-03                   2.0                   5.0
        1970-01-04                   3.0                   6.0
        1970-01-05                   4.0                   7.0
        1970-01-06                   5.0                   NaN
        1970-01-07                   6.0                   NaN
        1970-01-08                   7.0                   NaN

        >>> TimeSeries(data, index, 'ts').sampling(2, -3).to_pandas()
                    ts.sampling(2, -3)[-1]  ts.sampling(2, -3)[0]
        1970-01-01                     NaN                    0.0
        1970-01-02                     NaN                    1.0
        1970-01-03                     NaN                    2.0
        1970-01-04                     0.0                    3.0
        1970-01-05                     1.0                    4.0
        1970-01-06                     2.0                    5.0
        1970-01-07                     3.0                    6.0
        1970-01-08                     4.0                    7.0

        """
        values = self._data.to_numpy()
        index = self._index.values
        if np.array_equal(index, self._indices.data[:, 0]):
            columns = [f'{self._name}[{v}]' for v in range(values.shape[1])]
        else:
            columns = [f'{self._name}[{v}]'
                       for v in range(-values.shape[1] + 1, 1)]
        return pd.DataFrame(values, index=index, columns=columns)

    def to_dict(self) -> Dict[datetime.date, TimeSeries]:
        """Return a copy of the object as dict of time-series.

        Returns
        -------
        dict

        Examples
        --------
        >>> index = np.array(['1970-01-01', '1970-01-02', '1970-01-03',
                              '1970-01-04', '1970-01-05', '1970-01-06',
                              '1970-01-07', '1970-01-08'], 'datetime64')
        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        >>> TimeSeries(data, index, 'ts').sampling(3, 2).to_dict()
        {datetime.date(1970, 1, 1): 1970-01-01	0
         1970-01-03	2
         1970-01-05	4
         Name: ts.sampling(3, 2)[1970-01-01], dtype: int32,
         datetime.date(1970, 1, 2): 1970-01-02	1
         1970-01-04	3
         1970-01-06	5
         Name: ts.sampling(3, 2)[1970-01-02], dtype: int32,
         datetime.date(1970, 1, 3): 1970-01-03	2
         1970-01-05	4
         1970-01-07	6
         Name: ts.sampling(3, 2)[1970-01-03], dtype: int32,
         datetime.date(1970, 1, 4): 1970-01-04	3
         1970-01-06	5
         1970-01-08	7
         Name: ts.sampling(3, 2)[1970-01-04], dtype: int32,
         datetime.date(1970, 1, 5): 1970-01-05	4
         1970-01-07	6
         Name: ts.sampling(3, 2)[1970-01-05], dtype: int32,
         datetime.date(1970, 1, 6): 1970-01-06	5
         1970-01-08	7
         Name: ts.sampling(3, 2)[1970-01-06], dtype: int32,
         datetime.date(1970, 1, 7): 1970-01-07	6
         Name: ts.sampling(3, 2)[1970-01-07], dtype: int32,
         datetime.date(1970, 1, 8): 1970-01-08	7
         Name: ts.sampling(3, 2)[1970-01-08], dtype: int32}

        >>> TimeSeries(data, index, 'ts').sampling(3, -2).to_dict()
        {datetime.date(1970, 1, 1): 1970-01-01	0
         Name: ts.sampling(3, -2)[1970-01-01], dtype: int32,
         datetime.date(1970, 1, 2): 1970-01-02	1
         Name: ts.sampling(3, -2)[1970-01-02], dtype: int32,
         datetime.date(1970, 1, 3): 1970-01-01	0
         1970-01-03	2
         Name: ts.sampling(3, -2)[1970-01-03], dtype: int32,
         datetime.date(1970, 1, 4): 1970-01-02	1
         1970-01-04	3
         Name: ts.sampling(3, -2)[1970-01-04], dtype: int32,
         datetime.date(1970, 1, 5): 1970-01-01	0
         1970-01-03	2
         1970-01-05	4
         Name: ts.sampling(3, -2)[1970-01-05], dtype: int32,
         datetime.date(1970, 1, 6): 1970-01-02	1
         1970-01-04	3
         1970-01-06	5
         Name: ts.sampling(3, -2)[1970-01-06], dtype: int32,
         datetime.date(1970, 1, 7): 1970-01-03	2
         1970-01-05	4
         1970-01-07	6
         Name: ts.sampling(3, -2)[1970-01-07], dtype: int32,
         datetime.date(1970, 1, 8): 1970-01-04	3
         1970-01-06	5
         1970-01-08	7
         Name: ts.sampling(3, -2)[1970-01-08], dtype: int32}

        """
        ret = {}
        for idx, date in enumerate(self._index.values.tolist()):
            values = self._data[idx]
            index = self._indices[idx]
            available = ~index.isna()
            name = f'{self._name}[{date}]'
            ret[date] = TimeSeries(values[available], index[available].data,
                                   name, sort=False)
        return ret

    def __repr__(self):
        recv = self.to_dict()
        ret = []
        for date, ts_ in recv.items():
            title = str(date)
            sep = '\n' + ' ' * (len(title) + 2)
            ts_ = sep.join(str(ts_).split('\n')[:-1])
            ret.append(f'{title}:\t{ts_}')
        ret = '\n'.join(ret) + f'\nName: {self._name}, dtype: {self._data.dtype}'
        return ret
