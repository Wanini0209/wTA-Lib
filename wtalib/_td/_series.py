# -*- coding: utf-8 -*-
"""Time Series.

This module provides the base of time-series and its derivatives.

Classes
-------
TimeSeries : A sequence of data points indexed by time.

"""

from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._array import MaskedArray
from ._index import TimeIndex

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
            self._data = data
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
