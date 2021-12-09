# -*- coding: utf-8 -*-
"""Time Index.

This module implement time-index which used as index of time-indexing data in
current package.

"""

import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from ._array import MaskedArray
from ._unit import TimeUnit


class _IndexGroupByTimeUnit:
    """Index-grouper by time-unit.

    Parameters
    ----------
    index : array of datetimes
    unit : TimeUnit

    Methods
    -------
    shift : Shift given values along the index by given period.
    sampling : Get moving samples of values along the index by desired step.

    Notes
    -----
    Because it is a private class and only used in this module, ignore
    unnecessary dynamic checking for better performance.

    """
    def __init__(self, index: np.ndarray, unit: TimeUnit):
        # Only used in current module, ignore dynamic type checking.
        values = unit.encode(index)
        is_changed = values[1:] != values[:-1]
        is_begin = np.concatenate([[True], is_changed])
        is_end = np.concatenate([is_changed, [True]])
        self._group_id = np.cumsum(is_begin) - 1
        self._begin_idx = np.where(is_begin)[0]  # only 1-D
        self._end_idx = np.where(is_end)[0]  # only 1-D

    def shift(self, values: MaskedArray, period: int) -> MaskedArray:
        """Shift values by desired period along the index.

        Parameters
        ----------
        values : MaskedArray
            A data array has equal length of the index.
        period : int
            Number of periods to shift. It could be positive or negative but
            not be zero. If `period` is set as a positive integer, n, it return
            the elements of `values` corresponding to the index shifted
            backward n units and the elements corresponging to the first n
            units are set to be N/A. If `period` is set as a negative integer,
            -n, it return the elements of `values` corresponding to the index
            shifted forward n units and the elements corresponging to the last
            n units are set to be N/A.

        """
        # Only used in current module, ignore length checking and value checking
        if period > 0:
            idxs = self._end_idx[self._group_id - period]
        else:
            idxs = self._begin_idx[self._group_id - period - len(self._begin_idx)]
        ret = values[idxs]
        if period > 0:
            ret[:self._begin_idx[period]] = np.nan
        else:
            ret[self._begin_idx[period]:] = np.nan
        return ret

    def sampling(self, values: MaskedArray, samples: int, step: int
                 ) -> MaskedArray:
        """Get moving samples along the index by desired step.

        Parameters
        ----------
        values : MaskedArray
            A data array has equal length of the index.
        samples : int
            Number of samples.
        step : int
            Number of step between each sample. It could be positive or
            negative but not be zero. If `step` is set as a positive integer,
            n, it get samples forward per n units along the index. If `step` is
            set as a negative integer, -n, it get samples backward per n units
            along the index.

        Returns
        -------
        MaskedArray
            An extended masked-array which 1st dimension is equal to the 1st
            dimension of `values` and the 2nd dimension is equal to `samples`.
            The rest dimensions are equal to which of `values`. For example, if
            the shape of `values` is ``(m, n)`` and `samples` is `k`, the shape
            of result is ``(m, k, n)``.

        """
        # Only used in current module, ignore length checking and value checking
        if step > 0:
            ret = values[self._begin_idx
                         ].moving_sampling(samples, step)[self._group_id]
            ret[:, 0] = values
        else:
            ret = values[self._end_idx
                         ].moving_sampling(samples, step)[self._group_id]
            ret[:, -1] = values
        return ret


class TimeIndex:
    """Time-Index.

    Immutable sequence of datetimes, used as index of time-indexing data.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Data used to construct time-index. It should be a 1-dimensional array
        or any sequence, in which all elements can be converted to
        `numpy.datetime64`.
    sort : bool
        If ``True``, sort `data`; otherwise do not. Default is ``True``.

    Notes
    -----
    `TimeIndex` is used as index of time-indexing data supported in current
    package, such as `TimeSeries` and `PanelData`. For better performance, it
    is immutable and always used as a reference instead of a copy. Because it
    should only be used in current package internally, so unnecessary dynamic
    checks are ignored.

    Properties
    ----------
    values : numpy.ndarray
        Return a copy of the time-index as an array of datetimes.

    Built-in Functions
    ------------------
    len : int
        Length of the time-index. It is equivalent to the number of elements
        in the time-index.

    Methods
    -------
    equals : bool
        Determine if two time-index are equal.
    argsort: numpy.array
        Return the indices that would sort the time-index.
    shift : MaskedArray
        Shift values equivalent to the time-index shifted by desired period.
    sampling : MaskedArray
        Get moving samples of values along the time-index by desired step.

    Examples
    --------
    >>> dates = np.arange(3).astype('datetime64[D]')
    >>> TimeIndex(dates)
    TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    With `data` as a list of `datetime.date`:

    >>> dates = [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]
    >>> TimeIndex(dates)
    TimeIndex(['1970-01-01', '1970-01-02'], dtype='datetime64[D]')

    With `data` as an array of `datetimes.date`:

    >>> dates = [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]
    >>> TimeIndex(np.array(dates))
    TimeIndex(['1970-01-01', '1970-01-02'], dtype='datetime64[D]')

    With `data` as a list of string representation of datetime:

    >>> dates = ['1970-01-01', '1970-01-02', '1970-01-03']
    >>> TimeIndex(dates)
    TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    With `data` as an array of string representation of datetime:

    >>> dates = ['1970-01-01', '1970-01-02', '1970-01-03']
    >>> TimeIndex(np.array(dates))
    TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    Without setting `sort`:
    >>> dates = np.arange(3).astype('datetime64[D]')[::-1]
    >>> TimeIndex(dates)
    TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    With `sort = False`:
    >>> dates = np.arange(3).astype('datetime64[D]')[::-1]
    >>> TimeIndex(dates, sort=False)
    TimeIndex(['1970-01-03', '1970-01-02', '1970-01-01'], dtype='datetime64[D]')

    """
    def __init__(self, data: ArrayLike, sort: bool = True):
        self._values = np.array(data, dtype=np.datetime64)
        if sort:
            self._values.sort()
        self._grouper: Dict[TimeUnit, _IndexGroupByTimeUnit] = {}

    @property
    def values(self) -> np.ndarray:
        """
        Return a copy of the time-index as an array of datetimes.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> dates = np.arange(3).astype('datetime64[D]')
        >>> TimeIndex(dates).values
        array(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

        """
        return self._values.copy()

    def __len__(self) -> int:
        """Length of the time-index.

        Examples
        --------
        >>> dates = np.arange(5).astype('datetime64[D]')
        >>> len(TimeIndex(dates))
        5

        """
        return len(self._values)

    def equals(self, other: 'TimeIndex') -> bool:
        """Determine if two time-index are equal.

        The things that are being compared are:
        - the elements inside the `TimeIndex` object, and
        - the order of the elements inside the `TimeIndex` object.

        Parameters
        ----------
        other : TimeIndex
            The other time-index to compare against.

        Returns
        -------
        bool
            ``True`` if `other` is an `TimeIndex` object and it has the same
            elements and order as the calling object; ``False`` otherwise.

        See Also
        --------
        numpy.array_equal

        Examples
        --------
        >>> dates = np.arange(5).astype('datetime64[D]')
        >>> TimeIndex(dates).equals(TimeIndex(dates))
        True

        Same elements but different order:

        >>> TimeIndex(dates).equals(TimeIndex(dates[::-1], sort=False))
        False

        Non-TimeIndex:

        >>> TimeIndex(dates).equals(dates)
        False

        """
        if self is other:
            return True
        if isinstance(other, TimeIndex):
            # pylint: disable=protected-access
            return np.array_equal(self._values, other._values)
            # pylint: enable=protected-access
        return False

    def argsort(self) -> np.ndarray:
        """Return the indices that would sort the time-index.

        See Also
        --------
        numpy.ndarray.argsort

        Examples
        --------
        >>> dates = np.arange(3).astype('datetime64[D]')
        >>> TimeIndex(dates).argsort()
        array([0, 1, 2], dtype=int64)

        Built with unsorted `data` and default `sort`:

        >>> dates = np.arange(3)[::-1].astype('datetime64[D]')
        >>> TimeIndex(dates).argsort()
        array([0, 1, 2], dtype=int64)

        Built with unsorted `data` and `sort=False`:

        >>> dates = np.arange(3)[::-1].astype('datetime64[D]')
        >>> TimeIndex(dates, sort=False).argsort()
        array([2, 1, 0], dtype=int64)

        """
        return self._values.argsort()

    def __getitem__(self, key: Any) -> Union[datetime.date, 'TimeIndex']:
        """Subscript Operator.

        Parameters
        ----------
        key:
            Like the subscript operator of NumPy array.

        Returns
        -------
        datetime or MaskedArray:
            If the result of subscript operator is an array, return it as a
            time-index; otherwise return it as a datetime object.

        See Also
        --------
        numpy.ndarray

        Examples
        --------
        >>> dates = np.arange(5).astype('datetime64[D]')
        >>> tindex = TimeIndex(dates)
        >>> tindex
        TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03', '1970-01-04',
                   '1970-01-05'], dtype='datetime64[D]')

        1. supscript with slice:

        >>> tindex[1:]
        TimeIndex(['1970-01-02', '1970-01-03', '1970-01-04', '1970-01-05'],
                  dtype='datetime64[D]')
        >>> tindex[:-1]
        TimeIndex(['1970-01-01', '1970-01-02', '1970-01-03', '1970-01-04'],
                  dtype='datetime64[D]')
        >>> tindex[1:-1]
        TimeIndex(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')
        >>> tindex[::2]
        TimeIndex(['1970-01-01', '1970-01-03', '1970-01-05'], dtype='datetime64[D]')

        2a. supscript with integer:

        >>> tindex[2]
        datetime.date(1970, 1, 3)

        2b. supscript with integer(raise out-of-range):

        >>> tindex[5]
        IndexError: index 5 is out of bounds for axis 0 with size 5

        3a. supscript with integer array:

        >>> tindex[np.array([1, 3])]
        TimeIndex(['1970-01-02', '1970-01-04'], dtype='datetime64[D]')

        3b. supscript with integer array(raise out-of-range):

        >>> tindex[np.array([1, 3, 5])]
        IndexError: index 5 is out of bounds for axis 0 with size 5

        4a. supscript with boolean array(same length):

        >>> tindex[np.arange(5) % 2 == 0]
        TimeIndex(['1970-01-01', '1970-01-03', '1970-01-05'], dtype='datetime64[D]')

        4b. supscript with boolean array(different length):

        >>> tindex[np.arange(6) % 2 == 0]
        IndexError: boolean index did not match indexed array along dimension 0;
        dimension is 5 but corresponding boolean dimension is 6

        """
        values = self._values[key]
        if isinstance(values, np.ndarray):
            # return a time-index
            return TimeIndex(values, sort=False)
        # return a datetime
        return values.tolist()

    def shift(self, values: MaskedArray, period: int,
              punit: Optional[TimeUnit] = None) -> MaskedArray:
        """Shift values equivalent to the time-index shifted by desired period.

        Parameters
        ----------
        values : MaskedArray
            A data array has same length of the time-index.
        period : int
            Number of periods to shift. It could be positive or negative but
            not be zero. If `period` is set as a positive integer, n, it return
            the elements of `values` corresponding to the index shifted
            backward n units and the elements corresponging to the first n
            units are set to be N/A. If `period` is set as a negative integer,
            -n, it return the elements of `values` corresponding to the index
            shifted forward n units and the elements corresponging to the last
            n units are set to be N/A.
        punit : TimeUnit, optional
            It is optional. It it is specified, it must be an instance of
            `TimeUnit` which is a super-unit of or equivalent to the dtype of
            index.

        See Also
        --------
        _IndexGroupByTimeUnit.shift

        Examples
        --------
        >>> tindex = TimeIndex(['2021-11-01', '2021-11-03', '2021-11-06',
                                '2021-11-10', '2021-11-13', '2021-11-15',
                                '2021-11-18', '2021-11-22', '2021-11-25',
                                '2021-11-27', '2021-11-30', '2021-12-03'])
        >>> values = MaskedArray(np.arange(12))
        >>> values
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], dtype=int32)

        1A. positive `period` and no specified `punit`:

        >>> tindex.shift(values, 2)
        array([nan, nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

        1B. negative `period` and no specified `punit`:

        >>> tindex.shift(values, -2)
        array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, nan, nan], dtype=int32)

        2A. positive `period` and equivalent `punit`:

        >>> tindex.shift(values, 2, TimeUnit.DAY)
        array([nan, nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

        2B. negative `period` and equivalent `punit`:

        >>> tindex.shift(values, -2, TimeUnit.DAY)
        array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, nan, nan], dtype=int32)

        3A. positive `period` and super-unit `punit`:

        >>> tindex.shift(values, 2, TimeUnit.WEEK)
        array([nan, nan, nan, nan, nan, 2, 2, 4, 4, 4, 6, 6], dtype=int32)

        3B. negative `period` and super-unit `punit`:

        >>> tindex.shift(values, -2, TimeUnit.WEEK)
        array([5, 5, 5, 7, 7, 10, 10, nan, nan, nan, nan, nan], dtype=int32)

        4. sub-unit `punit`:

        >>> tindex.shift(values, 1, TimeUnit.HOUR)
        ValueError: not support shift 'TimeUnit.HOUR' on 'datetime64[D]' datetimes

        5. non-integer `period`:

        >>> tindex.shift(values, 1.)
        TypeError: 'period' must be 'int' not 'float'

        6. zero `period`:

        >>> tindex.shift(values, 0)
        ValueError: 'period' can not be zero

        7. invalid `punit`:

        >>> tindex.shift(values, 1, 'day')
        TypeError: 'punit' must be 'TimeUnit' not 'str'

        """
        if not isinstance(period, int):
            raise TypeError("'period' must be 'int' not '%s'"
                            % type(period).__name__)
        if period == 0:
            raise ValueError("'period' can not be zero")
        if punit is not None and not isinstance(punit, TimeUnit):
            raise TypeError("'punit' must be 'TimeUnit' not '%s'"
                            % type(punit).__name__)
        if punit is None or punit.isequiv(self._values.dtype):
            return values.shift(period)
        if punit.issub(self._values.dtype):
            raise ValueError("not support shift '%s' on '%s' datetimes"
                             % (punit, self._values.dtype.name))
        if punit not in self._grouper:
            self._grouper[punit] = _IndexGroupByTimeUnit(self._values, punit)
        return self._grouper[punit].shift(values, period)

    def sampling(self, values: MaskedArray, samples: int, step: int,
                 sunit: Optional[TimeUnit] = None) -> MaskedArray:
        """Get moving samples along the time-index by desired step.

        Parameters
        ----------
        values : MaskedArray
            A data array has equal length of the index.
        samples : int
            Number of samples.
        step : int
            Number of step between each sample. It could be positive or
            negative but not be zero. If `step` is set as a positive integer,
            n, it get samples forward per n units along the time-index. If
            `step` is set as a negative integer, -n, it get samples backward
            per n units along the time-index.
        sunit : TimeUnit
            Time-unit of step.

        Returns
        -------
        MaskedArray
            An extended masked-array which 1st dimension is equal to the 1st
            dimension of `values` and the 2nd dimension is equal to `samples`.
            The rest dimensions are equal to which of `values`. For example, if
            the shape of `values` is ``(m, n)`` and `samples` is `k`, the shape
            of result masked-array is ``(m, k, n)``.

        Examples
        --------
        >>> tindex = TimeIndex(['2021-11-01', '2021-11-03', '2021-11-06',
                                '2021-11-10', '2021-11-13', '2021-11-15',
                                '2021-11-18', '2021-11-22', '2021-11-25',
                                '2021-11-27', '2021-11-30', '2021-12-03'])
        >>> values = MaskedArray(np.arange(12), np.arange(12) % 5 == 0)
        >>> values
        array([nan, 1, 2, 3, 4, nan, 6, 7, 8, 9, nan, 11], dtype=int32)

        1A. positive `step` and on specified `sunit`:

        >>> tindex.sampling(values, 3, 2)
        array([[nan, 2, 4],
               [1, 3, nan],
               [2, 4, 6],
               [3, nan, 7],
               [4, 6, 8],
               [nan, 7, 9],
               [6, 8, nan],
               [7, 9, 11],
               [8, nan, nan],
               [9, 11, nan],
               [nan, nan, nan],
               [11, nan, nan]], dtype=int32)

        1B. negative `step` and on specified `sunit`:

        >>> tindex.sampling(values, 3, -2)
        array([[nan, nan, nan],
               [nan, nan, 1],
               [nan, nan, 2],
               [nan, 1, 3],
               [nan, 2, 4],
               [1, 3, nan],
               [2, 4, 6],
               [3, nan, 7],
               [4, 6, 8],
               [nan, 7, 9],
               [6, 8, nan],
               [7, 9, 11]], dtype=int32)

        2A. positive `step` and equivalent `sunit`:

        >>> tindex.sampling(values, 3, 2, TimeUnit.DAY)
        array([[nan, 2, 4],
               [1, 3, nan],
               [2, 4, 6],
               [3, nan, 7],
               [4, 6, 8],
               [nan, 7, 9],
               [6, 8, nan],
               [7, 9, 11],
               [8, nan, nan],
               [9, 11, nan],
               [nan, nan, nan],
               [11, nan, nan]], dtype=int32)

        2B. negative `step` and equivalent `sunit`:

        >>> tindex.sampling(values, 3, -2, TimeUnit.DAY)
        array([[nan, nan, nan],
               [nan, nan, 1],
               [nan, nan, 2],
               [nan, 1, 3],
               [nan, 2, 4],
               [1, 3, nan],
               [2, 4, 6],
               [3, nan, 7],
               [4, 6, 8],
               [nan, 7, 9],
               [6, 8, nan],
               [7, 9, 11]], dtype=int32)

        3A. positive `step` and super-unit `sunit`:

        >>> tindex.sampling(values, 3, 2, TimeUnit.WEEK)
        array([[nan, nan, nan],
               [1, nan, nan],
               [2, nan, nan],
               [3, 7, nan],
               [4, 7, nan],
               [nan, nan, nan],
               [6, nan, nan],
               [7, nan, nan],
               [8, nan, nan],
               [9, nan, nan],
               [nan, nan, nan],
               [11, nan, nan]], dtype=int32)

        3B. negative `step` and super-unit `sunit`:

        >>> tindex.sampling(values, 3, -2, TimeUnit.WEEK)
        array([[nan, nan, nan],
               [nan, nan, 1],
               [nan, nan, 2],
               [nan, nan, 3],
               [nan, nan, 4],
               [nan, 2, nan],
               [nan, 2, 6],
               [nan, 4, 7],
               [nan, 4, 8],
               [nan, 4, 9],
               [2, 6, nan],
               [2, 6, 11]], dtype=int32)

        4. sub-unit `sunit`:

        >>> tindex.sampling(values, 2, 1, TimeUnit.HOUR)
        ValueError: not support sampling 'TimeUnit.HOUR' on 'datetime64[D]' datetimes

        5A. non-integer `samples`:

        >>> tindex.sampling(values, 2., 1)
        TypeError: 'samples' must be 'int' not 'float'

        5B. invalid `samples`:

        >>> tindex.sampling(values, 1, 1)
        ValueError: 'samples' must be larger than 1

        >>> tindex.sampling(values, 0, 1)
        ValueError: 'samples' must be larger than 1

        >>> tindex.sampling(values, -1, 1)
        ValueError: 'samples' must be larger than 1

        6A. non-integer `step`:

        >>> tindex.sampling(values, 2, 1.)
        TypeError: 'step' must be 'int' not 'float'

        6B. zero `step`:

        >>> tindex.sampling(values, 2, 0)
        ValueError: 'step' must be non-zero

        7. invalid `sunit`:

        >>> tindex.sampling(values, 2, 1, 'day')
        TypeError: 'sunit' must be 'TimeUnit' not 'str'

        """
        # Dynamic checks for `samples` and `step` are done by `moving_sampling`
        # in 'MaskedArray`.
        if sunit is not None and not isinstance(sunit, TimeUnit):
            raise TypeError("'sunit' must be 'TimeUnit' not '%s'"
                            % type(sunit).__name__)
        if sunit is None or sunit.isequiv(self._values.dtype):
            return values.moving_sampling(samples, step)
        if sunit.issub(self._values.dtype):
            raise ValueError("not support sampling '%s' on '%s' datetimes"
                             % (sunit, self._values.dtype.name))
        if sunit not in self._grouper:
            self._grouper[sunit] = _IndexGroupByTimeUnit(self._values, sunit)
        return self._grouper[sunit].sampling(values, samples, step)

    def __repr__(self) -> str:
        """String representation for the time-index.

        See Also
        --------
        numpy.ndarray

        """
        ret = repr(self._values)
        ret = ret.replace('array', 'TimeIndex')
        # fix indentation for the difference between 'array' and 'TimeIndex'
        ret = ret.replace('\n', '\n    ')
        return ret
