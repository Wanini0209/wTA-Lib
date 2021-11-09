# -*- coding: utf-8 -*-
"""Time Index.

This module implement time-index which used as index of time-indexing data in
current package.

"""

import datetime
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike


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

    Examples
    --------
    >>> dates = np.arange(3).astype('datetime64[D]')
    >>> TimeIndex(dates).values
    array(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    With `data` as a list of `datetime.date`:

    >>> dates = [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]
    >>> TimeIndex(dates).values
    array(['1970-01-01', '1970-01-02'], dtype='datetime64[D]')

    With `data` as an array of `datetimes.date`:

    >>> dates = [datetime.date(1970, 1, 1), datetime.date(1970, 1, 2)]
    >>> TimeIndex(np.array(dates)).values
    array(['1970-01-01', '1970-01-02'], dtype='datetime64[D]')

    With `data` as a list of string representation of datetime:

    >>> dates = ['1970-01-01', '1970-01-02', '1970-01-03']
    >>> TimeIndex(dates).values
    array(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    With `data` as an array of string representation of datetime:

    >>> dates = ['1970-01-01', '1970-01-02', '1970-01-03']
    >>> TimeIndex(np.array(dates)).values
    array(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    Without setting `sort`:
    >>> dates = np.arange(3).astype('datetime64[D]')[::-1]
    >>> TimeIndex(dates).values
    array(['1970-01-01', '1970-01-02', '1970-01-03'], dtype='datetime64[D]')

    With `sort = False`:
    >>> dates = np.arange(3).astype('datetime64[D]')[::-1]
    >>> TimeIndex(dates, sort=False).values
    array(['1970-01-03', '1970-01-02', '1970-01-01'], dtype='datetime64[D]')

    """
    def __init__(self, data: ArrayLike, sort: bool = True):
        self._values = np.array(data, dtype=np.datetime64)
        if sort:
            self._values.sort()

    @property
    def values(self) -> np.ndarray:
        """
        Return a copy of the time-index as an array of datetimes.

        Returns
        -------
        numpy.ndarray

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
        >>> tindex.values
        array(['1970-01-01', '1970-01-02', '1970-01-03', '1970-01-04',
               '1970-01-05'], dtype='datetime64[D]')

        1. supscript with slice:

        >>> tindex[1:].values
        array(['1970-01-02', '1970-01-03', '1970-01-04', '1970-01-05'],
              dtype='datetime64[D]')
        >>> tindex[:-1].values
        array(['1970-01-01', '1970-01-02', '1970-01-03', '1970-01-04'],
              dtype='datetime64[D]')
        >>> tindex[1:-1].values
        array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')
        >>> tindex[::2].values
        array(['1970-01-01', '1970-01-03', '1970-01-05'], dtype='datetime64[D]')

        2a. supscript with integer:

        >>> tindex[2]
        datetime.date(1970, 1, 3)

        2b. supscript with integer(raise out-of-range):

        >>> tindex[5]
        IndexError: index 5 is out of bounds for axis 0 with size 5

        3a. supscript with integer array:

        >>> tindex[np.array([1, 3])].values
        array(['1970-01-02', '1970-01-04'], dtype='datetime64[D]')

        3b. supscript with integer array(raise out-of-range):

        >>> tindex[np.array([1, 3, 5])].values
        IndexError: index 5 is out of bounds for axis 0 with size 5

        4a. supscript with boolean array(same length):

        >>> tindex[np.arange(5) % 2 == 0].values
        array(['1970-01-01', '1970-01-03', '1970-01-05'], dtype='datetime64[D]')

        4b. supscript with boolean array(different length):

        >>> tindex[np.arange(6) % 2 == 0].values
        IndexError: boolean index did not match indexed array along dimension 0;
        dimension is 5 but corresponding boolean dimension is 6

        """
        values = self._values[key]
        if isinstance(values, np.ndarray):
            # return a time-index
            return TimeIndex(values, sort=False)
        # return a datetime
        return values.tolist()
