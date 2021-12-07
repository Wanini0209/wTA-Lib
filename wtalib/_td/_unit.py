# -*- coding: utf-8 -*-
"""Time Unit.

This module provides definitions and utilities of time-units.

"""

from enum import Enum
from typing import NamedTuple

import numpy as np


class _TimeUnit(NamedTuple):
    """Unit of Time.

    Attributes
    ----------
    name : str
    symbol : str
        A single char or a single char follows an integer. It is case-sensitive.
    dtype : np.dtype
        The base unit in NumPy's datetime representation.
    factor : int
        The facotr from the time-unit to `dtype`. Default is ``1``.
        For example, one week is 7days then the factor of 'week' based on
        NumPy's dates is ``7``.
    offset : int
        The offset from the time-unit to `dtype`. Default is ``0``.
        For example, in NumPy the integer ``0`` is corresponding date of
        Thirsday but the first day of one week is Monday, so the offset of
        'week' based on NumPy's dates is ``3``.

    Methods
    -------
    issuper : Determine the time-unit is a super-unit of another NumPy's dtype
              of datetime.
    issub : Determine the time-unit is a sub-unit of another NumPy's dtype
            of datetime.
    isequiv : Determine the time-unit is equivalent to another NumPy's dtype
              of datetime.
    encode : Convert NumPy's datetimes to the integer representation of
             the time-unit.
    decode : Convert integers representation of the time-unit to the NumPy's
             datetimes.

    """
    name: str
    symbol: str
    dtype: np.dtype
    factor: int = 1
    offset: int = 0

    def issuper(self, other: np.dtype) -> bool:
        """Determine the time-unit is a super-unit of the given NumPy's dtype.

        Parameters
        ----------
        other : numpy.dtype

        Returns
        -------
        bool
            Return ``True`` if the time-unit is a super-unit of `other`;
            otherwise return ``False``.

        """
        if not isinstance(other, np.dtype):
            raise TypeError("'other' must be a 'numpy.dtype' not a '%s'"
                            % type(other).__name__)
        if not np.issubdtype(other, np.datetime64):
            raise ValueError(f"'{other}' is not a dtype of 'numpy.datetime64'")
        if self.dtype == other:
            return self.factor > 1
        return self.dtype < other

    def issub(self, other: np.dtype) -> bool:
        """Determine the time-unit is a sub-unit of the given NumPy's dtype.

        Parameters
        ----------
        other : numpy.dtype

        Returns
        -------
        bool
            Return ``True`` if the time-unit is a sub-unit of `other`;
            otherwise return ``False``.

        """
        if not isinstance(other, np.dtype):
            raise TypeError("'other' must be a 'numpy.dtype' not a '%s'"
                            % type(other).__name__)
        if not np.issubdtype(other, np.datetime64):
            raise ValueError(f"'{other}' is not a dtype of 'numpy.datetime64'")
        return self.dtype > other

    def isequiv(self, other: np.dtype) -> bool:
        """Determine the time-unit is equivalent to another dtype of time.

        Parameters
        ----------
        other : numpy.dtype

        Returns
        -------
        bool
            Return ``True`` if the time-unit is equivalent to `other`;
            otherwise return ``False``.

        """
        if not isinstance(other, np.dtype):
            raise TypeError("'other' must be a 'numpy.dtype' not a '%s'"
                            % type(other).__name__)
        if not np.issubdtype(other, np.datetime64):
            raise ValueError(f"'{other}' is not a dtype of 'numpy.datetime64'")
        return self.dtype == other and self.factor == 1

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encoder of time-unit.

        Convert NumPy's datetimes to the integer representation of the
        time-unit.

        Parameters
        ----------
        values : numpy.ndarray
            The input array of datetimes.

        Returns
        -------
        numpy.ndarray
            The output array of integers.

        """
        if not isinstance(values, np.ndarray):
            raise TypeError("'values' must be an array not a '%s'"
                            % type(values).__name__)
        if not np.issubdtype(values.dtype, np.datetime64):
            raise ValueError("'values' must be a array of 'datetime64' not '%s'"
                             % values.dtype)
        if self.issub(values.dtype):
            raise ValueError("not support up-casting encode")
        ret = values.astype(self.dtype).astype(int)
        if self.factor > 1:
            ret = (ret + self.offset) // self.factor
        return ret

    def decode(self, values: np.ndarray) -> np.ndarray:
        """Decoder of time-unit.

        Convert integers representation of the time-unit to the NumPy's
        datetimes.

        Parameters
        ----------
        values : numpy.ndarray
            The input array of integers.

        Returns
        -------
        numpy.ndarray
            The output array of datetimes.

        """
        if not isinstance(values, np.ndarray):
            raise TypeError("'values' must be an array not a '%s'"
                            % type(values).__name__)
        if not np.issubdtype(values.dtype, int):
            raise ValueError("'values' must be a array of 'int' not '%s'"
                             % values.dtype)
        if self.factor > 1:
            values = values * self.factor - self.offset
        ret = values.astype(self.dtype)
        return ret


class TimeUnit(_TimeUnit, Enum):
    """Enumerator of time-units.

    Members
    -------
    SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, QUARTER, YEAR.

    """
    SECOND = _TimeUnit('second', 's', np.dtype('datetime64[s]'))
    MINUTE = _TimeUnit('minute', 'm', np.dtype('datetime64[m]'))
    HOUR = _TimeUnit('hour', 'h', np.dtype('datetime64[h]'))
    DAY = _TimeUnit('day', 'D', np.dtype('datetime64[D]'))
    WEEK = _TimeUnit('week', 'W', np.dtype('datetime64[D]'),
                     factor=7,  # 7 days per week
                     # NumPy's integer representation of Thursday is Multiples
                     # of 7. In order to shift it to 3, set `offset` to be ``3``.
                     offset=3)
    MONTH = _TimeUnit('month', 'M', np.dtype('datetime64[M]'))
    QUARTER = _TimeUnit('quarter', 'Q', np.dtype('datetime64[M]'),
                        factor=3)  # 3 months per quarter
    YEAR = _TimeUnit('year', 'Y', np.dtype('datetime64[Y]'))
