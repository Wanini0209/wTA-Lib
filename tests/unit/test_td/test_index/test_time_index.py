# -*- coding: utf-8 -*-
"""Unit-Tests related to `TimeIndex`."""

import datetime

import numpy as np

from .._context import TimeIndex

# pylint: disable=no-self-use, too-few-public-methods


def array_equal(tar: np.ndarray, ref: np.ndarray) -> bool:
    """Determine if two array are equal.

    Two array are equal, it means they have
    1. the same shape,
    2. the same dtype, and
    3. the same elements.

    """
    if tar.dtype == ref.dtype:
        return np.array_equal(tar, ref)
    return False


class TestTimeIndex:
    """Tests related to builder of `TimeIndex`.

    """
    def test_always_copy_data(self):
        """Should always copy `data`.

        When the data array used to construct an instance of `TimeIndex`
        being modified, the content of the built object would not be changed.

        """
        src = np.arange(2).astype('datetime64[D]')
        tindex = TimeIndex(src)
        cond_1 = array_equal(src, tindex.values)  # before being modified
        src[0] = src[1]
        cond_2 = array_equal(src, tindex.values)  # after being modified
        assert cond_1 and not cond_2

    def test_sort_data_by_default(self):
        """Return an instance of `TimeIndex` with sorted values.

        When constructing without setting `sort`(By default), it should return
        an instance of `TimeIndex` with expected values, an sorted array of
        `numpy.datetime64`.

        """
        src = np.arange(2).astype('datetime64[D]')  # sorted array
        tindex = TimeIndex(src[::-1])  # construct with unsorted `data`
        assert array_equal(src, tindex.values)

    def test_disable_sort(self):
        """Return an instance of `TimeIndex` with unsorted values.

        When constructing with `sort=False`, it should return an instance of
        `TimeIndex` with expected values, an unsorted array of
        `numpy.datetime64`.

        """
        src = np.arange(2).astype('datetime64[D]')[::-1]  # unsorted array
        # construct with unsorted `data` and disable `sort`
        tindex = TimeIndex(src, sort=False)
        assert array_equal(src, tindex.values)

    def test_with_array_of_datetime_objects(self):
        """Return an instance of `TimeIndex` with expected values.

        When constructing with an array of `datatime.date`, it should return an
        instance of `TimeIndex` with expected values, an equivalent array of
        `numpy.datetime64`.

        """
        src = np.arange(2).astype('datetime64[D]')
        # construct with array of `datetime.date`
        tindex = TimeIndex(src.astype(datetime.date))
        assert array_equal(src, tindex.values)

    def test_with_list_of_datetime_objects(self):
        """Return an instance of `TimeIndex` with expected values.

        When constructing with a list of `datatime.date`, it should return an
        instance of `TimeIndex` with expected values, an equivalent array of
        `numpy.datetime64`.

        """
        src = np.arange(2).astype('datetime64[D]')
        # construct with list of `datetime.date`
        tindex = TimeIndex(src.tolist())
        assert array_equal(src, tindex.values)

    def test_with_array_of_string_of_datetimes(self):
        """Return an instance of `TimeIndex` with expected values.

        When constructing with an array of string representation of datetimes,
        it should return an instance of `TimeIndex` with expected values, an
        equivalent array of `numpy.datetime64`.

        """
        src = np.arange(2).astype('datetime64[D]')
        # construct with array of string representation of datetime
        tindex = TimeIndex(src.astype(str))
        assert array_equal(src, tindex.values)

    def test_with_list_of_string_of_datetimes(self):
        """Return an instance of `TimeIndex` with expected values.

        When constructing with a list of string representation of datetimes,
        it should return an instance of `TimeIndex` with expected values, an
        equivalent array of `numpy.datetime64`.

        """
        src = np.arange(2).astype('datetime64[D]')
        # construct with list of string representation of datetime
        tindex = TimeIndex(src.astype(str).tolist())
        assert array_equal(src, tindex.values)


class TestValues:
    """Tests related to `TimeIndex.values`.

    See Also
    --------
    TestTimeIndex

    """
    def test_always_return_a_copy(self):
        """Should always return a copy.

        When the array received from `value` of an instance of `TimeIndex`
        being modified, the content of the instance would not be changed.

        """
        tindex = TimeIndex(np.arange(2).astype('datetime64[D]'))
        recv = tindex.values
        cond_1 = array_equal(recv, tindex.values)  # before being modified
        recv[0] = recv[1]
        cond_2 = array_equal(recv, tindex.values)  # after being modified
        assert cond_1 and not cond_2


class TestLen:
    """Tests related to `len` on 'TimeIndex`.

    """
    def test_return_expected_integer(self):
        """Should return an expected integer.

        When calling `len` with an instance of `TimeIndex`, it should return an
        integer equals to the length of array used to construct the instance.

        """
        src = np.arange(2).astype('datetime64[D]')
        tindex = TimeIndex(src)
        assert len(tindex) == len(src)


class TestEquals:
    """Tests related to `TimeIndex.equals`.

    See Also
    --------
    TestTimeIndex

    """
    def test_compare_against_identical_object(self):
        """Should return ``True``."""
        src = np.arange(2).astype('datetime64[D]')
        tindex = TimeIndex(src)
        assert tindex.equals(tindex)

    def test_compare_against_object_with_equal_values(self):
        """Should return ``True``.

        When comparing an instance of `TimeSeries` against another with equal
        values, it always return ``True`` even if they are constructed with
        unequal but equivalent `data`.

        """
        src = np.arange(2).astype('datetime64[D]')
        tindex_1 = TimeIndex(src)
        tindex_2 = TimeIndex(src.tolist())
        assert tindex_1.equals(tindex_2)

    def test_compare_against_non_timeindex_object(self):
        """Should return ``False``."""
        src = np.arange(2).astype('datetime64[D]')
        tindex = TimeIndex(src)
        assert not tindex.equals(src)


class TestArgsort:
    """Tests related to `TimeIndex.argsort`.

    """
    def test_return_expected_integer_array(self):
        """Should return an expected integer array.

        When calling `argsort` with an instance of `TimeIndex`, it should
        return an integer array which can used to sort the time-index.

        See Also
        --------
        numpy.ndarray.argsort

        """
        src = np.array([1, 3, 5, 2, 4, 6]).astype('datetime64[D]')
        tindex = TimeIndex(src, sort=False)
        result = tindex.argsort()
        answer = src.argsort()
        assert array_equal(result, answer)


class TestSubscript:
    """Tests related to subscript operator of `TimeIndex`.

    """
    def test_subscript_by_boolean_array(self):
        """Return an instance of `TimeIndex` with expected contents.

        Return a new instance of `TimeSeries` which content is equal to the
        result of making same subscript operator on the content of the calling
        object.

        """
        values = np.arange(10)
        dates = values.astype('datetime64[D]')
        subscript = values % 3 == 2
        result = TimeIndex(dates)[subscript]
        answer = TimeIndex(dates[subscript])
        assert result.equals(answer)

    def test_subscript_by_integer_array(self):
        """Return an instance of `TimeIndex` with expected contents.

        Return a new instance of `TimeSeries` which content is equal to the
        result of making same subscript operator on the content of the calling
        object.

        """
        values = np.arange(10)
        dates = values.astype('datetime64[D]')
        subscript = np.arange(0, 10, 2)
        result = TimeIndex(dates)[subscript]
        answer = TimeIndex(dates[subscript])
        assert result.equals(answer)

    def test_subscript_by_slice(self):
        """Return an instance of `TimeIndex` with expected contents.

        Return a new instance of `TimeSeries` which content is equal to the
        result of making same subscript operator on the content of the calling
        object.

        """
        values = np.arange(10)
        dates = values.astype('datetime64[D]')
        result = TimeIndex(dates)[1: 9: 2]
        answer = TimeIndex(dates[1: 9: 2])
        assert result.equals(answer)

    def test_subscript_by_integer(self):
        """Return an expected `datetime.date` object.

        Return an instance of `datetime.date` equivalent to the element of the
        time-index corresponding to given integer.

        """
        values = np.array(['2021-11-02', '2021-11-03'], dtype=np.datetime64)
        result = TimeIndex(values)[0]
        answer = datetime.date(2021, 11, 2)
        assert result == answer
