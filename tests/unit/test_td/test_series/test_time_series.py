# -*- coding: utf-8 -*-
"""Unit-Tests related to `TimeSeries`."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from .._context import MaskedArray, TimeIndex, TimeSeries

# pylint: disable=no-self-use, too-few-public-methods


class TestTimeSeries:
    """Tests related to builder of `TimeSeries`.

    See Also
    --------
    TimeIndex, MaskedArray

    """
    def test_with_a_masked_array_as_data(self):
        """Should store `data` as a copy sorted by `index`.

        When constructing a time-series with `sort=True`(by default), if `data`
        is a masked-array object, it would be stored in the time-series as a
        copy which sorted by `index`.

        See Also
        --------
        MaskedArray.equals

        """
        data = MaskedArray([1, 2, 3, 4])
        index = TimeIndex(['2021-11-04', '2021-11-03',
                           '2021-11-02', '2021-11-01'], sort=False)
        tseries = TimeSeries(data, index, name='ts')
        answer = data[::-1]
        assert answer.equals(tseries.data)

    def test_with_a_masked_array_as_data_and_disabled_sort(self):
        """Should store `data` directly, not a copy.

        When constructing a time-series with `sort=False`, if `data` is a
        masked-array object, it would be stored as a reference in the
        time-series not a copy.

        """
        data = MaskedArray([1, 2, 3, 4])
        index = TimeIndex(['2021-11-04', '2021-11-03',
                           '2021-11-02', '2021-11-01'], sort=False)
        tseries = TimeSeries(data, index, name='ts', sort=False)
        assert data is tseries.data

    def test_with_a_array_like_as_data(self):
        """Should store `data` as a masked-arry sorted by `index`.

        When constructing a time-series with `sort=True`(by default), if `data`
        is an array-like object, it would be stored in the time-series as a
        masked-array which is equivalent to `data` sorted by `index`.

        See Also
        --------
        MaskedArray.equals

        """
        data = [1, 2, 3, 4]
        index = TimeIndex(['2021-11-04', '2021-11-03',
                           '2021-11-02', '2021-11-01'], sort=False)
        tseries = TimeSeries(data, index, name='ts')
        answer = MaskedArray(data[::-1])
        assert answer.equals(tseries.data)

    def test_with_a_array_like_as_data_and_disabled_sort(self):
        """Should store `data` as an equivalent masked-arry.

        When constructing a time-series with `sort=False`, if `data` is an
        array-like object, it would be stored in the time-series as a
        masked-array which is equivalent to `data`.

        See Also
        --------
        MaskedArray.equals

        """
        data = [1, 2, 3, 4]
        index = TimeIndex(['2021-11-04', '2021-11-03',
                           '2021-11-02', '2021-11-01'], sort=False)
        tseries = TimeSeries(data, index, name='ts', sort=False)
        answer = MaskedArray(data)
        assert answer.equals(tseries.data)

    def test_with_a_time_index_as_index(self):
        """Should store `index` as a sorted copy.

        When constructing a time-series with `sort=True`(by default), if
        `index` is a time-index object, it would be stored in the time-series
        as a sorted copy.

        See Also
        --------
        TimeIndex.equals

        """
        data = MaskedArray([1, 2, 3, 4])
        index = TimeIndex(['2021-11-04', '2021-11-03',
                           '2021-11-02', '2021-11-01'], sort=False)
        tseries = TimeSeries(data, index, name='ts')
        answer = index[::-1]
        assert answer.equals(tseries.index)

    def test_with_a_time_index_as_index_and_disabled_sort(self):
        """Should store `index` directly, not a copy.

        When constructing a time-series with `sort=False`, if `index` is a
        time-index object, it would be stored as a reference in the
        time-series not a copy.

        """
        data = MaskedArray([1, 2, 3, 4])
        index = TimeIndex(['2021-11-04', '2021-11-03',
                           '2021-11-02', '2021-11-01'], sort=False)
        tseries = TimeSeries(data, index, name='ts', sort=False)
        assert index is tseries.index

    def test_with_a_array_like_as_index(self):
        """Should store `index` as a sorted time-index.

        When constructing a time-series with `sort=True`(by default), if
        `index` is an array-like object, it would be stored in the time-series
        as a time-index which is equivalent to `index` but sorted.

        See Also
        --------
        TimeIndex.equals

        """
        data = MaskedArray([1, 2, 3, 4])
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        answer = TimeIndex(index)
        assert answer.equals(tseries.index)

    def test_with_a_array_like_as_index_and_disabled_sort(self):
        """Should store `index` as an equivalent time-index.

        When constructing a time-series with `sort=False`, if `index` is an
        array-like object, it would be stored in the time-series as a
        time-index which is equivalent to `index`.

        See Also
        --------
        TimeIndex.equals

        """
        data = MaskedArray([1, 2, 3, 4])
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts', sort=False)
        answer = TimeIndex(index, sort=False)
        assert answer.equals(tseries.index)

    def test_with_inconsistent_length_between_data_and_index(self):
        """Should rasie `ValueError`.

        When constructing a time-series with `data` and `index`, if the length
        of `data` and `index` are different, it would raise a `ValueError`
        exception.

        """
        data = MaskedArray([1, 2, 3])
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        with pytest.raises(ValueError):
            TimeSeries(data, index, name='ts', sort=False)


class TestData:
    """Tests related to `TimeSeries.data`.

    See Also
    --------
    TestTimeSeries

    """
    def test_always_return_a_reference(self):
        """Should always return a reference."""
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert tseries.data is tseries.data


class TestIndex:
    """Tests related to `TimeSeries.index`.

    See Also
    --------
    TestTimeSeries

    """
    def test_always_return_a_reference(self):
        """Should always return a reference."""
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert tseries.index is tseries.index


class TestName:
    """Tests related to `MaskedArray.name`.

    """
    def test_as_a_getter(self):
        """Should return an expected string."""
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert tseries.name == 'ts'

    def test_as_a_setter(self):
        """Should change the name of the time-series to an expected string."""
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        tseries.name = 'ts1'
        assert tseries.name == 'ts1'


class TestDtype:
    """Tests related to `MaskedArray.dtype`.

    See Also
    --------
    MaskedArray.dtype

    """
    def test_with_masked_array_as_data(self):
        """Should return an expected object of `numpy.dtype`.

        When a masked-array used to construct an instance of `TimeSeries`,
        the `dtype` of the instance must be the same as the `dtype` of the
        masked-array.

        """

        data = MaskedArray([1, 2, 3, 4])
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert tseries.dtype == data.dtype

    def test_with_array_as_data(self):
        """Should return an expected object of `numpy.dtype`.

        When an array data used to construct an instance of `TimeSeries`, the
        `dtype` of the instance must be the same as the `dtype` of the array.

        """
        data = np.array([1, 2, 3, 4])
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert tseries.dtype == data.dtype

    def test_with_list_as_data(self):
        """Should return an expected object of `numpy.dtype`.

        When a list data used to construct an instance of `TimeSeries`, the `dtype`
        of the instance must be the same as the `dtype` of the NumPy array
        generated by the same list.

        """
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert tseries.dtype == np.array(data).dtype


class TestLen:
    """Tests related to `len` on `TimeSeries`.

    """
    def test_return_expected_integer(self):
        """Should return an expected integer.

        When calling `len` with an instance of `TimeSeries`, it should return
        an integer both equals to the length of `data` and `index` used to
        construct the instance.

        """
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts')
        assert len(tseries) == 4


class TestRename:
    """Tests related to `TimeSeries.rename`.

    See Also
    --------
    TestName

    """
    def test_return_a_copy_with_expceted_name(self):
        """Should return a copy of the time-series with expected name."""
        data = [1, 2, 3, 4]
        index = ['2021-11-04', '2021-11-03', '2021-11-02', '2021-11-01']
        tseries = TimeSeries(data, index, name='ts').rename('ts1')
        assert tseries.name == 'ts1'


class TestEquals:
    """Tests related to `TimeSeries.equals`.

    See Also
    --------
    TimeIndex.equals, MaskedArray.equals

    """
    def test_against_another_with_equal_data_and_equal_index(self):
        """Should return ``True``.

        Compare against another time-series with equal `data` and
        equal `index`.

        Notes
        -----
        The method, 'TimeSeries.equals', must compare `data` and `index`
        between two instances. It delegate the comparsion of `data` to
        `MaskedArray.equals`, and the comparsion of `index` to
        `TimeIndex.equals`. In this test, the test situation is that
        two instances have equal `data` and equal `index`. We use two magic
        methods which always return ``True`` to simulate the test situation.

        """
        ts1 = TimeSeries([1], ['2021-11-01'], 'ts1')
        ts2 = TimeSeries([2], ['2021-11-02'], 'ts2')
        ts1.data.equals = Mock(return_value=True)
        ts1.index.equals = Mock(return_value=True)
        assert ts1.equals(ts2)

    def test_against_another_with_unequal_index(self):
        """Should return ``False``.

        Compare against another time-series with equal `data` but unequal
        `index`.

        Notes
        -----
        The method, 'TimeSeries.equals', must compare `data` and `index`
        between two instances. It delegate the comparsion of `data` to
        `MaskedArray.equals`, and the comparsion of `index` to
        `TimeIndex.equals`. In this test, the test situation is that
        two instances have equal `data` but unequal equal `index`. We use two
        magic methods, one always return ``True`` to mock `MaskedArray.equals`
        and another always return ``False`` to mock `TimeIndex.equals`, to
        simulate the test situation.

        """
        ts1 = TimeSeries([1], ['2021-11-01'], 'ts1')
        ts2 = TimeSeries([1], ['2021-11-01'], 'ts2')
        ts1.data.equals = Mock(return_value=True)
        ts1.index.equals = Mock(return_value=False)
        assert not ts1.equals(ts2)

    def test_against_another_with_unequal_data(self):
        """Should return ``False``.

        Compare against another time-series with equal `index` but unequal
        `data`.

        Notes
        -----
        The method, 'TimeSeries.equals', must compare `data` and `index`
        between two instances. It delegate the comparsion of `data` to
        `MaskedArray.equals`, and the comparsion of `index` to
        `TimeIndex.equals`. In this test, the test situation is that
        two instances have equal `index` but unequal equal `data`. We use two
        magic methods, one always return ``False`` to mock `MaskedArray.equals`
        and another always return ``True`` to mock `TimeIndex.equals`, to
        simulate the test situation.

        """
        ts1 = TimeSeries([1], ['2021-11-01'], 'ts1')
        ts2 = TimeSeries([1], ['2021-11-01'], 'ts2')
        ts1.data.equals = Mock(return_value=False)
        ts1.index.equals = Mock(return_value=True)
        assert not ts1.equals(ts2)

    def test_compare_against_non_timeseries_object(self):
        """Should rasie `TypeError`."""
        ts1 = TimeSeries([1], ['2021-11-01'], 'ts1')
        ts2 = ts1.to_pandas()
        with pytest.raises(TypeError):
            ts1.equals(ts2)


class TestFillNa:
    """Tests related to `TimeSeries.fillna`.

    See Also
    --------
    MaskedArray.fillna

    """
    def test_keep_the_calling_object_unchanged(self):
        """The contents of the calling object could not be changed.

        When call `fillna` method on an instance of `TimeSeries`, the
        contents of the calling object must remain unchanged.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        src = TimeSeries(data, index, 'ts')
        tseries = TimeSeries(data, index, 'ts')
        recv = tseries.fillna(0)
        cond_1 = not tseries.equals(recv)
        cond_2 = tseries.equals(src)
        assert cond_1 and cond_2

    def test_on_timeseries_with_na_elements(self):
        """Return an instance of `TimeSeries` with expected contents.

        When calling `fillna` method with specified `value` on an instance of
        `TimeSeries` which contains some N/A elements, it return a copy of the
        calling object in which all N/A elements are filled using `value`.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(data, index, 'ts').fillna(0)
        answer = TimeSeries([0, 2, 0, 4], index, 'ts')
        assert result.equals(answer)


class TestFfill:
    """Tests related to `TimeSeries.ffill`.

    """
    def test_keep_the_calling_object_unchanged(self):
        """The contents of the calling object could not be changed.

        When calling `ffill` method of a `TimeSeries` instance, the
        contents of the calling object must remain unchanged.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        src = TimeSeries(data, index, 'ts')
        tseries = TimeSeries(data, index, 'ts')
        recv = tseries.ffill()
        cond_1 = not tseries.equals(recv)
        cond_2 = tseries.equals(src)
        assert cond_1 and cond_2

    def test_on_timeseries_without_na_elements(self):
        """Return a copy of the time-series.

        When calling `ffill` method of a `TimeSeries` instance which contains
        no N/A element, it return a copy of the calling object.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries([1, 2, 3, 4], index, 'ts').ffill()
        answer = TimeSeries([1, 2, 3, 4], index, 'ts')
        assert result.equals(answer)

    def test_on_timeseries_with_leading_na_elements(self):
        """Return an instance of `TimeSeries` with expected contents.

        When calling `ffill` method of a `TimeSeries` instance which contains
        some leading N/A elements, it return a copy of the calling object in
        which the non-leading N/A elements are filled using forward-fill method
        and the leading N/A elements keep unavailable.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(MaskedArray([1, 2, 3, 4],
                                        [True, False, True, False]),
                            index, 'ts').ffill()
        answer = TimeSeries(MaskedArray([1, 2, 2, 4],
                                        [True, False, False, False]),
                            index, 'ts')
        assert result.equals(answer)

    def test_on_timeseries_without_leading_na_elements(self):
        """Return an instance of `TimeSeries` with expected contents.

        When calling `ffill` method of a `TimeSeries` instance which contains
        no leading N/A element, it return a copy of the calling object in which
        all N/A elements are filled using forward-fill method.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(MaskedArray([1, 2, 3, 4],
                                        [False, True, False, True]),
                            index, 'ts').ffill()
        answer = TimeSeries([1, 1, 3, 3], index, 'ts')
        assert result.equals(answer)


class TestBfill:
    """Tests related to `TimeSeries.bfill`.

    """
    def test_keep_the_calling_object_unchanged(self):
        """The contents of the calling object could not be changed.

        When calling `bfill` method of a `TimeSeries` instance, the
        contents of the calling object must remain unchanged.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        src = TimeSeries(data, index, 'ts')
        tseries = TimeSeries(data, index, 'ts')
        recv = tseries.bfill()
        cond_1 = not tseries.equals(recv)
        cond_2 = tseries.equals(src)
        assert cond_1 and cond_2

    def test_on_timeseries_without_na_elements(self):
        """Return a copy of the time-series.

        When calling `bfill` method of a `TimeSeries` instance which conatins
        no N/A element, it return a copy of the calling object.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries([1, 2, 3, 4], index, 'ts').bfill()
        answer = TimeSeries([1, 2, 3, 4], index, 'ts')
        assert result.equals(answer)

    def test_on_timeseries_with_tailing_na_elements(self):
        """Return an instance of `TimeSeries` with expected contents.

        When calling `bfill` method of a `TimeSeries` instance which conatins
        some tailing N/A elements, it return a copy of the calling object in
        which the non-tailing N/A elements are filled using backward-fill
        method and the tailing N/A elements keep unavailable.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(MaskedArray([1, 2, 3, 4],
                                        [False, True, False, True]),
                            index, 'ts').bfill()
        answer = TimeSeries(MaskedArray([1, 3, 3, 4],
                                        [False, False, False, True]),
                            index, 'ts')
        assert result.equals(answer)

    def test_on_timeseries_without_tailing_na_elements(self):
        """Return an instance of `TimeSeries` with expected contents.

        When calling `bfill` method of a `TimeSeries` instance which conatins
        no tailing N/A element, it should return a copy of the calling object
        in which all N/A elements are filled using backward-fill method.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(MaskedArray([1, 2, 3, 4],
                                        [True, False, True, False]),
                            index, 'ts').bfill()
        answer = TimeSeries([2, 2, 4, 4], index, 'ts')
        assert result.equals(answer)


class TestDropNa:
    """Tests related to `TimeSeries.dropna`.

    """
    def test_keep_the_calling_object_unchanged(self):
        """The contents of the calling object could not be changed.

        When call `dropna` method of a `TimeSeries` instance, the contents of
        the calling object must remain unchanged.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        src = TimeSeries(data, index, 'ts')
        tseries = TimeSeries(data, index, 'ts')
        recv = tseries.dropna()
        cond_1 = not tseries.equals(recv)
        cond_2 = tseries.equals(src)
        assert cond_1 and cond_2

    def test_on_timeseries_with_na_elements(self):
        """Return an instance of `TimeSeries` with expected contents.

        When call `dropna` method of a `TimeSeries` instance which contains
        some N/A elements, it should return a copy of the calling object
        in which N/A elements are removed.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(data, index, 'ts').dropna()
        answer = TimeSeries([2, 4], ['2021-11-02', '2021-11-04'], 'ts')
        assert result.equals(answer)

    def test_on_timeseries_without_na_elements(self):
        """Return a copy of the time-series.

        When calling `dropna` method of a `TimeSeries` instance which contains
        no N/A element, it should return a copy of the calling object with which
        shared `data` and `index`.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        tseries = TimeSeries([1, 2, 3, 4], index, 'ts')
        result = tseries.dropna()
        cond_1 = tseries.index is result.index
        cond_2 = tseries.data is result.data
        assert cond_1 and cond_2


class TestToPandas:
    """Tests related to `MaskedArray.to_pandas`.

    """
    def test_on_timeseries_without_na_elements(self):
        """Return an expected pandas series.

        When calling `to_pandas` method of a `TimeSeries` instance which
        contains no N/A element, it should return an equivalent
        'pandas.Series`.

        """
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries([1, 2, 3, 4], index, 'ts').to_pandas()
        answer = pd.Series(np.array([1, 2, 3, 4], int),
                           index=np.array(index, 'datetime64[D]'), name='ts')
        cond_1 = result.name == answer.name
        cond_2 = result.equals(answer)
        assert cond_1 and cond_2

    def test_on_timeseries_with_integer_data_and_na_elements(self):
        """Return an expected pandas series.

        When calling `to_pandas` method of a `TimeSeries` instance which `data`
        is consisted of integer values and some N/A elements, it should return
        an equivalent 'pandas.Series` but `dtype` is changed to `float` and
        the N/A elements are replaced by ``numpy.nan``.

        """
        data = MaskedArray([1, 2, 3, 4], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(data, index, 'ts').to_pandas()
        answer = pd.Series(np.array([np.nan, 2, np.nan, 4], float),
                           index=np.array(index, 'datetime64[D]'), name='ts')
        cond_1 = result.name == answer.name
        cond_2 = result.equals(answer)
        assert cond_1 and cond_2

    def test_on_timeseries_with_float_data_and_na_elements(self):
        """Return an expected pandas series.

        When calling `to_pandas` method of a `TimeSeries` instance which `data`
        is consisted of float values and some N/A elements, it should return
        an equivalent 'pandas.Series` with N/A elements replaced by
        ``numpy.nan``.

        """
        data = MaskedArray([1., 2., 3., 4.], [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(data, index, 'ts').to_pandas()
        answer = pd.Series(np.array([np.nan, 2, np.nan, 4], float),
                           index=np.array(index, 'datetime64[D]'), name='ts')
        cond_1 = result.name == answer.name
        cond_2 = result.equals(answer)
        assert cond_1 and cond_2

    def test_on_timeseries_with_boolean_data_and_na_elements(self):
        """Return an expected pandas series.

        When calling `to_pandas` method of a `TimeSeries` instance which `data`
        is consisted of boolean values and some N/A elements, it should return
        an equivalent 'pandas.Series` but `dtype` is changed to `object` and
        the N/A elements are replaced by ``numpy.nan``.

        """
        data = MaskedArray([True, True, False, False],
                           [True, False, True, False])
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        result = TimeSeries(data, index, 'ts').to_pandas()
        answer = pd.Series(np.array([np.nan, True, np.nan, False], object),
                           index=np.array(index, 'datetime64[D]'), name='ts')
        cond_1 = result.name == answer.name
        cond_2 = result.equals(answer)
        assert cond_1 and cond_2
