# -*- coding: utf-8 -*-
"""Unit-Tests related to `TimeSeries.rolling` and `TimeSeriesRolling`."""

import numpy as np
import pytest

from .._context import (
    NP_DATETIME_DTYPES,
    UNIT_VS_EQUIV_DTYPES,
    UNIT_VS_SUB_DTYPES,
    UNIT_VS_SUPER_DTYPES,
    MaskedArray,
    TimeIndex,
    TimeSeries,
    TimeSeriesRolling,
)
from ._context import dts_identical

# pylint: disable=no-self-use, too-few-public-methods


def _tsr_identical(target: TimeSeriesRolling,
                   reference: TimeSeriesRolling) -> bool:
    """Determine whether two rolling result of time-series are identical.

    Two `TimeSeriesRolling` are identical if they have identical result of
    `to_dict` method.

    """
    return dts_identical(target.to_dict(), reference.to_dict())


class TestRolling:
    """Tests related to `rolling` of `TimeSeries`.

    """
    @classmethod
    def _generate_dates_for_test_on_superunit(cls, length, dtype, unit):
        codes = unit.decode(np.arange(length // 3 + 1)
                            ).astype(dtype).astype(int)
        codes = [codes[:-1], (codes[:-1] + codes[1:]) // 2, codes[1:] - 1]
        dates = np.array(codes).T.flatten().astype(dtype)
        return dates

    def test_with_invalid_unit(self):
        """Should raise TypeError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(TypeError):
            tseries.rolling(1, 'day')

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUPER_DTYPES)
    def test_with_subunit(self, unit, dtype):
        """Should raise ValueError."""
        values = np.arange(8)
        tseries = TimeSeries(values, values.astype(dtype), 'ts')
        with pytest.raises(ValueError):
            tseries.rolling(1, unit)

    def test_with_non_integer_period(self):
        """Should raise TypeError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(TypeError):
            tseries.rolling(1.)

    def test_with_zero_period(self):
        """Should raise ValueError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(ValueError):
            tseries.rolling(0)

    @pytest.mark.parametrize('dtype', NP_DATETIME_DTYPES)
    def test_rolling_by_positive_period_without_specified_unit(self, dtype):
        """Should return an expected instance of `TimeSeriesRolling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.rolling(3)
        # generate answer
        values = [np.arange(v, v + 3) for v in values]
        answer = TimeSeriesRolling([MaskedArray(each[each < 8],
                                                each[each < 8] % 2 == 0)
                                    for each in values],
                                   index, 'ts.rolling(3)', period=3)
        cond_1 = isinstance(result, TimeSeriesRolling)
        cond_2 = _tsr_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('dtype', NP_DATETIME_DTYPES)
    def test_rolling_by_negative_period_without_specified_unit(self, dtype):
        """Should return an expected instance of `TimeSeriesRolling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.rolling(-3)
        # generate answer
        values = [np.arange(v - 2, v + 1) for v in values]
        answer = TimeSeriesRolling([MaskedArray(each[each >= 0],
                                                each[each >= 0] % 2 == 0)
                                    for each in values],
                                   index, 'ts.rolling(-3)', period=-3)
        cond_1 = isinstance(result, TimeSeriesRolling)
        cond_2 = _tsr_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_rolling_by_positive_period_of_equivalent_unit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesRolling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.rolling(3, unit)
        # generate answer
        values = [np.arange(v, v + 3) for v in values]
        answer = TimeSeriesRolling([MaskedArray(each[each < 8],
                                                each[each < 8] % 2 == 0)
                                    for each in values],
                                   index, f'ts.rolling(3, {unit.name})',
                                   period=3)
        cond_1 = isinstance(result, TimeSeriesRolling)
        cond_2 = _tsr_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_rolling_by_negative_period_of_equivalent_unit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesRolling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.rolling(-3, unit)
        # generate answer
        values = [np.arange(v - 2, v + 1) for v in values]
        answer = TimeSeriesRolling([MaskedArray(each[each >= 0],
                                                each[each >= 0] % 2 == 0)
                                    for each in values],
                                   index, f'ts.rolling(-3, {unit.name})',
                                   period=-3)
        cond_1 = isinstance(result, TimeSeriesRolling)
        cond_2 = _tsr_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_rolling_by_positive_period_of_superunit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesRolling`.

        In this test, we generate an array of datetimes with desired dtype,
        in which per three elements are in the same offset of desired time-unit.

        """
        values = np.arange(12)
        dates = self._generate_dates_for_test_on_superunit(12, dtype, unit)
        index = TimeIndex(dates)
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.rolling(3, unit)
        # generate answer
        values = [np.arange(v, (v // 3 + 3) * 3) for v in values]
        answer = TimeSeriesRolling([MaskedArray(each[each < 12],
                                                each[each < 12] % 2 == 0)
                                    for each in values],
                                   index, f'ts.rolling(3, {unit.name})',
                                   period=3)
        cond_1 = isinstance(result, TimeSeriesRolling)
        cond_2 = _tsr_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_rolling_by_negative_period_of_superunit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesRolling`.

        In this test, we generate an array of datetimes with desired dtype,
        in which per three elements are in the same offset of desired time-unit.

        """
        values = np.arange(12)
        dates = self._generate_dates_for_test_on_superunit(12, dtype, unit)
        index = TimeIndex(dates)
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.rolling(-3, unit)
        # generate answer
        values = [np.arange((v // 3 - 2) * 3, v + 1) for v in values]
        answer = TimeSeriesRolling([MaskedArray(each[each >= 0],
                                                each[each >= 0] % 2 == 0)
                                    for each in values],
                                   index, f'ts.rolling(-3, {unit.name})',
                                   period=-3)
        cond_1 = isinstance(result, TimeSeriesRolling)
        cond_2 = _tsr_identical(answer, result)
        assert cond_1 and cond_2


class TestToDict:
    """Tests related to `to_dict` of `TimeSeriesRolling`.

    """
    def test_rolling_by_positive_period(self):
        """Should return an expected dict of time-series."""
        values = np.arange(8)
        dates = values.astype('datetime64[D]')
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             dates, 'ts')
        result = tseries.rolling(3).to_dict()
        # generate answer
        answer = {}
        values = [np.arange(v, v + 3) for v in values]
        for key, val in zip(dates.tolist(), values):
            answer[key] = TimeSeries(MaskedArray(val[val < 8],
                                                 val[val < 8] % 2 == 0),
                                     dates[val[val < 8]],
                                     f'ts.rolling(3)[{key}]')
        assert dts_identical(result, answer)

    def test_rolling_by_negative_period(self):
        """Should return an expected dict of time-series."""
        values = np.arange(8)
        dates = values.astype('datetime64[D]')
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             dates, 'ts')
        result = tseries.rolling(-3).to_dict()
        # generate answer
        answer = {}
        values = [np.arange(v - 2, v + 1) for v in values]
        for key, val in zip(dates.tolist(), values):
            answer[key] = TimeSeries(MaskedArray(val[val >= 0],
                                                 val[val >= 0] % 2 == 0),
                                     dates[val[val >= 0]],
                                     f'ts.rolling(-3)[{key}]')
        assert dts_identical(result, answer)
