# -*- coding: utf-8 -*-
"""Unit-Tests related to `TimeSeries.sampling` and `TimeSeriesSampling`."""

import numpy as np
import pandas as pd
import pytest

from .._context import (
    NP_DATETIME_DTYPES,
    UNIT_VS_EQUIV_DTYPES,
    UNIT_VS_SUB_DTYPES,
    UNIT_VS_SUPER_DTYPES,
    MaskedArray,
    TimeIndex,
    TimeSeries,
    TimeSeriesSampling,
)
from ._context import dts_identical

# pylint: disable=no-self-use, too-few-public-methods


def _tss_identical(target: TimeSeriesSampling,
                   reference: TimeSeriesSampling) -> bool:
    """Determine whether two sampling result of time-series are identical.

    Two `TimeSeriesSampling` are identical if they have identical result of
    `to_dict` method.

    """
    return dts_identical(target.to_dict(), reference.to_dict())


class TestSampling:
    """Tests related to `sampling` of `TimeSeries`.

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
            tseries.sampling(2, 1, 'day')

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUPER_DTYPES)
    def test_with_subunit(self, unit, dtype):
        """Should raise ValueError."""
        values = np.arange(8)
        tseries = TimeSeries(values, values.astype(dtype), 'ts')
        with pytest.raises(ValueError):
            tseries.sampling(2, 1, unit)

    def test_with_non_integer_samples(self):
        """Should raise TypeError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(TypeError):
            tseries.sampling(2., 1)

    def test_with_samples_equal_to_one(self):
        """Should raise ValueError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(ValueError):
            tseries.sampling(1, 1)

    def test_with_zero_samples(self):
        """Should raise ValueError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(ValueError):
            tseries.sampling(0, 1)

    def test_with_negative_samples(self):
        """Should raise ValueError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(ValueError):
            tseries.sampling(-1, 1)

    def test_with_non_integer_step(self):
        """Should raise TypeError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(TypeError):
            tseries.sampling(2, 1.)

    def test_with_zero_step(self):
        """Should raise ValueError."""
        values = np.arange(8)
        index = values.astype('datetime64[D]')
        tseries = TimeSeries(values, index, 'ts')
        with pytest.raises(ValueError):
            tseries.sampling(2, 0)

    @pytest.mark.parametrize('dtype', NP_DATETIME_DTYPES)
    def test_sampling_by_positive_step_without_specified_unit(self, dtype):
        """Should return an expected instance of `TimeSeriesSampling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.sampling(2, 3)
        # generate answer
        values = np.array([[v, v + 3] for v in values])
        masks = (values >= 8) | (values % 2 == 0)
        answer = TimeSeriesSampling(MaskedArray(values, masks),
                                    MaskedArray(values.astype(dtype), values >= 8),
                                    index, 'ts.sampling(2, 3)')
        cond_1 = isinstance(result, TimeSeriesSampling)
        cond_2 = _tss_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('dtype', NP_DATETIME_DTYPES)
    def test_sampling_by_negative_step_without_specified_unit(self, dtype):
        """Should return an expected instance of `TimeSeriesSampling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.sampling(2, -3)
        # generate answer
        values = np.array([[v - 3, v] for v in values])
        masks = (values < 0) | (values % 2 == 0)
        answer = TimeSeriesSampling(MaskedArray(values, masks),
                                    MaskedArray(values.astype(dtype), values < 0),
                                    index, 'ts.sampling(2, -3)')
        cond_1 = isinstance(result, TimeSeriesSampling)
        cond_2 = _tss_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_sampling_by_positive_step_of_equivalent_unit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesSampling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.sampling(2, 3, unit)
        # generate answer
        values = np.array([[v, v + 3] for v in values])
        masks = (values >= 8) | (values % 2 == 0)
        answer = TimeSeriesSampling(MaskedArray(values, masks),
                                    MaskedArray(values.astype(dtype), values >= 8),
                                    index, f'ts.sampling(2, 3, {unit.name})')
        cond_1 = isinstance(result, TimeSeriesSampling)
        cond_2 = _tss_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_sampling_by_negative_step_of_equivalent_unit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesSampling`."""
        values = np.arange(8)
        index = TimeIndex(values.astype(dtype))
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.sampling(2, -3, unit)
        # generate answer
        values = np.array([[v - 3, v] for v in values])
        masks = (values < 0) | (values % 2 == 0)
        answer = TimeSeriesSampling(MaskedArray(values, masks),
                                    MaskedArray(values.astype(dtype), values < 0),
                                    index, f'ts.sampling(2, -3, {unit.name})')
        cond_1 = isinstance(result, TimeSeriesSampling)
        cond_2 = _tss_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_sampling_by_positive_step_of_superunit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesSampling`.

        In this test, we generate an array of datetimes with desired dtype,
        in which per three elements are in the same offset of desired time-unit.

        """
        values = np.arange(12)
        dates = self._generate_dates_for_test_on_superunit(12, dtype, unit)
        index = TimeIndex(dates)
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.sampling(2, 3, unit)
        # generate answer
        values = np.array([[v, (v // 3 + 3) * 3] for v in values])
        masks = (values >= 12) | (values % 2 == 0)
        answer = TimeSeriesSampling(MaskedArray(values, masks),
                                    MaskedArray(dates[values % 12], values >= 12),
                                    index, f'ts.sampling(2, 3, {unit.name})')
        cond_1 = isinstance(result, TimeSeriesSampling)
        cond_2 = _tss_identical(answer, result)
        assert cond_1 and cond_2

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_sampling_by_negative_step_of_superunit(self, unit, dtype):
        """Should return an expected instance of `TimeSeriesSampling`.

        In this test, we generate an array of datetimes with desired dtype,
        in which per three elements are in the same offset of desired time-unit.

        """
        values = np.arange(12)
        dates = self._generate_dates_for_test_on_superunit(12, dtype, unit)
        index = TimeIndex(dates)
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             index, 'ts')
        result = tseries.sampling(2, -3, unit)
        # generate answer
        values = np.array([[(v // 3 - 3) * 3 + 2, v] for v in values])
        masks = (values < 0) | (values % 2 == 0)
        answer = TimeSeriesSampling(MaskedArray(values, masks),
                                    MaskedArray(dates[values], values < 0),
                                    index, f'ts.sampling(2, -3, {unit.name})')
        cond_1 = isinstance(result, TimeSeriesSampling)
        cond_2 = _tss_identical(answer, result)
        assert cond_1 and cond_2


class TestToPandas:
    """Tests related to `to_pandas` of `TimeSeriesSampling`.

    """
    def test_sampling_by_positive_step(self):
        """Should return an expected `pandas.DataFrame`."""
        values = np.arange(8)
        dates = values.astype('datetime64[D]')
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             dates, 'ts')
        result = tseries.sampling(2, 3).to_pandas()
        # generate answer
        values = np.array([[v, v + 3] for v in values])
        masks = (values >= 8) | (values % 2 == 0)
        values = values.astype(float)
        values[masks] = np.nan
        columns = [f'ts.sampling(2, 3)[{i}]' for i in range(2)]
        answer = pd.DataFrame(values, dates, columns)
        assert result.equals(answer)

    def test_sampling_by_negative_step(self):
        """Should return an expected `pandas.DataFrame`."""
        values = np.arange(8)
        dates = values.astype('datetime64[D]')
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             dates, 'ts')
        result = tseries.sampling(2, -3).to_pandas()
        # generate answer
        values = np.array([[v - 3, v] for v in values])
        masks = (values < 0) | (values % 2 == 0)
        values = values.astype(float)
        values[masks] = np.nan
        columns = [f'ts.sampling(2, -3)[{i+1}]' for i in range(-2, 0)]
        answer = pd.DataFrame(values, dates, columns)
        assert result.equals(answer)


class TestToDict:
    """Tests related to `to_dict` of `TimeSeriesSampling`.

    """
    def test_sampling_by_positive_step(self):
        """Should return an expected dict of time-series."""
        values = np.arange(8)
        dates = values.astype('datetime64[D]')
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             dates, 'ts')
        result = tseries.sampling(2, 3).to_dict()
        # generate answer
        answer = {}
        values = np.array([[v, v + 3] for v in values])
        masks = values % 2 == 0
        for key, val, mask in zip(dates.tolist(), values, masks):
            answer[key] = TimeSeries(MaskedArray(val[val < 8], mask[val < 8]),
                                     dates[val % 8][val < 8],
                                     f'ts.sampling(2, 3)[{key}]')
        assert dts_identical(result, answer)

    def test_sampling_by_negative_step(self):
        """Should return an expected dict of time-series."""
        values = np.arange(8)
        dates = values.astype('datetime64[D]')
        tseries = TimeSeries(MaskedArray(values, values % 2 == 0),
                             dates, 'ts')
        result = tseries.sampling(2, -3).to_dict()
        # generate answer
        answer = {}
        values = np.array([[v - 3, v] for v in values])
        masks = values % 2 == 0
        for key, val, mask in zip(dates.tolist(), values, masks):
            answer[key] = TimeSeries(MaskedArray(val[val >= 0], mask[val >= 0]),
                                     dates[val][val >= 0],
                                     f'ts.sampling(2, -3)[{key}]')
        assert dts_identical(result, answer)
