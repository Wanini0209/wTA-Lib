# -*- coding: utf-8 -*-
"""Unit-Tests related to `TimeUnit`."""

import datetime

import numpy as np
import pytest

from .._context import (
    UNIT_VS_EQUIV_DTYPES,
    UNIT_VS_SUB_DTYPES,
    UNIT_VS_SUPER_DTYPES,
    TimeUnit,
    array_equal,
)

# pylint: disable=no-self-use, too-few-public-methods, too-many-public-methods


class TestTimeUnit:
    """Tests related to members of `TimeUnit`.

    1. The `name` of each member is unique.
    2. The `symbol` of each member is unique.
    3. The `factor` of each member is a positive integer.
    4. The `offset` of eahc member is a integer.

    """
    def test_names_unique(self):
        """Name of each member in `TimeUnit` must be unique."""
        names = [each.name for each in TimeUnit]
        assert len(names) == len(set(names))  # check uniqueness

    def test_symbols_unique(self):
        """Symbol of each member in `TimeUnit` must be unique."""
        symbols = [each.symbol for each in TimeUnit]
        assert len(symbols) == len(set(symbols))  # check uniqueness

    def test_factor_is_positive_integer(self):
        """Factor of each member in `TimeUnit` must be a positive integer."""
        factors = np.array([each.factor for each in TimeUnit])
        cond_1 = np.issubdtype(factors.dtype, int)
        cond_2 = (factors > 0).all()
        assert cond_1 and cond_2

    def test_offset_is_integer(self):
        """Offset of each member in `TimeUnit` must be an integer."""
        offsets = np.array([each.offset for each in TimeUnit])
        assert np.issubdtype(offsets.dtype, int)


class TestIssuper:
    """Tests related to `issuper` of `TimeUnit`.

    """
    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_issuper_with_equivalent_dtype(self, unit, dtype):
        """Should return ``False``."""
        assert not unit.issuper(dtype)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_issuper_with_dtype_of_subunit(self, unit, dtype):
        """Should return ``True``."""
        assert unit.issuper(dtype)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUPER_DTYPES)
    def test_issuper_with_dtype_of_superunit(self, unit, dtype):
        """Should return ``False``."""
        assert not unit.issuper(dtype)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_with_not_numpy_dtype(self, unit):
        """Should raise `TypeError`"""
        with pytest.raises(TypeError):
            _ = unit.issuper(datetime.date)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_with_not_numpy_datetime64(self, unit):
        """Should raise `ValueError`"""
        with pytest.raises(ValueError):
            _ = unit.issuper(np.dtype(int))


class TestIssub:
    """Tests related to `issub` of `TimeUnit`."""
    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_issub_with_equivalent_dtype(self, unit, dtype):
        """Should return ``False``."""
        assert not unit.issub(dtype)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_issub_with_dtype_of_subunit(self, unit, dtype):
        """Should return ``False``."""
        assert not unit.issub(dtype)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUPER_DTYPES)
    def test_issub_with_dtype_of_superunit(self, unit, dtype):
        """Should return ``True``."""
        assert unit.issub(dtype)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_with_not_numpy_dtype(self, unit):
        """Should raise `TypeError`"""
        with pytest.raises(TypeError):
            _ = unit.issub(datetime.date)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_with_not_numpy_datetime64(self, unit):
        """Should raise `ValueError`"""
        with pytest.raises(ValueError):
            _ = unit.issub(np.dtype(int))


class TestIsEquivalent:
    """Tests related to `isequiv` to `TimeUnit`."""
    # tests for SECOND
    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_isequiv_with_equivalent_dtype(self, unit, dtype):
        """Should return ``True``."""
        assert unit.isequiv(dtype)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUB_DTYPES)
    def test_isequiv_with_dtype_of_subunit(self, unit, dtype):
        """Should return ``False``."""
        assert not unit.isequiv(dtype)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUPER_DTYPES)
    def test_isequiv_with_dtype_of_superunit(self, unit, dtype):
        """Should return ``False``."""
        assert not unit.isequiv(dtype)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_with_not_numpy_dtype(self, unit):
        """Should raise `TypeError`"""
        with pytest.raises(TypeError):
            _ = unit.isequiv(datetime.date)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_with_not_numpy_datetime64(self, unit):
        """Should raise `ValueError`"""
        with pytest.raises(ValueError):
            _ = unit.isequiv(np.dtype(int))


class TestEncode:
    """Tests related to `encode` of `TimeUnit`.

    This class includes common tests related to `encode` of `TimeUnit`, but
    ignores the tests specified on individual time-unit.

    """
    @pytest.mark.parametrize('unit', TimeUnit)
    def test_encode_on_not_array_data(self, unit):
        """Should raise `TypeError`"""
        dates = np.arange(10).astype('datetime64[D]').tolist()
        with pytest.raises(TypeError):
            _ = unit.encode(dates)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_encode_on_not_numpy_datetime_array(self, unit):
        """Should raise `ValueError`"""
        dates = np.arange(10)
        with pytest.raises(ValueError):
            _ = unit.encode(dates)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_SUPER_DTYPES)
    def test_encode_on_datetimes_with_dtype_of_superunit(self, unit, dtype):
        """Should raise `ValueError`"""
        dates = np.arange(10).astype(dtype)
        with pytest.raises(ValueError):
            _ = unit.encode(dates)

    @pytest.mark.parametrize('unit, dtype', UNIT_VS_EQUIV_DTYPES)
    def test_encode_on_datetimes_with_dtype_of_equivalent_unit(self, unit, dtype):
        """Should return expected integers.

        The result must be equal as the result of NumPy's casting.

        """
        answer = np.arange(-100, 100)
        result = unit.encode(answer.astype(dtype))
        assert array_equal(answer, result)

    @pytest.mark.parametrize('unit, dtype',
                             [each for each in UNIT_VS_SUB_DTYPES
                              if each[0] not in [TimeUnit.WEEK, TimeUnit.QUARTER]])
    def test_encode_on_datetimes_with_dtype_of_subunit(self, unit, dtype):
        """Should return expected integers.

        The result must be equal as the result of NumPy's casting.

        """
        dates = np.arange(-100, 100).astype(dtype)
        result = unit.encode(dates)
        answer = dates.astype(unit.dtype).astype(int)
        assert array_equal(answer, result)


class TestDecode:
    """Tests related to `decode` of `TimeUnit`.

    This class includes common tests related to `decode` of `TimeUnit`, but
    ignores the tests specified on individual time-unit.

    """
    @pytest.mark.parametrize('unit', TimeUnit)
    def test_decode_on_not_array_data(self, unit):
        """Should raise `TypeError`"""
        dates = np.arange(10).tolist()
        with pytest.raises(TypeError):
            _ = unit.decode(dates)

    @pytest.mark.parametrize('unit', TimeUnit)
    def test_decode_on_not_integer_array(self, unit):
        """Should raise `ValueError`"""
        dates = np.arange(10).astype(float)
        with pytest.raises(ValueError):
            _ = unit.decode(dates)

    @pytest.mark.parametrize('unit',
                             [each for each in TimeUnit
                              if each not in [TimeUnit.WEEK, TimeUnit.QUARTER]])
    def test_decode_on_integers(self, unit):
        """Should return expected datetimes.

        The result must be equal as the result of NumPy's casting.

        """
        src = np.arange(-100, 100)
        result = unit.decode(src)
        answer = src.astype(unit.dtype)
        assert array_equal(answer, result)


class TestWeek:
    """Tests specified on `TimeUnit.WEEK`.

    """
    def test_encode_on_datetimes(self):
        """Should return expected integers.

        The result must equal to the difference of the corresponding week of
        datetimes and the week of '1969-12-29~1970-01-04'.

        """
        src = np.arange(-100, 100).astype('datetime64[D]')
        result = TimeUnit.WEEK.encode(src)
        dates = src.astype(datetime.date)
        bdate = datetime.date(1969, 12, 29)
        answer = np.floor([(each - bdate).days / 7 for each in dates]).astype(int)
        assert array_equal(answer, result)

    def test_decode_on_integers_must_always_return_monday_of_week(self):
        """Should return expected datetimes.

        The result of decode must be Monday(The first day of week).

        """
        src = np.arange(-100, 100)
        result = TimeUnit.WEEK.decode(src).tolist()
        weekdays = np.array([each.isocalendar().weekday for each in result])
        assert (weekdays == 1).any()

    def test_decode_and_encode_are_invertable(self):
        """Encode follows decode must return the integers before decoding."""
        src = np.arange(-100, 100)
        result = TimeUnit.WEEK.encode(TimeUnit.WEEK.decode(src))
        assert array_equal(src, result)


class TestQuarter:
    """Tests specified on `TimeUnit.Quarter`.

    """
    def test_encode_on_datetimes(self):
        """Should return expected integers.

        The result must equal to the difference of the corresponding quarter of
        datetimes and the quarter of '1970-01-01~1970-03-31'.

        """
        src = np.arange(-365, 365).astype('datetime64[D]')
        result = TimeUnit.QUARTER.encode(src)
        dates = src.astype(datetime.date)
        bdate = datetime.date(1970, 1, 1)
        answer = np.floor([((each.year - bdate.year
                             ) * 12 + (each.month - bdate.month)) / 3
                           for each in dates]).astype(int)
        assert array_equal(answer, result)

    def test_decode_on_integers_must_always_return_firstday_of_quarter(self):
        """Should return expected datetimes.

        The result of decode must be the first day of quarter(01-01, 04-01,
        07-01 or 10-01).

        """
        src = np.arange(-10, 10)
        result = TimeUnit.QUARTER.decode(src).tolist()
        months = np.array([each.month for each in result])
        days = np.array([each.day for each in result])
        assert (days == 1).any() and (months % 3 == 1).any()

    def test_decode_and_encode_are_invertable(self):
        """Encode follows decode must return the integers before decoding."""
        src = np.arange(-10, 10)
        result = TimeUnit.QUARTER.encode(TimeUnit.QUARTER.decode(src))
        assert array_equal(src, result)
