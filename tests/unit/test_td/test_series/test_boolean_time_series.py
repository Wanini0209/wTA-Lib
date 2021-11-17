# -*- coding: utf-8 -*-
"""Unit-Tests related to `BooleanTimeSeries`."""

import numpy as np
import pytest

from .._context import (
    BooleanTimeSeries,
    LogicalBinaryOperator,
    LogicalUnaryOperator,
    MaskedArray,
    TimeSeries,
)

# pylint: disable=no-self-use, too-few-public-methods


class TestBooleanTimeSeries:
    """Tests related to builder of `BooleanTimeSeries`.

    """
    def test_with_boolean_data(self):
        """Should construct an instance of 'BooleanTimeSeries'."""
        data = [True, False, True, False]
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        bts = BooleanTimeSeries(data, index, 'bts')
        assert isinstance(bts, BooleanTimeSeries)

    def test_with_integer_data(self):
        """Should rasie `ValueError`."""
        data = [1, 2, 3, 4]
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        with pytest.raises(ValueError):
            BooleanTimeSeries(data, index, 'bts')

    def test_with_float_data(self):
        """Should rasie `ValueError`."""
        data = [1., 2., 3., 4.]
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        with pytest.raises(ValueError):
            BooleanTimeSeries(data, index, 'bts')

    def test_with_string_data(self):
        """Should rasie `ValueError`."""
        data = ['True', 'False', 'True', 'False']
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        with pytest.raises(ValueError):
            BooleanTimeSeries(data, index, 'bts')

    def test_with_non_boolean_data(self):
        """Should rasie `ValueError`."""
        data = np.array([True, False, True, False], object)
        index = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
        with pytest.raises(ValueError):
            BooleanTimeSeries(data, index, 'bts')


@pytest.mark.parametrize('operator', LogicalUnaryOperator)
class TestLogicalUnaryOperator:
    """Tests related to logical unary operators of `BooleanTimeSeries`.

    Logical unary operators only include NOT(~).

    See Also
    --------
    MaskedArray, LogicalUnaryOperator

    """
    _DATA = MaskedArray([True, False, True, False], [True, False, False, True])
    _INDEX = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']

    def test_result_with_expected_contents(self, operator):
        """Return an instance of `BooleanTimeSeries` with expected contents."""
        bts = BooleanTimeSeries(self._DATA, self._INDEX, 'bts')
        result = operator.func(bts)
        answer = BooleanTimeSeries(operator.func(self._DATA), self._INDEX, '_')
        assert result.equals(answer)

    def test_result_with_expected_name(self, operator):
        """Return an instance of `BooleanTimeSeries` with expected name."""
        bts = BooleanTimeSeries(self._DATA, self._INDEX, 'bts')
        result = operator.func(bts).name
        answer = f'{operator.symbol}bts'
        assert result == answer

    def test_result_with_shared_index(self, operator):
        """Return an instance of `BooleanTimeSeries` with shared index."""
        bts = BooleanTimeSeries(self._DATA, self._INDEX, 'bts')
        result = operator.func(bts)
        assert bts.index is result.index


@pytest.mark.parametrize('operator', LogicalBinaryOperator)
class TestLogicalBinaryOperator:
    """Tests related to logical binary operators of `BooleanTimeSeries`.

    Logical binary operators includes AND(&), OR(|) and XOR(^).

    See Also
    --------
    MaskedArray, LogicalBinaryOperator

    """
    _DATA_1 = MaskedArray([True, False, True, False],
                          [False, False, True, True])
    _DATA_2 = MaskedArray([True, True, False, False],
                          [True, False, False, True])
    _INDEX_1 = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04']
    _INDEX_2 = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-05']

    def test_with_non_boolean_time_series(self, operator):
        """Should rasie `TypeError`."""
        operand_1 = BooleanTimeSeries(self._DATA_1, self._INDEX_1, 'bts')
        operand_2 = TimeSeries(self._DATA_2, self._INDEX_2, 'ts')
        with pytest.raises(TypeError):
            _ = operator.func(operand_1, operand_2)

    def test_on_two_boolean_time_series_with_different_index(self, operator):
        """Should rasie `ValueError`."""
        operand_1 = BooleanTimeSeries(self._DATA_1, self._INDEX_1, 'bts1')
        operand_2 = BooleanTimeSeries(self._DATA_2, self._INDEX_2, 'bts2')
        with pytest.raises(ValueError):
            _ = operator.func(operand_1, operand_2)

    def test_result_with_expected_contents(self, operator):
        """Return an instance of `BooleanTimeSeries` with expected contents."""
        operand_1 = BooleanTimeSeries(self._DATA_1, self._INDEX_1, 'bts1')
        operand_2 = BooleanTimeSeries(self._DATA_2, self._INDEX_1, 'bts2')
        result = operator.func(operand_1, operand_2)
        answer = BooleanTimeSeries(operator.func(self._DATA_1, self._DATA_2),
                                   self._INDEX_1, '_')
        assert result.equals(answer)

    def test_result_with_expected_name(self, operator):
        """Return an instance of `BooleanTimeSeries` with expected name."""
        operand_1 = BooleanTimeSeries(self._DATA_1, self._INDEX_1, 'bts1')
        operand_2 = BooleanTimeSeries(self._DATA_2, self._INDEX_1, 'bts2')
        result = operator.func(operand_1, operand_2).name
        answer = f'bts1 {operator.symbol} bts2'
        assert result == answer

    def test_result_with_shared_index(self, operator):
        """Return an instance of `BooleanTimeSeries` with shared index."""
        operand_1 = BooleanTimeSeries(self._DATA_1, self._INDEX_1, 'bts1')
        operand_2 = BooleanTimeSeries(self._DATA_2, self._INDEX_1, 'bts2')
        result = operator.func(operand_1, operand_2)
        assert operand_1.index is result.index
