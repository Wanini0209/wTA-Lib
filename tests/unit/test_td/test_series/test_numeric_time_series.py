# -*- coding: utf-8 -*-
"""Unit-Tests related to `NumericTimeSeries`."""

import itertools

import numpy as np
import pytest

from .._context import (
    ArithmeticBinaryOperator,
    ArithmeticUnaryFunction,
    ArithmeticUnaryOperator,
    BooleanTimeSeries,
    ComparisonOperator,
    MaskedArray,
    NumericTimeSeries,
    TimeSeries,
)
from ._context import ts_identical

# pylint: disable=no-self-use, too-few-public-methods

INDEX = ['2021-11-01', '2021-11-02', '2021-11-03', '2021-11-04', '2021-11-05']
MASK_1 = [True, False, True, False, False]
MASK_2 = [True, True, False, False, False]
MASK_R = [True, True, True, False, False]
INT_SEQS = [[-2, -1, 0, 1, 2], [2, -2, 1, -1, 0]]
FLT_SEQS = [[-1., -.5, 0., .5, 1.], [1., -1., .5, -.5, 0.]]
INTS = [-2, -1, 0, 1, 2]
FLTS = [-1., -.5, 0., .5, 1.]


class TestNumericTimeSeries:
    """Tests related to builder of `NumericTimeSeries`.

    """
    def test_with_integer_data(self):
        """Should construct an instance of 'NumericTimeSeries'."""
        data = [-1, 0, 1]
        index = ['2021-11-01', '2021-11-02', '2021-11-03']
        nts = NumericTimeSeries(data, index, 'nts')
        assert isinstance(nts, NumericTimeSeries)

    def test_with_float_data(self):
        """Should construct an instance of 'NumericTimeSeries'."""
        data = [-1., 0., 1.]
        index = ['2021-11-01', '2021-11-02', '2021-11-03']
        nts = NumericTimeSeries(data, index, 'nts')
        assert isinstance(nts, NumericTimeSeries)

    def test_with_boolean_data(self):
        """Should rasie `ValueError`."""
        data = [True, False]
        index = ['2021-11-01', '2021-11-02']
        with pytest.raises(ValueError):
            _ = NumericTimeSeries(data, index, 'nts')

    def test_with_string_data(self):
        """Should rasie `ValueError`."""
        data = ['-1', '0', '1']
        index = ['2021-11-01', '2021-11-02', '2021-11-03']
        with pytest.raises(ValueError):
            _ = NumericTimeSeries(data, index, 'nts')

    def test_with_non_boolean_data(self):
        """Should rasie `ValueError`."""
        data = np.array([-1., 0, 1.], object)
        index = ['2021-11-01', '2021-11-02', '2021-11-03']
        with pytest.raises(ValueError):
            _ = NumericTimeSeries(data, index, 'nts')


@pytest.mark.parametrize('operator', ArithmeticUnaryOperator)
class TestArithmeticUnaryOperator:
    """Tests related to arithmetic unary operators of `NumericTimeSeries`.

    Arithmetic unary operators only include Negative(-).

    See Also
    --------
    MaskedArray, ArithmeticUnaryOperator

    """
    def test_with_integer_data(self, operator):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        data = [-1, 0, 1]
        masks = [True, False, True]
        index = ['2021-11-01', '2021-11-02', '2021-11-03']
        nts = NumericTimeSeries(MaskedArray(data, masks), index, 'nts')
        result = operator.func(nts)
        answer = operator.func(np.array(data))
        answer = NumericTimeSeries(MaskedArray(answer, masks),
                                   index, f'{operator.symbol}nts')
        assert ts_identical(result, answer)

    def test_with_float_data(self, operator):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        data = [-1., -0.5, 0., 0.5, 1.]
        masks = [True, False, True, False, False]
        index = ['2021-11-01', '2021-11-02', '2021-11-03',
                 '2021-11-04', '2021-11-05']
        nts = NumericTimeSeries(MaskedArray(data, masks), index, 'nts')
        result = operator.func(nts)
        answer = operator.func(np.array(data))
        answer = NumericTimeSeries(MaskedArray(answer, masks),
                                   index, f'{operator.symbol}nts')
        assert ts_identical(result, answer)


@pytest.mark.parametrize('operator', ArithmeticUnaryFunction)
class TestArithmeticUnaryFunction:
    """Tests related to arithmetic unary functions of `NumericTimeSeries`.

    Arithmetic unary functions only include Absolute value(abs).

    See Also
    --------
    MaskedArray, ArithmeticUnaryFunction

    """
    def test_with_integer_data(self, operator):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        data = [-1, 0, 1]
        masks = [True, False, True]
        index = ['2021-11-01', '2021-11-02', '2021-11-03']
        nts = NumericTimeSeries(MaskedArray(data, masks), index, 'nts')
        result = operator.func(nts)
        answer = operator.func(np.array(data))
        answer = NumericTimeSeries(MaskedArray(answer, masks),
                                   index, f'{operator.symbol}(nts)')
        assert ts_identical(result, answer)

    def test_with_float_data(self, operator):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        data = [-1., -0.5, 0., 0.5, 1.]
        masks = [True, False, True, False, False]
        index = ['2021-11-01', '2021-11-02', '2021-11-03',
                 '2021-11-04', '2021-11-05']
        nts = NumericTimeSeries(MaskedArray(data, masks), index, 'nts')
        result = operator.func(nts)
        answer = operator.func(np.array(data))
        answer = NumericTimeSeries(MaskedArray(answer, masks),
                                   index, f'{operator.symbol}(nts)')
        assert ts_identical(result, answer)


@pytest.mark.parametrize('operator',
                         list(ArithmeticBinaryOperator) + list(ComparisonOperator))
class TestNumericBinaryOperator:
    """Tests related to all numeric binary operator of `NumericTimeSeries`.

    Numeric binary operators include:
    - Arithmetic Operators:
        Addition(+), Subtraction(-), Multiplication(*), Division(/),
        Modulus(%), Power(**), Floor division(//).
    - Comparison operators:
        Equal(==), Not equal(!=), Greater than(>), Less than(<),
        Greater than or equal to(>=), Less than or equal to(<=).

    See Also
    --------
    MaskedArray, ArithmeticBinaryOperator, ComparisonOperator

    """
    def test_with_non_numeric_timeseries(self, operator):
        """Should raise `TypeError`."""
        op1 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-03'],
                                'nts1')
        op2 = TimeSeries([1, 2, 3],
                         ['2021-11-01', '2021-11-02', '2021-11-03'],
                         'ts2')
        with pytest.raises(TypeError):
            _ = operator.func(op1, op2)

    @pytest.mark.parametrize('op2', [[1, 2, 3],
                                     np.array([1, 2, 3])])
    def test_with_same_size_numeric_array_like(self, operator, op2):
        """Should raise `TypeError`."""
        op1 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-03'],
                                'nts1')
        with pytest.raises(TypeError):
            _ = operator.func(op1, op2)

    @pytest.mark.parametrize('op2', ['1', True])
    def test_with_non_numeric_scalar(self, operator, op2):
        """Should raise `TypeError`."""
        op1 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-03'],
                                'nts1')
        with pytest.raises(TypeError):
            _ = operator.func(op1, op2)


@pytest.mark.parametrize('operator',
                         [each for each in ArithmeticBinaryOperator
                          if each is not ArithmeticBinaryOperator.POW
                          ] + list(ComparisonOperator))
class TestNumericBinaryOperatorExceptPower:
    """Tests related to all numeric binary operator of `NumericTimeSeries`.

    Numeric binary operators include:
    - Arithmetic Operators:
        Addition(+), Subtraction(-), Multiplication(*), Division(/),
        Modulus(%), Power(**), Floor division(//).
    - Comparison operators:
        Equal(==), Not equal(!=), Greater than(>), Less than(<),
        Greater than or equal to(>=), Less than or equal to(<=).
    In this test, Power(**) is not inclued.

    See Also
    --------
    MaskedArray, ArithmeticBinaryOperator, ComparisonOperator

    """
    def test_on_two_numeric_timeseries_with_inconsistent_index(self, operator):
        """Should raise `ValueError`."""
        op1 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-03'],
                                'nts1')
        op2 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-04'],
                                'nts2')
        with pytest.raises(ValueError):
            _ = operator.func(op1, op2)


@pytest.mark.parametrize('operator', [ArithmeticBinaryOperator.ADD,
                                      ArithmeticBinaryOperator.SUB,
                                      ArithmeticBinaryOperator.MUL])
class TestWellDefinedArithmeticBinaryOperator:
    """Tests related to well-defined arithmetic binary operator of `NumericTimeSeries`.

    Well-defined arithmetic binary operators include: Addition(+),
    Subtraction(-), and Multiplication(*).

    See Also
    --------
    MaskedArray, ArithmeticBinaryOperator

    """
    @pytest.mark.parametrize(
        'op1, op2',
        itertools.product(
            [NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1),
                               INDEX, 'nts1'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[0], MASK_1),
                               INDEX, 'nts1')],
            [NumericTimeSeries(MaskedArray(INT_SEQS[1], MASK_2),
                               INDEX, 'nts2'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[1], MASK_2),
                               INDEX, 'nts2')] + INTS + FLTS))
    def test_result_with_shared_index(self, operator, op1, op2):
        """Return an instance of `NumericTimeSeries` with shared index."""
        result = operator.func(op1, op2)
        assert op1.index is result.index

    @pytest.mark.parametrize('data_1, data_2',
                             itertools.product([INT_SEQS[0], FLT_SEQS[0]],
                                               [INT_SEQS[1], FLT_SEQS[1]]))
    def test_on_two_numeric_timeseries(self, operator, data_1, data_2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(data_1, MASK_1), INDEX, 'nts1')
        op2 = NumericTimeSeries(MaskedArray(data_2, MASK_2), INDEX, 'nts2')
        result = operator.func(op1, op2)
        answer = operator.func(np.array(data_1), np.array(data_2))
        answer = NumericTimeSeries(MaskedArray(answer, MASK_R), INDEX,
                                   f'nts1 {operator.symbol} nts2')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize('data_1, op2',
                             itertools.product([INT_SEQS[0], FLT_SEQS[0]],
                                               INTS + FLTS))
    def test_with_numeric_scalar(self, operator, data_1, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(data_1, MASK_1), INDEX, 'nts1')
        result = operator.func(op1, op2)
        answer = operator.func(np.array(data_1), op2)
        answer = NumericTimeSeries(MaskedArray(answer, MASK_1), INDEX,
                                   f'nts1 {operator.symbol} {op2}')
        assert ts_identical(result, answer)


@pytest.mark.parametrize('operator', [ArithmeticBinaryOperator.DIV,
                                      ArithmeticBinaryOperator.MOD,
                                      ArithmeticBinaryOperator.FDIV])
class TestArithmeticDivisionRelatedOperator:
    """Tests related to arithmetic division-related operator of `NumericTimeSeries`.

    Division-related arithmetic binary operators include: Division(/),
    Modulus(%), Power(**), Floor division(//).

    See Also
    --------
    MaskedArray, ArithmeticBinaryOperator

    """
    @pytest.mark.parametrize(
        'op1, op2',
        itertools.product(
            [NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1),
                               INDEX, 'nts1'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[0], MASK_1),
                               INDEX, 'nts1')],
            [NumericTimeSeries(MaskedArray(INT_SEQS[1], MASK_2),
                               INDEX, 'nts2'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[1], MASK_2),
                               INDEX, 'nts2')
             ] + [v for v in INTS + FLTS if v != 0]))
    def test_result_with_shared_index(self, operator, op1, op2):
        """Return an instance of `NumericTimeSeries` with shared index."""
        result = operator.func(op1, op2)
        assert op1.index is result.index

    @pytest.mark.parametrize('data_1, data_2',
                             itertools.product([INT_SEQS[0], FLT_SEQS[0]],
                                               [INT_SEQS[1], FLT_SEQS[1]]))
    def test_on_two_numeric_timeseries(self, operator, data_1, data_2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(data_1, MASK_1), INDEX, 'nts1')
        op2 = NumericTimeSeries(MaskedArray(data_2, MASK_2), INDEX, 'nts2')
        result = operator.func(op1, op2)
        answer = operator.func(np.array(data_1), np.array(data_2))
        masks = MASK_R | np.isnan(answer)
        answer = NumericTimeSeries(MaskedArray(answer, masks), INDEX,
                                   f'nts1 {operator.symbol} nts2')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize(
        'data_1, op2',
        itertools.product([INT_SEQS[0], FLT_SEQS[0]],
                          [v for v in INTS + FLTS if v != 0]))
    def test_with_non_zero_numeric_scalar(self, operator, data_1, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(data_1, MASK_1), INDEX, 'nts1')
        result = operator.func(op1, op2)
        answer = operator.func(np.array(data_1), op2)
        answer = NumericTimeSeries(MaskedArray(answer, MASK_1), INDEX,
                                   f'nts1 {operator.symbol} {op2}')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize(
        'op1, op2',
        itertools.product(
            [NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1),
                               INDEX, 'nts1'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[0], MASK_1),
                               INDEX, 'nts1')], [0, 0.]))
    def test_with_numeric_zero(self, operator, op1, op2):
        """Should raise `ZeroDivisionError`."""
        with pytest.raises(ZeroDivisionError):
            _ = operator.func(op1, op2)


class TestArithmeticPowerOperator:
    """Tests related to arithmetic power operator of `NumericTimeSeries`.

    Division-related arithmetic binary operators include: Division(/),
    Modulus(%), Power(**), Floor division(//).

    See Also
    --------
    MaskedArray, ArithmeticBinaryOperator

    """
    def test_with_numeric_timeseries(self):
        """Should raise `TypeError`."""
        op1 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-03'],
                                'nts1')
        op2 = NumericTimeSeries([1, 2, 3],
                                ['2021-11-01', '2021-11-02', '2021-11-03'],
                                'nts2')
        with pytest.raises(TypeError):
            _ = op1 ** op2

    @pytest.mark.parametrize(
        'op1, op2',
        itertools.product(
            [NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1),
                               INDEX, 'nts1'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[0], MASK_1),
                               INDEX, 'nts1')], INTS + FLTS))
    def test_result_with_shared_index(self, op1, op2):
        """Return an instance of `NumericTimeSeries` with shared index."""
        result = op1 ** op2
        assert op1.index is result.index

    @pytest.mark.parametrize('op2', INTS + FLTS)
    def test_between_float_timeseries_and_numeric_scalar(self, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(FLT_SEQS[0], MASK_1), INDEX, 'nts1')
        result = op1 ** op2
        answer = np.array(FLT_SEQS[0]) ** op2
        masks = MASK_1 | np.isnan(answer)
        answer = NumericTimeSeries(MaskedArray(answer, masks), INDEX,
                                   f'nts1 ** {op2}')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize('op2', [v for v in INTS if v >= 0])
    def test_between_integer_timeseries_and_nonnegative_integer(self, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1), INDEX, 'nts1')
        result = op1 ** op2
        answer = np.array(INT_SEQS[0]) ** op2
        answer = NumericTimeSeries(MaskedArray(answer, MASK_1), INDEX,
                                   f'nts1 ** {op2}')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize('op2', [v for v in INTS if v < 0])
    def test_between_integer_timeseries_and_negative_integer(self, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1), INDEX, 'nts1')
        result = op1 ** op2
        answer = np.array(INT_SEQS[0], float) ** op2
        masks = MASK_1 | np.isnan(answer)
        answer = NumericTimeSeries(MaskedArray(answer, masks), INDEX,
                                   f'nts1 ** {op2}')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize('op2', FLTS)
    def test_between_integer_timeseries_and_float_scalar(self, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1), INDEX, 'nts1')
        result = op1 ** op2
        answer = np.array(INT_SEQS[0]) ** op2
        masks = MASK_1 | np.isnan(answer)
        answer = NumericTimeSeries(MaskedArray(answer, masks), INDEX,
                                   f'nts1 ** {op2}')
        assert ts_identical(result, answer)


@pytest.mark.parametrize('operator', ComparisonOperator)
class TestComparisonOperator:
    """Tests related to Comparison operator of `NumericTimeSeries`.

    Comparison operators include: Equal(==), Not equal(!=), Greater than(>),
    Less than(<), Greater than or equal to(>=), Less than or equal to(<=).

    See Also
    --------
    MaskedArray, ComparisonOperator

    """
    @pytest.mark.parametrize(
        'op1, op2',
        itertools.product(
            [NumericTimeSeries(MaskedArray(INT_SEQS[0], MASK_1),
                               INDEX, 'nts1'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[0], MASK_1),
                               INDEX, 'nts1')],
            [NumericTimeSeries(MaskedArray(INT_SEQS[1], MASK_2),
                               INDEX, 'nts2'),
             NumericTimeSeries(MaskedArray(FLT_SEQS[1], MASK_2),
                               INDEX, 'nts2')] + INTS + FLTS))
    def test_result_with_shared_index(self, operator, op1, op2):
        """Return an instance of `NumericTimeSeries` with shared index."""
        result = operator.func(op1, op2)
        assert op1.index is result.index

    @pytest.mark.parametrize('data_1, data_2',
                             itertools.product([INT_SEQS[0], FLT_SEQS[0]],
                                               [INT_SEQS[1], FLT_SEQS[1]]))
    def test_on_two_numeric_timeseries(self, operator, data_1, data_2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(data_1, MASK_1), INDEX, 'nts1')
        op2 = NumericTimeSeries(MaskedArray(data_2, MASK_2), INDEX, 'nts2')
        result = operator.func(op1, op2)
        answer = operator.func(np.array(data_1), np.array(data_2))
        answer = BooleanTimeSeries(MaskedArray(answer, MASK_R), INDEX,
                                   f'nts1 {operator.symbol} nts2')
        assert ts_identical(result, answer)

    @pytest.mark.parametrize('data_1, op2',
                             itertools.product([INT_SEQS[0], FLT_SEQS[0]],
                                               INTS + FLTS))
    def test_with_numeric_scalar(self, operator, data_1, op2):
        """Return an expected instance of 'NumericTimeSeries'.

        An instance with expected contents, index, and name.

        """
        op1 = NumericTimeSeries(MaskedArray(data_1, MASK_1), INDEX, 'nts1')
        result = operator.func(op1, op2)
        answer = operator.func(np.array(data_1), op2)
        answer = BooleanTimeSeries(MaskedArray(answer, MASK_1), INDEX,
                                   f'nts1 {operator.symbol} {op2}')
        assert ts_identical(result, answer)
