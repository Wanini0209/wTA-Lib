# -*- coding: utf-8 -*-
"""Unit-Tests related to arithmetic operators on `MaskedArray`."""

import itertools

import numpy as np
import pytest

from .._context import (
    ArithmeticBinaryOperator,
    ArithmeticUnaryFunction,
    ArithmeticUnaryOperator,
    MaskedArray,
)
from ._context import (
    gen_broadcastable_dataset,
    gen_masked_array_like_dataset,
    get_unbroadcastable_dataset,
)

# pylint: disable=no-self-use, too-few-public-methods
# pylint: disable=too-many-arguments

NUMERIC_DATA = [[0], [0.]]  # dtypes : int and float
NON_NUMERIC_DATA = [['0'], [True]]  # dtypes : str and bool
ANY_DATA = NUMERIC_DATA + NON_NUMERIC_DATA

UNBROADCASTABLE_DATA = get_unbroadcastable_dataset([-1, 0, 1])
BROADCASTABLE_DATA = []
for data_1 in ([-2, -1, 0, 1, 2], [-1., -.5, 0., .5, 1.]):
    for data_2 in ([2, -2, 1, -1, 0], [1., -1., .5, -.5, 0.]):
        BROADCASTABLE_DATA += gen_broadcastable_dataset(
            data_1=data_1, data_2=data_2,
            mask_1=[True, False, True, False, False],
            mask_2=[True, True, False, False, False],
            mask_r=[True, True, True, False, False])

VALID_DATA_FOR_POWER = gen_broadcastable_dataset(
    data_1=[-2, -1, 0, 1, 2], data_2=[1., -1., .5, -.5, 0.],
    mask_1=[True, False, True, False, False],
    mask_2=[True, True, False, False, False],
    mask_r=[True, True, True, False, False])
VALID_DATA_FOR_POWER += gen_broadcastable_dataset(
    data_1=[-1., -.5, 0., .5, 1.], data_2=[1., -1., .5, -.5, 0.],
    mask_1=[True, False, True, False, False],
    mask_2=[True, True, False, False, False],
    mask_r=[True, True, True, False, False])
VALID_DATA_FOR_POWER += gen_broadcastable_dataset(
    data_1=[-1., -.5, 0., .5, 1.], data_2=[1., -1., .5, -.5, 0.],
    mask_1=[True, False, True, False, False],
    mask_2=[True, True, False, False, False],
    mask_r=[True, True, True, False, False])
VALID_DATA_FOR_POWER += gen_broadcastable_dataset(
    data_1=[-2, -1, 0, 1, 2], data_2=[0, 1, 2, 3, 4],
    mask_1=[True, False, True, False, False],
    mask_2=[True, True, False, False, False],
    mask_r=[True, True, True, False, False])


@pytest.mark.parametrize('operator', ArithmeticUnaryFunction)
class TestArithmeticUnaryFunctions:
    """Tests related to arithmetic unary functions of `MaskedArray`.

    Arithmetic unary functions only include `abs`.

    See Also
    --------
    numpy.abs, ArithmeticUnaryFunction

    """
    @pytest.mark.parametrize('data', NON_NUMERIC_DATA)
    def test_on_non_numeric_masked_array(self, operator, data):
        """Should rasie `ValueError`."""
        with pytest.raises(ValueError):
            _ = operator.func(MaskedArray(data))

    @pytest.mark.parametrize('data', NUMERIC_DATA)
    def test_on_numeric_masked_array(self, operator, data):
        """Return an instance of `MaskedArray` with expected contents."""
        result = operator.func(MaskedArray(data))
        answer = MaskedArray(operator.func(np.array(data)))
        assert result.equals(answer)


@pytest.mark.parametrize('operator', ArithmeticUnaryOperator)
class TestArithmeticUnaryOperators:
    """Tests related to arithmetic unary operators of `MaskedArray`.

    Arithmetic unary operators only include Negative(-).

    See Also
    --------
    numpy.negative, ArithmeticUnaryOperator

    """
    @pytest.mark.parametrize('data', NON_NUMERIC_DATA)
    def test_on_non_numeric_masked_array(self, operator, data):
        """Should rasie `ValueError`."""
        with pytest.raises(ValueError):
            _ = operator.func(MaskedArray(data))

    @pytest.mark.parametrize('data', NUMERIC_DATA)
    def test_on_numeric_masked_array(self, operator, data):
        """Return an instance of `MaskedArray` with expected contents."""
        result = operator.func(MaskedArray(data))
        answer = MaskedArray(operator.func(np.array(data)))
        assert result.equals(answer)


@pytest.mark.parametrize('operator', ArithmeticBinaryOperator)
class TestArithmeticBinaryOperators:
    """Tests related to arithmetic binary operators of `MaskedArray`.

    Arithmetic binary operators includes: Addition(+), Subtraction(-),
    Multiplication(*), Division(/), Modulus(%), Power(**), and
    Floor division(//).

    See Also
    --------
    numpy.add, numpy.subtract, numpy.multiply, numpy.divide, numpy.mod,
    numpy.power, numpy.floor_divide, ArithmeticBinaryOperator.

    """
    # Test on two operands, at least one of which has non-numeric dtype
    @pytest.mark.parametrize(
        'operand_1, operand_2',
        itertools.product(
            map(MaskedArray, NON_NUMERIC_DATA),
            itertools.chain(*map(gen_masked_array_like_dataset, ANY_DATA))))
    def test_on_non_numeric_masked_array(self, operator, operand_1, operand_2):
        """Should rasie `ValueError`.

        test condition:
            - operand_1 : non-numeric masked-array, and
            - operand_2 : masked-array like.

        results :
            Raise `ValueError`

        """
        with pytest.raises(ValueError):
            _ = operator.func(operand_1, operand_2)

    @pytest.mark.parametrize(
        'operand_1, operand_2',
        itertools.product(
            map(MaskedArray, NUMERIC_DATA),
            itertools.chain(*map(gen_masked_array_like_dataset,
                                 NON_NUMERIC_DATA))))
    def test_with_non_numeric_masked_array_like(
            self, operator, operand_1, operand_2):
        """Should rasie `ValueError`.

        test condition:
            - operand_1 : numeric masked-array, and
            - operand_2 : non-numeric masked-array like.

        results :
            Raise `ValueError`

        """
        with pytest.raises(ValueError):
            _ = operator.func(operand_1, operand_2)

    # Test on two operands with unbroadcastable shapes
    @pytest.mark.parametrize('operand_1, operand_2', UNBROADCASTABLE_DATA)
    def test_on_two_unbroadcastable_operands(self, operator, operand_1, operand_2):
        """Should rasie `ValueError`.

        test condition:
            - operand_1 : a numeric masked-array, and
            - operand_2 : a numeric masked-array like,
            which shapes are unbroadcastable.

        results :
            Raise `ValueError`

        """
        with pytest.raises(ValueError):
            _ = operator.func(operand_1, operand_2)


@pytest.mark.parametrize('operator',
                         [ArithmeticBinaryOperator.ADD,
                          ArithmeticBinaryOperator.SUB,
                          ArithmeticBinaryOperator.MUL,
                          ArithmeticBinaryOperator.DIV,
                          ArithmeticBinaryOperator.MOD,
                          ArithmeticBinaryOperator.FDIV])
class TestArithmeticBinaryOperatorsExceptPower:
    """Tests related to arithmetic binary operators of `MaskedArray` except power.

    Arithmetic binary operators includes: Addition(+), Subtraction(-),
    Multiplication(*), Division(/), Modulus(%), Power(**), and
    Floor division(//). In this test Power operator is excluded, because
    integers to negative integer powers are not allowed in NumPy.

    See Also
    --------
    numpy.add, numpy.subtract, numpy.multiply, numpy.divide, numpy.mod,
    numpy.floor_divide, ArithmeticBinaryOperator.

    """
    # Test on two operands with broadcastable shapes
    @pytest.mark.parametrize('array_1, array_2, masks, operand_1, operand_2',
                             BROADCASTABLE_DATA)
    def test_on_two_broadcastable_operands(self, operator, array_1, array_2,
                                           masks, operand_1, operand_2):
        """Return an instance of `MaskedArray` with expected contents.

        test condition:
            - operand_1 : a numeric masked-array, and
            - operand_2 : a numeric masked-array like,
            which shapes are broadcastable.

        results :
            an instance of `MaskedArray` with expected contents generated by
            the same operator.

        """
        result = operator.func(operand_1, operand_2)
        answer = operator.func(array_1, array_2)
        if masks is None:
            masks = np.isnan(answer)
        else:
            masks = np.array(masks) | np.isnan(answer)
        answer = MaskedArray(answer, masks)
        assert result.equals(answer)


class TestArithmeticPowerOperator:
    """Tests related to arithmetic power operator of `MaskedArray`.

    Be careful, integers to negative integer powers are not allowed in NumPy.

    See Also
    --------
    numpy.add, numpy.subtract, numpy.multiply, numpy.divide, numpy.mod,
    numpy.floor_divide, ArithmeticBinaryOperator.

    """
    # Test on two operands with broadcastable shapes
    @pytest.mark.parametrize('array_1, array_2, masks, operand_1, operand_2',
                             VALID_DATA_FOR_POWER)
    def test_without_integer_to_negative_integer_power(
            self, array_1, array_2, masks, operand_1, operand_2):
        """Return an instance of `MaskedArray` with expected contents.

        test condition:
            - operand_1 : a numeric masked-array, and
            - operand_2 : a numeric masked-array like,
            which shapes are broadcastable, and not include interger to
            negative integer power.

        results :
            an instance of `MaskedArray` with expected contents generated by
            the same operator.

        """
        result = operand_1 ** operand_2
        answer = array_1 ** array_2
        if masks is None:
            masks = np.isnan(answer)
        else:
            masks = np.array(masks) | np.isnan(answer)
        answer = MaskedArray(answer, masks)
        assert result.equals(answer)

    @pytest.mark.parametrize('operand_1, operand_2',
                             itertools.product([MaskedArray([0, 1, 2])],
                                               [MaskedArray([-1, -2, -1]),
                                                np.array([-1, -2, -1]),
                                                [-1, -2, -1], -1, -2]))
    def test_integers_to_negative_integer_powers(self, operand_1, operand_2):
        """Should rasie `ValueError`.

        test condition:
            - operand_1 : a integer masked-array, and
            - operand_2 : a integer masked-array like with negative integers,
            which shapes are broadcastable.

        results :
            Raise `ValueError`

        """
        with pytest.raises(ValueError):
            _ = operand_1 ** operand_2
