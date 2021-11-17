# -*- coding: utf-8 -*-
"""Unit-Tests related to logical operators on `MaskedArray`."""

import itertools

import numpy as np
import pytest

from .._context import LogicalBinaryOperator, LogicalUnaryOperator, MaskedArray
from ._context import gen_broadcastable_dataset, get_unbroadcastable_dataset

# pylint: disable=no-self-use, too-few-public-methods
# pylint: disable=too-many-arguments

BOOLEAN_OPERANDS = [
    MaskedArray([True, True, False, False]),
    np.array([True, True, False, False]),
    [True, True, False, False],
    True, False]
NON_BOOLEAN_OPERANDS = [
    # `MaskedArray`
    MaskedArray([1, 0, 1, 0]),
    MaskedArray([1., 0., 1., 0.]),
    MaskedArray(['True', 'True', 'False', 'False']),
    MaskedArray(np.array([True, True, False, False], object)),
    # Numpy's array
    np.array([1, 0, 1, 0]),
    np.array([1., 0., 1., 0.]),
    np.array(['True', 'True', 'False', 'False']),
    np.array([True, True, False, False], object),
    # List
    [1, 0, 1, 0],
    [1., 0., 1., 0.],
    ['True', 'True', 'False', 'False'],
    # Scalar
    1, 1., 'True', 'False']
UNBROADCASTABLE_DATA = get_unbroadcastable_dataset([True, False, True, False])
BROADCASTABLE_DATA = gen_broadcastable_dataset(
    data_1=[True, False, True, False],
    data_2=[True, True, False, False],
    mask_1=[True, False, False, True],
    mask_2=[False, False, True, True],
    mask_r=[True, False, True, True])


@pytest.mark.parametrize('operator', LogicalUnaryOperator)
class TestLogicalUnaryOperators:
    """Tests related to logical unary operators of `MaskedArray`.

    Logical unary operators only include NOT(~).

    See Also
    --------
    numpy.logical_not, LogicalUnaryOperator

    """
    @pytest.mark.parametrize('operand', NON_BOOLEAN_OPERANDS[:4])
    def test_on_non_boolean_masked_array(self, operator, operand):
        """Should rasie `ValueError`."""
        with pytest.raises(ValueError):
            _ = operator.func(operand)

    def test_on_boolean_masked_array(self, operator):
        """Return an instance of `MaskedArray` with expected contents."""
        data = [True, False, True, False]
        result = operator.func(MaskedArray(data))
        answer = MaskedArray(operator.func(np.array(data)))
        assert result.equals(answer)


@pytest.mark.parametrize('operator', LogicalBinaryOperator)
class TestLogicalBinaryOperators:
    """Tests related to logical binary operators of `MaskedArray`.

    Logical binary operators includes AND(&), OR(|), and XOR(^).

    See Also
    --------
    numpy.logical_and, numpy.logical_or, numpy.logical_xor,
    LogicalBinaryOperator

    """
    @pytest.mark.parametrize('operand_1, operand_2',
                             itertools.product(
                                 NON_BOOLEAN_OPERANDS[:4],
                                 BOOLEAN_OPERANDS + NON_BOOLEAN_OPERANDS))
    # Test on two operands, at least one of which has non-boolean dtype
    def test_with_non_boolean_dtype(self, operator, operand_1, operand_2):
        """Should rasie `ValueError`.

        test condition:
            - operand_1 : a non-boolean masked-array, and
            - operand_2 : any.

        results :
            Raise `ValueError`

        """
        with pytest.raises(ValueError):
            _ = operator.func(operand_1, operand_2)

    @pytest.mark.parametrize('operand_1, operand_2',
                             itertools.product(BOOLEAN_OPERANDS[:1],
                                               NON_BOOLEAN_OPERANDS))
    def test_on_non_boolean_dtype(self, operator, operand_1, operand_2):
        """Should rasie `ValueError`.

        test condition:
            - operand_1 : a boolean masked-array, and
            - operand_2 : a non-boolean masked-array like.

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
            - operand_1 : a boolean masked-array, and
            - operand_2 : a non-boolean masked-array like,
            which shapes are unbroadcastable.

        results :
            Raise `ValueError`

        """
        with pytest.raises(ValueError):
            _ = operator.func(operand_1, operand_2)

    # Test on two operands with broadcastable shapes
    @pytest.mark.parametrize('array_1, array_2, masks, operand_1, operand_2',
                             BROADCASTABLE_DATA)
    def test_on_two_broadcastable_operands(self, operator, array_1, array_2,
                                           masks, operand_1, operand_2):
        """Return an instance of `MaskedArray` with expected contents.

        test condition:
            - operand_1 : a boolean masked-array, and
            - operand_2 : a boolean masked-array like,
            which shapes are broadcastable.

        results :
            an instance of `MaskedArray` with expected contents generated by
            the same operator.

        """
        result = operator.func(operand_1, operand_2)
        answer = MaskedArray(operator.func(array_1, array_2), masks)
        assert result.equals(answer)
