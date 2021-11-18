# -*- coding: utf-8 -*-
"""Unit-Tests related to comparison operators on `MaskedArray`."""

import itertools

import pytest

from .._context import ComparisonOperator, MaskedArray
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
for data_1 in ([1, -1, 1, -1], [1., -1., 1., -1.]):
    for data_2 in ([1, 1, -1, -1], [1., 1., -1., -1.]):
        BROADCASTABLE_DATA += gen_broadcastable_dataset(
            data_1=data_1, data_2=data_2,
            mask_1=[True, False, True, False],
            mask_2=[True, True, False, False],
            mask_r=[True, True, True, False])


@pytest.mark.parametrize('operator', ComparisonOperator)
class TestComparisonOperators:
    """Tests related to comparison operators of `MaskedArray`.

    Comparison operators includes: Equal(==), Not equal(!=), Greater than(>),
    Less than(<), Greater than or equal to(>=), Less than or equal to(<=).

    See Also
    --------
    numpy.equal, numpy.not_equal, numpy.greater, numpy.less,
    numpy.greater_equal, numpy.less_equal, ComparisonOperator.

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
        answer = MaskedArray(operator.func(array_1, array_2), masks)
        assert result.equals(answer)