# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""Data Array.

This module provides derivatives of array, which is used to store the content
of time-indexing data in current package.

"""

from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

_MaskedArrayLike = Union[ArrayLike, 'MaskedArray']


class Operator(NamedTuple):
    """Operation Definitions."""
    name: str
    symbol: str
    func: Callable


class LogicalUnaryOperator(Operator, Enum):
    """Logical unary operator."""
    NOT = Operator('Logical NOT', '~', lambda x: ~x)


class LogicalBinaryOperator(Operator, Enum):
    """Logical binary operator."""
    AND = Operator('Logical AND', '&', lambda x, y: x & y)
    OR = Operator('Logical OR', '|', lambda x, y: x | y)
    XOR = Operator('Logical Exclusive-OR', '^', lambda x, y: x ^ y)


def array_equal(target: np.ndarray, reference: np.ndarray) -> bool:
    """Determine if two array are equal.

    Two array are equal, it means they have
    1. the same shape,
    2. the same dtype, and
    3. the same elements.

    Notes
    -----
    `numpy.array_equal` support to compare two numeric arrays with `np.nan` by
    setting the `equal_nan` argument as ``True``. However, if the one of the
    arrays is not numeric, it would raise `TypeError`.

    """
    if target.dtype != reference.dtype:
        return False
    if target.shape != reference.shape:
        return False
    if not np.issubdtype(target.dtype, object):
        return np.array_equal(target, reference, equal_nan=True)
    # NumPy's `array_equal` method would raise an exception if
    # `dtype` of arrays is ``object`` and `equal_nan` is ``True``.
    # In this case, it should compare arrays element by element.
    for tar, ref in zip(target.flatten(), reference.flatten()):
        if (np.isnan(tar) and np.isnan(ref)) or (tar == ref):
            continue
        return False
    return True


class ArithmeticUnaryOperator(Operator, Enum):
    """Arithmetic unary operator."""
    NEG = Operator('Negative', '-', lambda x: -x)


class ArithmeticUnaryFunction(Operator, Enum):
    """Arithmetic unary function."""
    ABS = Operator('Absolute value', 'abs', abs)


ArithmeticUnaryOperation = Union[ArithmeticUnaryOperator,
                                 ArithmeticUnaryFunction]


class ArithmeticBinaryOperator(Operator, Enum):
    """Arithmetic binary operator."""
    ADD = Operator('Arithmetic Addition', '+', lambda x, y: x + y)
    SUB = Operator('Arithmetic Subtraction', '-', lambda x, y: x - y)
    MUL = Operator('Arithmetic Multiplication', '*', lambda x, y: x * y)
    DIV = Operator('Arithmetic Division', '/', lambda x, y: x / y)
    MOD = Operator('Arithmetic Modulus', '%', lambda x, y: x % y)
    POW = Operator('Arithmetic Power', '**', lambda x, y: x ** y)
    FDIV = Operator('Arithmetic Floor-division', '//', lambda x, y: x // y)


class ComparisonOperator(Operator, Enum):
    """Comparison (binary) operator."""
    EQ = Operator('Equal', '==', lambda x, y: x == y)
    NE = Operator('Not equal', '!=', lambda x, y: x != y)
    GT = Operator('Greater than', '>', lambda x, y: x > y)
    LT = Operator('Less than', '<', lambda x, y: x < y)
    GE = Operator('Greater than or equal to', '>=', lambda x, y: x >= y)
    LE = Operator('Less than or equal to', '<=', lambda x, y: x <= y)


NumericBinaryOperator = Union[ArithmeticBinaryOperator, ComparisonOperator]


class MaskedArray:
    """An array with masks indicating which elements are N/A.

    Like NumPy's array, `MaskedArray` supports operators as follows:
    1. Logical operators:
        NOT(~), AND(&), OR(|), XOR(^).
    2. Arithmetic Operators:
        Negative(-), Absolute value(abs), Addition(+), Subtraction(-),
        Multiplication(*), Division(/), Modulus(%), Power(**),
        Floor division(//).
    3. Comparison operators:
        Equal(==), Not equal(!=), Greater than(>), Less than(<),
        Greater than or equal to(>=), Less than or equal to(<=).

    The second operand of binary operators could be scalar, array-like or
    `MaskedArray`, and the shapes of two operands must be broadcastable.
    No matter unary or binary operator, the data-type of operand(s) must be
    supported for the operator. For example, the logical operators only support
    boolean data and arithmetic operators only support numeric data.

    Parameters
    ----------
    data : array-like
        The raw-data stored in the masked-array, in which the values
        corresponding to N/A elements are undefined.
    masks: array-like, optional
        Boolean values used to indicate which elements in `data` ara N/A. If it
        is not specified, all elements in `data` are available.

    Notes
    -----
    `data` and `masks`(if it is specified) must have same shape.

    See Also
    --------
    numpy.ndarray

    Properties
    ----------
    data : numpy.ndarray
        Return the raw-data of the masked-array, in which the value of N/A
        elements are undefined.
    dtype : numpy dtype object
        Data-type of elements of the masked-array.
    ndim : int
        Number of dimensions of the masked-array.
    shape : tuple of int
        Tuple of dimensions of the masked-array.

    Built-in Functions
    ------------------
    len : int
        Size of the first dimension of the masked-array.

    Methods
    ----------
    astype : MaskedArray
        Return a copy of the masked-array casting to a specified type.
    isna : numpy.ndarray
        Return a same-sized boolean array indicating if the elements in the
        masked-array are N/A.
    to_numpy : numpy.ndarray
        Return a copy of the masked-array as a NumPy array, in which the value
        of N/A elements are replaced with ``numpy.nan``.
    equals : bool
        Determine whether two masked-arrays contain the same elements.
    fillna : MaskedArray
        Fill N/A elements using given value.
    ffill : MaskedArray
        Fill N/A elements using forward-fill method. For each N/A element, fill
        it by the last available element preceding it.
    bfill : MaskedArray
        Fill N/A elements using backward-fill method. For each N/A element,
        fill it by the next available element.
    shift : MaskedArray
        Shift elements along a given axis by desired positions.

    Examples
    --------
    >>> MaskedArray([1, 2, 3, 4])
    array([1, 2, 3, 4], dtype=int32)

    With specified `masks`:

    >>> MaskedArray([1, 2, 3, 4], [True, False, True, False])
    array([nan, 2, nan, 4], dtype=int32)

    2-D array:
    >>> MaskedArray([[1, 2], [3, 4]], [[True, False], [True, False]])
    array([[nan, 2],
           [nan, 4]], dtype=int32)

    N-D array:
    >>> data = np.arange(120).reshape((2, 3, 4, 5))
    >>> MaskedArray(data, data % 2 == 0)
    array([[[[nan, 1, nan, 3, nan],
             [5, nan, 7, nan, 9],
             [nan, 11, nan, 13, nan],
             [15, nan, 17, nan, 19]],

            [[nan, 21, nan, 23, nan],
             [25, nan, 27, nan, 29],
             [nan, 31, nan, 33, nan],
             [35, nan, 37, nan, 39]],

            [[nan, 41, nan, 43, nan],
             [45, nan, 47, nan, 49],
             [nan, 51, nan, 53, nan],
             [55, nan, 57, nan, 59]]],


           [[[nan, 61, nan, 63, nan],
             [65, nan, 67, nan, 69],
             [nan, 71, nan, 73, nan],
             [75, nan, 77, nan, 79]],

            [[nan, 81, nan, 83, nan],
             [85, nan, 87, nan, 89],
             [nan, 91, nan, 93, nan],
             [95, nan, 97, nan, 99]],

            [[nan, 101, nan, 103, nan],
             [105, nan, 107, nan, 109],
             [nan, 111, nan, 113, nan],
             [115, nan, 117, nan, 119]]]], dtype=int32)

    """

    def __init__(self, data: ArrayLike, masks: Optional[ArrayLike] = None):
        self._data = np.array(data)
        self._masks = None if masks is None else np.array(masks, dtype=bool)

    @property
    def data(self) -> np.ndarray:
        """Raw-data stored in the masked-array.

        Return a copy of the raw-data of the masked-array, in which the value
        of N/A elements are undefined.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> MaskedArray(data).data
        array([1, 2, 3])

        With N/A elements(built with specified `masks`):

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([True, False, False])
        >>> MaskedArray(data, masks).data
        array([1, 2, 3])

        """
        return self._data.copy()

    @property
    def dtype(self) -> np.dtype:
        """Data-type of elements of the masked-array.

        Returns
        -------
        numpy.dtype

        See Also
        --------
        numpy.dtype, numpy.ndarray, numpy.ndarray.dtype

        Examples
        --------
        >>> data = np.array([1, 2, 3, 4])
        >>> MaskedArray(data).dtype
        dtype('int32')

        """
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of the masked-array.

        Returns
        -------
        int

        See Also
        --------
        numpy.ndarray.ndim

        Examples
        --------
        >>> data = np.array([1, 2, 3, 4])
        >>> MaskedArray(data).ndim
        1

        N-dimension:

        >>> data = np.zeros((2, 3, 4))
        >>> MaskedArray(data).ndim
        3

        """
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tuple of dimensions of the masked-array.

        Notes
        -----
        Not like the `shape` of `numpy.ndarray`, the `shape` property of
        `MaskedArray` is only used to get the current shape of the
        masked-array. It is read-only and it can not be used to reshape the
        masked-array by assigning a tuple of array dimensions to it.

        See Also
        --------
        numpy.ndarray.shape

        Examples
        --------
        >>> data = np.array([1, 2, 3, 4])
        >>> MaskedArray(data).shape
        (4,)

        N-dimension:

        >>> data = np.zeros((2, 3, 4))
        >>> MaskedArray(data).shape
        (2, 3, 4)

        """
        return self._data.shape

    def __len__(self) -> int:
        """Size of the first dimension of the masked-array.

        See Also
        --------
        numpy.ndarray

        Examples
        --------
        >>> data = np.array([1, 2, 3, 4])
        >>> len(MaskedArray(data))
        4

        N-dimension:

        >>> data = np.zeros((2, 3, 4))
        >>> len(MaskedArray(data))
        2

        """
        return len(self._data)

    def astype(self, dtype: Union[str, type, np.dtype]) -> 'MaskedArray':
        """Copy of the masked-array, cast to a specified data-type.

        Return a copy of the masked-array casting to a `dtype`.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the masked-array is cast.

        Returns
        -------
        MaskedArray

        See Also
        --------
        numpy.ndarray.astype

        Examples
        --------
        >>> MaskedArray([1, 2, 3]).astype(float)
        array([1., 2., 3.], dtype=float64)

        Down-casting:
        >>> MaskedArray([-1.75, -1.5, -1.25, -0.5, 0,
                         0.5, 1.25, 1.5, 1.75]).astype(int)
        array([-1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32)

        """
        return MaskedArray(self._data.astype(dtype), self._masks)

    def isna(self) -> np.ndarray:
        """Detect N/A elements of the masked-array.

        Return a same-sized boolean array indicating if the elements in the
        masked-array are N/A.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> MaskedArray(data).isna()
        array([False, False, False])

        With N/A elements(built with specified `masks`):

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([True, False, False])
        >>> MaskedArray(data, masks).isna()
        array([ True, False, False])

        """
        if self._masks is None:
            return np.full(shape=self._data.shape, fill_value=False)
        return self._masks.copy()

    def to_numpy(self) -> np.ndarray:
        """A NumPy array representing this object.

        Return a copy of the masked-array as a NumPy array, in which the value
        of N/A elements are replaced with ``numpy.nan``.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> MaskedArray(data).to_numpy()
        array([1, 2, 3])

        1. integer array with N/As:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).to_numpy()
        array([ 1., nan,  3.])

        2. float array with N/As:

        >>> data = np.array([1., 2., 3.])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).to_numpy()
        array([ 1., nan,  3.])

        3. non-numeric array with N/As:

        >>> data = np.array([False, True, False])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).to_numpy()
        array([False, nan, False], dtype=object)
        >>> data = np.array(['1', '2', '3'])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).to_numpy()
        array(['1', nan, '3'], dtype=object)

        """
        if self._masks is None or not self._masks.any():
            return self._data.copy()
        if np.issubdtype(self._data.dtype, np.number):
            ret = self._data.astype(float)
        else:
            ret = self._data.astype(object)
        ret[self._masks] = np.nan
        return ret

    def equals(self, other: 'MaskedArray') -> bool:
        """Determine whether two masked-arrays are equal.

        Two masked-array are equal, it means they have
        1. the same shape,
        2. the same dtype, and
        3. the same elements.

        The 3rd rule, two masked-array have the same elements, if the element
        of one masked-array is N/A, the corresponding element of another
        masked-array must be N/A; otherwise they should have same value.

        Parameters
        ----------
        other : MaskedArray
            The other masked-array to compare against.

        Returns
        -------
        bool
            ``True`` if `other` is an `MaskedArray` object and it has the same
            shape and elements as the calling object; ``False`` otherwise.

        See Also
        --------
        numpy.array_equal

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> MaskedArray(data).equals(MaskedArray(data))
        True

        1. Equivalent `data` but differnet `dtype`:

        >>> data = np.array([1, 2, 3])
        >>> MaskedArray(data).equals(MaskedArray(data.astype(float)))
        False

        2. Specified equivalent `masks`:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.full((3,), False)
        >>> MaskedArray(data).equals(MaskedArray(data, masks))
        True

        3. Equal `data` but different `masks`:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data).equals(MaskedArray(data, masks))
        False

        4. `data` only different on N/A elements:

        >>> data_t = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> target = MaskedArray(data_t, masks)
        >>> data_r = np.array([1, -2, 3])
        >>> target.equals(MaskedArray(data_r, masks))
        True

        5. compare to another object not a masked-array:

        >>> data = np.array([1, 2, 3])
        >>> MaskedArray(data).equals(data)
        False

        """
        if isinstance(other, MaskedArray):
            # pylint: disable=protected-access
            # Case1: Either `self` or `other` contains no N/A element.
            cond_1 = self._masks is None or not self._masks.any()
            cond_2 = other._masks is None or not other._masks.any()
            if cond_1:
                return cond_2 and array_equal(self._data, other._data)
            if cond_2:
                return False
            # Case2: Both `self` and `other` contains N/A elements
            # In this case, neither self._mask nor other._masks could be
            # ``None``. So, ignore `mypy` and `pylint` for wrong warning below.
            # pylint: disable=invalid-unary-operand-type
            cond_1 = array_equal(self._masks, other._masks)  # type: ignore
            cond_2 = array_equal(self._data[~self._masks],  # type: ignore
                                 other._data[~other._masks])  # type: ignore
            # pylint: enable=invalid-unary-operand-type, protected-access
            return cond_1 and cond_2
        return False

    def fillna(self, value: Any) -> 'MaskedArray':
        """Fill N/A elements using the given value.

        Parameters
        ----------
        value : scalar
            Value used to fill N/A elements.

        Returns
        -------
        MaskedArray
            A copy of the masked-array in which the N/A elements are filled
            with given value.

        Notes
        -----
        The data-type of `value` must be equal or equivalent of the `dtype` of
        the masked-array. If not, the contents of result may be unexpected.

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).fillna(0)
        array([1, 0, 3], dtype=int32)

        Fill float array with integer value:

        >>> data = np.array([1., 2., 3.])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).fillna(0)
        array([1., 0., 3.], dtype=float64)

        Fill integer array with float value:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).fillna(0.)
        array([1, 0, 3], dtype=int32)

        """
        data = self._data.copy()
        if self._masks is not None:
            data[self._masks] = value
        return MaskedArray(data)

    def ffill(self) -> 'MaskedArray':
        """Fill N/A elements using forward-fill method.

        For each N/A element, fill it by the last available element preceding
        it.

        Returns
        -------
        MaskedArray
            A copy of this masked-array in which the N/A elements are filled
            using forward-fill method.

        Notes
        -----
        The N/A elements preceding all available elements would not be filled.

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).ffill()
        array([1, 1, 3], dtype=int32)

        With leading N/A:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([True, False, False])
        >>> MaskedArray(data, masks).ffill()
        rray([nan, 2, 3], dtype=int32)

        With tailing N/A:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, False, True])
        >>> MaskedArray(data, masks).ffill()
        array([1, 2, 2], dtype=int32)

        """
        def get_idxs(masks):
            """
            for each element in masks:
            1. If it is ``False``, return the index of the element itself.
            2. If it is ``True`` and no preceding element with ``False`` value,
               return ``-1``.
            3. Otherwise, return the index of the last element whose value is
               ``False`` before it.
            """
            ret = np.arange(1, len(masks) + 1)
            ret[masks] = 0
            ret = np.maximum.accumulate(ret) - 1
            return ret

        if self._masks is not None:
            idxs = get_idxs(self._masks)
            data = self._data[idxs]
            masks = idxs < 0
        else:
            data = self._data.copy()
            masks = None
        return MaskedArray(data, masks)

    def bfill(self) -> 'MaskedArray':
        """Fill N/A elements using backward-fill method.

        For each N/A element, fill it by the next available element.

        Returns
        -------
        MaskedArray
            A copy of this masked-array in which the N/A elements are filled
            using backward-fill method.

        Notes
        -----
        The N/A elements after which there is no available element would not
        be filled.

        Examples
        --------
        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, True, False])
        >>> MaskedArray(data, masks).bfill()
        array([1, 3, 3], dtype=int32)

        With leading N/A:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([True, False, False])
        >>> MaskedArray(data, masks).bfill()
        array([2, 2, 3], dtype=int32)

        With tailing N/A:

        >>> data = np.array([1, 2, 3])
        >>> masks = np.array([False, False, True])
        >>> MaskedArray(data, masks).bfill()
        array([1, 2, nan], dtype=int32)

        """
        def get_idxs(masks):
            """
            for each element in masks:
            1. If it is ``False``, return the negative index of the element
               itself.
            2. If it is ``True`` and there is no element with ``False`` value
               after it, return ``0``.
            3. Otherwise, return the negative index of the next element
               whose value is ``False`` after it.
            """
            masks = masks[::-1]
            ret = np.arange(1, len(masks) + 1)
            ret[masks] = 0
            ret = -np.maximum.accumulate(ret)[::-1]
            return ret

        if self._masks is not None:
            idxs = get_idxs(self._masks)
            data = self._data[idxs]
            masks = idxs >= 0
        else:
            data = self._data.copy()
            masks = None
        return MaskedArray(data, masks)

    def shift(self, period: int, axis: int = 0) -> 'MaskedArray':
        """Shift elements along a given axis by desired positions.

        Parameters
        ----------
        period : int
            The number of places by which elements are shifted.It could be
            positive, negative or zero as follows:
            1. If `period` is ``0``, it return a copy of this array directly.
            2. If `period` is positive, it shift elements forward along given
               axis by desired positions.
            3. If `period` is negative, it shift elements backward along given
               axis by desired positions.
        axis : int
            axis along which elements shifted.

        Returns
        -------
        MaskedArray
            A copy of the masked-array with elements shifted along given axis
            by desired positions.

        Examples
        --------
        >>> data = np.arange(24).reshape((4,6))
        >>> masks = data % 7 == 0
        >>> array = MaskedArray(data, masks)
        >>> array
        array([[nan, 1, 2, 3, 4, 5],
               [6, nan, 8, 9, 10, 11],
               [12, 13, nan, 15, 16, 17],
               [18, 19, 20, nan, 22, 23]], dtype=int32)

        1. zero `period`:

        >>> array.shift(0)
        array([[nan, 1, 2, 3, 4, 5],
               [6, nan, 8, 9, 10, 11],
               [12, 13, nan, 15, 16, 17],
               [18, 19, 20, nan, 22, 23]], dtype=int32)

        2. positive `period`:

        >>> array.shift(2)
        array([[nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan],
               [nan, 1, 2, 3, 4, 5],
               [6, nan, 8, 9, 10, 11]], dtype=int32)

        >>> array.shift(2, axis=0)
        array([[nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan],
               [nan, 1, 2, 3, 4, 5],
               [6, nan, 8, 9, 10, 11]], dtype=int32)

        >>> array.shift(2, axis=1)
        array([[nan, nan, nan, 1, 2, 3],
               [nan, nan, 6, nan, 8, 9],
               [nan, nan, 12, 13, nan, 15],
               [nan, nan, 18, 19, 20, nan]], dtype=int32)

        3. negative `period`:

        >>> array.shift(-2)
        array([[12, 13, nan, 15, 16, 17],
               [18, 19, 20, nan, 22, 23],
               [nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan]], dtype=int32)

        >>> array.shift(-2, axis=0)
        array([[12, 13, nan, 15, 16, 17],
               [18, 19, 20, nan, 22, 23],
               [nan, nan, nan, nan, nan, nan],
               [nan, nan, nan, nan, nan, nan]], dtype=int32)

        >>> array.shift(-2, axis=1)
        array([[2, 3, 4, 5, nan, nan],
               [8, 9, 10, 11, nan, nan],
               [nan, 15, 16, 17, nan, nan],
               [20, nan, 22, 23, nan, nan]], dtype=int32)

        4. `axis` out of bounds:

        >>> array.shift(2, axis=2)
        AxisError: axis 2 is out of bounds for array of dimension 2

        See Also
        --------
        numpy.roll

        """
        if not isinstance(period, int):
            raise TypeError("'period' must be 'int' not '%s'"
                            % type(period).__name__)
        if period == 0:
            return MaskedArray(self._data, self._masks)
        data = np.roll(self._data, period, axis)
        masks = np.roll(self.isna(), period, axis)
        idxs = [slice(None, None, None)] * axis
        if period > 0:
            idxs.append(slice(None, period))
        else:
            idxs.append(slice(period, None))
        masks[tuple(idxs)] = True
        return MaskedArray(data, masks)

    def __getitem__(self, key: Any) -> Union[Any, 'MaskedArray']:
        """Subscript Operator.

        Parameters
        ----------
        key:
            Like the subscript operator of NumPy array.

        Returns
        -------
        Scalar or MaskedArray:
            If the result of subscript operater is an array, return it as a
            masked-array. If it is an available element, return it directly;
            otherwise return ``numpy.nan``.

        See Also
        --------
        numpy.ndarray

        Examples
        --------
        >>> data = np.arange(5)
        >>> masks = np.array([True, False, True, False, True])
        >>> array = MaskedArray(data, masks)
        >>> array
        array([nan, 1, nan, 3, nan], dtype=int32)

        1. supscript with slice:

        >>> array[1:]
        array([1, nan, 3, nan], dtype=int32)
        >>> array[:-1]
        array([nan, 1, nan, 3], dtype=int32)
        >>> array[1: -1]
        array([1, nan, 3], dtype=int32)
        >>> array[::2]
        array([nan, nan, nan], dtype=int32)

        2a. supscript with integer indexing an available element:

        >>> array[1]
        1

        2b. supscript with integer indexing an N/A element:

        >>> array[2]
        nan

        2c. supscript with integer(raise out-of-range):

        >>> array[5]
        IndexError: index 5 is out of bounds for axis 0 with size 5

        3a. supscript with integer array:

        >>> array[np.array([1, 3])]
        array([1, 3], dtype=int32)

        3b. supscript with integer array(raise out-of-range):

        >>> array[np.array([1, 3, 5])]
        IndexError: index 5 is out of bounds for axis 0 with size 5

        4a. supscript with boolean array(same length):

        >>> array[np.arange(5) % 2 == 0]
        array([nan, nan, nan], dtype=int32)

        4b. supscript with boolean array(different length):

        >>> array[np.arange(6) % 2 == 0]
        IndexError: boolean index did not match indexed array along dimension 0;
        dimension is 5 but corresponding boolean dimension is 6

        """
        values = self._data[key]
        masks = self._masks
        if masks is not None:
            masks = masks[key]
        if isinstance(values, np.ndarray):
            # return a masked-array
            if masks is None or not masks.any():
                return MaskedArray(values)
            return MaskedArray(values, masks)
        # return a scalar
        if masks:
            return np.nan
        return values

    def __repr__(self) -> str:
        """String representation for the masked-array.

        See Also
        --------
        numpy.ndarray

        """
        if self._masks is None or not self._masks.any():
            ret = repr(self._data)
            if 'dtype' not in ret:
                ret = f'{ret[:-1]}, dtype={self.dtype})'
            return ret
        data = self._data.astype(object)
        data[self._masks] = np.nan
        return repr(data).replace('dtype=object', f'dtype={self.dtype}')

    def _broadcast_to(self, shape: Tuple[int, ...]) -> 'MaskedArray':
        data = np.broadcast_to(self._data, shape)
        masks = self._masks
        if masks is not None:
            masks = np.broadcast_to(masks, shape)
        return MaskedArray(data, masks)

    def _unary_op(self, func: Callable[[np.ndarray, ], np.ndarray]
                  ) -> 'MaskedArray':
        return MaskedArray(func(self._data), self._masks)

    def _binary_op(self, other: _MaskedArrayLike,
                   func: Callable[[np.ndarray, np.ndarray], np.ndarray]
                   ) -> 'MaskedArray':
        if not isinstance(other, MaskedArray):
            # Case 1: `other` is not a `MaskedArray`
            other = MaskedArray(np.asarray(other))
            return self._binary_op(other, func)
        # pylint: disable=protected-access
        if self.ndim < other.ndim:
            # Case 2: `other` is a `MaskedArray` with larger dimension
            return self._broadcast_to(other.shape)._binary_op(other, func)
        if other.ndim < self.ndim:
            # Case 3: `other` is a `MaskedArray` with less dimension
            return self._binary_op(other._broadcast_to(self.shape), func)
        # Case end: self and other are two `MaskedArray` with same shape
        if self.shape != other.shape:
            raise ValueError("operands could not be broadcast together with "
                             f"shapes {self.shape} {other.shape}")
        masks = self._masks
        if other._masks is not None and other._masks.any():
            if masks is None or not masks.any():
                masks = other._masks
            else:
                masks = masks | other._masks
        data = func(self._data, other._data)
        # pylint: enable=protected-access

        # deal with `np.nan`
        if np.issubdtype(data.dtype, float):
            isnan = np.isnan(data)
            if isnan.any():
                if masks is None:
                    masks = isnan
                else:
                    masks = masks | isnan
        return MaskedArray(data, masks)

    def _logical_unary_op(self, operator: LogicalUnaryOperator
                          ) -> 'MaskedArray':
        if not np.issubdtype(self.dtype, bool):
            raise ValueError("unsupported operand dtype for %s: '%s'"
                             % (operator.symbol, self.dtype))
        return self._unary_op(operator.func)

    def _logical_binary_op(self, other: _MaskedArrayLike,
                           operator: LogicalBinaryOperator) -> 'MaskedArray':
        dtype1 = self.dtype
        if isinstance(other, MaskedArray):
            dtype2 = other.dtype
        else:
            dtype2 = np.asarray(other).dtype
        if not(np.issubdtype(dtype1, bool) and np.issubdtype(dtype2, bool)):
            raise ValueError("unsupported operand dtype(s) for %s: '%s' and '%s'"
                             % (operator.symbol, dtype1, dtype2))
        return self._binary_op(other, operator.func)

    def __invert__(self) -> 'MaskedArray':
        return self._logical_unary_op(LogicalUnaryOperator.NOT)

    def __and__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._logical_binary_op(other, LogicalBinaryOperator.AND)

    def __or__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._logical_binary_op(other, LogicalBinaryOperator.OR)

    def __xor__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._logical_binary_op(other, LogicalBinaryOperator.XOR)

    def _arithmetic_unary_op(self, operator: ArithmeticUnaryOperation
                             ) -> 'MaskedArray':
        if not np.issubdtype(self.dtype, np.number):
            raise ValueError("unsupported operand dtype for %s: '%s'"
                             % (operator.symbol, self.dtype))
        return self._unary_op(operator.func)

    def _numeric_binary_op(self, other: _MaskedArrayLike,
                           operator: NumericBinaryOperator
                           ) -> 'MaskedArray':
        dtype1 = self.dtype
        if isinstance(other, MaskedArray):
            dtype2 = other.dtype
        else:
            dtype2 = np.asarray(other).dtype
        if not(np.issubdtype(dtype1, np.number) and np.issubdtype(dtype2, np.number)):
            raise ValueError("unsupported operand dtype(s) for %s: '%s' and '%s'"
                             % (operator.symbol, dtype1, dtype2))
        return self._binary_op(other, operator.func)

    def __neg__(self) -> 'MaskedArray':
        return self._arithmetic_unary_op(ArithmeticUnaryOperator.NEG)

    def __abs__(self) -> 'MaskedArray':
        return self._arithmetic_unary_op(ArithmeticUnaryFunction.ABS)

    def __add__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.ADD)

    def __sub__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.SUB)

    def __mul__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.MUL)

    def __truediv__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.DIV)

    def __floordiv__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.FDIV)

    def __mod__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.MOD)

    def __pow__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ArithmeticBinaryOperator.POW)

    # When overriding `__eq__` and `__ne__` methods with specified object and
    # return non-boolean, `mypy` would raise a 'incompatible-override' waring.
    # So we ignore `mypy` above.
    def __eq__(self, other: _MaskedArrayLike) -> 'MaskedArray':  # type: ignore
        return self._numeric_binary_op(other, ComparisonOperator.EQ)

    def __ne__(self, other: _MaskedArrayLike) -> 'MaskedArray':  # type: ignore
        return self._numeric_binary_op(other, ComparisonOperator.NE)

    def __gt__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ComparisonOperator.GT)

    def __lt__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ComparisonOperator.LT)

    def __ge__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ComparisonOperator.GE)

    def __le__(self, other: _MaskedArrayLike) -> 'MaskedArray':
        return self._numeric_binary_op(other, ComparisonOperator.LE)
