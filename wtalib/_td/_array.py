# -*- coding: utf-8 -*-
"""Data Array.

This module provides derivatives of array, which is used to store the content
of time-indexing data in current package.

"""

from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


class MaskedArray:
    """An array with masks indicating which elements are N/A.

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

        """
        def array_equal(tar: np.ndarray, ref: np.ndarray) -> bool:
            """Determine if two array are equal.

            Two array are equal, it means they have
            1. the same shape,
            2. the same dtype, and
            3. the same elements.

            """
            if tar.dtype == ref.dtype:
                return np.array_equal(tar, ref)
            return False

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
