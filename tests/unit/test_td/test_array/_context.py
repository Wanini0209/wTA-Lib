# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:28:36 2021

@author: Wanini
"""

from typing import Any, List, Tuple

import numpy as np

from .._context import MaskedArray

# pylint: disable=too-many-arguments


def gen_masked_array_like_dataset(data: List[Any]) -> List[Any]:
    """Generate masked-array-likes equivalent to given data."""
    return [MaskedArray(data), np.array(data), data] + list(set(data))


def get_unbroadcastable_dataset(data: List[Any]) -> List[Tuple]:
    """Generate unbroadcastable dataset for binary operator on `MaskedArray`.

    Parameters
    ----------
    data : List[Any]

    Returns
    -------
    List[Tuple]

    """
    def _gen_dataset(data, ddim: int) -> List[Tuple]:
        data_1 = data
        data_2 = data[:-1]
        if ddim <= 0:
            data_2 = [data_2, data_2]
        if ddim >= 0:
            data_1 = [data_1, data_1]
        ret = [(MaskedArray(data_1), data_2),
               (MaskedArray(data_1), np.array(data_2)),
               (MaskedArray(data_1), MaskedArray(data_2))]
        return ret
    ret = []
    for ddim in [-1, 0, 1]:
        ret += _gen_dataset(data, ddim)
    return ret


def gen_broadcastable_dataset(data_1: List[Any], data_2: List[Any],
                              mask_1: List[Any], mask_2: List[Any],
                              mask_r: List[Any]) -> List[Tuple]:
    """Generate broadcastable dataset for binary operator on `MaskedArray`.

    Parameters
    ----------
    data_1 : List[Any]
    data_2 : List[Any]
    mask_1 : List[bool]
    mask_2 : List[bool]
    mask_r : List[bool]

    Returns
    -------
    List[Tuple]

    """
    def _gen_dataset(data_1, data_2, mask_1, mask_2, mask_r, scalars, ddim: int
                     ) -> List[Tuple]:
        mask_r1, mask_r2 = [mask_1, mask_1], [mask_2, mask_2]
        mask_r = [mask_r, mask_r]
        if ddim <= 0:
            data_2, mask_2 = [data_2, data_2], [mask_2, mask_2]
        if ddim >= 0:
            data_1, mask_1 = [data_1, data_1], [mask_1, mask_1]
        ar1, ar2 = np.array(data_1), np.array(data_2)
        # without masks
        op1 = MaskedArray(data_1)
        ret = [(ar1, each, None, op1, each) for each in scalars]
        ret += [(ar1, ar2, None, op1, data_2),
                (ar1, ar2, None, op1, ar2),
                (ar1, ar2, None, op1, MaskedArray(data_2)),
                (ar1, ar2, mask_r2, op1, MaskedArray(data_2, mask_2))]
        # with masks
        op1 = MaskedArray(data_1, mask_1)
        ret += [(ar1, each, mask_1, op1, each) for each in scalars]
        ret += [(ar1, ar2, mask_r1, op1, data_2),
                (ar1, ar2, mask_r1, op1, ar2),
                (ar1, ar2, mask_r1, op1, MaskedArray(data_2)),
                (ar1, ar2, mask_r, op1, MaskedArray(data_2, mask_2))]
        return ret
    ret = []
    scalars = list(set(data_2))
    for ddim in [-1, 0, 1]:
        ret += _gen_dataset(data_1, data_2, mask_1, mask_2, mask_r,
                            scalars, ddim)
    return ret
