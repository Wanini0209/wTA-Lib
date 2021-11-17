# -*- coding: utf-8 -*-
"""Unit-Tests related to `array_equal`."""

import numpy as np

from .._context import array_equal

# pylint: disable=no-self-use, too-few-public-methods


class TestArrayEqual:
    """Tests related to `array_equal`.

    It is an additional test to check whether ``array_equal`` works in two
    special cases: including ``np.nan`` and ``object`` dtype.

    """
    def test_equal_float_arrays_with_nan(self):
        """Should return ``True``."""
        array_1 = np.array([1, np.nan])
        array_2 = np.array([1, np.nan])
        assert array_equal(array_1, array_2)

    def test_unequal_float_arrays(self):
        """Should return ``False``."""
        array_1 = np.array([1, np.nan])
        array_2 = np.array([np.nan, 1])
        assert not array_equal(array_1, array_2)

    def test_equal_objct_arrays_with_nan(self):
        """Should return ``True``."""
        array_1 = np.array([1, np.nan], object)
        array_2 = np.array([1, np.nan], object)
        assert array_equal(array_1, array_2)

    def test_unequal_object_arrays(self):
        """Should return ``False``."""
        array_1 = np.array([1, np.nan], object)
        array_2 = np.array([np.nan, 1], object)
        assert not array_equal(array_1, array_2)
