# -*- coding: utf-8 -*-

#          _|__   __  | . |__
# \  /\  /  |    /  | | | |  |
#  \/  \/   |__ /__/| | | |__|
#

"""
Wanini's Financial Technical Analysis Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wTA-Lib is a financial technical analysis library, written in Python.

:copyright: (c) 2021 by Ting-Hsu Chang.
:license: MIT, see LICENSE for more details.
"""

# pylint: disable=unused-import
from ._td import BooleanTimeSeries, TimeSeries  # noqa: F401

# pylint: enable=unused-import

__ALL__ = ['BooleanTimeSeries', 'TimeSeries']
