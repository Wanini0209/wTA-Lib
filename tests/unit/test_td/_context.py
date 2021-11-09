# -*- coding: utf-8 -*-
"""Context commonly used by test modules in this package."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../../..')))

# pylint: disable=wrong-import-position, unused-import, wrong-import-order
from wtalib._td._index import TimeIndex  # noqa: E402, F401

# pylint: enable=wrong-import-position, unused-import, wrong-import-order
