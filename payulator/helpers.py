import math
from copy import copy
from typing import List, Union, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from . import constants as cs


def freq_to_num(freq: str, *, allow_cts: bool = False) -> Union[int, float]:
    """
    Map frequency name to number of occurrences per year via
    :const:`NUM_BY_FREQ`
    If not ``allow_cts``, then remove the ``"continuouly"`` option.
    Raise a ``ValueError`` in case of no frequency match.

    Arguments:
        freq {str} -- Frequence name of occurrences

    Keyword Arguments:
        allow_cts {bool} -- Is freq continuously allowed  (default: {False})

    Returns:
        Union[int, float] -- Return the number correspondent to frequency
    """

    d = copy(cs.NUM_BY_FREQ)
    if not allow_cts:
        del d["continuously"]

    try:
        return d[freq]
    except KeyError:
        raise ValueError(
            f"Invalid frequency {freq}. " f"Frequency must be on of {d.keys()}"
        )


def to_date_offset(num_per_year: int) -> Union[pd.DateOffset, None]:
    """
    Convert the given number of occurrences per year to its
    corresponding period (Pandas DateOffset object)

    Arguments:
        num_per_year {int} -- Number of occurrences per year | valid values [1, 2, 3, 4, 6, 12, 26, 52, 365]

    Returns:
        Union[pd.DateOffset, None] -- Pandas DateOffset Object or None if invalid period
    """
    k = num_per_year
    if k in [1, 2, 3, 4, 6, 12]:
        d = pd.DateOffset(months=12 // k)
    elif k == 26:
        d = pd.DateOffset(weeks=2)
    elif k == 52:
        d = pd.DateOffset(weeks=1)
    elif k == 365:
        d = pd.DateOffset(days=1)
    else:
        d = None
    return d
