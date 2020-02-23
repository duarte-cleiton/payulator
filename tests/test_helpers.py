import pandas as pdb
import numpy as np
import pytest

from .context import payulator
from payulator import *


def test_freq_to_num():
    d = {
        "annually": 1,
        "semiannually": 2,
        "triannually": 3,
        "quarterly": 4,
        "bimonthly": 6,
        "monthly": 12,
        "fortnightly": 26,
        "weekly": 52,
        "daily": 365,
        "continuously": np.inf,
    }

    for allow_cts in [True, False]:
        # Test on valid freq names
        for key, val in d.items():
            if key == "continuously" and not allow_cts:
                with pytest.raises(ValueError):
                    freq_to_num(key, allow_cts=allow_cts)
            else:
                assert freq_to_num(key, allow_cts=allow_cts) == val

        # Test on invalid freq name
        with pytest.raises(ValueError):
            freq_to_num("bingo", allow_cts=allow_cts)
