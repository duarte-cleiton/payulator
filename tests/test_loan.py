from collections import OrderedDict

import pandas as pd
import voluptuous as vt
import pytest

from .context import payulator, ROOT
from payulator import *

# def test_to_dict():
#     assert isinstance(Loan().to_dict(), dict)

def test_summarize():
    for kind in ["amortized", "interest_only"]:
        loan = Loan(kind=kind)
        get = loan.summarize()

        if kind == "amortized":
            expect = summarize_amortized_loan(
                loan.principal,
                loan.interest_rate,
                loan.compounding_freq,
                loan.payment_freq,
                loan.num_payments
            )
        elif kind == "interest_only":
            expect = summarize_interest_only_loan(
                loan.principal,
                loan.interest_rate,
                loan.payment_freq,
                loan.num_payments
            )
        
        for k, v in expect.items():
            if isinstance(v, pd.DataFrame):
                assert pd.DataFrame.equals(get[k], v)
            else:
                assert get[k] == v