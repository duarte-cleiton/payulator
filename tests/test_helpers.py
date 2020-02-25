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


def test_to_date_offset():
    for k in [1, 2, 3, 4, 6, 12, 26, 52, 365]:
        assert isinstance(to_date_offset(k), pd.DateOffset)

    assert to_date_offset(10) is None

def test_amortize():
    A = amortize(1000, 0.05, "quarterly", "monthly", 3 * 12)
    assert round(A, 2) == 29.96

    A = amortize(1000, 0.02, "continuously", "semiannually", 2 * 2)
    assert round(A, 2) == 256.31

def test_compute_period_interest_rate():
    I = compute_period_interest_rate(0.12, "monthly", "monthly")
    assert round(I, 2) == 0.01

def test_build_principal_fn():
    balances = [
        100.00,
        91.93,
        83.81,
        75.65,
        67.44,
        59.18,
        50.87,
        42.52,
        34.11,
        25.66,
        17.16,
        8.60,
        0,
    ]

    p = build_principal_fn(100, 0.07, "monthly", "monthly", 12)
    for i in range(13):
        assert round(p(i), 2) == balances[i]

def test_summarize_amortized_loan():
    s = summarize_amortized_loan(
        1000,
        0.05,
        "quarterly",
        "monthly",
        3 * 12,
        fee=10,
        first_payment_date="2018-01-01"
    )
    expect_keys = {
        "payment_schedule",
        "periodic_payment",
        "interest_total",
        "interest_and_fee_total",
        "payment_total",
        "interest_and_fee_total/principal",
        "first_payment_date",
        "last_payment_date"
    }
    assert set(s.keys()) == expect_keys
    assert round(s["periodic_payment"], 2) == 29.96
    assert round(s["interest_and_fee_total"], 2) == 88.62

    #Check payment schedule
    f = s["payment_schedule"]
    assert set(f.columns) == {
        "payment_sequence",
        "payment_date",
        "beginning_balance",
        "principal_payment",
        "ending_balance",
        "interest_payment",
        "fee_payment",
        "notes",
    }
    assert f.shape[0] == 3 * 12
    f["payment"] = f["interest_payment"] + f["principal_payment"]
    assert (abs(f["payment"] - 29.96) <= 0.015).all()