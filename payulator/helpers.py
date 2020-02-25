from datetime import date
import math
from copy import copy
from typing import List, Union, Dict, Optional

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

def compute_period_interest_rate(
    interest_rate: float, 
    compounding_freq: str, 
    payment_freq: str
) -> float:
    """
    Compute the interest rate per payment period given
    an annual interest rate, a compounding frequency, and a payment
    freq.
    See the function :func:`freq_to_num` for acceptable frequencies.
    Arguments:
        interest_rate {float} -- Return rate for a given operation
        compounding_freq {str} -- Interest compounding frequency period
        payment_freq {str} -- Payment frequence period
    
    Returns:
        float -- The period interest rate
    """
    i = interest_rate
    j = freq_to_num(compounding_freq, allow_cts=True)
    k = freq_to_num(payment_freq)

    if np.isinf(j):
        return math.exp(i / k) - 1
    else:
        return (1 + i / j) ** (j / k) - 1

def build_principal_fn(
    principal: float,
    interest_rate: float,
    compounding_freq: str,
    payment_freq: str,
    num_payments: int
):

    """
    Compute the remaining loan principal, the loan balance,
    as a function of the number of payments made.
    Return the resulting function.
    """
    P = principal
    I = compute_period_interest_rate(interest_rate, compounding_freq, payment_freq)
    n = num_payments

    def p(t):
        if I == 0:
            return P - t * P / n
        else:
            return P * (1 - ((1 + I) ** t - 1) / ((1 + I) ** n - 1))
    return p

def amortize(
    principal: float,
    interest_rate: float,
    compounding_freq: str,
    payment_freq: str,
    num_payments: int
) -> float:

    """
        Given the loan parameters

    - ``principal``: (float) amount of loan, that is, the principal
    - ``interest_rate``: (float) nominal annual interest rate
      (not as a percentage), e.g. 0.1 for 10%
    - ``compounding_freq``: (string) compounding frequency;
      one of the keys of :const:`NUM_BY_FREQ`, e.g. "monthly"
    - ``payment_freq``: (string) payments frequency;
      one of the keys of :const:`NUM_BY_FREQ`, e.g. "monthly"
    - ``num_payments``: (integer) number of payments in the loan
      term

    return the periodic payment amount due to
    amortize the loan into equal payments occurring at the frequency
    ``payment_freq``.
    See the function :func:`freq_to_num` for valid frequncies.

    Notes:

    - https://en.wikipedia.org/wiki/Amortization_calculator
    - https://www.vertex42.com/ExcelArticles/amortization-calculation.html
    """

    P = principal
    I = compute_period_interest_rate(interest_rate, compounding_freq, payment_freq)
    n = num_payments
    if I == 0:
        A = P / n
    else:
        A = P * I / (1 - (1 + I) ** (-n))

    return A

def summarize_amortized_loan(
    principal: float,
    interest_rate: float,
    compounding_freq: str,
    payment_freq: str,
    num_payments: int,
    fee: float = 0,
    first_payment_date: Optional[str] = None,
    decimals: int = 2
) -> Dict:

    A = amortize(principal, interest_rate, compounding_freq, payment_freq, num_payments)
    p = build_principal_fn(
        principal, interest_rate, compounding_freq, payment_freq, num_payments
    )

    n = num_payments
    f = (
        pd.DataFrame({"payment_sequence": range(1, n + 1)})
        .assign(beginning_balance=lambda x: (x.payment_sequence - 1).map(p))
        .assign(
            principal_payment=lambda x: x.beginning_balance.diff(-1).fillna(
                x.beginning_balance.iat[-1]
            )
        )
        .assign(ending_balance=lambda x: x.beginning_balance - x.principal_payment)
        .assign(interest_payment=lambda x: A - x.principal_payment)
        .assign(fee_payment=0)
        .assign(notes=np.nan)
    )
    f.fee_payment.iat[0] = fee

    date_offset = to_date_offset(freq_to_num(payment_freq))
    if first_payment_date and date_offset:
        f["payment_date"] = [
            pd.Timestamp(first_payment_date) + j * date_offset for j in range(n)
        ]

        # Put payment date first
        cols = f.columns.tolist()
        cols.remove("payment_date")
        cols.insert(1, "payment_date")
        f = f[cols].copy()

    # Bundle the result into dictionary
    d = {}
    d["payment_schedule"] = f
    d["periodic_payment"] = A
    d["interest_total"] = f["interest_payment"].sum()
    d["interest_and_fee_total"] = d["interest_total"] + fee
    d["payment_total"] = d["interest_and_fee_total"] + principal
    d["interest_and_fee_total/principal"] = d["interest_and_fee_total"] / principal
    if "payment_date" in f:
        d["first_payment_date"] = f.payment_date.iat[0].strftime("%Y-%m-%d")
        d["last_payment_date"] = f.payment_date.iat[-1].strftime("%Y-%m-%d")
    
    if decimals is not None:
        for key, val in d.items():
            if isinstance(val, pd.DataFrame):
                d[key] = val.round(decimals)
            elif isinstance(val, float):
                d[key] = round(val, 2)
    return d
