import numpy as np
import pandas as pd
import pytest

from app.services import survival


def test_kaplan_meier_hand_calculation():
    # durations 1,2,2,3 ; churned, churned, censored, censored
    # t=1: at risk = #(d>1)=3 + 1 churner = 4, 1 churn  -> S = 0.75
    # t=2: at risk = #(d>2)=1 + 1 churner = 2, 1 churn  -> S = 0.375
    # t=3: nobody can churn observably (only a censored customer) -> flat
    km = survival.kaplan_meier([1, 2, 2, 3], [True, True, False, False]).set_index("k")
    assert km.loc[0, "survival"] == 1.0
    assert km.loc[1, "survival"] == pytest.approx(0.75)
    assert km.loc[2, "survival"] == pytest.approx(0.375)
    assert km.loc[3, "survival"] == pytest.approx(0.375)


def test_all_censored_means_nobody_churned():
    km = survival.kaplan_meier([5, 5, 5], [False, False, False])
    assert (km["survival"] == 1.0).all()


def test_survival_is_non_increasing():
    rng = np.random.default_rng(0)
    d = rng.integers(1, 20, 500)
    e = rng.random(500) < 0.4
    km = survival.kaplan_meier(d, e)
    assert (km["survival"].diff().dropna() <= 1e-12).all()


def test_expected_months_and_median():
    km = pd.DataFrame(dict(k=[0, 1, 2, 3], survival=[1.0, 0.8, 0.4, 0.4]))
    assert survival.expected_paid_months(km, horizon=3) == pytest.approx(1 + 0.8 + 0.4)
    assert survival.expected_paid_months(km, horizon=99) == pytest.approx(2.6)  # clipped, no extrapolation
    assert survival.median_lifetime(km) == 2
