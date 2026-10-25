import numpy as np
import pandas as pd
import pytest

from app.services import metrics


@pytest.fixture
def mrr_df():
    # A: 1000 all three months, upgrades to 3000 in Mar
    # B: joins Feb at 3000, churns in Mar
    # C: joins Mar at 1000
    rows = [
        (1, "2026-01-01", 1000), (1, "2026-02-01", 1000), (1, "2026-03-01", 3000),
        (2, "2026-02-01", 3000),
        (3, "2026-03-01", 1000),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "month", "mrr"])


def test_waterfall_values(mrr_df):
    wf = metrics.mrr_waterfall(mrr_df).set_index("month")
    feb = wf.loc["2026-02-01"]
    assert (feb.opening, feb.new, feb.closing) == (1000, 3000, 4000)
    mar = wf.loc["2026-03-01"]
    assert mar.expansion == 2000      # A upgraded 1000 -> 3000
    assert mar.churned == 3000        # B left
    assert mar.new == 1000            # C joined
    assert mar.closing == 4000


def test_waterfall_identity(mrr_df):
    wf = metrics.mrr_waterfall(mrr_df)
    calc = wf.opening + wf.new + wf.expansion - wf.contraction - wf.churned
    assert (calc == wf.closing).all()


def test_waterfall_handles_missing_months():
    df = pd.DataFrame(
        [(1, "2026-01-01", 500), (1, "2026-03-01", 500)],
        columns=["customer_id", "month", "mrr"],
    )
    wf = metrics.mrr_waterfall(df)
    assert len(wf) == 3                     # Feb is filled in
    assert wf.iloc[1].churned == 500        # gap month = churn
    assert wf.iloc[2].new == 500            # returns as new


def test_kpis(mrr_df):
    k = metrics.kpis(metrics.mrr_waterfall(mrr_df))
    assert k["mrr"] == 4000 and k["arr"] == 48000 and k["customers"] == 2
    assert k["logo_churn"] == pytest.approx(1 / 2)
    # (4000 + 2000 - 0 - 3000) / 4000
    assert k["nrr"] == pytest.approx(0.75)


def test_cohort_retention(mrr_df):
    c = metrics.cohort_retention(mrr_df)
    assert c.loc["Jan 2026", 0] == 1.0 and c.loc["Jan 2026", 2] == 1.0
    assert c.loc["Feb 2026", 1] == 0.0       # B churned
    assert np.isnan(c.loc["Mar 2026", 1])    # beyond the data horizon


def test_funnel_and_unit_economics():
    customers = pd.DataFrame(
        dict(
            id=[1, 2, 3, 4],
            channel=["A", "A", "B", "B"],
            converted_on=["2026-01-15", None, "2026-01-15", "2026-01-20"],
            churned_on=[None, None, "2026-03-01", None],
            acquisition_cost=[100, 100, 400, 400],
        )
    )
    f = metrics.trial_funnel(customers)
    assert f["stages"] == [("Free trials", 4), ("Became paying", 3), ("Still paying", 2)]

    mrr = pd.DataFrame(
        dict(
            customer_id=[1, 1, 3, 3, 4, 4],
            month=["2026-02-01", "2026-03-01"] * 3,
            mrr=[1000] * 6,
        )
    )
    ue = metrics.unit_economics(customers, mrr, gross_margin=0.8).set_index("channel")
    assert ue.loc["A", "cac"] == 200          # 200 spend / 1 payer
    assert ue.loc["B", "cac"] == 400          # 800 spend / 2 payers
    assert ue.loc["B", "churn"] == pytest.approx(1 / 4)
    # LTV must equal ARPA x margin x expected months, with expected months from the KM curve
    assert ue.loc["A", "ltv"] == pytest.approx(1000 * 0.8 * ue.loc["A", "exp_months"])
    assert ue.loc["B", "exp_months"] < ue.loc["A", "exp_months"]   # B has a churner, A none
