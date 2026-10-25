import numpy as np
import pandas as pd
import pytest

from app.services import anomaly, forecast, metrics
from data.seed_data import generate, shock_period

TODAY = "2026-10-23"


def synthetic_wf(n=24, nrr=0.9, new=100.0, start=1000.0):
    rows, prev = [], 0.0
    close = start
    for i in range(n):
        opening = close if i else 0.0
        churned = opening * (1 - nrr)
        close = opening - churned + new
        rows.append(
            dict(
                month=pd.Timestamp("2024-01-01") + pd.DateOffset(months=i),
                opening=opening, new=new, expansion=0.0, contraction=0.0,
                churned=churned, closing=close,
                opening_customers=int(opening > 0) * 10, churned_customers=0,
            )
        )
    return pd.DataFrame(rows)


def test_forecast_matches_hand_recursion():
    wf = synthetic_wf()
    f = forecast.forecast(wf, horizon=3, n_sims=0)
    c = wf["closing"].iloc[-1]
    expected = []
    for _ in range(3):
        c = c * 0.9 + 100
        expected.append(c)
    assert f["forecast"].to_numpy() == pytest.approx(expected)
    assert f["month"].iloc[0] == wf["month"].iloc[-1] + pd.DateOffset(months=1)


def test_forecast_interval_brackets_point():
    f = forecast.forecast(synthetic_wf(), horizon=4, n_sims=200)
    assert (f["p10"] <= f["forecast"] + 1e-6).all() and (f["forecast"] <= f["p90"] + 1e-6).all()


def test_backtest_on_clean_series_is_near_perfect():
    bt = forecast.backtest(synthetic_wf(), horizon=6)
    best = bt["summary"].iloc[0]
    assert best["method"].startswith("Bottom-up") and best["mape"] < 0.01
    assert bt["n_origins"] == 7


def test_robust_z_flags_a_spike_but_not_noise():
    rng = np.random.default_rng(0)
    s = pd.Series(0.03 + rng.normal(0, 0.003, 20))
    s.iloc[15] = 0.09
    z = anomaly.robust_z(s)
    assert z.iloc[15] > 3.5
    assert (z.drop(15).abs().dropna() < 3.5).all()


def test_detects_injected_churn_shock_and_blames_google_ads():
    customers, subs = generate(n_trials=3000, n_months=24, seed=42, today=TODAY)
    wf = metrics.mrr_waterfall(subs)
    flags = anomaly.detect(wf)
    shock = pd.Timestamp(shock_period(24, TODAY).to_timestamp())
    churn_flags = flags[flags["metric"] == "Logo churn rate"]
    assert shock in set(churn_flags["month"]), "injected shock was not detected"

    why = anomaly.explain_month(subs, customers, shock)
    assert why.iloc[0]["channel"] == "Google Ads"
    assert why.iloc[0]["share_of_excess"] > 0.5
    # the p-chart should raise no false alarms on this series
    assert len(churn_flags) == 1


def test_pchart_handles_small_bases_and_flags_real_spikes():
    # 3% churn on 400 customers is normal noise; 12% is a spike.
    opening = np.full(12, 400)
    churned = np.array([12, 11, 13, 12, 10, 14, 12, 13, 11, 12, 48, 12])
    z = anomaly.churn_pchart_z(churned, opening)
    assert z[10] > 3.5
    assert np.nanmax(np.delete(z, 10)) < 3.5
    # the same 8% rate on a tiny base (n=30) is NOT unusual
    z_small = anomaly.churn_pchart_z([1, 1, 1, 1, 1, 1, 3], [30] * 7)
    assert z_small[-1] < 3.5
