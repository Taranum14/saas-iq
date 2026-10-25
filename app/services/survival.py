"""Survival analysis for subscription lifetimes (Kaplan-Meier, written from scratch).

Why not just use 1 / churn_rate?  Customers who are still paying have an *unknown*
final lifetime ("censored"). Ignoring them biases lifetimes low; counting them as
churned is worse. Kaplan-Meier uses every customer for exactly as long as they were
observed.

Conventions
    duration = number of months the customer has paid so far
    event    = True if they churned after that, False if still paying (censored)
    S(k)     = probability a customer is still paying in their k-th month
               after starting (k = 0 is the first paid month, so S(0) = 1)
"""
import numpy as np
import pandas as pd


def build_durations(customers_df, mrr_df):
    """One row per paying customer: duration (months paid) and event flag."""
    months_paid = mrr_df.groupby("customer_id").size().rename("duration")
    paid = customers_df[customers_df["converted_on"].notna()].set_index("id")
    out = paid.join(months_paid, how="inner")
    out["event"] = out["churned_on"].notna()
    keep = [c for c in ("channel", "plan") if c in out.columns] + ["duration", "event"]
    return out[keep].reset_index()


def kaplan_meier(durations, events):
    """Return a DataFrame with columns k, survival, at_risk, churned.

    A customer with duration d and event=True was observed NOT paying in month index d.
    A censored customer with duration d tells us nothing about month index d, so they
    are not in the risk set for it.
    """
    d = np.asarray(durations, dtype=int)
    e = np.asarray(events, dtype=bool)
    if len(d) == 0:
        return pd.DataFrame(dict(k=[0], survival=[1.0], at_risk=[0], churned=[0]))

    max_k = int(d.max())
    rows, s = [dict(k=0, survival=1.0, at_risk=len(d), churned=0)], 1.0
    for k in range(1, max_k + 1):
        churned = int(((d == k) & e).sum())
        at_risk = int((d > k).sum()) + churned
        if at_risk > 0:
            s *= 1 - churned / at_risk
        rows.append(dict(k=k, survival=s, at_risk=at_risk, churned=churned))
    return pd.DataFrame(rows)


def expected_paid_months(km, horizon=24):
    """Expected number of paid months within `horizon` (restricted mean survival).

    Sum of S(k) for k = 0..horizon-1.  Beyond the longest observed tenure we do not
    extrapolate: the horizon is clipped to what the data can support.
    """
    horizon = min(horizon, int(km["k"].max()) + 1)
    return float(km.loc[km["k"] < horizon, "survival"].sum())


def median_lifetime(km):
    """First month index at which survival drops to 50% or below (NaN if it never does)."""
    below = km[km["survival"] <= 0.5]
    return float(below["k"].iloc[0]) if len(below) else float("nan")


def survival_by(durations_df, column):
    """Kaplan-Meier curve for each value of `column` (e.g. channel or plan)."""
    return {
        str(name): kaplan_meier(g["duration"], g["event"])
        for name, g in durations_df.groupby(column)
    }
