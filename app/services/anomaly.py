"""Anomaly detection on monthly KPIs, plus a root-cause breakdown by channel.

Detection
* Churn rate  -> p-chart: expected churners = trailing pooled churn rate x customers at
  risk; z = (observed - expected) / binomial std. Noise allowance shrinks as the base
  grows, so small early-stage months do not trigger false alarms.
* New MRR (a level, not a rate) -> ROBUST rolling z-score using median and MAD of the
  previous `window` months, so one bad month does not distort the baseline.

Root cause: for a flagged month, compare each channel's churn rate with its own recent
normal and rank channels by the number of EXTRA customers lost.
"""
import numpy as np
import pandas as pd

from .metrics import _pivot


def robust_z(values, window=6, min_history=3):
    s = pd.Series(values, dtype=float).reset_index(drop=True)
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        base = s.iloc[max(0, i - window) : i].dropna()
        if len(base) < min_history or pd.isna(s.iloc[i]):
            continue
        med = base.median()
        scale = max(1.4826 * (base - med).abs().median(), 0.10 * abs(med), 1e-9)
        out[i] = (s.iloc[i] - med) / scale
    return pd.Series(out, index=values.index if hasattr(values, "index") else None)


def churn_pchart_z(churned, opening, window=6, min_history=3):
    """Binomial z-score of each month's churn count against the trailing pooled rate."""
    churned = np.asarray(churned, dtype=float)
    opening = np.asarray(opening, dtype=float)
    z = np.full(len(churned), np.nan)
    for i in range(len(churned)):
        lo = max(0, i - window)
        hist_open = opening[lo:i]
        if (hist_open > 0).sum() < min_history or opening[i] <= 0:
            continue
        # +0.5 / +1 smoothing keeps the rate away from exactly 0
        p0 = (churned[lo:i].sum() + 0.5) / (hist_open.sum() + 1)
        expected = p0 * opening[i]
        sd = np.sqrt(opening[i] * p0 * (1 - p0))
        z[i] = (churned[i] - expected) / sd
    return z


def detect(wf, window=6, threshold=3.5):
    """Flag unusual months: churn too HIGH (p-chart) or new MRR too LOW (robust z)."""
    rows = []
    z = churn_pchart_z(wf["churned_customers"], wf["opening_customers"], window)
    rate = wf["churned_customers"] / wf["opening_customers"].replace(0, np.nan)
    for i, zi in enumerate(z):
        if not np.isnan(zi) and zi > threshold:
            rows.append(dict(month=wf["month"].iloc[i], metric="Logo churn rate",
                             value=float(rate.iloc[i]), z=float(zi)))

    new = wf["new"].astype(float).reset_index(drop=True)
    zn = robust_z(new, window)
    for i, zi in enumerate(zn):
        if not pd.isna(zi) and zi < -threshold:
            rows.append(dict(month=wf["month"].iloc[i], metric="New MRR",
                             value=float(new.iloc[i]), z=float(zi)))
    return pd.DataFrame(rows, columns=["month", "metric", "value", "z"])


def channel_churn_table(mrr_df, customers_df):
    """Per month and channel: customers at the start of the month and how many left."""
    p = _pivot(mrr_df)
    ch = customers_df.set_index("id")["channel"].reindex(p.index)
    rows = []
    cols = list(p.columns)
    for prev_m, m in zip(cols[:-1], cols[1:]):
        opening = p[prev_m] > 0
        churned = opening & (p[m] == 0)
        for channel in ch.dropna().unique():
            mask = ch == channel
            rows.append(
                dict(
                    month=m.to_timestamp(),
                    channel=channel,
                    opening=int((opening & mask).sum()),
                    churned=int((churned & mask).sum()),
                )
            )
    return pd.DataFrame(rows)


def explain_month(mrr_df, customers_df, month, window=6):
    """Which channels explain a churn spike in `month`?"""
    t = channel_churn_table(mrr_df, customers_df)
    t["rate"] = t["churned"] / t["opening"].replace(0, np.nan)
    month = pd.Timestamp(month)
    out = []
    for channel, g in t.groupby("channel"):
        g = g.sort_values("month")
        cur = g[g["month"] == month]
        base = g[g["month"] < month].tail(window)
        if cur.empty or len(base) < 3:
            continue
        # baseline rate = pooled over the baseline window (more stable than averaging rates)
        base_rate = base["churned"].sum() / max(base["opening"].sum(), 1)
        opening = int(cur["opening"].iloc[0])
        churned = int(cur["churned"].iloc[0])
        out.append(
            dict(
                channel=channel,
                opening=opening,
                churned=churned,
                rate=churned / opening if opening else np.nan,
                normal_rate=base_rate,
                excess_churned=churned - base_rate * opening,
            )
        )
    res = pd.DataFrame(out).sort_values("excess_churned", ascending=False).reset_index(drop=True)
    pos = res["excess_churned"].clip(lower=0)
    res["share_of_excess"] = pos / pos.sum() if pos.sum() > 0 else 0.0
    return res
