"""SaaS metrics, written as pure pandas functions (no Flask) so they are easy to test.

Inputs
    mrr_df       columns: customer_id, month (date), mrr
    customers_df columns: id, channel, trial_start, converted_on, churned_on, acquisition_cost
"""
import numpy as np
import pandas as pd

from . import survival


def _pivot(mrr_df):
    """Customer x month matrix of MRR, with every month present (gaps = 0)."""
    df = mrr_df.copy()
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
    p = df.pivot_table(
        index="customer_id", columns="month", values="mrr", aggfunc="sum", fill_value=0
    )
    full = pd.period_range(p.columns.min(), p.columns.max(), freq="M")
    return p.reindex(columns=full, fill_value=0)


def mrr_waterfall(mrr_df):
    """Monthly MRR movements.

    closing = opening + new + expansion - contraction - churned
    (contraction and churned are positive magnitudes.)
    A customer who returns after a gap is counted as 'new' again.
    """
    p = _pivot(mrr_df)
    prev = pd.Series(0, index=p.index)
    rows = []
    for m in p.columns:
        cur = p[m]
        is_new = (prev == 0) & (cur > 0)
        is_churn = (prev > 0) & (cur == 0)
        both = (prev > 0) & (cur > 0)
        diff = cur - prev
        rows.append(
            dict(
                month=m.to_timestamp(),
                opening=int(prev.sum()),
                new=int(cur[is_new].sum()),
                expansion=int(diff[both & (diff > 0)].sum()),
                contraction=int(-diff[both & (diff < 0)].sum()),
                churned=int(prev[is_churn].sum()),
                closing=int(cur.sum()),
                opening_customers=int((prev > 0).sum()),
                new_customers=int(is_new.sum()),
                churned_customers=int(is_churn.sum()),
                closing_customers=int((cur > 0).sum()),
            )
        )
        prev = cur
    return pd.DataFrame(rows)


def kpis(waterfall):
    """Headline numbers for the latest month."""
    cur = waterfall.iloc[-1]
    prev_close = waterfall.iloc[-2]["closing"] if len(waterfall) > 1 else 0
    opening = cur["opening"]
    return dict(
        month=cur["month"],
        mrr=int(cur["closing"]),
        arr=int(cur["closing"] * 12),
        mom_growth=(cur["closing"] / prev_close - 1) if prev_close else np.nan,
        customers=int(cur["closing_customers"]),
        logo_churn=(cur["churned_customers"] / cur["opening_customers"])
        if cur["opening_customers"]
        else np.nan,
        # net revenue retention: what share of last month's MRR survives, incl. upgrades
        nrr=((opening + cur["expansion"] - cur["contraction"] - cur["churned"]) / opening)
        if opening
        else np.nan,
        arpa=(cur["closing"] / cur["closing_customers"]) if cur["closing_customers"] else np.nan,
    )


def cohort_retention(mrr_df):
    """Share of each signup-month cohort still paying k months later.

    Rows = cohort (first paid month), columns = months since start (0, 1, 2...).
    Cells beyond the data horizon are NaN, so the table is a triangle.
    """
    p = _pivot(mrr_df)
    paying = (p > 0).to_numpy()
    n_months = paying.shape[1]
    first = paying.argmax(axis=1)  # first paying month index per customer

    cohorts = {}
    for ci in range(n_months):
        members = paying[first == ci]
        if len(members) == 0:
            continue
        row = []
        for k in range(n_months - ci):
            row.append(members[:, ci + k].mean())
        row += [np.nan] * (n_months - len(row))
        cohorts[p.columns[ci].to_timestamp().strftime("%b %Y")] = row
    out = pd.DataFrame.from_dict(cohorts, orient="index", columns=range(n_months))
    out.insert(0, "size", [int((first == i).sum()) for i, c in enumerate(p.columns)
                           if (first == i).sum() > 0])
    return out


def trial_funnel(customers_df):
    """Trial -> paid -> still active, overall and by channel."""
    df = customers_df
    trials = len(df)
    converted = int(df["converted_on"].notna().sum())
    active = int((df["converted_on"].notna() & df["churned_on"].isna()).sum())

    by_channel = (
        df.assign(converted=df["converted_on"].notna())
        .groupby("channel")
        .agg(trials=("id", "count"), paid=("converted", "sum"))
        .assign(conversion=lambda d: d["paid"] / d["trials"])
        .sort_values("conversion", ascending=False)
        .reset_index()
    )
    return dict(
        stages=[("Free trials", trials), ("Became paying", converted), ("Still paying", active)],
        by_channel=by_channel,
    )


def unit_economics(customers_df, mrr_df, gross_margin=0.8, horizon=24):
    """CAC, LTV and LTV:CAC by acquisition channel.

    CAC  = total spend on the channel / customers who actually paid
    LTV  = ARPA x gross margin x expected paid months within `horizon`,
           where expected months come from a Kaplan-Meier survival curve per channel.
           Still-active customers are handled correctly (censoring) and nothing is
           extrapolated past what the data has observed.
    churn = churn events / paying customer-months (simple rate, shown for reference)
    """
    paid = customers_df[customers_df["converted_on"].notna()]
    spend = customers_df.groupby("channel")["acquisition_cost"].sum()
    paying_n = paid.groupby("channel")["id"].count()
    churn_events = paid.assign(c=paid["churned_on"].notna()).groupby("channel")["c"].sum()

    m = mrr_df.merge(customers_df[["id", "channel"]], left_on="customer_id", right_on="id")
    cust_months = m.groupby("channel")["mrr"].count()
    arpa = m.groupby("channel")["mrr"].mean()

    durations = survival.build_durations(customers_df, mrr_df)
    exp_months = pd.Series(
        {
            ch: survival.expected_paid_months(
                survival.kaplan_meier(g["duration"], g["event"]), horizon
            )
            for ch, g in durations.groupby("channel")
        }
    ).rename_axis("channel")

    out = pd.DataFrame(
        dict(
            spend=spend,
            paying=paying_n,
            arpa=arpa,
            churn=churn_events / cust_months,
            exp_months=exp_months,
        )
    ).dropna()
    out["cac"] = out["spend"] / out["paying"]
    out["ltv"] = out["arpa"] * gross_margin * out["exp_months"]
    out["ltv_cac"] = out["ltv"] / out["cac"]
    out["payback_months"] = out["cac"] / (out["arpa"] * gross_margin)
    out.index.name = "channel"
    return out.sort_values("ltv_cac", ascending=False).reset_index()
