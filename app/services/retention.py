"""Turn churn probabilities into a decision: who should the team contact, and is it worth it?

For each active customer
    expected benefit = P(churn) x P(save | contacted) x MRR x gross margin x months kept
    net value        = expected benefit - cost of the intervention
Contact customers in order of net value, only while net value is positive (and the
budget lasts). `save_rate` and `months_kept` are ASSUMPTIONS: in real life they would
be measured with an A/B test, which is why the UI lets the user change them.
"""
import numpy as np
import pandas as pd


def build_call_list(scores, save_rate=0.30, cost=500.0, months_kept=6,
                    gross_margin=0.8, budget=None):
    """`scores` needs columns: probability, mrr (plus any identifying columns).

    Returns (call_list, summary). call_list holds only customers worth contacting.
    """
    df = scores.copy()
    df["expected_loss"] = df["probability"] * df["mrr"] * gross_margin * months_kept
    df["expected_benefit"] = df["expected_loss"] * save_rate
    df["net_value"] = df["expected_benefit"] - cost
    ranked = df.sort_values("net_value", ascending=False)
    chosen = ranked[ranked["net_value"] > 0].copy()

    if budget is not None and cost > 0:
        chosen = chosen.head(int(budget // cost))

    n = len(chosen)
    spend = n * cost
    saved = float(chosen["expected_benefit"].sum())

    # reference: contacting the same number of customers at random
    random_saved = float(df["expected_benefit"].mean() * n) if n else 0.0

    summary = dict(
        active=len(df),
        targeted=n,
        spend=float(spend),
        expected_saved_margin=saved,
        net_value=saved - spend,
        roi=(saved - spend) / spend if spend else float("nan"),
        random_saved_margin=random_saved,
        vs_random=(saved / random_saved) if random_saved else float("nan"),
        mrr_at_risk=float((df["probability"] * df["mrr"]).sum()),
        targeted_mrr=float(chosen["mrr"].sum()),
    )
    return chosen.reset_index(drop=True), summary
