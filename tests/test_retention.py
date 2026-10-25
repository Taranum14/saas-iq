import pandas as pd
import pytest

from app.services import retention


def scores():
    return pd.DataFrame(
        dict(
            customer_id=[1, 2, 3, 4],
            probability=[0.40, 0.05, 0.30, 0.20],
            mrr=[3000, 1000, 8000, 3000],
        )
    )


def test_ranks_by_net_value_and_drops_unprofitable():
    calls, s = retention.build_call_list(scores(), save_rate=0.3, cost=1500, months_kept=6, gross_margin=0.8)
    # customer 2: benefit = .05*.3*1000*.8*6 = 72 < cost -> excluded
    assert 2 not in calls["customer_id"].values
    # customer 3: .3*.3*8000*.8*6 = 3456 -> best
    assert calls.iloc[0]["customer_id"] == 3
    assert calls.iloc[0]["expected_benefit"] == pytest.approx(3456)
    assert (calls["net_value"] > 0).all()
    assert calls["net_value"].is_monotonic_decreasing


def test_budget_caps_number_of_contacts():
    calls, s = retention.build_call_list(scores(), cost=1500, budget=3000)
    assert len(calls) == 2 and s["spend"] == 3000


def test_summary_arithmetic_and_beats_random():
    calls, s = retention.build_call_list(scores(), cost=1500)
    assert s["net_value"] == pytest.approx(s["expected_saved_margin"] - s["spend"])
    assert s["vs_random"] > 1          # model targeting beats random calls
    assert s["active"] == 4


def test_nobody_worth_contacting():
    calls, s = retention.build_call_list(scores(), cost=1_000_000)
    assert calls.empty and s["targeted"] == 0 and s["spend"] == 0
