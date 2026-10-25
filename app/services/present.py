"""Turns metric tables into the plain-language pieces the dashboard shows.

Kept separate from Flask so the wording and bar geometry can be unit-tested.
"""
import math

from . import fmt


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def build_bridge(row):
    """Opening MRR -> four movements -> closing MRR, as bar positions in percent.

    Revenue is ~100x bigger than a month's movements, so drawing bars from zero
    would make every movement invisible. Instead the bars show *change from the
    opening figure*: the zero line is the opening level, each bar starts where the
    previous one ended.
    """
    steps = [
        ("New customers", float(row["new"]), _plural(int(row["new_customers"]), "customer") + " joined"),
        ("Upgrades", float(row["expansion"]), ""),
        ("Downgrades", -float(row["contraction"]), ""),
        ("Cancellations", -float(row["churned"]), _plural(int(row["churned_customers"]), "customer") + " left"),
    ]
    points, cum = [0.0], 0.0
    placed = []
    for label, value, note in steps:
        start, cum = cum, cum + value
        points += [cum]
        placed.append((label, value, note, start, cum))
    net = cum
    lo, hi = min(points), max(points)
    span = hi - lo or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05

    def pos(x):
        return (x - lo) / (hi - lo) * 100

    rows = []
    for label, value, note, start, end in placed:
        left = pos(min(start, end))
        rows.append(
            dict(
                label=label, note=note, value=value, kind="gain" if value >= 0 else "loss",
                left=round(left, 2), width=round(max(pos(max(start, end)) - left, 0.4), 2),
                from_left=value >= 0,
            )
        )
    net_left = pos(min(0, net))
    return dict(
        opening=float(row["opening"]),
        closing=float(row["closing"]),
        net=net,
        zero=round(pos(0), 2),
        rows=rows,
        net_left=round(net_left, 2),
        net_width=round(max(pos(max(0, net)) - net_left, 0.4), 2),
        net_kind="gain" if net >= 0 else "loss",
    )


def build_lead(month_ts, prev_ts, closing, growth, row):
    """The opening statement, as [(text, is_figure), ...] so figures can be set in heavier type."""
    month = month_ts.strftime("%B")
    out = [("MRR closed " + month + " at ", False), (fmt.inr_short(closing), True)]
    if growth is not None and not math.isnan(growth) and prev_ts is not None:
        word = "up" if growth >= 0 else "down"
        out += [(", " + word + " ", False), (fmt.pct(abs(growth)), True), (" on " + prev_ts.strftime("%B") + ". ", False)]
    else:
        out.append((". ", False))
    out += [("New customers added ", False), (fmt.inr(row["new"]), True),
            (" and upgrades ", False), (fmt.inr(row["expansion"]), True), (". ", False)]
    n = int(row["churned_customers"])
    if n == 0:
        out.append(("Nobody cancelled.", False))
    else:
        out += [(_plural(n, "customer") + " cancelled", False), (", taking ", False),
                (fmt.inr(row["churned"]), True), (" a month with them.", False)]
    return out


def build_funnel(stages):
    """[('Free trials', 2200), ('Became paying', 539), ('Still paying', 404)] -> bar rows."""
    top = stages[0][1] or 1
    rows = []
    for i, (label, count) in enumerate(stages):
        if i == 0:
            note = ""
        else:
            prev = stages[i - 1][1] or 1
            share = count / prev
            note = (f"{fmt.pct(share, 0)} of trials" if i == 1
                    else f"{fmt.pct(share, 0)} of those who paid")
        rows.append(dict(label=label, count=count, note=note, width=round(count / top * 100, 2)))
    return rows


def verdict(ltv_cac):
    """Plain-words reading of LTV:CAC."""
    if ltv_cac != ltv_cac:
        return "Not enough data"
    if ltv_cac >= 3:
        return "Pays back well"
    if ltv_cac >= 1:
        return "Marginal"
    return "Loses money"
