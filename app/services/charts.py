"""Plotly figure builders.

Colours are written as tokens ("@ink", "@c1", ...). The browser swaps each token for
the CSS variable of the same name before drawing (static/app.js), so the charts follow
the page's light/dark theme and share one palette with the rest of the interface.
"""
import json
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import fmt

FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"

# Plotly validates colour strings, so each token is built as a unique placeholder hex and
# swapped back to "@token" in the JSON that goes to the browser (see _to_dict).
_TOKENS = ["ink", "ink2", "muted", "rule", "rule-strong", "paper", "gain", "loss", "neutral", "band",
           "c1", "c2", "c3", "c4", "c5", "seq0", "seq1", "seq2", "seq3", "seq4"]
_PLACEHOLDER = {name: f"#a0{i:04x}" for i, name in enumerate(_TOKENS, start=1)}


def T(name):
    return _PLACEHOLDER[name]


# Colour follows the entity: a channel keeps its hue on every chart and every filter.
CHANNEL_COLOR = {
    "Referral": "c1",
    "Organic / SEO": "c2",
    "Outbound Sales": "c3",
    "LinkedIn Ads": "c4",
    "Google Ads": "c5",
}
_FALLBACK = ["c1", "c2", "c3", "c4", "c5"]


def _slot(name):
    if name in CHANNEL_COLOR:
        return CHANNEL_COLOR[name]
    return _FALLBACK[sum(map(ord, name)) % len(_FALLBACK)]


def channel_color(name):
    return T(_slot(name))


def channel_var(name):
    """CSS variable name for a channel colour, e.g. '--c1' (used by the table colour keys)."""
    return "--" + _slot(name)


def _to_dict(fig):
    raw = fig.to_json()
    for name, hex_value in _PLACEHOLDER.items():
        raw = raw.replace(f'"{hex_value}"', f'"@{name}"')
    return json.loads(raw)


def _base(**overrides):
    layout = dict(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=T("ink2")),
        hoverlabel=dict(bgcolor=T("paper"), bordercolor=T("rule"), font=dict(family=FONT, size=12, color=T("ink"))),
        legend=dict(orientation="h", x=0, y=1.1, xanchor="left", yanchor="bottom", font=dict(size=12, color=T("ink2"))),
    )
    layout.update(overrides)
    return layout


def _x(**kw):
    d = dict(showgrid=False, zeroline=False, linecolor=T("rule"), ticks="outside", tickcolor=T("rule"),
             tickfont=dict(color=T("muted")), automargin=True)
    d.update(kw)
    return d


def _y(**kw):
    d = dict(gridcolor=T("rule"), gridwidth=1, zeroline=False, showline=False, ticks="",
             tickfont=dict(color=T("muted")), automargin=True)
    d.update(kw)
    return d


def _nice_step(raw):
    if raw <= 0:
        return 1
    exp = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * exp:
            return m * exp
    return 10 * exp


def inr_ticks(lo, hi, n=4):
    step = _nice_step((hi - lo) / n)
    start = math.floor(lo / step) * step
    vals = []
    v = start
    while v <= hi + step * 0.001:
        vals.append(v)
        v += step
    return dict(tickmode="array", tickvals=vals, ticktext=[fmt.inr_axis(x) for x in vals])


# ------------------------------------------------------------------ overview
def history_chart(wf):
    """MRR on top, the month's gains and losses underneath. Two plots sharing one time axis,
    never two scales on one plot."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36], vertical_spacing=0.07)
    x = wf["month"]
    gains = wf["new"] + wf["expansion"]
    losses = -(wf["contraction"] + wf["churned"])

    fig.add_scatter(
        x=x, y=wf["closing"], name="MRR", mode="lines", line=dict(color=T("ink"), width=2),
        hovertemplate="%{x|%b %Y}<br>MRR ₹%{y:,.0f}<extra></extra>", row=1, col=1,
    )
    fig.add_scatter(
        x=[x.iloc[-1]], y=[wf["closing"].iloc[-1]], mode="markers+text", showlegend=False, hoverinfo="skip",
        marker=dict(color=T("ink"), size=9, line=dict(color=T("paper"), width=2)),
        text=[fmt.inr_short(wf["closing"].iloc[-1])], textposition="middle right", cliponaxis=False,
        textfont=dict(color=T("ink"), size=12),
        row=1, col=1,
    )
    bar_edge = dict(color=T("paper"), width=2)
    fig.add_bar(x=x, y=gains, name="Gained", marker=dict(color=T("gain"), line=bar_edge),
                hovertemplate="%{x|%b %Y}<br>Gained ₹%{y:,.0f}<extra></extra>", row=2, col=1)
    fig.add_bar(x=x, y=losses, name="Lost", marker=dict(color=T("loss"), line=bar_edge),
                hovertemplate="%{x|%b %Y}<br>Lost ₹%{customdata:,.0f}<extra></extra>",
                customdata=-losses, row=2, col=1)

    lo, hi = float(wf["closing"].min()), float(wf["closing"].max())
    pad = (hi - lo) * 0.1 or 1
    fig.update_layout(
        **_base(margin=dict(l=8, r=96, t=8, b=8), barmode="relative", bargap=0.35),
        xaxis=_x(showticklabels=False, ticks=""),
        xaxis2=_x(tickformat="%b %Y", nticks=7),
        yaxis=_y(range=[lo - pad, hi + pad], **inr_ticks(lo - pad, hi + pad, 3)),
        yaxis2=_y(**inr_ticks(float(losses.min()) * 1.15, float(gains.max()) * 1.15, 2)),
    )
    return _to_dict(fig)


def ltv_cac_chart(ue):
    d = ue.sort_values("ltv_cac", ascending=False)
    top = float(d["ltv_cac"].max())
    fig = go.Figure(
        go.Bar(
            x=d["ltv_cac"], y=d["channel"], orientation="h", width=0.42,
            marker=dict(color=[channel_color(c) for c in d["channel"]]),
            text=[fmt.times(v) for v in d["ltv_cac"]], textposition="outside", cliponaxis=False,
            textfont=dict(color=T("ink"), size=13),
            hovertemplate="%{y}<br>LTV:CAC %{x:.1f}×<extra></extra>",
        )
    )
    fig.add_vline(x=3, line=dict(color=T("rule-strong"), width=1))
    fig.add_annotation(x=3, y=1, yref="paper", yanchor="bottom", xanchor="left", showarrow=False,
                       text=" 3× is the usual bar", font=dict(color=T("muted"), size=12))
    fig.update_layout(
        **_base(margin=dict(l=8, r=40, t=24, b=8), showlegend=False),
        xaxis=_x(range=[0, top * 1.18], showticklabels=False, ticks="", showline=False),
        yaxis=dict(autorange="reversed", showgrid=False, ticks="", tickfont=dict(color=T("ink"), size=13), automargin=True),
    )
    return _to_dict(fig)


def cohort_heatmap(cohorts):
    z = cohorts.drop(columns="size").to_numpy(dtype=float) * 100
    zmin = float(np.floor(np.nanmin(z) / 10) * 10)  # start the scale where the data starts, so differences show
    sizes = np.repeat(cohorts["size"].to_numpy()[:, None], z.shape[1], axis=1)
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(range(z.shape[1])), y=list(cohorts.index), customdata=sizes,
            colorscale=[[0, T("seq0")], [0.25, T("seq1")], [0.5, T("seq2")], [0.75, T("seq3")], [1, T("seq4")]],
            zmin=zmin, zmax=100, xgap=2, ygap=2, hoverongaps=False,
            colorbar=dict(title=dict(text="% still paying", side="right", font=dict(size=12, color=T("ink2"))),
                          thickness=10, len=0.6, outlinewidth=0, tickfont=dict(color=T("muted"), size=11),
                          ticksuffix="%"),
            hovertemplate=("Started paying in %{y} (%{customdata} customers)"
                           "<br>%{z:.0f}% still paying after %{x} months<extra></extra>"),
        )
    )
    fig.update_layout(
        **_base(margin=dict(l=8, r=8, t=8, b=8), showlegend=False),
        xaxis=_x(title=dict(text="Months since first payment", font=dict(color=T("muted"), size=12)),
                 side="bottom", dtick=3, ticks=""),
        yaxis=dict(autorange="reversed", showgrid=False, ticks="", tickfont=dict(color=T("muted"), size=11),
                   automargin=True, dtick=2),
    )
    return _to_dict(fig)


# ------------------------------------------------------------------ insights
def survival_chart(curves):
    fig = go.Figure()
    for name, km in curves.items():
        fig.add_scatter(
            x=km["k"], y=km["survival"] * 100, name=name, mode="lines",
            line=dict(shape="hv", width=2, color=channel_color(name)),
            hovertemplate=name + "<br>after %{x} months: %{y:.0f}% still paying<extra></extra>",
        )
    fig.update_layout(
        **_base(margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified"),
        xaxis=_x(title=dict(text="Months since first payment", font=dict(color=T("muted"), size=12))),
        yaxis=_y(range=[0, 102], ticksuffix="%", dtick=25),
    )
    return _to_dict(fig)


def churn_anomaly_chart(wf, flagged_months):
    rate = (wf["churned_customers"] / wf["opening_customers"].replace(0, np.nan)) * 100
    fig = go.Figure()
    fig.add_scatter(x=wf["month"], y=rate, name="Customers lost", mode="lines",
                    line=dict(color=T("ink"), width=2),
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}% of customers left<extra></extra>")
    hit = wf["month"].isin(flagged_months)
    fig.add_scatter(x=wf.loc[hit, "month"], y=rate[hit], name="Flagged month", mode="markers",
                    marker=dict(color=T("loss"), size=11, symbol="diamond", line=dict(color=T("paper"), width=2)),
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}% left. Flagged<extra></extra>")
    for m, v in list(zip(wf.loc[hit, "month"], rate[hit]))[:3]:
        fig.add_annotation(x=m, y=v, text=f"{m.strftime('%b %Y')}: {v:.1f}%", showarrow=False, yshift=18,
                           font=dict(color=T("ink"), size=12))
    fig.update_layout(
        **_base(margin=dict(l=8, r=8, t=8, b=8)),
        xaxis=_x(tickformat="%b %Y", nticks=7),
        yaxis=_y(ticksuffix="%", rangemode="tozero"),
    )
    return _to_dict(fig)


def forecast_chart(wf, fc, history=18):
    h = wf.tail(history)
    last_x, last_y = h["month"].iloc[-1], h["closing"].iloc[-1]
    bx = [last_x] + list(fc["month"])
    fig = go.Figure()
    fig.add_scatter(x=bx, y=[last_y] + list(fc["p90"]), mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=bx, y=[last_y] + list(fc["p10"]), mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=T("band"), name="Likely range (P10 to P90)", hoverinfo="skip")
    fig.add_scatter(x=h["month"], y=h["closing"], mode="lines", name="Actual MRR",
                    line=dict(color=T("ink"), width=2),
                    hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra>Actual</extra>")
    fig.add_scatter(x=bx, y=[last_y] + list(fc["forecast"]), mode="lines", name="Forecast",
                    line=dict(color=T("c1"), width=2, dash="dash"),
                    hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra>Forecast</extra>")
    fig.add_scatter(x=[fc["month"].iloc[-1]], y=[fc["forecast"].iloc[-1]], mode="markers+text", showlegend=False,
                    hoverinfo="skip", marker=dict(color=T("c1"), size=9, line=dict(color=T("paper"), width=2)),
                    text=[fmt.inr_short(fc["forecast"].iloc[-1])], textposition="middle right", cliponaxis=False,
                    textfont=dict(color=T("ink"), size=12))
    lo = float(min(h["closing"].min(), fc["p10"].min()))
    hi = float(max(h["closing"].max(), fc["p90"].max()))
    pad = (hi - lo) * 0.08
    fig.update_layout(
        **_base(margin=dict(l=8, r=96, t=8, b=8)),
        xaxis=_x(tickformat="%b %Y", nticks=8),
        yaxis=_y(range=[lo - pad, hi + pad], **inr_ticks(lo - pad, hi + pad, 4)),
    )
    return _to_dict(fig)


def backtest_chart(by_horizon):
    fig = go.Figure()
    cycle = iter([T("c1"), T("c2"), T("c3")])
    for col in by_horizon.columns:
        color = T("neutral") if "naive" in str(col).lower() or "flat" in str(col).lower() else next(cycle)
        fig.add_bar(x=[f"{h} mo" for h in by_horizon.index], y=by_horizon[col], name=str(col),
                    marker=dict(color=color, line=dict(color=T("paper"), width=2)),
                    hovertemplate="%{x} ahead<br>" + str(col) + ": %{y:.1f}% off<extra></extra>")
    fig.update_layout(
        **_base(margin=dict(l=8, r=8, t=8, b=8), barmode="group", bargap=0.4, bargroupgap=0.05),
        xaxis=_x(title=dict(text="How far ahead", font=dict(color=T("muted"), size=12))),
        yaxis=_y(ticksuffix="%", rangemode="tozero"),
    )
    return _to_dict(fig)


def decile_chart(rates, base_rate):
    n = len(rates)
    fig = go.Figure(
        go.Bar(x=[str(i) for i in range(1, n + 1)], y=[r * 100 for r in rates], width=0.55,
               marker=dict(color=[T("c1")] + [T("neutral")] * (n - 1)),
               hovertemplate="Risk group %{x} of " + str(n) + "<br>%{y:.1f}% actually churned<extra></extra>")
    )
    fig.add_hline(y=base_rate * 100, line=dict(color=T("rule-strong"), width=1))
    fig.add_annotation(x=1, xref="paper", y=base_rate * 100, xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"Average {base_rate * 100:.1f}%", font=dict(color=T("muted"), size=12))
    fig.update_layout(
        **_base(margin=dict(l=8, r=8, t=8, b=8), showlegend=False),
        xaxis=_x(title=dict(text="Risk group (1 is the model's riskiest tenth)", font=dict(color=T("muted"), size=12))),
        yaxis=_y(ticksuffix="%", rangemode="tozero"),
    )
    return _to_dict(fig)


def importance_chart(importance):
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", width=0.5,
                           marker=dict(color=T("c1")),
                           hovertemplate="%{y}<br>accuracy drops by %{x:.3f} when shuffled<extra></extra>"))
    fig.update_layout(
        **_base(margin=dict(l=8, r=8, t=8, b=8), showlegend=False),
        xaxis=_x(title=dict(text="Drop in accuracy when shuffled", font=dict(color=T("muted"), size=12))),
        yaxis=dict(showgrid=False, ticks="", tickfont=dict(color=T("ink"), size=12), automargin=True),
    )
    return _to_dict(fig)
