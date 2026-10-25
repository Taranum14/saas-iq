from functools import lru_cache, wraps

import numpy as np
import pandas as pd
from flask import (Blueprint, Response, abort, current_app, flash, jsonify, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required, login_user, logout_user

from . import db
from .ml.train import load_metrics
from .models import AuditLog, ChurnScore, Customer, SubscriptionMonth, User
from .services import anomaly, charts, fmt, metrics, present, retention, survival
from .services import forecast as forecast_svc

auth_bp = Blueprint("auth", __name__)
main_bp = Blueprint("main", __name__)


def role_required(*roles):
    def deco(fn):
        @wraps(fn)
        @login_required
        def wrapper(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)
            return fn(*args, **kwargs)

        return wrapper

    return deco


# ---------------------------------------------------------------- auth
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user)
            return redirect(request.args.get("next") or url_for("main.dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


# ----------------------------------------------------------- dashboard
def _load_frames():
    mrr_df = pd.read_sql(
        db.select(SubscriptionMonth.customer_id, SubscriptionMonth.month, SubscriptionMonth.mrr),
        db.engine,
    )
    customers_df = pd.read_sql(db.select(Customer), db.engine)
    return mrr_df, customers_df


@lru_cache(maxsize=1)
def _plotly_source():
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


@main_bp.route("/vendor/plotly.js")
def plotly_js():
    """Serve the chart library from the installed plotly package, so the app works offline."""
    resp = Response(_plotly_source(), mimetype="application/javascript")
    resp.cache_control.public = True
    resp.cache_control.max_age = 60 * 60 * 24 * 30
    return resp


@main_bp.route("/")
@login_required
def dashboard():
    mrr_df, customers_df = _load_frames()
    if mrr_df.empty:
        return render_template("empty.html")

    wf = metrics.mrr_waterfall(mrr_df)
    months = [m.strftime("%Y-%m") for m in wf["month"]]
    selected = request.args.get("month", months[-1])
    if selected not in months:
        selected = months[-1]
    idx = months.index(selected)
    row = wf.iloc[idx]
    k = metrics.kpis(wf.iloc[: idx + 1])  # figures follow the month being shown
    prev_ts = wf["month"].iloc[idx - 1] if idx > 0 else None

    funnel = metrics.trial_funnel(customers_df)
    margin = current_app.config["GROSS_MARGIN"]
    ue = metrics.unit_economics(customers_df, mrr_df, margin)
    cohorts = metrics.cohort_retention(mrr_df)

    table = funnel["by_channel"].merge(ue, on="channel", how="left").to_dict("records")
    for r in table:
        r["color"] = charts.channel_var(r["channel"])
        r["verdict"] = present.verdict(r["ltv_cac"])

    history_rows = [
        dict(month=m.strftime("%b %Y"), closing=c, gained=n + e, lost=-(ct + ch))
        for m, c, n, e, ct, ch in zip(wf["month"], wf["closing"], wf["new"], wf["expansion"],
                                      wf["contraction"], wf["churned"])
    ][::-1]

    return render_template(
        "dashboard.html",
        k=k,
        lead=present.build_lead(row["month"], prev_ts, row["closing"], k["mom_growth"], row),
        month_name=row["month"].strftime("%B %Y"),
        month_options=[(m, t.strftime("%B %Y")) for m, t in zip(months, wf["month"])],
        selected=selected,
        bridge=present.build_bridge(row),
        funnel=present.build_funnel(funnel["stages"]),
        fig_history=charts.history_chart(wf),
        fig_cohort=charts.cohort_heatmap(cohorts),
        fig_ltv=charts.ltv_cac_chart(ue),
        channel_table=table,
        history_rows=history_rows,
        margin_pct=round(margin * 100),
    )


# ----------------------------------------------------------- insights
def _scores_frame():
    """Latest churn scores joined with customer details and current MRR."""
    scores = pd.read_sql(db.select(ChurnScore), db.engine)
    if scores.empty:
        return scores
    mrr_df, customers_df = _load_frames()
    month = pd.to_datetime(scores["month"].iloc[0])
    latest = mrr_df[pd.to_datetime(mrr_df["month"]) == month][["customer_id", "mrr"]]
    scores = scores.drop(columns=["id"], errors="ignore")  # keep customers' id as the only "id"
    return (
        scores.merge(customers_df[["id", "company", "channel", "plan"]], left_on="customer_id", right_on="id")
        .merge(latest, on="customer_id")
        .drop(columns="id")
    )


def _num(name, default, lo, hi):
    try:
        v = float(request.args.get(name, default))
    except (TypeError, ValueError):
        v = default
    return min(max(v, lo), hi)


@main_bp.route("/retention")
@role_required("manager", "admin")
def retention_page():
    m = load_metrics(current_app)
    df = _scores_frame()
    if m is None or df.empty:
        return render_template("retention.html", ready=False)

    save_rate = _num("save_rate", 30, 1, 100) / 100
    cost = _num("cost", 500, 0, 1e6)
    months_kept = int(_num("months_kept", 6, 1, 36))
    budget = _num("budget", 0, 0, 1e9) or None

    calls, summary = retention.build_call_list(
        df, save_rate=save_rate, cost=cost, months_kept=months_kept,
        gross_margin=current_app.config["GROSS_MARGIN"], budget=budget,
    )
    sel = m["selected_test"]
    return render_template(
        "retention.html",
        ready=True, m=m, sel=sel, s=summary,
        form=dict(save_rate=round(save_rate * 100), cost=int(cost), months_kept=months_kept,
                  budget=int(budget) if budget else ""),
        calls=calls.head(50).to_dict("records"),
        n_calls=len(calls),
        fig_deciles=charts.decile_chart(sel["decile_churn_rates"], sel["base_rate"]),
        fig_importance=charts.importance_chart(m["importance"]),
    )


@main_bp.route("/forecast")
@login_required
def forecast_page():
    mrr_df, _ = _load_frames()
    if mrr_df.empty:
        return render_template("empty.html")
    wf = metrics.mrr_waterfall(mrr_df)
    fc = forecast_svc.forecast(wf)
    bt = forecast_svc.backtest(wf)
    return render_template(
        "forecast.html",
        fig_forecast=charts.forecast_chart(wf, fc),
        fig_backtest=charts.backtest_chart(bt["by_horizon"]),
        summary=bt["summary"].to_dict("records"),
        fc=fc.to_dict("records"),
        n_origins=bt["n_origins"],
        current=float(wf["closing"].iloc[-1]),
    )


@main_bp.route("/insights")
@login_required
def insights():
    mrr_df, customers_df = _load_frames()
    if mrr_df.empty:
        return render_template("empty.html")
    wf = metrics.mrr_waterfall(mrr_df)

    # survival by channel
    durations = survival.build_durations(customers_df, mrr_df)
    curves = survival.survival_by(durations, "channel")
    surv_rows = []
    for ch, km in curves.items():
        g = durations[durations["channel"] == ch]

        def at(k):
            return float(km.loc[km["k"] == k, "survival"].iloc[0]) if k in set(km["k"]) else float("nan")

        surv_rows.append(dict(channel=ch, color=charts.channel_var(ch), customers=len(g), churned=int(g["event"].sum()),
                              s6=at(6), s12=at(12), median=survival.median_lifetime(km)))
    surv_rows.sort(key=lambda r: -(r["s12"] if r["s12"] == r["s12"] else 0))

    # anomalies + root cause
    flags = anomaly.detect(wf)
    churn_flags = flags[flags["metric"] == "Logo churn rate"].sort_values("month", ascending=False)
    rca = []
    for _, f in churn_flags.head(3).iterrows():
        rca.append(dict(month=f["month"], rate=f["value"], z=f["z"],
                        rows=[dict(r, color=charts.channel_var(r["channel"]))
                              for r in anomaly.explain_month(mrr_df, customers_df, f["month"]).to_dict("records")]))
    return render_template(
        "insights.html",
        fig_surv=charts.survival_chart(curves),
        fig_anom=charts.churn_anomaly_chart(wf, set(churn_flags["month"])),
        surv_rows=surv_rows,
        rca=rca,
        other_flags=flags[flags["metric"] != "Logo churn rate"].to_dict("records"),
    )


# ---------------------------------------------------------------- API
@main_bp.route("/health")
def health():
    return jsonify(status="ok")


@main_bp.route("/api/churn-risk")
@role_required("manager", "admin")
def api_churn_risk():
    """Customers ranked by churn probability. ?limit=20&min_prob=0.1"""
    limit = int(_num("limit", 20, 1, 200))
    min_prob = _num("min_prob", 0, 0, 1)
    df = _scores_frame()
    m = load_metrics(current_app)
    if df.empty or m is None:
        return jsonify(error="model not trained; run python -m app.ml.train"), 503
    df = df[df["probability"] >= min_prob].sort_values("probability", ascending=False).head(limit)
    return jsonify(
        model=m["selected_model"],
        trained_at=m["trained_at"],
        score_month=str(df["month"].iloc[0]) if len(df) else None,
        count=len(df),
        customers=[
            dict(customer_id=int(r.customer_id), company=r.company, plan=r.plan, channel=r.channel,
                 mrr=int(r.mrr), churn_probability=round(float(r.probability), 4), reasons=r.reasons)
            for r in df.itertuples()
        ],
    )


# --------------------------------------------------------------- admin
@main_bp.route("/admin/customers")
@role_required("admin")
def admin_customers():
    page = request.args.get("page", 1, type=int)
    pagination = Customer.query.order_by(Customer.id.desc()).paginate(page=page, per_page=25)
    logs = AuditLog.query.order_by(AuditLog.id.desc()).limit(10).all()
    return render_template("admin_customers.html", pagination=pagination, logs=logs)


@main_bp.route("/admin/customers/<int:cid>/delete", methods=["POST"])
@role_required("admin")
def delete_customer(cid):
    c = db.get_or_404(Customer, cid)
    db.session.add(
        AuditLog(username=current_user.username, action="delete", detail=f"Customer #{c.id} {c.company}")
    )
    db.session.delete(c)
    db.session.commit()
    flash(f"Deleted {c.company}.", "ok")
    return redirect(url_for("main.admin_customers"))


@main_bp.app_errorhandler(403)
def forbidden(_):
    return render_template("403.html"), 403
