"""Churn prediction: "will this paying customer cancel next month?"

Design choices that matter
* The training table is a customer-month PANEL. Each row only uses information that
  was available at the end of that month (no future leakage).
* Validation is TEMPORAL: train on early months, validate on the next ones, test on
  the most recent ones. Random splits would let the model peek at the future.
* Metrics suit a rare event: PR-AUC (vs the base rate), lift and capture in the top
  decile, Brier score for calibration. Plain accuracy would be ~97% for a model that
  predicts "nobody churns".
* Each score comes with plain-English reasons, from one-at-a-time ablation: replace a
  feature group with the typical value and see how much the risk drops.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PLANS = ("Starter", "Pro", "Business")
NUMERIC = ["tenure", "logins", "logins_prev3", "logins_ratio", "tickets", "tickets_3m", "plan_changed"]


# ------------------------------------------------------------------ data
def build_panel(customers_df, mrr_df):
    """One row per customer per paid month, with features and the next-month label.

    churn_next = 1 if the customer paid this month but not next month,
                 NaN for the latest month (outcome not yet known).
    """
    df = mrr_df.merge(customers_df[["id", "channel"]], left_on="customer_id", right_on="id")
    df = df.drop(columns="id")
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["customer_id", "month"]).reset_index(drop=True)
    df["mi"] = df["month"].dt.year * 12 + df["month"].dt.month

    g = df.groupby("customer_id")
    df["tenure"] = g.cumcount()
    df["logins_prev3"] = g["logins"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["logins_ratio"] = df["logins"] / df["logins_prev3"].clip(lower=1)
    df["tickets_3m"] = g["tickets"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    prev_plan = g["plan"].shift(1)
    df["plan_changed"] = (prev_plan.notna() & (prev_plan != df["plan"])).astype(int)

    next_mi = g["mi"].shift(-1)
    churn = (next_mi.isna() | (next_mi != df["mi"] + 1)).astype(float)
    churn[df["mi"] == df["mi"].max()] = np.nan
    df["churn_next"] = churn
    return df


def design_matrix(panel, channels):
    X = panel[NUMERIC].astype(float).copy()
    for p in PLANS:
        X[f"plan={p}"] = (panel["plan"] == p).astype(float)
    for c in channels:
        X[f"channel={c}"] = (panel["channel"] == c).astype(float)
    return X


def feature_groups(columns):
    groups = {c: [c] for c in NUMERIC}
    groups["plan"] = [c for c in columns if c.startswith("plan=")]
    groups["channel"] = [c for c in columns if c.startswith("channel=")]
    return groups


# ---------------------------------------------------------------- models
def _models():
    return {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000),
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0,
        ),
    }


def temporal_split(panel):
    """Month indices for train / validation / test, with 1-month gaps between them.

    A row's label depends on the NEXT month, so the month right before a boundary is
    dropped to keep outcomes from straddling the split.
    """
    ms = sorted(panel.loc[panel["churn_next"].notna(), "mi"].unique())
    if len(ms) < 12:
        raise ValueError("Need at least 12 labelled months of history to train.")
    return dict(
        train=ms[:-8], val=ms[-7:-4], test=ms[-3:], refit=ms[:-4],
    )


def evaluate(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    base = float(y.mean())
    order = np.argsort(-p)
    top = max(1, int(round(0.10 * len(y))))
    top_hits = float(y[order[:top]].sum())
    deciles = np.array_split(y[order], 10)
    return dict(
        n=int(len(y)),
        churners=int(y.sum()),
        base_rate=base,
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        brier_baseline=float(brier_score_loss(y, np.full(len(y), base))),
        top_decile_churn_rate=top_hits / top,
        top_decile_lift=(top_hits / top) / base if base else float("nan"),
        top_decile_capture=top_hits / y.sum() if y.sum() else float("nan"),
        decile_churn_rates=[float(d.mean()) for d in deciles],
    )


def train(customers_df, mrr_df):
    """Fit, select on validation, refit, evaluate once on the held-out test months."""
    panel = build_panel(customers_df, mrr_df)
    channels = sorted(panel["channel"].unique())
    X_all = design_matrix(panel, channels)
    labelled = panel["churn_next"].notna()
    split = temporal_split(panel)

    def rows(months):
        return labelled & panel["mi"].isin(months)

    tr, va, te, rf = (rows(split[k]) for k in ("train", "val", "test", "refit"))
    y = panel["churn_next"]

    # 1) model selection on validation months
    val_scores = {}
    for name, model in _models().items():
        model.fit(X_all[tr], y[tr])
        val_scores[name] = average_precision_score(y[va], model.predict_proba(X_all[va])[:, 1])
    best = max(val_scores, key=val_scores.get)

    # 2) refit the winner on train+val, evaluate once on test; also report both on test
    test_report = {}
    final = None
    for name, model in _models().items():
        model.fit(X_all[rf], y[rf])
        test_report[name] = evaluate(y[te], model.predict_proba(X_all[te])[:, 1])
        if name == best:
            final = model

    # 3) global importance on the test months (drop in PR-AUC when a column is shuffled)
    perm = permutation_importance(
        final, X_all[te], y[te], scoring="average_precision", n_repeats=5, random_state=0
    )
    importance = (
        pd.Series(perm.importances_mean, index=X_all.columns).sort_values(ascending=False).head(8)
    )

    baseline = X_all[rf].median()
    dummy_cols = [c for c in X_all.columns if "=" in c]
    baseline[dummy_cols] = X_all.loc[rf, dummy_cols].mean()

    def to_month(mi):
        return f"{(mi - 1) // 12}-{(mi - 1) % 12 + 1:02d}"

    metrics = dict(
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        selected_model=best,
        validation_pr_auc={k: float(v) for k, v in val_scores.items()},
        test=test_report,
        selected_test=test_report[best],
        importance={k: float(v) for k, v in importance.items()},
        windows={k: [to_month(min(v)), to_month(max(v))] for k, v in split.items()},
        n_features=int(X_all.shape[1]),
    )
    artifact = dict(model=final, columns=list(X_all.columns), channels=channels,
                    baseline=baseline, metrics=metrics)
    return artifact


# ------------------------------------------------------------- scoring
def _n(count, word):
    count = int(count)
    return f"{count} {word}" + ("" if count == 1 else "s")


def _describe(group, row):
    if group == "logins_ratio":
        return f"logins at {row['logins_ratio']:.0%} of prior 3-month average"
    if group == "logins":
        return f"only {_n(row['logins'], 'login')} this month"
    if group == "logins_prev3":
        return f"low usage history ({row['logins_prev3']:.0f} logins/month)"
    if group == "tickets":
        return f"{_n(row['tickets'], 'support ticket')} this month"
    if group == "tickets_3m":
        return f"{_n(row['tickets_3m'], 'support ticket')} in 3 months"
    if group == "tenure":
        return f"early in subscription (month {int(row['tenure']) + 1})"
    if group == "plan_changed":
        return "recent plan change"
    if group == "plan":
        return f"{row['plan']} plan"
    if group == "channel":
        return f"acquired via {row['channel']}"
    return group


def score_latest(artifact, customers_df, mrr_df, top_reasons=3):
    """Churn probability + reasons for every customer paying in the latest month."""
    panel = build_panel(customers_df, mrr_df)
    latest = panel[panel["mi"] == panel["mi"].max()].reset_index(drop=True)
    cols = artifact["columns"]
    X = design_matrix(latest, artifact["channels"])[cols]
    model = artifact["model"]
    p = model.predict_proba(X)[:, 1]

    contrib = {}
    for group, members in feature_groups(cols).items():
        Xa = X.copy()
        Xa[members] = artifact["baseline"][members].to_numpy()
        contrib[group] = p - model.predict_proba(Xa)[:, 1]  # risk added vs a typical customer
    contrib = pd.DataFrame(contrib)

    reasons = []
    for i in range(len(latest)):
        top = contrib.iloc[i].sort_values(ascending=False)
        top = top[top > 0.005].head(top_reasons)  # at least half a point of risk
        reasons.append(
            "; ".join(f"{_describe(g, latest.iloc[i])} (+{v * 100:.1f} pts)" for g, v in top.items())
            or "no strong risk signals"
        )
    out = latest[["customer_id", "month", "mrr", "plan", "channel"]].copy()
    out["probability"] = p
    out["reasons"] = reasons
    return out
