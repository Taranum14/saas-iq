"""Train the churn model, save it, and store the latest scores.

Run:  python -m app.ml.train
"""
import json
import os

import joblib
import pandas as pd

from .. import db
from ..models import ChurnScore, Customer, SubscriptionMonth
from . import churn

MODEL_FILE = "churn_model.joblib"
METRICS_FILE = "churn_metrics.json"


def _model_dir(app):
    return app.config.get("MODEL_DIR") or app.instance_path


def load_frames():
    mrr_df = pd.read_sql(db.select(SubscriptionMonth), db.engine).drop(columns="id")
    customers_df = pd.read_sql(db.select(Customer), db.engine)
    return mrr_df, customers_df


def train_and_store():
    from flask import current_app

    mrr_df, customers_df = load_frames()
    artifact = churn.train(customers_df, mrr_df)
    scores = churn.score_latest(artifact, customers_df, mrr_df)

    out_dir = _model_dir(current_app)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifact, os.path.join(out_dir, MODEL_FILE))
    with open(os.path.join(out_dir, METRICS_FILE), "w") as f:
        json.dump(artifact["metrics"], f, indent=2)

    ChurnScore.query.delete()
    rows = [
        dict(
            customer_id=int(r.customer_id),
            month=pd.Timestamp(r.month).date(),
            probability=float(r.probability),
            reasons=r.reasons[:255],
        )
        for r in scores.itertuples()
    ]
    db.session.bulk_insert_mappings(ChurnScore, rows)
    db.session.commit()

    m = artifact["metrics"]["selected_test"]
    print(
        f"Churn model: {artifact['metrics']['selected_model']} | test PR-AUC {m['pr_auc']:.3f} "
        f"(base rate {m['base_rate']:.3f}) | ROC-AUC {m['roc_auc']:.3f} | "
        f"top-decile lift {m['top_decile_lift']:.1f}x | scored {len(rows)} active customers."
    )
    return artifact["metrics"]


def load_metrics(app):
    path = os.path.join(_model_dir(app), METRICS_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from .. import create_app

    with create_app().app_context():
        train_and_store()
