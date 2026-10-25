"""Synthetic SaaS data generator.

Real subscription data is private, so this builds a realistic, slightly messy dataset:

* five acquisition channels with different conversion, churn and cost profiles
* higher churn in the first months, occasional upgrades and downgrades
* MONTHLY usage (logins, support tickets). Most churners show warning signs in the
  3 months before they leave, but ~30% churn "silently" with no warning, so a
  model cannot reach perfect accuracy.
* one injected churn SHOCK (Google Ads customers churn 6x more in a single month,
  e.g. a billing incident),
  so anomaly detection can be validated against a known ground truth.

Run from the project root:   python -m data.seed_data
"""
import numpy as np
import pandas as pd
from faker import Faker

PLANS = {"Starter": 999, "Pro": 2999, "Business": 7999}  # INR per month
PLAN_LIST = list(PLANS)

# conversion = trial -> paid, churn_mult = scales monthly churn,
# cac = average spend to bring in one *trial* (INR)
CHANNELS = {
    "Google Ads": dict(share=0.30, conversion=0.17, churn_mult=1.25, cac=4500),
    "LinkedIn Ads": dict(share=0.15, conversion=0.22, churn_mult=0.90, cac=11000),
    "Organic / SEO": dict(share=0.25, conversion=0.24, churn_mult=0.85, cac=3000),
    "Referral": dict(share=0.15, conversion=0.38, churn_mult=0.70, cac=6500),
    "Outbound Sales": dict(share=0.15, conversion=0.30, churn_mult=0.70, cac=18000),
}
PLAN_WEIGHTS = {  # channel -> (Starter, Pro, Business)
    "Google Ads": (0.62, 0.32, 0.06),
    "LinkedIn Ads": (0.30, 0.50, 0.20),
    "Organic / SEO": (0.55, 0.35, 0.10),
    "Referral": (0.40, 0.45, 0.15),
    "Outbound Sales": (0.15, 0.45, 0.40),
}
BASE_CHURN = {"Starter": 0.045, "Pro": 0.028, "Business": 0.016}  # monthly
BASE_LOGINS = {"Starter": 7, "Pro": 13, "Business": 22}  # per month
SHOCK_CHANNEL = "Google Ads"
SHOCK_MULT = 6.0
SILENT_CHURN_SHARE = 0.30


# The demo data is generated "as of" a fixed date so the numbers in the README and on the
# pages are the same on every machine and every month. Pass `today=` to move the window.
DEMO_TODAY = "2026-10-25"


def make_months(n_months=24, today=None):
    today = pd.Timestamp(today if today is not None else DEMO_TODAY)
    last_month = today.to_period("M") - 1  # last complete month
    return pd.period_range(end=last_month, periods=n_months, freq="M")


def shock_period(n_months=24, today=None):
    """The month in which the injected churn shock happens (ground truth for tests)."""
    return make_months(n_months, today)[-6]


def generate(n_trials=4000, n_months=24, seed=42, today=None):
    rng = np.random.default_rng(seed)
    Faker.seed(seed)
    fake = Faker("en_IN")

    months = make_months(n_months, today)
    last_month = months[-1]
    shock_month = months[-6]

    # more signups in recent months = a growing company
    weights = 1 + 0.07 * np.arange(n_months)
    weights = weights / weights.sum()

    channel_names = list(CHANNELS)
    channel_probs = [CHANNELS[c]["share"] for c in channel_names]

    customers, sub_rows = [], []
    for cid in range(1, n_trials + 1):
        ch = str(rng.choice(channel_names, p=channel_probs))
        cfg = CHANNELS[ch]
        start_month = months[rng.choice(n_months, p=weights)]
        day = int(rng.integers(1, 28))
        trial_start = start_month.to_timestamp() + pd.Timedelta(days=day - 1)
        cost = float(cfg["cac"] * rng.lognormal(0, 0.25))

        row = dict(
            id=cid,
            company=fake.company(),
            email=fake.company_email(),
            channel=ch,
            trial_start=trial_start.date(),
            converted_on=None,
            churned_on=None,
            plan=None,
            acquisition_cost=round(cost, 2),
        )
        customers.append(row)

        if rng.random() >= cfg["conversion"]:
            continue
        conv_date = trial_start + pd.Timedelta(days=14)
        conv_month = conv_date.to_period("M")
        if conv_month > last_month:
            continue  # still in trial at the end of the data
        row["converted_on"] = conv_date.date()

        # ---- phase 1: simulate the paid lifetime month by month
        plan = str(rng.choice(PLAN_LIST, p=PLAN_WEIGHTS[ch]))
        life = []  # (period, plan)
        m, tenure, churn_period = conv_month, 0, None
        while m <= last_month:
            if tenure > 0:
                early = 1.8 if tenure < 3 else 1.0
                p_churn = BASE_CHURN[plan] * cfg["churn_mult"] * early
                if ch == SHOCK_CHANNEL and m == shock_month:
                    p_churn *= SHOCK_MULT
                if rng.random() < p_churn:
                    churn_period = m
                    break
                r = rng.random()
                if r < 0.035 and plan != "Business":
                    plan = PLAN_LIST[PLAN_LIST.index(plan) + 1]
                elif r > 0.985 and plan != "Starter":
                    plan = PLAN_LIST[PLAN_LIST.index(plan) - 1]
            life.append((m, plan))
            tenure += 1
            m += 1
        row["plan"] = plan
        if churn_period is not None:
            row["churned_on"] = churn_period.to_timestamp().date()

        # ---- phase 2: monthly usage, with warning signs before most churns
        engagement = rng.lognormal(0, 0.3)
        silent = rng.random() < SILENT_CHURN_SHARE
        n_rows = len(life)
        for i, (period, p) in enumerate(life):
            to_churn = (n_rows - i) if churn_period is not None else None  # 1 = last paid month
            factor, ticket_rate = 1.0, 0.4
            if to_churn is not None and to_churn <= 3 and not silent:
                factor = {1: 0.30, 2: 0.55, 3: 0.80}[to_churn]
                ticket_rate += {1: 1.6, 2: 1.0, 3: 0.5}[to_churn]
            if rng.random() < 0.08:  # random dips happen to healthy customers too
                factor *= 0.5
            logins = int(rng.poisson(BASE_LOGINS[p] * engagement * factor))
            tickets = int(rng.poisson(ticket_rate))
            sub_rows.append(
                dict(
                    customer_id=cid,
                    month=period.to_timestamp().date(),
                    plan=p,
                    mrr=PLANS[p],
                    logins=logins,
                    tickets=tickets,
                )
            )

    return pd.DataFrame(customers), pd.DataFrame(sub_rows)


def seed_db(reset=True, train=True, **kwargs):
    from app import create_app, db
    from app.models import AuditLog, Customer, SubscriptionMonth, User

    app = create_app()
    with app.app_context():
        if reset:
            db.drop_all()
            db.create_all()

        customers_df, subs_df = generate(**kwargs)
        db.session.bulk_insert_mappings(
            Customer,
            customers_df.astype(object).where(customers_df.notna(), None).to_dict("records"),
        )
        db.session.bulk_insert_mappings(SubscriptionMonth, subs_df.to_dict("records"))

        for username, role in [("analyst", "analyst"), ("manager", "manager"), ("admin", "admin")]:
            u = User(username=username, role=role)
            u.set_password("demo123")
            db.session.add(u)
        db.session.add(AuditLog(username="system", action="seed", detail="Synthetic data generated"))
        db.session.commit()
        print(
            f"Seeded {len(customers_df)} trials, {len(subs_df)} subscription-months, 3 demo users."
        )

        if train:
            from app.ml.train import train_and_store

            train_and_store()


if __name__ == "__main__":
    seed_db()
