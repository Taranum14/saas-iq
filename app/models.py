from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from . import db

ROLES = ("analyst", "manager", "admin")


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(16), nullable=False, default="analyst")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Customer(db.Model):
    """One row per account that started a free trial."""

    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    channel = db.Column(db.String(32), nullable=False, index=True)
    trial_start = db.Column(db.Date, nullable=False)
    converted_on = db.Column(db.Date)  # null = never became a paying customer
    churned_on = db.Column(db.Date)  # null = still active (or never paid)
    plan = db.Column(db.String(16))  # latest paid plan
    acquisition_cost = db.Column(db.Float, nullable=False, default=0)

    months = db.relationship(
        "SubscriptionMonth", backref="customer", cascade="all, delete-orphan"
    )
    scores = db.relationship("ChurnScore", cascade="all, delete-orphan")


class SubscriptionMonth(db.Model):
    """MRR snapshot: one row per customer per month in which they paid."""

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(
        db.Integer, db.ForeignKey("customer.id"), nullable=False, index=True
    )
    month = db.Column(db.Date, nullable=False, index=True)  # first day of month
    plan = db.Column(db.String(16), nullable=False)
    mrr = db.Column(db.Integer, nullable=False)
    logins = db.Column(db.Integer, default=0)  # product usage that month
    tickets = db.Column(db.Integer, default=0)  # support tickets raised that month


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    username = db.Column(db.String(64))
    action = db.Column(db.String(32))
    detail = db.Column(db.String(255))


class ChurnScore(db.Model):
    """Model output: probability each active customer churns next month."""

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customer.id"), nullable=False, index=True)
    month = db.Column(db.Date, nullable=False)  # month the score was made for
    probability = db.Column(db.Float, nullable=False)
    reasons = db.Column(db.String(255))  # top drivers, plain text
