import pytest

from app import create_app, db
from app.models import Customer, User
from data.seed_data import generate


class TestConfig:
    SECRET_KEY = "test"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GROSS_MARGIN = 0.8
    TESTING = True


@pytest.fixture
def client():
    app = create_app(TestConfig)
    with app.app_context():
        from app.models import SubscriptionMonth

        customers_df, subs_df = generate(n_trials=400, n_months=12, seed=1, today="2026-10-23")
        db.session.bulk_insert_mappings(
            Customer, customers_df.astype(object).where(customers_df.notna(), None).to_dict("records")
        )
        db.session.bulk_insert_mappings(SubscriptionMonth, subs_df.to_dict("records"))
        for name in ("analyst", "admin"):
            u = User(username=name, role=name)
            u.set_password("demo123")
            db.session.add(u)
        db.session.commit()
    return app.test_client()


def login(client, user):
    return client.post("/login", data={"username": user, "password": "demo123"}, follow_redirects=True)


def test_dashboard_requires_login(client):
    r = client.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_bad_password(client):
    r = client.post("/login", data={"username": "analyst", "password": "nope"})
    assert b"Invalid username or password" in r.data


def test_analyst_sees_dashboard_but_not_admin(client):
    r = login(client, "analyst")
    assert b"How MRR moved" in r.data
    assert client.get("/admin/customers").status_code == 403


def test_admin_can_delete(client):
    login(client, "admin")
    assert client.get("/admin/customers").status_code == 200
    r = client.post("/admin/customers/1/delete", follow_redirects=True)
    assert r.status_code == 200 and b"Deleted" in r.data
