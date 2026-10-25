import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me-in-production")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///" + os.path.join(INSTANCE_DIR, "saas.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GROSS_MARGIN = 0.80  # used for LTV
    MODEL_DIR = INSTANCE_DIR  # where the trained churn model is saved
