import os

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./churn.db",
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    support_calls = Column(Integer, nullable=False)
    contract_type = Column(String(50), nullable=False)
    payment_method = Column(String(50), nullable=False)
    internet_service = Column(String(50), nullable=False)
    paperless_billing = Column(Boolean, nullable=False)
    senior_citizen = Column(Boolean, nullable=False)
    churn_prediction = Column(Boolean, nullable=False)
    churn_probability = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
