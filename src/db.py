"""Independent database configuration for quantylab-trainer.

The default backend is a project-local SQLite file. Set TRAINER_DATABASE_URL
or DATABASE_URL to use PostgreSQL, MySQL, or another SQLAlchemy backend.
"""

from contextlib import contextmanager
import os
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker


def _database_url() -> str:
    configured = os.getenv("TRAINER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if configured:
        return configured
    default_path = Path(__file__).resolve().parents[1] / "data" / "trainer.sqlite3"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{default_path}"


DATABASE_URL = _database_url()
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("TRAINER_SQL_ECHO", "0") == "1",
    future=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class _DatabaseCompat:
    @contextmanager
    def get_session(self):
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def get_df_from_result(result):
        rows = []
        for item in result:
            if isinstance(item, dict):
                rows.append(item)
                continue
            try:
                mapper = sa.inspect(item).mapper
                rows.append({column.key: getattr(item, column.key) for column in mapper.column_attrs})
            except sa.exc.NoInspectionAvailable:
                rows.append(dict(item._mapping) if hasattr(item, "_mapping") else dict(item))
        return pd.DataFrame(rows)

    @staticmethod
    def upsert(session, instance, **_kwargs):
        session.merge(instance)

    @staticmethod
    def insert_or_ignore(session, instance):
        try:
            session.merge(instance)
        except sa.exc.IntegrityError:
            session.rollback()

    @staticmethod
    def create_tables(base):
        base.metadata.create_all(engine)

    @staticmethod
    def get_last_date(model):
        with SessionLocal() as session:
            return session.execute(select(model.date).order_by(model.date.desc()).limit(1)).scalar()


db = _DatabaseCompat()
# Backward-compatible alias for modules that have not migrated yet.
psql = db


class _DataAccess:
    """Small compatibility layer for the feature loader's named datasets."""

    @staticmethod
    def _load(model, filters=(), start_date=None, end_date=None):
        with SessionLocal() as session:
            try:
                query = select(model).where(*filters)
                if hasattr(model, "date") and start_date:
                    query = query.where(model.date >= start_date)
                if hasattr(model, "date") and end_date:
                    query = query.where(model.date <= end_date)
                query = query.order_by(model.date.asc()) if hasattr(model, "date") else query
                return psql.get_df_from_result(session.execute(query).scalars().all())
            except sa.exc.SQLAlchemyError:
                return pd.DataFrame()

    def get_stock_market_day_candles(self, code, start_date, end_date):
        from .models import StockMarketDayCandle
        return self._load(StockMarketDayCandle, (StockMarketDayCandle.code == code,), start_date, end_date)

    def get_foreign_stock_market_day_candles(self, code, start_date, end_date):
        from .models import ForeignStockMarketDayCandle
        return self._load(ForeignStockMarketDayCandle, (ForeignStockMarketDayCandle.code == code,), start_date, end_date)

    def get_fx(self, code, start_date, end_date):
        from .models import Fx
        return self._load(Fx, (Fx.code == code,), start_date, end_date)

    def get_vix(self, start_date, end_date):
        from .models import Vix
        return self._load(Vix, (), start_date, end_date)

    def get_kospi_vix(self, start_date, end_date):
        from .models import KospiVix
        return self._load(KospiVix, (), start_date, end_date)

    def get_us_bond_10_year_yield(self, start_date, end_date):
        from .models import UsBond10YearYield
        return self._load(UsBond10YearYield, (), start_date, end_date)

    def get_kr_bond_3_year_yield(self, start_date, end_date):
        from .models import KrBond3YearYield
        return self._load(KrBond3YearYield, (), start_date, end_date)

    def get_sox(self, start_date, end_date):
        from .models import Sox
        return self._load(Sox, (), start_date, end_date)

    def get_gsci(self, start_date, end_date):
        from .models import Gsci
        return self._load(Gsci, (), start_date, end_date)

    def get_dx(self, start_date, end_date):
        from .models import Dx
        return self._load(Dx, (), start_date, end_date)

    def get_cnn_fear_greed_index(self, start_date, end_date):
        from .models import CnnFearGreed
        return self._load(CnnFearGreed, (), start_date, end_date)

    def get_stock_index_per_pbr(self, name, start_date, end_date):
        from .models import StockIndexPerPbr
        return self._load(StockIndexPerPbr, (StockIndexPerPbr.name == name,), start_date, end_date)

    def get_citi_surprise(self, group_name, start_date, end_date):
        from .models import CitiSurprise
        return self._load(CitiSurprise, (CitiSurprise.group_name == group_name,), start_date, end_date)


data_access = _DataAccess()
