# models.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from config import DB_URI

Base = declarative_base()

engine = create_engine(
    DB_URI,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Tick(Base):
    __tablename__ = "ticks"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, index=True)
    symbol = Column(String, index=True)
    price = Column(Float)
    qty = Column(Float)

    __table_args__ = (
        Index("idx_ticks_symbol_ts", "symbol", "ts"),
    )


class OHLCKline(Base):
    __tablename__ = "ohlc_kline"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    ts = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


def init_db():
    Base.metadata.create_all(bind=engine)
