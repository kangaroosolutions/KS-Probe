"""
Database engine factory for KS-Probe.

Uses SQLite by default (zero-config, fully reproducible).
Set DATABASE_URL environment variable for PostgreSQL.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ks_probe.core.config import get_db_url
from ks_probe.db.schema import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine(db_url: str | None = None) -> Engine:
    """Return (or create) the shared database engine."""
    global _engine
    if _engine is None:
        url = db_url or get_db_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, echo=False)
    return _engine


def get_session_factory(db_url: str | None = None) -> sessionmaker:
    """Return (or create) the shared session factory."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine(db_url)
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory


def init_db(db_url: str | None = None) -> None:
    """Create all tables (idempotent — safe to call multiple times)."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)


def get_session(db_url: str | None = None) -> Session:
    """Return a new database session."""
    factory = get_session_factory(db_url)
    return factory()
