"""SQLAlchemy engine, session factory and declarative base.

Provides the central declarative ``Base``, an ``Engine`` and a session
factory. ``init_db`` creates the tables for the PoC via ``create_all``
(Alembic is intentionally out of scope).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

engine = create_engine(_settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Shared declarative base class for all ORM models."""


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager with commit/rollback handling."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (idempotent). Models are imported here on purpose so
    they are registered on ``Base.metadata``."""
    from app import models  # noqa: F401  (register mappers)

    Base.metadata.create_all(bind=engine)
