"""SQLAlchemy-Engine, Session-Factory und Basisklasse.

Stellt die zentrale ``Base`` (declarative), eine ``Engine`` sowie eine
Session-Factory bereit. ``init_db`` legt die Tabellen fuer den PoC per
``create_all`` an (Alembic ist bewusst out of scope).
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
    """Gemeinsame declarative Basisklasse aller ORM-Modelle."""


@contextmanager
def session_scope() -> Iterator[Session]:
    """Kontextmanager mit Commit/Rollback-Handling."""
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
    """Legt alle Tabellen an (idempotent). Importiert die Modelle bewusst hier,
    damit sie bei ``Base.metadata`` registriert sind."""
    from app import models  # noqa: F401  (Registrierung der Mapper)

    Base.metadata.create_all(bind=engine)
