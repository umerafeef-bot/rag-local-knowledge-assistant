"""SQLite engine construction, schema creation and a lightweight ping."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import Settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


def create_database(settings: Settings) -> tuple[Engine, sessionmaker[Session]]:
    """Build the engine and session factory.

    ``check_same_thread=False`` is required because FastAPI runs synchronous
    endpoints in a worker thread pool; SQLAlchemy's connection pool still hands
    each thread its own connection.
    """
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        # WAL keeps readers (list/detail requests) from blocking the writer
        # during ingestion.
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
        future=True,
    )
    return engine, session_factory


def init_schema(engine: Engine) -> None:
    """Create tables that do not exist yet."""
    from app.db import models  # noqa: F401  (registers the ORM models)

    Base.metadata.create_all(engine)


def ping_database(engine: Engine) -> None:
    """Raise if the database cannot be reached. Used by GET /health."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
