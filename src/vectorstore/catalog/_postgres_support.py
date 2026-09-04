"""Low-level PostgreSQL DB-API adaptation shared by catalog components."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

ConnectionFactory = Callable[[], Any]


class _PostgresDatabase:
    """Open operation-scoped connections from an injected factory.

    The object owns transaction and resource lifecycle only. Repositories
    receive this boundary instead of creating driver connections themselves.
    """

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory

    @contextmanager
    def cursor(self, *, write: bool = False) -> Iterator[Any]:
        """Yield a cursor and consistently finalize its operation."""
        connection = self._connection_factory()
        cursor: Any | None = None
        try:
            cursor = connection.cursor()
            yield cursor
            if write:
                connection.commit()
        except BaseException:
            if write:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise
        finally:
            try:
                if cursor is not None:
                    cursor.close()
            finally:
                connection.close()


def _postgres_connection_factory(connection_string: str) -> ConnectionFactory:
    """Build the default lazy Psycopg connection factory."""

    def open_connection() -> Any:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PostgresDocumentCatalog requires the postgres extra; install it "
                "with uv sync --extra postgres"
            ) from exc
        return psycopg.connect(connection_string)

    return open_connection


def _row_value(row: Any, position: int, name: str) -> Any:
    """Read a value from positional, mapping, or attribute-style driver rows."""
    try:
        return row[position]
    except KeyError, IndexError, TypeError:
        try:
            return row[name]
        except KeyError, IndexError, TypeError:
            return getattr(row, name)


def _datetime_from_value(value: object) -> datetime:
    """Normalize a driver timestamp while preserving timezone information."""
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value)
    else:
        raise ValueError("stored timestamp must be a datetime or ISO string")
    if result.tzinfo is None:
        raise ValueError("stored timestamp must be timezone-aware")
    return result


def _iso_string(value: object) -> str:
    """Normalize PostgreSQL date and timestamp values to ISO text."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))
