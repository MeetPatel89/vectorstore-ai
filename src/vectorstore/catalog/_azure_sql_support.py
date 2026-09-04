"""Low-level Azure SQL DB-API adaptation shared by catalog components."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any

ConnectionFactory = Callable[[], Any]


class _AzureSqlDatabase:
    """Open operation-scoped connections from an injected factory."""

    def __init__(self, connection_factory: ConnectionFactory) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory

    @contextmanager
    def cursor(
        self,
        *,
        write: bool = False,
        autocommit: bool = False,
    ) -> Iterator[Any]:
        """Yield a cursor and consistently finalize its operation."""
        if write and autocommit:
            raise ValueError("write and autocommit modes are mutually exclusive")
        connection = self._connection_factory()
        cursor: Any | None = None
        try:
            if autocommit:
                try:
                    connection.autocommit = True
                except AttributeError, TypeError:
                    setter = getattr(connection, "setautocommit", None)
                    if not callable(setter):
                        raise TypeError(
                            "Azure SQL connection must support autocommit for "
                            "Full-Text DDL"
                        ) from None
                    setter(True)
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


def _azure_sql_connection_factory(connection_string: str) -> ConnectionFactory:
    """Build the default lazy mssql-python connection factory."""

    def open_connection() -> Any:
        try:
            from mssql_python import connect
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "AzureSqlDocumentCatalog requires the Azure SQL extra; install it "
                "with uv sync --extra azure-sql"
            ) from exc
        return connect(connection_string)

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
    """Normalize a driver timestamp and require timezone information."""
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
    """Normalize date and timestamp values to ISO text."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))
