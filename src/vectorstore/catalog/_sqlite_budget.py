"""SQLite persistence for the embedding budget ledger.

This component owns usage-schema migration, reservations, reconciliation, and
spend aggregation. ``SqliteDocumentCatalog`` composes it so the catalog keeps
its historical ``BudgetLedger``-compatible API without mixing budget policy
persistence into document retrieval code.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite
from uuid import uuid4

from vectorstore.embeddings.policy import (
    BudgetPeriod,
    BudgetReservation,
    BudgetReservationDecision,
    EmbeddingUsageRecord,
    UsageStatus,
    _coerce_record_charge,
    _validate_reconciliation,
)
from vectorstore.embeddings.pricing import EmbeddingCharge, UsdAmount, nanos_to_usd

Clock = Callable[[], datetime]

_CREATE_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS embedding_usage (
    event_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    processing_mode TEXT NOT NULL,
    tokens INTEGER NOT NULL CHECK (tokens >= 0),
    rate_nanos_per_million INTEGER CHECK (rate_nanos_per_million >= 0),
    price_version TEXT,
    charge_nanos INTEGER CHECK (charge_nanos >= 0),
    status TEXT NOT NULL CHECK (
        status IN ('reserved', 'committed', 'released', 'expired')
    ),
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (rate_nanos_per_million IS NULL
            AND price_version IS NULL
            AND charge_nanos IS NULL)
        OR
        (rate_nanos_per_million IS NOT NULL
            AND price_version IS NOT NULL
            AND charge_nanos IS NOT NULL)
    )
)
"""

_USAGE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_embedding_usage_date ON embedding_usage(date)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_usage_status_date "
    "ON embedding_usage(status, date)",
    "CREATE INDEX IF NOT EXISTS idx_embedding_usage_provider_model_date "
    "ON embedding_usage(provider, model, date)",
)

_USAGE_COLUMNS = frozenset(
    {
        "event_id",
        "date",
        "provider",
        "model",
        "processing_mode",
        "tokens",
        "rate_nanos_per_million",
        "price_version",
        "charge_nanos",
        "status",
        "expires_at",
        "created_at",
        "updated_at",
    }
)
_LEGACY_USAGE_COLUMNS = frozenset({"date", "provider", "tokens", "estimated_usd"})


class _SqliteBudgetLedger:
    """Persist budget reservations and usage in an injected SQLite connection."""

    def __init__(self, connection: sqlite3.Connection, now: Clock) -> None:
        self._connection = connection
        self._now = now

    def initialize_schema(self) -> None:
        """Create or migrate the usage schema within the caller's transaction."""
        table = self._connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'embedding_usage'"
        ).fetchone()
        if table is None:
            self._create_schema()
            return

        columns = frozenset(
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(embedding_usage)"
            ).fetchall()
        )
        if _USAGE_COLUMNS <= columns:
            self._create_indexes()
            return
        if columns == _LEGACY_USAGE_COLUMNS:
            self._migrate_legacy_schema()
            return
        raise RuntimeError(
            "unsupported embedding_usage schema; expected the current or legacy "
            f"columns, got {sorted(columns)!r}"
        )

    def reserve(
        self,
        charge: EmbeddingCharge,
        *,
        daily_limit_nanos: int | None,
        monthly_limit_nanos: int | None,
        ttl_seconds: float,
    ) -> BudgetReservationDecision:
        """Atomically reserve predicted spend across SQLite connections."""
        _validate_reservation_inputs(
            charge, daily_limit_nanos, monthly_limit_nanos, ttl_seconds
        )
        assert charge.charge_nanos is not None

        moment = self._now()
        day = moment.strftime("%Y-%m-%d")
        month = moment.strftime("%Y-%m")
        reservation_id = uuid4().hex
        expires_at = datetime.fromtimestamp(
            moment.timestamp() + ttl_seconds,
            tz=UTC,
        )

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._expire_reservations(moment)
            if (
                daily_limit_nanos is not None
                and self._sum_active_nanos(day=day) + charge.charge_nanos
                > daily_limit_nanos
            ):
                self._connection.commit()
                return BudgetReservationDecision(exceeded=BudgetPeriod.DAILY)
            if (
                monthly_limit_nanos is not None
                and self._sum_active_nanos(month=month) + charge.charge_nanos
                > monthly_limit_nanos
            ):
                self._connection.commit()
                return BudgetReservationDecision(exceeded=BudgetPeriod.MONTHLY)

            self._insert_usage_record(
                event_id=reservation_id,
                date=day,
                charge=charge,
                status=UsageStatus.RESERVED,
                expires_at=expires_at,
                created_at=moment.isoformat(),
                updated_at=moment.isoformat(),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

        return BudgetReservationDecision(
            reservation=BudgetReservation(
                reservation_id=reservation_id,
                date=day,
                charge=charge,
                expires_at=expires_at,
            )
        )

    def commit(
        self,
        reservation: BudgetReservation,
        actual_charge: EmbeddingCharge,
    ) -> None:
        """Reconcile a reservation with authoritative token usage."""
        _validate_reconciliation(reservation, actual_charge)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT * FROM embedding_usage WHERE event_id = ?",
                (reservation.reservation_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown budget reservation")
            status = UsageStatus(row["status"])
            if status in (UsageStatus.COMMITTED, UsageStatus.RELEASED):
                raise ValueError(f"cannot commit a {status} reservation")
            expected = reservation.charge
            stored_identity = (
                row["provider"],
                row["model"],
                row["processing_mode"],
                row["rate_nanos_per_million"],
                row["price_version"],
            )
            reservation_identity = (
                expected.provider,
                expected.model,
                expected.processing_mode,
                expected.rate_nanos_per_million,
                expected.price_version,
            )
            if stored_identity != reservation_identity:
                raise ValueError("stored reservation identity does not match")
            self._connection.execute(
                """
                UPDATE embedding_usage
                SET tokens = ?, rate_nanos_per_million = ?, price_version = ?,
                    charge_nanos = ?, status = ?, expires_at = NULL,
                    updated_at = ?
                WHERE event_id = ?
                """,
                (
                    actual_charge.tokens,
                    actual_charge.rate_nanos_per_million,
                    actual_charge.price_version,
                    actual_charge.charge_nanos,
                    str(UsageStatus.COMMITTED),
                    self._now().isoformat(),
                    reservation.reservation_id,
                ),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def release(self, reservation: BudgetReservation) -> None:
        """Release an unused reservation while retaining its audit row."""
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT status FROM embedding_usage WHERE event_id = ?",
                (reservation.reservation_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown budget reservation")
            status = UsageStatus(row["status"])
            if status in (UsageStatus.RELEASED, UsageStatus.EXPIRED):
                self._connection.commit()
                return
            if status is UsageStatus.COMMITTED:
                raise ValueError("cannot release a committed reservation")
            self._connection.execute(
                """
                UPDATE embedding_usage
                SET status = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (
                    str(UsageStatus.RELEASED),
                    self._now().isoformat(),
                    reservation.reservation_id,
                ),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def record(
        self,
        charge_or_provider: EmbeddingCharge | str,
        tokens: int | None = None,
        usd: UsdAmount | None = None,
        *,
        model: str = "<unspecified>",
        processing_mode: str = "standard",
        price_version: str = "legacy-explicit-total",
    ) -> None:
        """Record committed usage with complete price provenance."""
        charge = _coerce_record_charge(
            charge_or_provider,
            tokens,
            usd,
            model=model,
            processing_mode=processing_mode,
            price_version=price_version,
        )
        moment = self._now()
        with self._connection:
            self._insert_usage_record(
                event_id=uuid4().hex,
                date=moment.strftime("%Y-%m-%d"),
                charge=charge,
                status=UsageStatus.COMMITTED,
                expires_at=None,
                created_at=moment.isoformat(),
                updated_at=moment.isoformat(),
            )

    def spent_today_nanos(self) -> int:
        """Return committed plus reserved nanodollars for today."""
        moment = self._now()
        with self._connection:
            self._expire_reservations(moment)
            return self._sum_active_nanos(day=moment.strftime("%Y-%m-%d"))

    def spent_month_nanos(self) -> int:
        """Return committed plus reserved nanodollars for this month."""
        moment = self._now()
        with self._connection:
            self._expire_reservations(moment)
            return self._sum_active_nanos(month=moment.strftime("%Y-%m"))

    def spent_today(self) -> Decimal:
        """Return exact committed plus reserved USD spend for today."""
        return nanos_to_usd(self.spent_today_nanos())

    def spent_month(self) -> Decimal:
        """Return exact committed plus reserved USD spend for this month."""
        return nanos_to_usd(self.spent_month_nanos())

    def tokens_today(self, provider: str, model: str | None = None) -> int:
        """Return committed tokens today, optionally filtered by model."""
        statement = (
            "SELECT COALESCE(SUM(tokens), 0) AS total FROM embedding_usage "
            "WHERE date = ? AND provider = ? AND status = ?"
        )
        parameters: list[object] = [
            self._now().strftime("%Y-%m-%d"),
            provider,
            str(UsageStatus.COMMITTED),
        ]
        if model is not None:
            statement += " AND model = ?"
            parameters.append(model)
        row = self._connection.execute(statement, parameters).fetchone()
        return int(row["total"])

    def usage_records(self) -> tuple[EmbeddingUsageRecord, ...]:
        """Return all usage and reservation audit records."""
        moment = self._now()
        with self._connection:
            self._expire_reservations(moment)
            rows = self._connection.execute(
                "SELECT * FROM embedding_usage ORDER BY created_at, event_id"
            ).fetchall()
        return tuple(_usage_record_from_row(row) for row in rows)

    def _create_schema(self) -> None:
        self._connection.execute(_CREATE_USAGE_TABLE)
        self._create_indexes()

    def _create_indexes(self) -> None:
        for statement in _USAGE_INDEXES:
            self._connection.execute(statement)

    def _migrate_legacy_schema(self) -> None:
        rows = self._connection.execute(
            "SELECT date, provider, tokens, estimated_usd FROM embedding_usage"
        ).fetchall()
        self._connection.execute(
            "ALTER TABLE embedding_usage RENAME TO embedding_usage_legacy"
        )
        self._connection.execute("DROP INDEX IF EXISTS idx_embedding_usage_date")
        self._create_schema()
        for row in rows:
            tokens = int(row["tokens"])
            try:
                charge = EmbeddingCharge.from_total_usd(
                    provider=str(row["provider"]),
                    model="<legacy>",
                    processing_mode="standard",
                    tokens=tokens,
                    usd=float(row["estimated_usd"]),
                    price_version="legacy-float-migration",
                )
            except TypeError, ValueError:
                charge = EmbeddingCharge.unpriced(
                    provider=str(row["provider"]),
                    model="<legacy>",
                    processing_mode="standard",
                    tokens=tokens,
                )
            created_at = f"{row['date']}T00:00:00+00:00"
            self._insert_usage_record(
                event_id=uuid4().hex,
                date=str(row["date"]),
                charge=charge,
                status=UsageStatus.COMMITTED,
                expires_at=None,
                created_at=created_at,
                updated_at=created_at,
            )
        self._connection.execute("DROP TABLE embedding_usage_legacy")

    def _insert_usage_record(
        self,
        *,
        event_id: str,
        date: str,
        charge: EmbeddingCharge,
        status: UsageStatus,
        expires_at: datetime | None,
        created_at: str,
        updated_at: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO embedding_usage (
                event_id, date, provider, model, processing_mode, tokens,
                rate_nanos_per_million, price_version, charge_nanos, status,
                expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                date,
                charge.provider,
                charge.model,
                charge.processing_mode,
                charge.tokens,
                charge.rate_nanos_per_million,
                charge.price_version,
                charge.charge_nanos,
                str(status),
                expires_at.isoformat() if expires_at is not None else None,
                created_at,
                updated_at,
            ),
        )

    def _expire_reservations(self, moment: datetime) -> None:
        self._connection.execute(
            """
            UPDATE embedding_usage
            SET status = ?, updated_at = ?
            WHERE status = ? AND expires_at <= ?
            """,
            (
                str(UsageStatus.EXPIRED),
                moment.isoformat(),
                str(UsageStatus.RESERVED),
                moment.isoformat(),
            ),
        )

    def _sum_active_nanos(
        self,
        *,
        day: str | None = None,
        month: str | None = None,
    ) -> int:
        if (day is None) == (month is None):
            raise ValueError("exactly one of day or month must be supplied")
        date_clause = "date = ?" if day is not None else "date LIKE ?"
        date_value = day if day is not None else f"{month}-%"
        row = self._connection.execute(
            "SELECT COALESCE(SUM(charge_nanos), 0) AS total "
            "FROM embedding_usage WHERE "
            f"{date_clause} AND status IN (?, ?)",
            (
                date_value,
                str(UsageStatus.RESERVED),
                str(UsageStatus.COMMITTED),
            ),
        ).fetchone()
        return int(row["total"])


def _validate_reservation_inputs(
    charge: EmbeddingCharge,
    daily_limit_nanos: int | None,
    monthly_limit_nanos: int | None,
    ttl_seconds: float,
) -> None:
    if not charge.is_priced:
        raise ValueError("budget reservations require a priced charge")
    for value, label in (
        (daily_limit_nanos, "daily_limit_nanos"),
        (monthly_limit_nanos, "monthly_limit_nanos"),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"{label} must be a non-negative integer or None")
    if (
        not isinstance(ttl_seconds, (int, float))
        or isinstance(ttl_seconds, bool)
        or not isfinite(ttl_seconds)
        or ttl_seconds <= 0
    ):
        raise ValueError("ttl_seconds must be finite and greater than zero")


def _usage_record_from_row(row: sqlite3.Row) -> EmbeddingUsageRecord:
    if row["charge_nanos"] is None:
        charge = EmbeddingCharge.unpriced(
            row["provider"],
            row["model"],
            row["processing_mode"],
            row["tokens"],
        )
    else:
        charge = EmbeddingCharge(
            provider=row["provider"],
            model=row["model"],
            processing_mode=row["processing_mode"],
            tokens=row["tokens"],
            charge_nanos=row["charge_nanos"],
            rate_nanos_per_million=row["rate_nanos_per_million"],
            price_version=row["price_version"],
        )
    return EmbeddingUsageRecord(
        event_id=row["event_id"],
        date=row["date"],
        charge=charge,
        status=UsageStatus(row["status"]),
        expires_at=(
            datetime.fromisoformat(row["expires_at"])
            if row["expires_at"] is not None
            else None
        ),
    )
