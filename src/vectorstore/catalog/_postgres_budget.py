"""PostgreSQL persistence for the embedding budget ledger."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite
from typing import Any
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

from ._postgres_support import (
    _datetime_from_value,
    _iso_string,
    _optional_int,
    _optional_str,
    _PostgresDatabase,
    _row_value,
)

Clock = Callable[[], datetime]

_USAGE_FIELDS = (
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
)


class _PostgresBudgetLedger:
    """Persist atomic budget reservations through an injected DB boundary."""

    def __init__(
        self,
        database: _PostgresDatabase,
        *,
        usage_table: str,
        schema_table: str,
        schema_version: int,
        now: Clock,
    ) -> None:
        self._database = database
        self._usage_table = usage_table
        self._schema_table = schema_table
        self._schema_version = schema_version
        self._now = now

    def reserve(
        self,
        charge: EmbeddingCharge,
        *,
        daily_limit_nanos: int | None,
        monthly_limit_nanos: int | None,
        ttl_seconds: float,
    ) -> BudgetReservationDecision:
        """Atomically reserve predicted spend across PostgreSQL sessions."""
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

        with self._database.cursor(write=True) as cursor:
            self._lock_budget(cursor)
            self._expire_reservations(cursor, moment)
            if (
                daily_limit_nanos is not None
                and self._sum_active_nanos(cursor, day=day) + charge.charge_nanos
                > daily_limit_nanos
            ):
                return BudgetReservationDecision(exceeded=BudgetPeriod.DAILY)
            if (
                monthly_limit_nanos is not None
                and self._sum_active_nanos(cursor, month=month) + charge.charge_nanos
                > monthly_limit_nanos
            ):
                return BudgetReservationDecision(exceeded=BudgetPeriod.MONTHLY)
            self._insert_usage_record(
                cursor,
                event_id=reservation_id,
                date_value=day,
                charge=charge,
                status=UsageStatus.RESERVED,
                expires_at=expires_at,
                created_at=moment,
                updated_at=moment,
            )

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
        statement = f"""
SELECT provider, model, processing_mode, rate_nanos_per_million,
    price_version, status
FROM {self._usage_table}
WHERE event_id = %s
FOR UPDATE
""".strip()
        with self._database.cursor(write=True) as cursor:
            cursor.execute(statement, (reservation.reservation_id,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError("unknown budget reservation")
            status = UsageStatus(str(_row_value(row, 5, "status")))
            if status in (UsageStatus.COMMITTED, UsageStatus.RELEASED):
                raise ValueError(f"cannot commit a {status} reservation")
            expected = reservation.charge
            stored_identity = (
                str(_row_value(row, 0, "provider")),
                str(_row_value(row, 1, "model")),
                str(_row_value(row, 2, "processing_mode")),
                _optional_int(_row_value(row, 3, "rate_nanos_per_million")),
                _optional_str(_row_value(row, 4, "price_version")),
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
            cursor.execute(
                f"""
UPDATE {self._usage_table}
SET tokens = %s, rate_nanos_per_million = %s, price_version = %s,
    charge_nanos = %s, status = %s, expires_at = NULL, updated_at = %s
WHERE event_id = %s
""".strip(),
                (
                    actual_charge.tokens,
                    actual_charge.rate_nanos_per_million,
                    actual_charge.price_version,
                    actual_charge.charge_nanos,
                    str(UsageStatus.COMMITTED),
                    self._now(),
                    reservation.reservation_id,
                ),
            )

    def release(self, reservation: BudgetReservation) -> None:
        """Release an unused reservation while retaining its audit row."""
        with self._database.cursor(write=True) as cursor:
            cursor.execute(
                f"SELECT status FROM {self._usage_table} "
                "WHERE event_id = %s FOR UPDATE",
                (reservation.reservation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError("unknown budget reservation")
            status = UsageStatus(str(_row_value(row, 0, "status")))
            if status in (UsageStatus.RELEASED, UsageStatus.EXPIRED):
                return
            if status is UsageStatus.COMMITTED:
                raise ValueError("cannot release a committed reservation")
            cursor.execute(
                f"UPDATE {self._usage_table} "
                "SET status = %s, updated_at = %s WHERE event_id = %s",
                (
                    str(UsageStatus.RELEASED),
                    self._now(),
                    reservation.reservation_id,
                ),
            )

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
        """Record committed usage with exact price provenance."""
        charge = _coerce_record_charge(
            charge_or_provider,
            tokens,
            usd,
            model=model,
            processing_mode=processing_mode,
            price_version=price_version,
        )
        moment = self._now()
        with self._database.cursor(write=True) as cursor:
            self._insert_usage_record(
                cursor,
                event_id=uuid4().hex,
                date_value=moment.strftime("%Y-%m-%d"),
                charge=charge,
                status=UsageStatus.COMMITTED,
                expires_at=None,
                created_at=moment,
                updated_at=moment,
            )

    def spent_today_nanos(self) -> int:
        """Return committed plus reserved nanodollars for today."""
        moment = self._now()
        with self._database.cursor(write=True) as cursor:
            self._expire_reservations(cursor, moment)
            return self._sum_active_nanos(cursor, day=moment.strftime("%Y-%m-%d"))

    def spent_month_nanos(self) -> int:
        """Return committed plus reserved nanodollars for this month."""
        moment = self._now()
        with self._database.cursor(write=True) as cursor:
            self._expire_reservations(cursor, moment)
            return self._sum_active_nanos(cursor, month=moment.strftime("%Y-%m"))

    def spent_today(self) -> Decimal:
        """Return exact committed plus reserved USD spend for today."""
        return nanos_to_usd(self.spent_today_nanos())

    def spent_month(self) -> Decimal:
        """Return exact committed plus reserved USD spend for this month."""
        return nanos_to_usd(self.spent_month_nanos())

    def tokens_today(self, provider: str, model: str | None = None) -> int:
        """Return committed tokens today, optionally filtered by model."""
        statement = (
            f"SELECT COALESCE(SUM(tokens), 0) AS total "
            f"FROM {self._usage_table} "
            "WHERE date = %s::date AND provider = %s AND status = %s"
        )
        parameters: list[object] = [
            self._now().strftime("%Y-%m-%d"),
            provider,
            str(UsageStatus.COMMITTED),
        ]
        if model is not None:
            statement += " AND model = %s"
            parameters.append(model)
        with self._database.cursor() as cursor:
            cursor.execute(statement, tuple(parameters))
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL token total query returned no row")
        return int(_row_value(row, 0, "total"))

    def usage_records(self) -> tuple[EmbeddingUsageRecord, ...]:
        """Return all usage and reservation audit records."""
        moment = self._now()
        fields = ", ".join(_USAGE_FIELDS)
        with self._database.cursor(write=True) as cursor:
            self._expire_reservations(cursor, moment)
            cursor.execute(
                f"SELECT {fields} FROM {self._usage_table} "
                "ORDER BY created_at, event_id"
            )
            rows = cursor.fetchall()
        return tuple(_usage_record_from_row(row) for row in rows)

    def _lock_budget(self, cursor: Any) -> None:
        cursor.execute(
            f"SELECT version FROM {self._schema_table} "
            "WHERE singleton = TRUE FOR UPDATE"
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL catalog schema metadata is missing")
        version = int(_row_value(row, 0, "version"))
        if version != self._schema_version:
            raise RuntimeError(
                f"unsupported PostgreSQL catalog schema version {version}; "
                f"expected {self._schema_version}"
            )

    def _insert_usage_record(
        self,
        cursor: Any,
        *,
        event_id: str,
        date_value: str,
        charge: EmbeddingCharge,
        status: UsageStatus,
        expires_at: datetime | None,
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        cursor.execute(
            f"""
INSERT INTO {self._usage_table} (
    event_id, date, provider, model, processing_mode, tokens,
    rate_nanos_per_million, price_version, charge_nanos, status,
    expires_at, created_at, updated_at
) VALUES (%s, %s::date, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""".strip(),
            (
                event_id,
                date_value,
                charge.provider,
                charge.model,
                charge.processing_mode,
                charge.tokens,
                charge.rate_nanos_per_million,
                charge.price_version,
                charge.charge_nanos,
                str(status),
                expires_at,
                created_at,
                updated_at,
            ),
        )

    def _expire_reservations(self, cursor: Any, moment: datetime) -> None:
        cursor.execute(
            f"""
UPDATE {self._usage_table}
SET status = %s, updated_at = %s
WHERE status = %s AND expires_at <= %s
""".strip(),
            (
                str(UsageStatus.EXPIRED),
                moment,
                str(UsageStatus.RESERVED),
                moment,
            ),
        )

    def _sum_active_nanos(
        self,
        cursor: Any,
        *,
        day: str | None = None,
        month: str | None = None,
    ) -> int:
        if (day is None) == (month is None):
            raise ValueError("exactly one of day or month must be supplied")
        parameters: list[object]
        if day is not None:
            date_clause = "date = %s::date"
            parameters = [day]
        else:
            first_day = f"{month}-01"
            date_clause = "date >= %s::date AND date < (%s::date + INTERVAL '1 month')"
            parameters = [first_day, first_day]
        parameters.extend((str(UsageStatus.RESERVED), str(UsageStatus.COMMITTED)))
        cursor.execute(
            f"""
SELECT COALESCE(SUM(charge_nanos), 0) AS total
FROM {self._usage_table}
WHERE {date_clause} AND status IN (%s, %s)
""".strip(),
            tuple(parameters),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL spend total query returned no row")
        return int(_row_value(row, 0, "total"))


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


def _usage_record_from_row(row: Any) -> EmbeddingUsageRecord:
    provider = str(_row_value(row, 2, "provider"))
    model = str(_row_value(row, 3, "model"))
    processing_mode = str(_row_value(row, 4, "processing_mode"))
    tokens = int(_row_value(row, 5, "tokens"))
    charge_nanos = _optional_int(_row_value(row, 8, "charge_nanos"))
    if charge_nanos is None:
        charge = EmbeddingCharge.unpriced(provider, model, processing_mode, tokens)
    else:
        rate = _optional_int(_row_value(row, 6, "rate_nanos_per_million"))
        version = _optional_str(_row_value(row, 7, "price_version"))
        if rate is None or version is None:
            raise ValueError("stored priced usage row has incomplete price provenance")
        charge = EmbeddingCharge(
            provider=provider,
            model=model,
            processing_mode=processing_mode,
            tokens=tokens,
            rate_nanos_per_million=rate,
            price_version=version,
            charge_nanos=charge_nanos,
        )
    expires_value = _row_value(row, 10, "expires_at")
    return EmbeddingUsageRecord(
        event_id=str(_row_value(row, 0, "event_id")),
        date=_iso_string(_row_value(row, 1, "date")),
        charge=charge,
        status=UsageStatus(str(_row_value(row, 9, "status"))),
        expires_at=(
            _datetime_from_value(expires_value) if expires_value is not None else None
        ),
    )
