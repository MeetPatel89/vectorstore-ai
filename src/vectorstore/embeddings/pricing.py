"""Precise, provider-aware pricing for embedding token usage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from types import MappingProxyType

NANODOLLARS_PER_USD = 1_000_000_000
TOKENS_PER_MILLION = 1_000_000
MAX_NANODOLLARS = 2**63 - 1

type UsdAmount = Decimal | int | float | str


def _validate_non_negative_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _as_decimal(value: UsdAmount, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative USD amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative USD amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{label} must be a finite non-negative USD amount")
    return amount


def usd_to_nanos(value: UsdAmount, *, label: str = "USD amount") -> int:
    """Convert a USD amount to integer nanodollars, rounding upward.

    Upward rounding makes budget admission conservative when a caller supplies
    more than nine decimal places. The resulting integer is safe to persist in
    a SQLite ``INTEGER`` column.
    """
    amount = _as_decimal(value, label)
    nanos = int(
        (amount * NANODOLLARS_PER_USD).to_integral_value(rounding=ROUND_CEILING)
    )
    if nanos > MAX_NANODOLLARS:
        raise ValueError(f"{label} exceeds the supported monetary range")
    return nanos


def nanos_to_usd(value: int) -> Decimal:
    """Convert non-negative integer nanodollars to an exact USD decimal."""
    _validate_non_negative_int(value, "nanodollars")
    return Decimal(value) / NANODOLLARS_PER_USD


@dataclass(frozen=True)
class EmbeddingCharge:
    """Auditable token usage and its computed charge, when priced.

    ``charge_nanos`` and its pricing fields are all ``None`` only for an
    unpriced, unbudgeted usage event. Budget reservations always require a
    fully priced charge.
    """

    provider: str
    model: str
    processing_mode: str
    tokens: int
    charge_nanos: int | None
    rate_nanos_per_million: int | None
    price_version: str | None

    def __post_init__(self) -> None:
        for label in ("provider", "model", "processing_mode"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        _validate_non_negative_int(self.tokens, "tokens")

        pricing_fields = (
            self.charge_nanos,
            self.rate_nanos_per_million,
            self.price_version,
        )
        if all(value is None for value in pricing_fields):
            return
        if any(value is None for value in pricing_fields):
            raise ValueError("charge pricing fields must be all present or all absent")
        assert self.charge_nanos is not None
        assert self.rate_nanos_per_million is not None
        assert self.price_version is not None
        _validate_non_negative_int(self.charge_nanos, "charge_nanos")
        _validate_non_negative_int(
            self.rate_nanos_per_million, "rate_nanos_per_million"
        )
        if self.charge_nanos > MAX_NANODOLLARS:
            raise ValueError("charge_nanos exceeds the supported monetary range")
        if self.rate_nanos_per_million > MAX_NANODOLLARS:
            raise ValueError(
                "rate_nanos_per_million exceeds the supported monetary range"
            )
        if not self.price_version:
            raise ValueError("price_version must be a non-empty string")

    @classmethod
    def unpriced(
        cls,
        provider: str,
        model: str,
        processing_mode: str,
        tokens: int,
    ) -> EmbeddingCharge:
        """Create an explicitly unpriced usage event."""
        return cls(provider, model, processing_mode, tokens, None, None, None)

    @classmethod
    def from_total_usd(
        cls,
        provider: str,
        model: str,
        processing_mode: str,
        tokens: int,
        usd: UsdAmount,
        *,
        price_version: str = "explicit-total",
    ) -> EmbeddingCharge:
        """Create a charge from an explicitly supplied total USD amount."""
        _validate_non_negative_int(tokens, "tokens")
        charge_nanos = usd_to_nanos(usd, label="usd")
        rate = (
            (charge_nanos * TOKENS_PER_MILLION + tokens - 1) // tokens if tokens else 0
        )
        return cls(
            provider,
            model,
            processing_mode,
            tokens,
            charge_nanos,
            rate,
            price_version,
        )

    @property
    def is_priced(self) -> bool:
        """Whether this event has an applied price and computed charge."""
        return self.charge_nanos is not None

    @property
    def usd(self) -> Decimal | None:
        """The exact computed charge in USD, or ``None`` when unpriced."""
        if self.charge_nanos is None:
            return None
        return nanos_to_usd(self.charge_nanos)


@dataclass(frozen=True)
class EmbeddingPrice:
    """One versioned token price for a provider/model/processing mode."""

    provider: str
    model: str
    processing_mode: str
    rate_nanos_per_million: int
    version: str

    def __post_init__(self) -> None:
        for label in ("provider", "model", "processing_mode", "version"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        _validate_non_negative_int(
            self.rate_nanos_per_million, "rate_nanos_per_million"
        )
        if self.rate_nanos_per_million > MAX_NANODOLLARS:
            raise ValueError(
                "rate_nanos_per_million exceeds the supported monetary range"
            )

    @classmethod
    def from_usd_per_million(
        cls,
        provider: str,
        model: str,
        usd_per_million_tokens: UsdAmount,
        *,
        version: str,
        processing_mode: str = "standard",
    ) -> EmbeddingPrice:
        """Build a price from a human-readable USD-per-million rate."""
        return cls(
            provider=provider,
            model=model,
            processing_mode=processing_mode,
            rate_nanos_per_million=usd_to_nanos(
                usd_per_million_tokens,
                label="usd_per_million_tokens",
            ),
            version=version,
        )

    @property
    def usd_per_million_tokens(self) -> Decimal:
        """The exact USD-per-million rate."""
        return nanos_to_usd(self.rate_nanos_per_million)

    def charge(self, tokens: int) -> EmbeddingCharge:
        """Compute a conservative nanodollar charge for ``tokens``."""
        _validate_non_negative_int(tokens, "tokens")
        numerator = tokens * self.rate_nanos_per_million
        charge_nanos = (
            (numerator + TOKENS_PER_MILLION - 1) // TOKENS_PER_MILLION
            if numerator
            else 0
        )
        if charge_nanos > MAX_NANODOLLARS:
            raise ValueError("computed charge exceeds the supported monetary range")
        return EmbeddingCharge(
            provider=self.provider,
            model=self.model,
            processing_mode=self.processing_mode,
            tokens=tokens,
            charge_nanos=charge_nanos,
            rate_nanos_per_million=self.rate_nanos_per_million,
            price_version=self.version,
        )


class PricingUnavailableError(LookupError):
    """No price is configured for a provider/model/processing mode."""

    def __init__(self, provider: str, model: str, processing_mode: str) -> None:
        super().__init__(
            "no embedding price is configured for "
            f"provider={provider!r}, model={model!r}, "
            f"processing_mode={processing_mode!r}; provide EmbeddingPricing or "
            "cost_per_million_tokens"
        )
        self.provider = provider
        self.model = model
        self.processing_mode = processing_mode


class EmbeddingPricing:
    """Immutable lookup table of versioned embedding token prices."""

    def __init__(self, prices: Iterable[EmbeddingPrice] = ()) -> None:
        resolved: dict[tuple[str, str, str], EmbeddingPrice] = {}
        for price in prices:
            key = (price.provider, price.model, price.processing_mode)
            if key in resolved:
                raise ValueError(f"duplicate embedding price for {key!r}")
            resolved[key] = price
        self._prices = MappingProxyType(resolved)

    def get(
        self,
        provider: str,
        model: str,
        processing_mode: str = "standard",
    ) -> EmbeddingPrice | None:
        """Return a configured price, or ``None`` when the key is unknown."""
        return self._prices.get((provider, model, processing_mode))

    def require(
        self,
        provider: str,
        model: str,
        processing_mode: str = "standard",
    ) -> EmbeddingPrice:
        """Return a price or fail closed with a typed error."""
        price = self.get(provider, model, processing_mode)
        if price is None:
            raise PricingUnavailableError(provider, model, processing_mode)
        return price

    @property
    def prices(self) -> tuple[EmbeddingPrice, ...]:
        """All configured prices in insertion order."""
        return tuple(self._prices.values())


OPENAI_PRICING_VERSION = "openai-public-2026-08-28"

DEFAULT_EMBEDDING_PRICING = EmbeddingPricing(
    (
        EmbeddingPrice.from_usd_per_million(
            "openai",
            "text-embedding-3-small",
            "0.02",
            version=OPENAI_PRICING_VERSION,
        ),
        EmbeddingPrice.from_usd_per_million(
            "openai",
            "text-embedding-3-large",
            "0.13",
            version=OPENAI_PRICING_VERSION,
        ),
        EmbeddingPrice.from_usd_per_million(
            "openai",
            "text-embedding-ada-002",
            "0.10",
            version=OPENAI_PRICING_VERSION,
        ),
    )
)

# Backward-compatible view of the standard OpenAI catalog. Values are exact
# Decimal amounts rather than binary floats.
EMBEDDING_COST_PER_MILLION_TOKENS: Mapping[str, Decimal] = MappingProxyType(
    {
        price.model: price.usd_per_million_tokens
        for price in DEFAULT_EMBEDDING_PRICING.prices
        if price.provider == "openai" and price.processing_mode == "standard"
    }
)


def estimate_cost_usd(
    model: str,
    tokens: int,
    *,
    provider: str = "openai",
    processing_mode: str = "standard",
    pricing: EmbeddingPricing = DEFAULT_EMBEDDING_PRICING,
) -> Decimal:
    """Compute an exact USD charge and fail when pricing is unavailable."""
    price = pricing.require(provider, model, processing_mode)
    charge = price.charge(tokens)
    assert charge.usd is not None
    return charge.usd
