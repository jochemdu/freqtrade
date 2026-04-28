"""Internal ccxt-API shim voor de IBKR exchange plugin.

Achtergrond
-----------
Identiek aan `_alpaca_shim.py`: Freqtrade's :class:`freqtrade.exchange.Exchange`
base class is diep verweven met ccxt — ``self._api`` en ``self._api_async``
worden overal gebruikt.

Voor IBKR *willen* we ccxt NIET gebruiken omdat:

1. ccxt's IBKR-driver bestaat NIET — IBKR's eigen API is socket-gebaseerd
   (TWS Gateway op port 4002/4001) en past niet in ccxt's REST-model.
2. ``ib_async>=2.0`` (de 2024-rebrand van het abandoned ``ib_insync``)
   is de moderne keuze: persistent socket connection + asyncio + eventkit
   pattern voor event-driven updates.

Oplossing
---------
Deze module levert :class:`IbkrCcxtShim` — een minimaal object dat:

- de juiste type-signatuur heeft (heeft de attributen die base Exchange
  uitleest of toewijst)
- alle netwerk-calls die base Exchange zou doen, blokkeert (raise
  ``_ShimError``) zodat we vroeg merken dat we een methode vergeten
  zijn te overriden in :class:`InteractiveBrokers`
- een paar attributen heeft (zoals ``markets``, ``id``, ``name``) die
  base Exchange tijdens init NODIG heeft maar niet kritisch zijn

De shim is **niet** een volledige ccxt-emulator. Hij is bewust dom.
Iedere methode die base Exchange erop zou aanroepen, MOET in
:class:`InteractiveBrokers` worden overriden — anders raise de shim.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class _ShimError(NotImplementedError):
    """Raised wanneer base Exchange een ccxt-method aanroept die de IBKR
    plugin nog niet heeft overriden. Loud error tijdens development zodat
    we gauw zien welke methods we missen."""


class IbkrCcxtShim:
    """Minimal stand-in voor een ccxt.Exchange instance.

    Wordt gemaakt door :meth:`InteractiveBrokers._init_ccxt` en op
    ``self._api`` / ``self._api_async`` gezet om base Exchange happy te
    houden. Mocht base Exchange een echte ccxt-method aanroepen, raise
    deze shim ``_ShimError`` zodat we de regressie meteen zien.
    """

    # ---------- Class identity (used by base Exchange logging) -----------

    # `id` matches the `exchange.name` in config.json (lowercase).
    # Freqtrade uses this name in many places (data dir, strategy resolution).
    id: str = "interactivebrokers"
    name: str = "Interactive Brokers"
    version: str = "shim-0.1"

    # ---------- ccxt-style state attributes ------------------------------

    has: dict[str, bool]
    markets: dict[str, Any]
    session: Any | None = None
    options: dict[str, Any]
    timeout: int = 30000  # ms
    enableRateLimit: bool = True

    def __init__(
        self,
        *,
        config: dict[str, Any],
        is_async: bool = False,
        fetch_ohlcv_callback: Any = None,
    ) -> None:
        self._config = config
        self._is_async = is_async
        self._fetch_ohlcv_callback = fetch_ohlcv_callback
        self.markets = {}
        self.has = {
            # Capabilities IBKR DOES support
            "fetchOHLCV": True,
            "fetchTicker": True,
            "fetchBalance": True,
            "fetchMarkets": True,  # via reqContractDetails (lazy)
            "createOrder": True,
            "fetchOrder": True,
            "cancelOrder": True,
            "fetchPositions": True,
            "fetchOrderBook": True,  # IBKR supports L2 depth (paid data)
            # Capabilities IBKR does NOT support (or not via this plugin)
            "watchOHLCV": False,  # Could be added via tickByTickData
            "fetchFundingRate": False,
            "fetchFundingRates": False,
            "fetchFundingHistory": False,
            "fetchTrades": False,
            "createMarketOrder": True,
            "createLimitOrder": True,
        }
        self.options = {}
        # IBKR barSize strings via Freqtrade timeframe mapping. The actual
        # mapping happens in InteractiveBrokers._to_ibkr_barsize().
        self.timeframes = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
            "1w": "1W",
            "1M": "1M",
        }
        # Multi-currency: IBKR supports trading in many currencies. Account
        # base may be EUR but trades may settle USD/GBP/JPY/etc.
        self.currencies = {
            "USD": {"id": "USD", "code": "USD", "precision": 0.01},
            "EUR": {"id": "EUR", "code": "EUR", "precision": 0.01},
            "GBP": {"id": "GBP", "code": "GBP", "precision": 0.01},
            "JPY": {"id": "JPY", "code": "JPY", "precision": 1.0},
            "CHF": {"id": "CHF", "code": "CHF", "precision": 0.01},
            "CAD": {"id": "CAD", "code": "CAD", "precision": 0.01},
            "AUD": {"id": "AUD", "code": "AUD", "precision": 0.01},
            "HKD": {"id": "HKD", "code": "HKD", "precision": 0.01},
        }
        # ccxt has these as common probe-able attributes
        self.symbols: list[str] = []
        # Base URL is informational only (ccxt-style). Real connection is
        # via socket to TWS Gateway.
        self.urls: dict = {"api": "https://www.interactivebrokers.com"}
        # IBKR fees are tiered + per-share + venue-dependent. Real fees
        # are computed by Alpaca-py-style calculate_fee(); these defaults
        # are placeholder. Realistic stocks: ~$0.005/share min $1.
        self.fees: dict = {"trading": {"maker": 0.0005, "taker": 0.0005}}
        self.precisionMode: int = 2  # ccxt DECIMAL_PLACES (constant value 2)
        # ccxt's `features` is a nested dict of capability flags. Base
        # Exchange reads `features.spot` for stoploss-related capability
        # checks. Provide an empty-but-valid structure.
        self.features: dict = {
            "spot": {
                "createOrder": {
                    # IBKR DOES support stoploss-on-exchange via STP order type,
                    # but our first version manages stops in-strategy for
                    # consistency with Alpaca plugin. Could be enabled later.
                    "stopLossPrice": False,
                    "takeProfitPrice": False,
                    "trailing": False,
                },
            },
            "swap": None,
            "future": None,
        }

    # ---------- Methods that base Exchange might call --------------------
    #
    # These all raise `_ShimError`. If any of these gets hit, it means
    # InteractiveBrokers subclass forgot to override a method.
    # ---------------------------------------------------------------------

    async def load_markets(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async ccxt's load_markets. Base Exchange calls this from
        ``_api_reload_markets``. We override that method in
        :class:`InteractiveBrokers` so this shim should never be hit."""
        raise _ShimError(
            "IbkrCcxtShim.load_markets called. Did "
            "InteractiveBrokers._api_reload_markets fail to override "
            "base Exchange behaviour?"
        )

    def fetch_markets(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("IbkrCcxtShim.fetch_markets called - override in InteractiveBrokers")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: Any = None,
    ) -> list[list[float]]:
        """Async fetch_ohlcv called by base Exchange's download-data flow.
        Delegates to the InteractiveBrokers subclass sync method via
        asyncio.to_thread to avoid blocking the event loop on ib_async's
        socket calls.

        ib_async actually has both sync (ib.reqHistoricalData) and async
        (ib.reqHistoricalDataAsync) variants. We use the sync variant
        wrapped in to_thread for two reasons:
        1. Consistency with Alpaca plugin pattern.
        2. ib_async manages its own asyncio loop internally; mixing event
           loops can cause deadlocks.
        """
        if self._fetch_ohlcv_callback is None:
            raise _ShimError(
                "fetch_ohlcv called on shim without callback -- "
                "InteractiveBrokers._init_ccxt should pass ib_async-backed "
                "callback."
            )
        import asyncio
        since_ms = int(since) if since else None
        return await asyncio.to_thread(
            self._fetch_ohlcv_callback,
            symbol,
            timeframe,
            since_ms,
            None,  # candle_type (not used by shim)
            False,  # is_new_pair
            None,  # until_ms
        )

    def fetch_ticker(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("IbkrCcxtShim.fetch_ticker called - override in InteractiveBrokers")

    def fetch_balance(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("IbkrCcxtShim.fetch_balance called - override in InteractiveBrokers")

    def create_order(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("IbkrCcxtShim.create_order called - override in InteractiveBrokers")

    def fetch_order(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("IbkrCcxtShim.fetch_order called - override in InteractiveBrokers")

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("IbkrCcxtShim.cancel_order called - override in InteractiveBrokers")

    # ---------- ccxt stub methods that base Exchange calls ---------------

    def calculate_fee(
        self,
        symbol: str,
        type: str,  # noqa: A002
        side: str,
        amount: float,
        price: float,
        takerOrMaker: str = "taker",
        params: Any = None,
    ) -> dict[str, Any]:
        """IBKR fees are tiered + per-share + venue-dependent. For this
        plugin's first version we use a flat estimate: $0.005/share with
        $1 minimum (typical IBKR Pro Tiered for US stocks). Currency is
        derived from the pair-suffix downstream.

        Real fee computation should query IBKR's commission report after
        order fill. This estimate is sufficient for backtesting and
        Freqtrade's planning logic.
        """
        # $0.005 per share, $1 minimum, capped at 1% of trade value
        per_share = 0.005
        min_fee = 1.0
        max_fee_pct = 0.01
        notional = amount * price
        cost = max(min_fee, min(amount * per_share, notional * max_fee_pct))
        return {
            "type": takerOrMaker,
            "currency": "USD",  # Default; real currency is from pair suffix
            "rate": cost / notional if notional > 0 else 0.0,
            "cost": cost,
        }

    def describe(self) -> dict[str, Any]:
        """ccxt's describe() returns the exchange's full config dict.
        Base Exchange occasionally inspects this. Return minimal."""
        return {
            "id": self.id,
            "name": self.name,
            "rateLimit": 50,  # ms — IBKR allows 50 messages/second
            "has": self.has,
            "timeframes": self.timeframes,
            "version": self.version,
        }

    def fetch_trading_fees(self, params: Any = None) -> dict[str, Any]:
        """Per-pair fee discovery. IBKR is tiered; return placeholder."""
        return {
            "info": {"note": "IBKR uses tiered per-share commissions"},
            "USD": self.fees["trading"],
        }

    async def close(self) -> None:
        """Base Exchange.close() awaits this on the async shim. The actual
        socket close happens in InteractiveBrokers.close() via ib.disconnect()."""
        return None

    def __getattr__(self, item: str) -> Any:
        """Catch-all for unexpected attribute access — raises _ShimError so
        we see exactly which ccxt-method base Exchange tried to use that
        we didn't anticipate. Beats a silent ``None`` everywhere."""
        # Don't raise on dunder attribute access (e.g. __class__, __repr__,
        # debugger introspection) — that breaks logging and dataclasses.
        if item.startswith("__"):
            raise AttributeError(item)
        raise _ShimError(
            f"IbkrCcxtShim has no attribute {item!r}. Either:\n"
            f"  1. Add a concrete impl in InteractiveBrokers subclass "
            f"(interactive_brokers.py), OR\n"
            f"  2. Add a stub on this shim if base Exchange just reads it.\n"
            f"Don't add ccxt-style logic here; this shim must stay thin."
        )

    def __repr__(self) -> str:
        async_marker = "async" if self._is_async else "sync"
        return f"<IbkrCcxtShim id={self.id} ({async_marker})>"
