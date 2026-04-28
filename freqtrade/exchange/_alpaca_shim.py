"""Internal ccxt-API shim voor de Alpaca exchange plugin.

Achtergrond
-----------
Freqtrade's :class:`freqtrade.exchange.Exchange` base class is diep verweven
met ccxt — ``self._api`` en ``self._api_async`` worden overal gebruikt.
Voor Alpaca *willen* we ccxt NIET gebruiken, omdat ccxt's Alpaca-driver
faalt op paper accounts (HTTP 401 / Alpaca code 40110000).

In plaats daarvan gebruikt de :class:`Alpaca` plugin de officiële
``alpaca-py`` SDK rechtstreeks. Maar base ``Exchange.__init__`` wil koste
wat kost een ccxt-achtig object op ``self._api`` zetten.

Oplossing
---------
Deze module levert :class:`AlpacaCcxtShim` — een minimaal object dat:

- de juiste type-signatuur heeft (heeft de attributen die base Exchange
  uitleest of toewijst)
- alle netwerk-calls die base Exchange zou doen, blokkeert (raise
  ``NotImplementedError``) zodat we vroeg merken dat we een methode
  vergeten zijn te overriden in :class:`Alpaca`
- een paar attributen heeft (zoals ``markets``, ``id``, ``name``) die
  base Exchange tijdens init NODIG heeft maar niet kritisch zijn

De shim is **niet** een volledige ccxt-emulator. Hij is bewust dom.
Iedere methode die base Exchange erop zou aanroepen, MOET in
:class:`Alpaca` worden overriden — anders raise de shim.
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class _ShimError(NotImplementedError):
    """Raised wanneer base Exchange een ccxt-method aanroept die de Alpaca
    plugin nog niet heeft overriden. Dit is ontwerpgewijs een loud error
    zodat we tijdens development gauw zien welke methods we missen."""


class AlpacaCcxtShim:
    """Minimal stand-in voor een ccxt.Exchange instance.

    Wordt gemaakt door :meth:`Alpaca._init_ccxt` en op
    ``self._api`` / ``self._api_async`` gezet om base Exchange happy te
    houden. Mocht base Exchange een echte ccxt-method aanroepen, raise
    deze shim ``_ShimError`` zodat we de regressie meteen zien.
    """

    # ---------- Class identity (used by base Exchange logging) -----------

    id: str = "alpaca"
    name: str = "Alpaca"
    version: str = "shim-0.1"

    # ---------- ccxt-style state attributes ------------------------------

    # ccxt exposes 'has' as a dict of feature flags. Base Exchange queries
    # this via ``exchange_has(name)`` to check for e.g. fetchOHLCV support.
    # We declare the bare minimum that Alpaca actually supports.
    has: dict[str, bool]

    # Markets dict — populated by Alpaca.reload_markets(); base Exchange
    # reads this in a few places.
    markets: dict[str, Any]

    # ccxt clients have a session; base Exchange's close() calls
    # `self._api_async.session` truthiness + `self._api_async.close()`.
    session: Any | None = None

    # Used by some ccxt utility code paths; we keep them as no-op defaults.
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
            # Capabilities Alpaca DOES support
            "fetchOHLCV": True,
            "fetchTicker": True,
            "fetchBalance": True,
            "fetchMarkets": True,
            "createOrder": True,
            "fetchOrder": True,
            "cancelOrder": True,
            # Capabilities Alpaca does NOT support
            "watchOHLCV": False,  # Could be added later via Alpaca's WS API
            "fetchFundingRate": False,
            "fetchFundingRates": False,
            "fetchFundingHistory": False,
            "fetchPositions": False,  # We override to use Alpaca positions
            "fetchOrderBook": False,  # No L2 orderbook in Alpaca's free tier
            "fetchTrades": False,
            "createMarketOrder": True,
            "createLimitOrder": True,
        }
        self.options = {}
        # ccxt exposes `timeframes` as dict of supported timeframe strings.
        # Base Exchange reads this in validate_timeframes(). Provide ours.
        self.timeframes = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "1h": "1Hour", "4h": "4Hour",
            "1d": "1Day", "1w": "1Week", "1M": "1Month",
        }
        # ccxt has `currencies` dict — base Exchange may inspect.
        self.currencies = {"USD": {"id": "USD", "code": "USD", "precision": 0.01}}
        # ccxt has these as common probe-able attributes
        self.symbols: list[str] = []
        self.urls: dict = {"api": "https://api.alpaca.markets"}
        self.fees: dict = {"trading": {"maker": 0.0, "taker": 0.0}}
        self.precisionMode: int = 2  # ccxt DECIMAL_PLACES (constant value 2)
        # ccxt's `features` is a nested dict of capability flags. Base
        # Exchange reads `features.spot` for stoploss-related capability
        # checks. Provide an empty-but-valid structure.
        self.features: dict = {
            "spot": {
                "createOrder": {
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
    # Alpaca subclass forgot to override a method. We'd then add a
    # concrete impl in alpaca.py, not in this shim.
    # ---------------------------------------------------------------------

    async def load_markets(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async ccxt's load_markets. Base Exchange calls this from
        ``_api_reload_markets``. We override that method in :class:`Alpaca`
        so this shim should never be hit."""
        raise _ShimError(
            "AlpacaCcxtShim.load_markets called. Did Alpaca._api_reload_markets "
            "fail to override base Exchange behaviour?"
        )

    def fetch_markets(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("AlpacaCcxtShim.fetch_markets called - override in Alpaca")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: Any = None,
    ) -> list[list[float]]:
        """Async fetch_ohlcv called by base Exchange's download-data flow.
        Delegates to the Alpaca subclass sync method via asyncio.to_thread
        to avoid blocking the event loop on alpaca-py's HTTP calls."""
        if self._fetch_ohlcv_callback is None:
            raise _ShimError(
                "fetch_ohlcv called on shim without callback -- "
                "Alpaca._init_ccxt should pass alpaca-py-backed callback."
            )
        import asyncio
        # Convert to ms-epoch since args we receive
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
        raise _ShimError("AlpacaCcxtShim.fetch_ticker called - override in Alpaca")

    def fetch_balance(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("AlpacaCcxtShim.fetch_balance called - override in Alpaca")

    def create_order(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("AlpacaCcxtShim.create_order called - override in Alpaca")

    def fetch_order(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("AlpacaCcxtShim.fetch_order called - override in Alpaca")

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        raise _ShimError("AlpacaCcxtShim.cancel_order called - override in Alpaca")

    # ---------- Lifecycle methods ---------------------------------------

    # ---- ccxt stub methods that base Exchange calls but Alpaca doesn't need ---

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
        """Stocks via Alpaca: commission-free for US equities. SEC/TAF fees
        are negligible ($0.01/trade) for our purposes. Return zero-fee."""
        return {
            "type": takerOrMaker,
            "currency": "USD",
            "rate": 0.0,
            "cost": 0.0,
        }

    def describe(self) -> dict[str, Any]:
        """ccxt's describe() returns the exchange's full config dict.
        Base Exchange occasionally inspects this. Return minimal."""
        return {
            "id": self.id,
            "name": self.name,
            "rateLimit": 200,  # ms between requests
            "has": self.has,
            "timeframes": self.timeframes,
            "version": self.version,
        }

    def fetch_trading_fees(self, params: Any = None) -> dict[str, Any]:
        """Per-pair fee discovery. Alpaca: zero across the board for stocks."""
        return {"info": {}, "USD": self.fees["trading"]}

    async def close(self) -> None:
        """Base Exchange.close() awaits this on the async shim."""
        # No-op: alpaca-py uses standard requests, no persistent session
        # to close.
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
            f"AlpacaCcxtShim has no attribute {item!r}. Either:\n"
            f"  1. Add a concrete impl in Alpaca subclass (alpaca.py), OR\n"
            f"  2. Add a stub on this shim if base Exchange just reads it.\n"
            f"Don't add ccxt-style logic here; this shim must stay thin."
        )

    def __repr__(self) -> str:
        async_marker = "async" if self._is_async else "sync"
        return f"<AlpacaCcxtShim id={self.id} ({async_marker})>"
