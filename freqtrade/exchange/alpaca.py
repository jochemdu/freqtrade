"""Alpaca exchange plugin voor Freqtrade (stocks via alpaca-py).

Doel
----
Een `freqtrade.exchange.Exchange` subclass die transparant Alpaca als
broker behandelt voor US stocks (NYSE/NASDAQ/AMEX) en eventueel crypto.
Bypasst ccxt entirely omdat ccxt's Alpaca-driver problemen heeft met
paper-account autorisatie (HTTP 401 / Alpaca code 40110000).

Architectuur
------------
Drie lagen:

1. **alpaca-py SDK** (de officiële Alpaca Python client)
   - `TradingClient`: orders, posities, account-info
   - `StockHistoricalDataClient`: historische OHLCV bars
   - Auth via `APCA-API-KEY-ID` + `APCA-API-SECRET-KEY` HTTP headers

2. **AlpacaCcxtShim** (in `_alpaca_shim.py`)
   - Minimale stand-in voor ccxt.Exchange instance
   - Voldoet aan de attributen die `Exchange.__init__` verwacht
   - Methods raise `_ShimError` als ze worden gehit (defensief)

3. **Alpaca(Exchange)** (deze class)
   - Override `_init_ccxt` om de shim te returnen ipv echte ccxt.alpaca
   - Override `_api_reload_markets` om markets te laden via alpaca-py
   - Override `fetch_ohlcv`, `fetch_ticker`, `create_order`, `cancel_order`,
     `fetch_order`, `fetch_balance` om alpaca-py te gebruiken

Pair-encoding
-------------
Volgt de research-aanbeveling uit `02_upstream_compatible_fork.md` §"De
pair string encoding trick":

    SYMBOL[_VENUE]/CURRENCY

  - `SPY/USD`        → symbol=SPY, venue=XNAS (default), currency=USD
  - `SPY_XNAS/USD`   → symbol=SPY, venue=XNAS, currency=USD (explicit)
  - `BRK.B_XNYS/USD` → symbol=BRK.B, venue=XNYS, currency=USD

`_parse_pair()` doet de decoding. Freqtrade's pair-validation regex slaagt
omdat `SYMBOL_VENUE` als één base-string telt.

Configuratie
------------
In `config.json`:

    {
      "exchange": {
        "name": "alpaca",
        "key": "${ALPACA_API_KEY_ID}",       // resolved by run_freqtrade.py
        "secret": "${ALPACA_API_SECRET_KEY}",
        "ccxt_config": {
          "apiBackend": "paper"              // or "live"
        },
        "pair_whitelist": ["SPY/USD", "QQQ/USD", "AAPL/USD"]
      }
    }

`apiBackend: "paper"` => https://paper-api.alpaca.markets endpoint.
`apiBackend: "live"`  => https://api.alpaca.markets endpoint.

Limitaties / TODO
-----------------
- Geen crypto support (alleen stocks via StockHistoricalDataClient).
  Crypto kan later toegevoegd door `CryptoHistoricalDataClient` te
  integreren — buiten scope voor deze eerste versie.
- Geen options support — buiten scope voor v1, zit in Year-2 roadmap
  (zie WP2v2 §3.5).
- Geen WebSocket / live tick data. Polling-mode only. Voor live trading
  is dit fine (5-minute candles), voor lagere timeframes zou WS toegevoegd
  moeten worden.
- Pacing: Alpaca free-tier rate limits zijn niet expliciet ge-implementeerd.
  Voor backtest met 1 paar werkt dit; bij grote pair-lijsten + frequent
  refreshing moet rate-limiter toegevoegd worden.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from freqtrade.exceptions import (
    DDosProtection,
    OperationalException,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange._alpaca_shim import AlpacaCcxtShim
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping: Freqtrade timeframe string -> alpaca-py TimeFrame
# ---------------------------------------------------------------------------

# We construct alpaca-py TimeFrame objects lazily at call-time to keep
# this module importable even if alpaca-py is not yet imported.
_TIMEFRAME_MAP_HINT = {
    "1m": (1, "Minute"),
    "5m": (5, "Minute"),
    "15m": (15, "Minute"),
    "30m": (30, "Minute"),
    "1h": (1, "Hour"),
    "4h": (4, "Hour"),
    "1d": (1, "Day"),
    "1w": (1, "Week"),
    "1M": (1, "Month"),
}


# ---------------------------------------------------------------------------
# The Alpaca exchange plugin
# ---------------------------------------------------------------------------


class Alpaca(Exchange):
    """Freqtrade exchange plugin for Alpaca (paper + live).

    See module docstring for architecture overview.
    """

    # _ft_has overrides the base Exchange feature flags. These tell
    # Freqtrade what Alpaca can/cannot do, which influences validation,
    # backtesting modes, and order-handling logic.
    _ft_has: FtHas = {
        "ohlcv_candle_limit": 10000,  # alpaca-py max bars/page
        "stoploss_on_exchange": False,  # we manage stops in strategy
        "trades_pagination": "id",
        "trades_pagination_arg": "page_token",
        # Critical: prevents base Exchange from stripping our credentials
        # in dry_run mode. We need them to instantiate alpaca-py clients
        # even for backtest (download-data step needs market access).
        "always_require_api_keys": True,
        # Custom flags (read by our own code, ignored by base Exchange)
        "supports_fractional_shares": True,
        "market_hours_aware": True,
        "trading_calendar": "XNYS",  # NYSE calendar (covers most US listings)
        "ccxt_async_support": False,  # we don't use ccxt at all
    }

    # ---- Construction ---------------------------------------------------

    def __init__(
        self,
        config: dict[str, Any],
        *,
        exchange_config: dict[str, Any] | None = None,
        validate: bool = True,
        load_leverage_tiers: bool = False,
    ) -> None:
        """Initialize the Alpaca exchange.

        We rely on the base Exchange.__init__ flow but redirect the ccxt
        init via our overridden `_init_ccxt`. After base init completes,
        we instantiate the actual alpaca-py clients.
        """
        # alpaca-py clients — constructed AFTER base init runs
        # (because we need the resolved exchange config first).
        # CRITICAL: read credentials + paper mode BEFORE super().__init__()
        # because base Exchange.__init__ calls remove_exchange_credentials()
        # which mutates the exchange_conf dict and clears keys in dry_run mode.
        # We extract here as defensive belt-and-braces.
        ex_conf_for_creds = (
            exchange_config if exchange_config is not None else config["exchange"]
        )
        api_key = (
            ex_conf_for_creds.get("key")
            or ex_conf_for_creds.get("api_key")
            or ex_conf_for_creds.get("apiKey")
        )
        api_secret = ex_conf_for_creds.get("secret")
        api_backend = (
            ex_conf_for_creds.get("ccxt_config", {})
            .get("apiBackend", "paper")
            .lower()
        )
        self._alpaca_paper: bool = api_backend != "live"

        # alpaca-py clients (constructed below after super completes)
        self._alpaca_trading: Any | None = None
        self._alpaca_data: Any | None = None

        # Defer base init; call super now that defaults are set
        super().__init__(
            config,
            exchange_config=exchange_config,
            validate=False,  # SKIP base reload_markets at init time —
                              # we'll do it ourselves below if validate=True
            load_leverage_tiers=load_leverage_tiers,
        )

        # Validate captured credentials (already extracted above before super)
        if not api_key or not api_secret:
            if config.get("dry_run"):
                logger.warning(
                    "Alpaca: no credentials in dry_run mode. Backtest with cached "
                    "data works; live data fetches will fail. Set ALPACA_API_KEY_ID "
                    "+ ALPACA_API_SECRET_KEY in .env for live data."
                )
                # Build stub markets from pair_whitelist so backtest can proceed.
                # reload_markets has the same logic but is not called when we early-return,
                # so we duplicate it inline here.
                whitelist = self._config.get("exchange", {}).get("pair_whitelist", [])
                stub_markets = {
                    pair: {
                        "id": pair.replace("/", ""),
                        "symbol": pair,
                        "base": pair.split("/")[0],
                        "quote": pair.split("/")[1] if "/" in pair else "USD",
                        "active": True,
                        "type": "spot",
                        "spot": True,
                        "margin": False,
                        "future": False,
                        "swap": False,
                        "option": False,
                        "contract": False,
                        "linear": None,
                        "inverse": None,
                        "precision": {"amount": 8, "price": 0.01},
                        "limits": {
                            "amount": {"min": 0.0001, "max": None},
                            "price": {"min": 0.01, "max": None},
                            "cost": {"min": None, "max": None},
                        },
                        "info": {"alpaca_dry_run_stub": True},
                    }
                    for pair in whitelist
                }
                self._markets = stub_markets
                if hasattr(self._api, "markets"):
                    self._api.markets = stub_markets
                logger.info(
                    "Alpaca dry_run: built %d stub markets from pair_whitelist",
                    len(stub_markets),
                )
                if validate:
                    self.validate_config(self._config)
                return
            raise OperationalException(
                "Alpaca exchange requires `key` and `secret` in config "
                "(set ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY in .env "
                "and use the run_freqtrade.py wrapper to resolve placeholders)."
            )

        # Build alpaca-py clients (deferred imports keep base module
        # importable for unit tests that don't need real network).
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise OperationalException(
                "alpaca-py is not installed. Run: "
                "pip install 'alpaca-py>=0.40,<1.0'"
            ) from e

        # Trading client — used for orders, balance, positions
        self._alpaca_trading = TradingClient(
            api_key=api_key or "",
            secret_key=api_secret or "",
            paper=self._alpaca_paper,
        )

        # Data client — historical bars (free-tier IEX feed by default)
        self._alpaca_data = StockHistoricalDataClient(
            api_key=api_key or "",
            secret_key=api_secret or "",
        )

        logger.info(
            "Alpaca client initialized (paper=%s, key=%s...)",
            self._alpaca_paper,
            (api_key or "")[:4],
        )

        # Now run validation (which calls reload_markets via OUR override)
        if validate:
            self.reload_markets(True, load_leverage_tiers=False)
            self.validate_config(self._config)

    # ---- ccxt-bypass: override _init_ccxt ------------------------------

    def _init_ccxt(
        self,
        exchange_config: dict[str, Any],
        sync: bool,
        ccxt_kwargs: dict[str, Any],
    ) -> Any:
        """Override base: do NOT initialize real ccxt. Return our shim
        that satisfies base Exchange's `self._api` / `self._api_async`
        attribute access without actually connecting to anything.

        ccxt's own Alpaca driver fails on paper accounts (auth error
        40110000). By bypassing ccxt entirely we avoid that bug; we
        implement everything via alpaca-py SDK in this subclass.
        """
        logger.debug(
            "Alpaca._init_ccxt called (sync=%s) — returning shim, skipping real ccxt",
            sync,
        )
        # Pass our sync fetch_ohlcv as callback. The async shim wraps it
        # via asyncio.to_thread so base Exchange's await-flow (download-data,
        # historic_ohlcv) gets data without blocking the event loop.
        return AlpacaCcxtShim(
            config=exchange_config,
            is_async=not sync,
            fetch_ohlcv_callback=self._fetch_ohlcv_pair_args,
        )

    # ---- Markets reload ------------------------------------------------

    def reload_markets(
        self, force: bool = False, *, load_leverage_tiers: bool = True
    ) -> None:
        """Override base: reload our markets dict from Alpaca's REST API.

        Base Exchange's reload_markets calls into async ccxt; we replace
        it entirely with a sync alpaca-py call. Result is stored on
        ``self._markets`` AND on ``self._api.markets`` (for any base
        code that reads from the shim).
        """
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        if not self._alpaca_trading:
            # In dry_run without API credentials: build minimal stub markets
            # from pair_whitelist so backtest can proceed against cached data.
            # No live network calls required.
            if self._config.get("dry_run"):
                whitelist = self._config.get("exchange", {}).get("pair_whitelist", [])
                stub_markets = {
                    pair: {
                        "id": pair.replace("/", ""),
                        "symbol": pair,
                        "base": pair.split("/")[0],
                        "quote": pair.split("/")[1] if "/" in pair else "USD",
                        "active": True,
                        "type": "spot",
                        "spot": True,
                        "margin": False,
                        "future": False,
                        "swap": False,
                        "option": False,
                        "contract": False,
                        "linear": None,
                        "inverse": None,
                        "precision": {"amount": 8, "price": 0.01},
                        "limits": {
                            "amount": {"min": 0.0001, "max": None},
                            "price": {"min": 0.01, "max": None},
                            "cost": {"min": None, "max": None},
                        },
                        "info": {"alpaca_dry_run_stub": True},
                    }
                    for pair in whitelist
                }
                self._markets = stub_markets
                if hasattr(self._api, "markets"):
                    self._api.markets = stub_markets
                logger.info(
                    "Alpaca dry_run: built %d stub markets from pair_whitelist",
                    len(stub_markets),
                )
                return
            # Should not happen — __init__ initialized this. Guard for
            # tests where the exchange might be partially constructed.
            logger.warning("reload_markets called before _alpaca_trading set")
            return

        # Configurable: stocks-only by default. Multi-asset support
        # (crypto, options) would need additional asset-class loops.
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self._alpaca_trading.get_all_assets(request)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(
                f"Could not fetch Alpaca markets: {type(e).__name__}: {e}"
            ) from e

        # Build markets dict in ccxt-compatible shape
        markets: dict[str, dict[str, Any]] = {}
        for asset in assets:
            symbol = asset.symbol
            venue = asset.exchange.value if asset.exchange else "XNAS"
            # Build BOTH plain (`SPY/USD`) and venue-encoded (`SPY_XNAS/USD`)
            # entries so users can use either form. Plain wins by default.
            for pair in (f"{symbol}/USD", f"{symbol}_{venue}/USD"):
                markets[pair] = {
                    "id": asset.id.hex if hasattr(asset.id, "hex") else str(asset.id),
                    "symbol": pair,
                    "base": symbol,
                    "quote": "USD",
                    "active": asset.tradable,
                    "type": "spot",
                    "spot": True,
                    "margin": asset.marginable,
                    "future": False,
                    "swap": False,
                    "option": False,
                    "contract": False,
                    "linear": None,
                    "inverse": None,
                    # Alpaca-specific flags surfaced for our strategy code
                    "info": {
                        "alpaca_asset_id": str(asset.id),
                        "venue": venue,
                        "fractionable": asset.fractionable,
                        "shortable": asset.shortable,
                        "easy_to_borrow": asset.easy_to_borrow,
                        "marginable": asset.marginable,
                    },
                    # Required ccxt-style precision/limits stubs
                    "precision": {
                        "price": 0.01,  # most US stocks tick in $0.01
                        "amount": 1.0 if not asset.fractionable else 0.000001,
                    },
                    "limits": {
                        "amount": {"min": 0.000001 if asset.fractionable else 1, "max": None},
                        "price": {"min": None, "max": None},
                        "cost": {"min": 1.0, "max": None},
                    },
                }

        self._markets = markets
        # Also update the shim's markets attribute for any base code
        # that reads from self._api.markets
        if hasattr(self._api, "markets"):
            self._api.markets = markets
        if hasattr(self._api_async, "markets"):
            self._api_async.markets = markets

        self._last_markets_refresh = int(datetime.now(timezone.utc).timestamp() * 1000)
        logger.info(
            "Loaded %d Alpaca markets (US equities, active+tradable)",
            len([m for m in markets.values() if m["active"]]),
        )

    async def _api_reload_markets(self, reload: bool = False) -> None:
        """Async variant called by base Exchange. Delegates to the sync
        version since alpaca-py is sync-only."""
        # alpaca-py is sync-only; just delegate.
        self.reload_markets(force=reload)

    # ---- Pair-encoding helpers -----------------------------------------

    @staticmethod
    def _parse_pair(pair: str) -> tuple[str, str, str]:
        """Decode a pair-string into (symbol, venue, currency).

        Examples:
            >>> Alpaca._parse_pair("SPY/USD")
            ('SPY', 'XNAS', 'USD')
            >>> Alpaca._parse_pair("SPY_XNAS/USD")
            ('SPY', 'XNAS', 'USD')
            >>> Alpaca._parse_pair("BRK.B_XNYS/USD")
            ('BRK.B', 'XNYS', 'USD')

        Raises:
            ValueError if pair format is invalid.
        """
        if "/" not in pair:
            raise ValueError(f"Invalid pair {pair!r}: missing '/'")
        symbol_part, currency = pair.split("/", 1)
        if not symbol_part or not currency:
            raise ValueError(f"Invalid pair {pair!r}: empty symbol or currency")
        if "_" in symbol_part:
            # SYMBOL_VENUE encoding
            # Use rsplit so symbols with dots like BRK.B work
            symbol, venue = symbol_part.rsplit("_", 1)
        else:
            symbol, venue = symbol_part, "XNAS"
        return symbol, venue, currency

    @staticmethod
    def _to_alpaca_timeframe(timeframe: str) -> Any:
        """Convert Freqtrade timeframe string to alpaca-py TimeFrame object.

        Raises:
            OperationalException if the timeframe is not supported.
        """
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if timeframe not in _TIMEFRAME_MAP_HINT:
            raise OperationalException(
                f"Timeframe {timeframe!r} not supported by Alpaca plugin. "
                f"Supported: {list(_TIMEFRAME_MAP_HINT.keys())}"
            )
        amount, unit_name = _TIMEFRAME_MAP_HINT[timeframe]
        unit = getattr(TimeFrameUnit, unit_name)
        return TimeFrame(amount, unit)

    # ---- OHLCV ---------------------------------------------------------

    def _fetch_ohlcv_pair_args(
        self,
        pair: str,
        timeframe: str,
        since_ms: int | None = None,
        candle_type: Any = None,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ) -> list[list[float]]:
        """Shim-callable adapter for async fetch via asyncio.to_thread."""
        return self.fetch_ohlcv(
            pair=pair, timeframe=timeframe, since_ms=since_ms,
            candle_type=candle_type, is_new_pair=is_new_pair, until_ms=until_ms,
        )

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int | None = None,
        candle_type: Any = None,
        is_new_pair: bool = False,
        until_ms: int | None = None,
    ) -> list[list[float]]:
        """Fetch historical OHLCV bars from Alpaca.

        Returns ccxt-style list of [ts_ms, open, high, low, close, volume].
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest

        if not self._alpaca_data:
            raise OperationalException("Alpaca data client not initialized")

        symbol, _venue, _currency = self._parse_pair(pair)
        alpaca_tf = self._to_alpaca_timeframe(timeframe)
        # Free tier accounts can only access IEX feed; SIP requires paid plan.
        # Make this configurable later via _ft_has["alpaca_data_feed"].
        feed = DataFeed.IEX

        # Time bounds
        if since_ms is not None:
            start = datetime.fromtimestamp(since_ms / 1000.0, tz=timezone.utc)
        else:
            # Default: lookback 1000 candles worth of time
            tf_seconds = self._tf_to_seconds(timeframe)
            start = datetime.now(timezone.utc) - timedelta(
                seconds=tf_seconds * 1000
            )
        if until_ms is not None:
            end = datetime.fromtimestamp(until_ms / 1000.0, tz=timezone.utc)
        else:
            end = datetime.now(timezone.utc)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_tf,
                start=start,
                end=end,
                feed=feed,  # IEX = free tier; SIP requires paid Alpaca plan
            )
            bars_set = self._alpaca_data.get_stock_bars(req)
        except Exception as e:  # noqa: BLE001
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                raise DDosProtection(f"Alpaca rate limit: {e}") from e
            raise TemporaryError(f"Alpaca fetch_ohlcv failed: {e}") from e

        # Extract bars list. alpaca-py returns either a BarSet (single
        # symbol) or dict[str, BarSet] (multi-symbol). We're single-symbol.
        if hasattr(bars_set, "data"):
            # BarSet has .data dict of symbol -> list[Bar]
            bars = bars_set.data.get(symbol, [])
        elif isinstance(bars_set, dict):
            bars = bars_set.get(symbol, [])
        else:
            bars = []

        result: list[list[float]] = []
        for bar in bars:
            # Bar.timestamp is a datetime, convert to ms epoch
            ts_ms = int(bar.timestamp.timestamp() * 1000)
            result.append([
                ts_ms,
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
            ])
        logger.debug("fetch_ohlcv(%s, %s): %d bars", pair, timeframe, len(result))
        return result

    @staticmethod
    def _tf_to_seconds(timeframe: str) -> int:
        """Convert Freqtrade timeframe to seconds. Helper for date math."""
        amount, unit = _TIMEFRAME_MAP_HINT[timeframe]
        unit_seconds = {
            "Minute": 60,
            "Hour": 3600,
            "Day": 86400,
            "Week": 86400 * 7,
            "Month": 86400 * 30,
        }
        return amount * unit_seconds[unit]

    # ---- Account / balance ---------------------------------------------

    def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return account balance in ccxt-compatible shape."""
        if not self._alpaca_trading:
            raise OperationalException("Alpaca trading client not initialized")
        try:
            account = self._alpaca_trading.get_account()
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"Alpaca fetch_balance failed: {e}") from e

        cash = float(account.cash)
        equity = float(account.equity)
        return {
            "info": {
                "alpaca_account_id": str(account.id),
                "status": str(account.status),
                "buying_power": float(account.buying_power),
                "regt_buying_power": float(account.regt_buying_power),
                "daytrading_buying_power": float(account.daytrading_buying_power),
                "portfolio_value": equity,
                "pattern_day_trader": account.pattern_day_trader,
            },
            "USD": {
                "free": cash,
                "used": equity - cash,
                "total": equity,
            },
            "free": {"USD": cash},
            "used": {"USD": equity - cash},
            "total": {"USD": equity},
        }

    # ---- Ticker --------------------------------------------------------

    def fetch_ticker(self, pair: str) -> dict[str, Any]:
        """Fetch current bid/ask/last for a pair via the latest 1-minute bar.

        Alpaca's free tier has a SIP/IEX bid-ask quote endpoint but it's
        15-minute delayed for free. For backtest/dry_run we use the latest
        1m bar's close as a proxy. For live trading you'd want
        StockLatestQuoteRequest with paid SIP feed.
        """
        from alpaca.data.requests import StockLatestQuoteRequest

        if not self._alpaca_data:
            raise OperationalException("Alpaca data client not initialized")

        symbol, _venue, _currency = self._parse_pair(pair)
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._alpaca_data.get_stock_latest_quote(req)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"Alpaca fetch_ticker failed: {e}") from e

        # quotes is dict[symbol -> Quote]
        quote = quotes.get(symbol) if isinstance(quotes, dict) else None
        if not quote:
            raise OperationalException(
                f"No quote returned for {symbol!r} from Alpaca"
            )

        bid = float(quote.bid_price) if quote.bid_price else None
        ask = float(quote.ask_price) if quote.ask_price else None
        last = (bid + ask) / 2.0 if bid and ask else (bid or ask or 0.0)

        return {
            "symbol": pair,
            "timestamp": int(quote.timestamp.timestamp() * 1000),
            "datetime": quote.timestamp.isoformat(),
            "bid": bid,
            "ask": ask,
            "last": last,
            "close": last,
            "vwap": None,
            "open": None,
            "high": None,
            "low": None,
            "previousClose": None,
            "change": None,
            "percentage": None,
            "average": last,
            "baseVolume": float(quote.bid_size or 0) + float(quote.ask_size or 0),
            "quoteVolume": None,
            "info": {
                "bid_size": float(quote.bid_size or 0),
                "ask_size": float(quote.ask_size or 0),
                "bid_exchange": quote.bid_exchange,
                "ask_exchange": quote.ask_exchange,
            },
        }

    # ---- Orders --------------------------------------------------------

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: str,
        amount: float,
        rate: float | None = None,
        leverage: float = 1.0,
        reduceOnly: bool = False,
        time_in_force: str = "GTC",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Place an order via alpaca-py.

        Maps Freqtrade order semantics to Alpaca's:
        - ordertype 'market' -> MarketOrderRequest
        - ordertype 'limit'  -> LimitOrderRequest (rate required)
        - side 'buy' / 'sell'
        - time_in_force 'GTC' / 'DAY' / 'IOC' / 'FOK' / 'OPG' / 'CLS'
        """
        from alpaca.trading.enums import (
            OrderSide as AlpacaSide,
            TimeInForce as AlpacaTif,
        )
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
        )

        if not self._alpaca_trading:
            raise OperationalException("Alpaca trading client not initialized")

        symbol, _venue, _currency = self._parse_pair(pair)
        alpaca_side = AlpacaSide.BUY if side.lower() == "buy" else AlpacaSide.SELL

        # Map TIF; default GTC if unrecognized
        tif_map = {
            "GTC": AlpacaTif.GTC,
            "DAY": AlpacaTif.DAY,
            "IOC": AlpacaTif.IOC,
            "FOK": AlpacaTif.FOK,
            "OPG": AlpacaTif.OPG,
            "CLS": AlpacaTif.CLS,
        }
        alpaca_tif = tif_map.get(time_in_force.upper(), AlpacaTif.GTC)

        # Fractional shares only allowed for MARKET orders with DAY TIF
        # (per Alpaca's API docs). Round to whole share if not eligible.
        is_fractional = amount != int(amount)
        if is_fractional and (ordertype.lower() != "market" or alpaca_tif != AlpacaTif.DAY):
            logger.warning(
                "Fractional share %g requested for %s order but Alpaca only "
                "allows fractional on MARKET+DAY. Rounding down to %d.",
                amount, ordertype, int(amount),
            )
            amount = float(int(amount))

        try:
            if ordertype.lower() == "market":
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=amount,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                )
            elif ordertype.lower() == "limit":
                if rate is None:
                    raise OperationalException(
                        "Limit order requires `rate` parameter"
                    )
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=amount,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=rate,
                )
            else:
                raise OperationalException(
                    f"Order type {ordertype!r} not supported by Alpaca plugin "
                    f"(use 'market' or 'limit')"
                )
            order = self._alpaca_trading.submit_order(req)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"Alpaca create_order failed: {e}") from e

        return self._alpaca_order_to_ccxt(order, pair)

    def fetch_order(
        self, order_id: str, pair: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fetch an order by ID."""
        if not self._alpaca_trading:
            raise OperationalException("Alpaca trading client not initialized")
        try:
            order = self._alpaca_trading.get_order_by_id(order_id)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"Alpaca fetch_order({order_id}) failed: {e}") from e
        return self._alpaca_order_to_ccxt(order, pair)

    def cancel_order(
        self, order_id: str, pair: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Cancel an order by ID. Alpaca returns nothing on success;
        we synthesize a ccxt-style response."""
        if not self._alpaca_trading:
            raise OperationalException("Alpaca trading client not initialized")
        try:
            self._alpaca_trading.cancel_order_by_id(order_id)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"Alpaca cancel_order({order_id}) failed: {e}") from e
        return {
            "id": order_id,
            "symbol": pair,
            "status": "canceled",
            "info": {"alpaca_canceled": True},
        }

    # ---- Order conversion helpers --------------------------------------

    @staticmethod
    def _alpaca_order_to_ccxt(order: Any, pair: str) -> dict[str, Any]:
        """Convert an alpaca-py Order into a ccxt-compatible order dict."""
        # Alpaca's status values map to ccxt as follows:
        status_map = {
            "new": "open",
            "accepted": "open",
            "pending_new": "open",
            "accepted_for_bidding": "open",
            "partially_filled": "open",
            "filled": "closed",
            "done_for_day": "closed",
            "canceled": "canceled",
            "expired": "canceled",
            "replaced": "canceled",
            "pending_cancel": "open",
            "pending_replace": "open",
            "rejected": "canceled",
            "suspended": "canceled",
        }
        alpaca_status = str(order.status.value if hasattr(order.status, "value") else order.status)
        ccxt_status = status_map.get(alpaca_status.lower(), "open")

        filled_qty = float(order.filled_qty or 0)
        order_qty = float(order.qty or 0)
        avg_price = float(order.filled_avg_price or 0) if order.filled_avg_price else None
        limit_price = float(order.limit_price or 0) if order.limit_price else None

        return {
            "id": str(order.id),
            "clientOrderId": str(order.client_order_id) if order.client_order_id else None,
            "timestamp": int(order.submitted_at.timestamp() * 1000) if order.submitted_at else None,
            "datetime": order.submitted_at.isoformat() if order.submitted_at else None,
            "lastTradeTimestamp": (
                int(order.filled_at.timestamp() * 1000) if order.filled_at else None
            ),
            "symbol": pair,
            "type": str(order.order_type.value if hasattr(order.order_type, "value") else order.order_type),
            "timeInForce": str(order.time_in_force.value if hasattr(order.time_in_force, "value") else order.time_in_force),
            "side": str(order.side.value if hasattr(order.side, "value") else order.side),
            "price": limit_price,
            "average": avg_price,
            "amount": order_qty,
            "filled": filled_qty,
            "remaining": order_qty - filled_qty,
            "status": ccxt_status,
            "fee": None,
            "trades": [],
            "info": {
                "alpaca_order_id": str(order.id),
                "alpaca_status": alpaca_status,
            },
        }

    # ---- Validation overrides ------------------------------------------

    def validate_pairs(self, pairs: list[str]) -> None:
        """Override base: ensure pairs exist in our markets dict.

        Base Exchange has more elaborate validation (precision, limits,
        etc.) — for stocks the simpler check (does this asset exist on
        Alpaca + is it tradable) is enough.
        """
        unknown_pairs = [p for p in pairs if p not in self._markets]
        if unknown_pairs:
            raise OperationalException(
                f"Pairs not available on Alpaca: {unknown_pairs}. "
                f"Run reload_markets() to refresh, or check ticker spelling."
            )
        inactive = [
            p for p in pairs if not self._markets[p].get("active", True)
        ]
        if inactive:
            logger.warning(
                "Some pairs are not currently tradable: %s", inactive
            )
