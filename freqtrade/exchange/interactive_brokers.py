"""IBKR (Interactive Brokers) exchange plugin voor Freqtrade.

Doel
----
Een `freqtrade.exchange.Exchange` subclass die transparant Interactive
Brokers (IBKR Ireland / IBIE) als broker behandelt voor multi-asset
trading: US stocks (NYSE/NASDAQ/AMEX), EU stocks (Xetra/Euronext/LSE),
forex, futures, options.

Architectuur
------------
Drie lagen:

1. **ib_async SDK** (de 2024-rebrand van het abandoned `ib_insync`)
   - `IB`: de hoofdclient, persistent socket connectie naar TWS Gateway
   - Sync API met internal asyncio loop (we gebruiken sync vanwege
     thread-safety en Freqtrade's mixed sync/async patterns)
   - Auth: GEEN — de TWS Gateway zelf handelt auth via username/password
     bij gateway-startup; deze code communiceert alleen met localhost.

2. **IbkrCcxtShim** (in `_ibkr_shim.py`)
   - Minimale stand-in voor ccxt.Exchange instance
   - Voldoet aan de attributen die `Exchange.__init__` verwacht
   - Methods raise `_ShimError` als ze worden gehit (defensief)

3. **InteractiveBrokers(Exchange)** (deze class)
   - Override `_init_ccxt` om de shim te returnen ipv echte ccxt
   - Override `_api_reload_markets` om markets lazy te laden via reqContractDetails
   - Override `fetch_ohlcv`, `fetch_ticker`, `create_order`, `cancel_order`,
     `fetch_order`, `fetch_balance` om ib_async te gebruiken

Pair-encoding
-------------
Volgt de research-aanbeveling uit `02_upstream_compatible_fork.md` §"De
pair string encoding trick", aangepast voor IBKR's multi-venue model:

    SYMBOL[_PRIMARY_EXCHANGE]/CURRENCY

Waar PRIMARY_EXCHANGE een MIC code is (XNYS, XNAS, XETR, XAMS, XLON, ...).
Voor onbekende of ambigue symbols laten we het weg en gebruikt IBKR's
SMART routing (algorithmic best-fill).

Voorbeelden:
    `SPY/USD`        → Stock("SPY", "SMART", "USD")
    `SPY_ARCX/USD`   → Stock("SPY", "ARCA", "USD") (NYSE Arca expliciet)
    `AAPL_XNAS/USD`  → Stock("AAPL", "NASDAQ", "USD")
    `SAP_XETR/EUR`   → Stock("SAP", "IBIS", "EUR") (Xetra)
    `ASML_XAMS/EUR`  → Stock("ASML", "AEB", "EUR") (Euronext Amsterdam)
    `BRK.B_XNYS/USD` → Stock("BRK B", "NYSE", "USD") (note IB's space!)

Configuratie
------------
In `config.json`:

    {
      "exchange": {
        "name": "interactivebrokers",
        "key": "",       // niet nodig — TWS Gateway handelt auth
        "secret": "",    // niet nodig
        "ccxt_config": {
          "ibkr_host": "127.0.0.1",
          "ibkr_port": 4002,            // 4002=paper, 4001=live
          "ibkr_client_id": 1,          // unique per concurrent connection
          "ibkr_account": "",           // optional, leeg = first account
          "ibkr_paper": true,           // sanity check
          "ibkr_market_data_type": 1    // 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
        },
        "pair_whitelist": ["SPY/USD", "AAPL_XNAS/USD"]
      }
    }

Voorvereisten
-------------
- TWS Gateway draait op de geconfigureerde host:port
- API connections enabled in Gateway settings
- "Read-Only API" UIT als je orders wilt plaatsen
- Trusted IPs: 127.0.0.1 toegestaan
- Voor NL-residents: gebruik IBIE (Ireland), niet IBLLC (US)

Limitaties / TODO
-----------------
- Geen options support — buiten scope voor v1, zit in Year-2 roadmap
- Geen futures support — buiten scope voor v1
- Geen forex direct (kan via IDEALPRO maar buiten scope)
- Markets worden lazy geladen on-demand: bij eerste validate_pairs call
  doen we reqContractDetails per pair en cachen resultaat. Dit voorkomt
  het massive issue van IBKR die geen "list all symbols" endpoint heeft.
- Pacing: IBKR rate limit ~50 msg/sec is via ib_async automatisch gerespecteerd.
  Voor backtest met 1 paar werkt dit; bij grote pair-lijsten + frequent
  refreshing moet expliciete pacing toegevoegd worden.
- Cash account (NL/IBIE default) heeft T+1 settlement. We tracken dit
  niet in het plugin; strategy moet hier rekening mee houden.
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
from freqtrade.exchange._ibkr_shim import IbkrCcxtShim
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping: Freqtrade timeframe string -> IBKR barSize string
# ---------------------------------------------------------------------------
# IBKR's reqHistoricalData accepts these exact strings. Mismatch = error.

_TIMEFRAME_TO_BARSIZE = {
    "1s": "1 secs",
    "5s": "5 secs",
    "15s": "15 secs",
    "30s": "30 secs",
    "1m": "1 min",
    "2m": "2 mins",
    "3m": "3 mins",
    "5m": "5 mins",
    "10m": "10 mins",
    "15m": "15 mins",
    "20m": "20 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "2h": "2 hours",
    "3h": "3 hours",
    "4h": "4 hours",
    "8h": "8 hours",
    "1d": "1 day",
    "1w": "1W",
    "1M": "1M",
}

# Mapping: MIC code (in pair-encoding) -> IBKR exchange code (used by SDK)
# IBKR doesn't use MIC directly; it has its own short codes per venue.
# We translate at parse time.
_MIC_TO_IBKR_EXCHANGE = {
    # US
    "XNYS": "NYSE",       # New York Stock Exchange
    "XNAS": "NASDAQ",     # NASDAQ
    "ARCX": "ARCA",       # NYSE Arca (ETF venue)
    "BATS": "BATS",       # Cboe BZX (formerly BATS)
    "XASE": "AMEX",       # NYSE American (formerly AMEX)
    "IEXG": "IEX",        # Investors Exchange
    # Europe
    "XETR": "IBIS",       # Xetra (Frankfurt) — IBKR uses IBIS code
    "XFRA": "FWB",        # Frankfurt Stock Exchange
    "XAMS": "AEB",        # Euronext Amsterdam
    "XBRU": "ENEXT.BE",   # Euronext Brussels
    "XPAR": "SBF",        # Euronext Paris
    "XLIS": "BVL",        # Euronext Lisbon
    "XLON": "LSE",        # London Stock Exchange
    "XSWX": "EBS",        # SIX Swiss Exchange
    "XMIL": "BVME",       # Borsa Italiana / Euronext Milan
    "XMAD": "BM",         # Bolsa de Madrid
    "XSTO": "SFB",        # Stockholm
    "XCSE": "OMS",        # Copenhagen
    "XHEL": "OMSHEX",     # Helsinki
    "XOSL": "OSE",        # Oslo
    # Asia
    "XHKG": "SEHK",       # Hong Kong
    "XTKS": "TSEJ",       # Tokyo (TOKYO)
    "XSES": "SGX",        # Singapore
    "XSHG": "SEHKNTL",    # Shanghai (via Stock Connect)
    "XSHE": "SEHKSZSE",   # Shenzhen (via Stock Connect)
    # Default
    "SMART": "SMART",     # IBKR's algorithmic routing (default)
}


class InteractiveBrokers(Exchange):
    """Freqtrade exchange plugin for IBKR (paper + live via TWS Gateway).

    See module docstring for architecture overview.
    """

    # _ft_has overrides the base Exchange feature flags. These tell
    # Freqtrade what IBKR can/cannot do, which influences validation,
    # backtesting modes, and order-handling logic.
    _ft_has: FtHas = {
        "ohlcv_candle_limit": 5000,  # IBKR ~max bars per reqHistoricalData
        "stoploss_on_exchange": False,  # could enable later via STP order
        "trades_pagination": "id",
        "trades_pagination_arg": "page_token",
        # Critical: prevents base Exchange from stripping our credentials
        # in dry_run mode. IBKR doesn't use API keys (TWS Gateway handles
        # auth) but base Exchange may still try to validate the field.
        "always_require_api_keys": False,
        # Custom flags (read by our own code, ignored by base Exchange)
        "supports_fractional_shares": True,    # IBKR supports fractional
        "market_hours_aware": True,
        "trading_calendar": "MULTI",            # multi-venue: NYSE, XETR, ...
        "ccxt_async_support": False,            # we don't use ccxt at all
        "supports_multi_currency": True,        # IBKR account multi-CCY
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
        """Initialize the IBKR exchange.

        We rely on the base Exchange.__init__ flow but redirect the ccxt
        init via our overridden `_init_ccxt`. After base init completes,
        we instantiate the actual ib_async IB client and connect to TWS
        Gateway.
        """
        # Read connection params BEFORE super().__init__()  (defensive,
        # base Exchange may mutate the config dict during init)
        ex_conf = (
            exchange_config if exchange_config is not None else config["exchange"]
        )
        ccxt_conf = ex_conf.get("ccxt_config", {})
        self._ibkr_host: str = ccxt_conf.get("ibkr_host", "127.0.0.1")
        self._ibkr_port: int = int(ccxt_conf.get("ibkr_port", 4002))
        self._ibkr_client_id: int = int(ccxt_conf.get("ibkr_client_id", 1))
        self._ibkr_account: str = str(ccxt_conf.get("ibkr_account", ""))
        self._ibkr_paper: bool = bool(ccxt_conf.get("ibkr_paper", True))
        # Market data type:
        #   1 = real-time (subscription required for most exchanges)
        #   2 = frozen (last price seen, no streaming)
        #   3 = delayed (15min delay, free for most US/EU)
        #   4 = delayed + frozen
        self._ibkr_market_data_type: int = int(
            ccxt_conf.get("ibkr_market_data_type", 3)
        )

        # Sanity check: 4002 is paper, 4001 is live, 7497 is TWS paper, 7496 TWS live
        if self._ibkr_paper and self._ibkr_port in (4001, 7496):
            logger.warning(
                "IBKR config says paper=True but port %d is the LIVE port. "
                "Verify ibkr_port matches your Gateway/TWS configuration.",
                self._ibkr_port,
            )
        if not self._ibkr_paper and self._ibkr_port in (4002, 7497):
            logger.warning(
                "IBKR config says paper=False but port %d is the PAPER port. "
                "Verify ibkr_port matches your Gateway/TWS configuration.",
                self._ibkr_port,
            )

        # ib_async IB client (constructed below after super completes)
        self._ib: Any | None = None

        # Cache of resolved Contract objects per pair-string. IBKR has no
        # "list all symbols" endpoint, so we lazily resolve via
        # reqContractDetails per pair on first use.
        self._contract_cache: dict[str, Any] = {}

        # Defer base init; call super now that defaults are set
        super().__init__(
            config,
            exchange_config=exchange_config,
            validate=False,  # SKIP base reload_markets at init time —
                              # we'll do it ourselves below if validate=True
            load_leverage_tiers=load_leverage_tiers,
        )

        # Build ib_async IB client (deferred imports keep base module
        # importable for unit tests that don't need real network).
        try:
            from ib_async import IB
        except ImportError as e:
            raise OperationalException(
                "ib_async is not installed. Run: "
                "pip install 'ib_async>=2.0,<3.0'"
            ) from e

        self._ib = IB()

        # Connect to TWS Gateway (synchronous). Will block briefly until
        # handshake completes. If Gateway is not running, ConnectionRefused.
        try:
            self._ib.connect(
                host=self._ibkr_host,
                port=self._ibkr_port,
                clientId=self._ibkr_client_id,
                account=self._ibkr_account or "",
                readonly=False,  # we want order placement
                timeout=10,
            )
        except ConnectionRefusedError as e:
            if config.get("dry_run"):
                logger.warning(
                    "IBKR Gateway not reachable at %s:%d. dry_run=True so "
                    "continuing with disconnected client (cached data only).",
                    self._ibkr_host, self._ibkr_port,
                )
                self._ib = None
                if validate:
                    self.validate_config(self._config)
                return
            raise OperationalException(
                f"Cannot connect to IBKR Gateway at {self._ibkr_host}:"
                f"{self._ibkr_port}. Ensure TWS Gateway is running and API "
                f"connections enabled. (ClientId={self._ibkr_client_id})"
            ) from e
        except Exception as e:  # noqa: BLE001
            if config.get("dry_run"):
                logger.warning(
                    "IBKR connect failed (%s). dry_run=True; continuing "
                    "with disconnected client.", e,
                )
                self._ib = None
                if validate:
                    self.validate_config(self._config)
                return
            raise OperationalException(
                f"IBKR connect failed: {e}"
            ) from e

        # Set market data type (1=live, 3=delayed, etc.)
        try:
            self._ib.reqMarketDataType(self._ibkr_market_data_type)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to set market data type: %s", e)

        logger.info(
            "IBKR client connected (host=%s, port=%d, clientId=%d, paper=%s, mktDataType=%d)",
            self._ibkr_host, self._ibkr_port, self._ibkr_client_id,
            self._ibkr_paper, self._ibkr_market_data_type,
        )

        # Now run validation (which calls reload_markets via OUR override)
        if validate:
            self.reload_markets(True, load_leverage_tiers=False)
            self.validate_config(self._config)

    def __del__(self) -> None:
        """Disconnect from IBKR on garbage collection. Defensive."""
        try:
            ib = getattr(self, "_ib", None)
            if ib is not None and getattr(ib, "isConnected", lambda: False)():
                ib.disconnect()
        except Exception:  # noqa: BLE001
            pass

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

        ccxt has no IBKR driver; bypassing entirely is the only option.
        We implement everything via ib_async SDK in this subclass.
        """
        logger.debug(
            "InteractiveBrokers._init_ccxt called (sync=%s) — returning "
            "shim, skipping real ccxt", sync,
        )
        return IbkrCcxtShim(
            config=exchange_config,
            is_async=not sync,
            fetch_ohlcv_callback=self._fetch_ohlcv_pair_args,
        )

    # ---- Markets reload ------------------------------------------------

    def reload_markets(
        self, force: bool = False, *, load_leverage_tiers: bool = True
    ) -> None:
        """Override base: lazy-build markets dict from pair_whitelist.

        Unlike Alpaca which has a `get_all_assets()` bulk endpoint, IBKR
        only allows symbol lookup via `reqContractDetails(Contract)`.
        Calling this for thousands of symbols would be slow + rate-limited.

        Strategy: lazy-resolve contracts on demand. For each pair in
        `pair_whitelist`, call reqContractDetails and cache the result.
        Build a minimal ccxt-style market dict from those.
        """
        if not self._ib or not self._ib.isConnected():
            logger.warning(
                "IBKR reload_markets skipped: client not connected. Markets dict empty."
            )
            self._markets = {}
            if hasattr(self._api, "markets"):
                self._api.markets = {}
            return

        # Get the pair_whitelist from config (the only way we know which
        # symbols to resolve).
        whitelist = (
            self._config.get("exchange", {}).get("pair_whitelist", [])
        )
        if not whitelist:
            logger.warning(
                "IBKR reload_markets: pair_whitelist is empty. No markets "
                "loaded. Add pairs to your config to use IBKR."
            )
            self._markets = {}
            if hasattr(self._api, "markets"):
                self._api.markets = {}
            return

        markets: dict[str, Any] = {}
        for pair in whitelist:
            try:
                contract = self._resolve_contract(pair)
                if contract is None:
                    logger.warning("Could not resolve %s on IBKR", pair)
                    continue
                # Build ccxt-style market dict
                markets[pair] = self._contract_to_market_dict(pair, contract)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to resolve %s on IBKR: %s", pair, e)

        self._markets = markets
        if hasattr(self._api, "markets"):
            self._api.markets = markets

        logger.info(
            "Loaded %d IBKR markets (lazy from pair_whitelist of %d)",
            len(markets), len(whitelist),
        )

    async def _api_reload_markets(self, reload: bool = False) -> None:
        """Async variant called by base Exchange. Delegates to sync."""
        self.reload_markets(force=reload)


    # ---- Pair-encoding helpers -----------------------------------------

    @staticmethod
    def _parse_pair(pair: str) -> tuple[str, str, str]:
        """Decode a pair-string into (symbol, ibkr_exchange, currency).

        Examples:
            >>> InteractiveBrokers._parse_pair("SPY/USD")
            ('SPY', 'SMART', 'USD')
            >>> InteractiveBrokers._parse_pair("AAPL_XNAS/USD")
            ('AAPL', 'NASDAQ', 'USD')
            >>> InteractiveBrokers._parse_pair("SAP_XETR/EUR")
            ('SAP', 'IBIS', 'EUR')
            >>> InteractiveBrokers._parse_pair("BRK.B_XNYS/USD")
            ('BRK B', 'NYSE', 'USD')

        Note: For Berkshire Hathaway-style symbols with dots, IBKR uses
        a SPACE instead of a dot ("BRK B" not "BRK.B"). We translate
        automatically: any dot in the symbol part becomes a space.

        Raises:
            ValueError if pair format is invalid.
        """
        if "/" not in pair:
            raise ValueError(f"Invalid pair {pair!r}: missing '/'")
        symbol_part, currency = pair.split("/", 1)
        if not symbol_part or not currency:
            raise ValueError(f"Invalid pair {pair!r}: empty symbol or currency")
        if "_" in symbol_part:
            symbol, mic = symbol_part.rsplit("_", 1)
        else:
            symbol, mic = symbol_part, "SMART"

        # Translate IBKR's dot -> space convention
        ibkr_symbol = symbol.replace(".", " ")

        # Translate MIC code to IBKR's own exchange identifier
        ibkr_exchange = _MIC_TO_IBKR_EXCHANGE.get(mic.upper(), mic.upper())

        return ibkr_symbol, ibkr_exchange, currency

    @staticmethod
    def _to_ibkr_barsize(timeframe: str) -> str:
        """Convert Freqtrade timeframe string to IBKR barSize string.

        Raises:
            OperationalException if the timeframe is not supported.
        """
        if timeframe not in _TIMEFRAME_TO_BARSIZE:
            raise OperationalException(
                f"Timeframe {timeframe!r} not supported by IBKR plugin. "
                f"Supported: {list(_TIMEFRAME_TO_BARSIZE.keys())}"
            )
        return _TIMEFRAME_TO_BARSIZE[timeframe]

    def _resolve_contract(self, pair: str) -> Any | None:
        """Resolve a pair-string to an ib_async Contract object.

        Caches results to avoid repeated reqContractDetails calls.
        Returns None if the contract cannot be resolved (e.g. symbol
        doesn't exist on IBKR).
        """
        if pair in self._contract_cache:
            return self._contract_cache[pair]

        if not self._ib or not self._ib.isConnected():
            return None

        from ib_async import Stock

        symbol, exchange, currency = self._parse_pair(pair)
        # Build a Stock contract — most common case for our use.
        # For options/futures we'd need separate logic (out of scope v1).
        contract = Stock(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
        )

        try:
            details = self._ib.reqContractDetails(contract)
        except Exception as e:  # noqa: BLE001
            logger.warning("reqContractDetails(%s) failed: %s", pair, e)
            return None

        if not details:
            logger.warning(
                "No contract details returned for %s (%s/%s/%s)",
                pair, symbol, exchange, currency,
            )
            return None

        # Use the first matching contract
        resolved = details[0].contract
        self._contract_cache[pair] = resolved
        logger.debug(
            "Resolved %s -> conId=%d, exchange=%s, primaryExchange=%s",
            pair, resolved.conId, resolved.exchange, resolved.primaryExchange,
        )
        return resolved

    @staticmethod
    def _contract_to_market_dict(pair: str, contract: Any) -> dict[str, Any]:
        """Convert an ib_async Contract into a ccxt-style market dict."""
        symbol_part, currency = pair.split("/", 1) if "/" in pair else (pair, "USD")
        return {
            "id": str(contract.conId),
            "symbol": pair,
            "base": symbol_part,
            "quote": currency,
            "active": True,  # if it resolved, it's tradable
            "type": "spot",
            "spot": True,
            "margin": False,
            "future": False,
            "swap": False,
            "option": False,
            "contract": False,
            "linear": None,
            "inverse": None,
            "precision": {
                "amount": 8,  # IBKR supports fractional shares
                "price": 0.01,
            },
            "limits": {
                "amount": {"min": 0.0001, "max": None},
                "price": {"min": 0.01, "max": None},
                "cost": {"min": None, "max": None},
            },
            "info": {
                "ibkr_conId": contract.conId,
                "ibkr_exchange": contract.exchange,
                "ibkr_primaryExchange": contract.primaryExchange,
                "ibkr_currency": contract.currency,
                "ibkr_localSymbol": contract.localSymbol,
                "ibkr_secType": contract.secType,
            },
        }

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
        """Fetch historical OHLCV bars from IBKR.

        Returns ccxt-style list of [ts_ms, open, high, low, close, volume].

        IBKR's reqHistoricalData has these constraints:
        - Max bars per call: depends on barSize (e.g. ~5000 for 1d, ~1500 for 1h)
        - Pacing: don't request more than 60 historical requests per 10 min
        - whatToShow: "TRADES" for stocks (we use this; others: BID, ASK, MIDPOINT)
        - useRTH: True = regular trading hours only (recommended for stocks)
        """
        if not self._ib or not self._ib.isConnected():
            raise OperationalException(
                "IBKR client not connected. Cannot fetch OHLCV."
            )

        contract = self._resolve_contract(pair)
        if contract is None:
            raise OperationalException(
                f"Cannot resolve {pair!r} on IBKR — check pair_whitelist + "
                f"that symbol exists at the configured venue."
            )

        bar_size = self._to_ibkr_barsize(timeframe)

        # IBKR uses a "duration string" (e.g. "1 D", "1 W", "30 D", "2 Y")
        # plus an "endDateTime" instead of (since, until). Translate.
        if until_ms is not None:
            end_dt = datetime.fromtimestamp(until_ms / 1000.0, tz=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)

        if since_ms is not None:
            start_dt = datetime.fromtimestamp(since_ms / 1000.0, tz=timezone.utc)
            duration_seconds = int((end_dt - start_dt).total_seconds())
        else:
            # Default: lookback ~1000 candles worth of seconds
            tf_seconds = self._tf_to_seconds(timeframe)
            duration_seconds = tf_seconds * 1000

        duration_str = self._seconds_to_ibkr_duration(duration_seconds)

        # Format endDateTime: empty string = "now", otherwise yyyymmdd-HH:MM:SS
        # We pass "" if end is "now" (within last 60 sec) for simpler request.
        if (datetime.now(timezone.utc) - end_dt).total_seconds() < 60:
            end_str = ""
        else:
            end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")

        try:
            bars = self._ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,  # 2 = Unix epoch seconds
                keepUpToDate=False,
            )
        except Exception as e:  # noqa: BLE001
            err_str = str(e).lower()
            if "pacing" in err_str or "rate" in err_str or "violation" in err_str:
                raise DDosProtection(f"IBKR pacing/rate error: {e}") from e
            raise TemporaryError(f"IBKR fetch_ohlcv failed: {e}") from e

        from datetime import date as _date_cls

        result: list[list[float]] = []
        for bar in bars or []:
            # ib_async returns:
            #   - datetime.date for DAILY (1d, 1w, 1M) bars (no time component)
            #   - datetime.datetime for INTRADAY (1m, 5m, 1h) bars (UTC when formatDate=2)
            #   - int (epoch seconds) for some other formats
            #   - str for formatDate=1 (we don't use this)
            #
            # IMPORTANT: isinstance(bar.date, datetime) is True for datetime
            # but FALSE for date (because date is not a subclass of datetime).
            # We must check date FIRST since datetime IS a subclass of date.
            if isinstance(bar.date, datetime):
                # Intraday: full datetime (UTC because formatDate=2)
                ts_ms = int(bar.date.timestamp() * 1000)
            elif isinstance(bar.date, _date_cls):
                # Daily: date object — combine with midnight UTC for ts
                bar_dt = datetime.combine(
                    bar.date, datetime.min.time(), tzinfo=timezone.utc
                )
                ts_ms = int(bar_dt.timestamp() * 1000)
            elif isinstance(bar.date, (int, float)):
                # Epoch seconds
                ts_ms = int(bar.date * 1000)
            else:
                # Fallback: log warning and use now
                ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                logger.warning("Unexpected bar.date format: %r (%s)",
                               bar.date, type(bar.date).__name__)

            result.append([
                ts_ms,
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume) if bar.volume else 0.0,
            ])

        logger.debug(
            "fetch_ohlcv(%s, %s, dur=%s): %d bars",
            pair, timeframe, duration_str, len(result),
        )
        return result

    @staticmethod
    def _tf_to_seconds(timeframe: str) -> int:
        """Convert Freqtrade timeframe to seconds. Helper for date math."""
        seconds_map = {
            "1s": 1, "5s": 5, "15s": 15, "30s": 30,
            "1m": 60, "2m": 120, "3m": 180, "5m": 300, "10m": 600,
            "15m": 900, "20m": 1200, "30m": 1800,
            "1h": 3600, "2h": 7200, "3h": 10800, "4h": 14400, "8h": 28800,
            "1d": 86400, "1w": 604800, "1M": 2592000,
        }
        if timeframe not in seconds_map:
            raise OperationalException(
                f"Unknown timeframe {timeframe!r}"
            )
        return seconds_map[timeframe]

    @staticmethod
    def _seconds_to_ibkr_duration(seconds: int) -> str:
        """Convert a duration in seconds to an IBKR-compatible duration string.

        IBKR has tight per-call limits:
        - 1d bars: ~1 Y per request (we cap at "365 D" for safety)
        - 1h bars: ~1 M
        - 1m bars: ~1 D
        - 5m bars: ~1 W

        For ranges longer than 1 year, Freqtrade's download-data flow
        calls fetch_ohlcv multiple times with sliding windows, so we
        only need to satisfy a single-call.

        We cap at 365 D regardless of input. For longer ranges
        (e.g. 2-year backtests) the download-data flow chunks naturally.
        """
        if seconds < 86400:
            return f"{max(seconds, 60)} S"
        days = seconds // 86400
        # Cap at 365 D (1 year) for safety across all bar sizes.
        # Longer ranges: caller (Freqtrade) chunks via since_ms iteration.
        capped_days = min(days, 365)
        return f"{capped_days} D"


    # ---- Account / balance ---------------------------------------------

    def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return account balance in ccxt-compatible shape.

        IBKR account is multi-currency. We expose:
        - Total NetLiquidation in account base currency (typically EUR for IBIE)
        - Per-currency cash balances (USD, EUR, etc.)
        - Buying power and available funds
        """
        if not self._ib or not self._ib.isConnected():
            raise OperationalException(
                "IBKR client not connected. Cannot fetch balance."
            )

        try:
            account_values = self._ib.accountValues(account=self._ibkr_account or "")
            account_summary = self._ib.accountSummary(account=self._ibkr_account or "")
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"IBKR fetch_balance failed: {e}") from e

        # accountSummary returns a list of AccountValue objects with
        # tag like "NetLiquidation", "AvailableFunds", etc.
        # Each entry has: account, tag, value, currency, modelCode
        info: dict[str, Any] = {}
        for entry in account_summary:
            key = f"{entry.tag}_{entry.currency}" if entry.currency else entry.tag
            info[key] = entry.value

        # Per-currency cash balances. accountValues includes "CashBalance"
        # entries per currency.
        free: dict[str, float] = {}
        used: dict[str, float] = {}
        total: dict[str, float] = {}

        for av in account_values:
            if not av.currency or av.currency == "BASE":
                continue  # BASE = synthesized base-currency aggregate, skip
            try:
                value = float(av.value) if av.value else 0.0
            except (ValueError, TypeError):
                continue

            ccy = av.currency
            if av.tag == "CashBalance":
                free.setdefault(ccy, 0.0)
                free[ccy] += value
                total.setdefault(ccy, 0.0)
                total[ccy] += value
            elif av.tag == "StockMarketValue":
                # Stock holdings count as "used" buying power
                used.setdefault(ccy, 0.0)
                used[ccy] += value
                total.setdefault(ccy, 0.0)
                total[ccy] += value

        result: dict[str, Any] = {
            "info": info,
            "free": free,
            "used": used,
            "total": total,
        }
        # Per-currency dicts (ccxt format)
        for ccy in set(list(free.keys()) + list(used.keys()) + list(total.keys())):
            result[ccy] = {
                "free": free.get(ccy, 0.0),
                "used": used.get(ccy, 0.0),
                "total": total.get(ccy, 0.0),
            }
        return result

    # ---- Ticker --------------------------------------------------------

    def fetch_ticker(self, pair: str) -> dict[str, Any]:
        """Fetch current bid/ask/last via reqMktData (snapshot mode).

        IBKR returns a Ticker object that streams updates over time.
        For Freqtrade's fetch_ticker semantics we want a single snapshot,
        so we wait briefly then read whatever we have.
        """
        if not self._ib or not self._ib.isConnected():
            raise OperationalException(
                "IBKR client not connected. Cannot fetch ticker."
            )

        contract = self._resolve_contract(pair)
        if contract is None:
            raise OperationalException(
                f"Cannot resolve {pair!r} on IBKR for ticker."
            )

        try:
            # Snapshot: a single fetch. Returns a Ticker that fills in
            # asynchronously; we wait a few seconds for data.
            ticker = self._ib.reqMktData(
                contract=contract,
                genericTickList="",
                snapshot=True,
                regulatorySnapshot=False,
            )
            # Wait for snapshot to populate (max ~3s)
            self._ib.sleep(2)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"IBKR fetch_ticker failed: {e}") from e

        bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else None
        ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else None
        last = float(ticker.last) if ticker.last and ticker.last > 0 else None
        if not last and bid and ask:
            last = (bid + ask) / 2.0
        elif not last:
            last = bid or ask or 0.0

        ts_ms = (
            int(ticker.time.timestamp() * 1000) if ticker.time
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )

        return {
            "symbol": pair,
            "timestamp": ts_ms,
            "datetime": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
            "bid": bid,
            "ask": ask,
            "last": last,
            "close": last,
            "vwap": float(ticker.vwap) if ticker.vwap else None,
            "open": float(ticker.open) if ticker.open else None,
            "high": float(ticker.high) if ticker.high else None,
            "low": float(ticker.low) if ticker.low else None,
            "previousClose": float(ticker.close) if ticker.close else None,
            "change": None,
            "percentage": None,
            "average": last,
            "baseVolume": float(ticker.volume) if ticker.volume else 0.0,
            "quoteVolume": None,
            "info": {
                "ibkr_bid_size": float(ticker.bidSize) if ticker.bidSize else 0,
                "ibkr_ask_size": float(ticker.askSize) if ticker.askSize else 0,
                "ibkr_last_size": float(ticker.lastSize) if ticker.lastSize else 0,
                "ibkr_market_data_type": ticker.marketDataType,
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
        """Place an order via ib_async.

        Maps Freqtrade order semantics to IBKR's:
        - ordertype 'market' -> MarketOrder
        - ordertype 'limit'  -> LimitOrder (rate required)
        - side 'buy' / 'sell'
        - time_in_force 'GTC' / 'DAY' / 'IOC' / 'OPG'
        """
        from ib_async import LimitOrder, MarketOrder

        if not self._ib or not self._ib.isConnected():
            raise OperationalException(
                "IBKR client not connected. Cannot create order."
            )

        contract = self._resolve_contract(pair)
        if contract is None:
            raise OperationalException(
                f"Cannot resolve {pair!r} on IBKR for order."
            )

        action = "BUY" if side.lower() == "buy" else "SELL"
        tif_map = {"GTC": "GTC", "DAY": "DAY", "IOC": "IOC", "OPG": "OPG"}
        tif = tif_map.get(time_in_force.upper(), "GTC")

        try:
            if ordertype.lower() == "market":
                order = MarketOrder(
                    action=action,
                    totalQuantity=amount,
                    tif=tif,
                )
            elif ordertype.lower() == "limit":
                if rate is None:
                    raise OperationalException(
                        "Limit order requires `rate` parameter"
                    )
                order = LimitOrder(
                    action=action,
                    totalQuantity=amount,
                    lmtPrice=rate,
                    tif=tif,
                )
            else:
                raise OperationalException(
                    f"Order type {ordertype!r} not supported by IBKR plugin "
                    f"(use 'market' or 'limit')"
                )

            # Submit account if specified
            if self._ibkr_account:
                order.account = self._ibkr_account

            trade = self._ib.placeOrder(contract, order)
            # Wait briefly for the order to be acknowledged
            self._ib.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            raise TemporaryError(f"IBKR create_order failed: {e}") from e

        return self._ibkr_trade_to_ccxt(trade, pair)

    def fetch_order(
        self, order_id: str, pair: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fetch an order by ID. IBKR uses int orderId; we accept str
        for ccxt-compatibility and cast."""
        if not self._ib or not self._ib.isConnected():
            raise OperationalException("IBKR client not connected.")

        try:
            target_id = int(order_id)
        except ValueError as e:
            raise OperationalException(
                f"IBKR order_id must be int-castable, got {order_id!r}"
            ) from e

        # Search through all known trades (IBKR maintains a list)
        trades = self._ib.trades()
        for trade in trades:
            if trade.order.orderId == target_id:
                return self._ibkr_trade_to_ccxt(trade, pair)

        raise OperationalException(
            f"Order {order_id} not found in IBKR. May have been filled and "
            f"flushed; check fetch_order history if needed."
        )

    def cancel_order(
        self, order_id: str, pair: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Cancel an order by ID."""
        if not self._ib or not self._ib.isConnected():
            raise OperationalException("IBKR client not connected.")

        try:
            target_id = int(order_id)
        except ValueError as e:
            raise OperationalException(
                f"IBKR order_id must be int-castable, got {order_id!r}"
            ) from e

        trades = self._ib.trades()
        for trade in trades:
            if trade.order.orderId == target_id:
                try:
                    self._ib.cancelOrder(trade.order)
                    self._ib.sleep(0.3)
                    return self._ibkr_trade_to_ccxt(trade, pair)
                except Exception as e:  # noqa: BLE001
                    raise TemporaryError(
                        f"IBKR cancel_order({order_id}) failed: {e}"
                    ) from e

        raise OperationalException(
            f"Order {order_id} not found for cancel."
        )

    # ---- Order conversion helpers --------------------------------------

    @staticmethod
    def _ibkr_trade_to_ccxt(trade: Any, pair: str) -> dict[str, Any]:
        """Convert an ib_async Trade into a ccxt-compatible order dict.

        IBKR Trade is a richer object than a single Order — it includes
        OrderStatus, fills, log, etc. We map relevant fields.
        """
        # Map IBKR status -> ccxt status. Status field is on Trade.orderStatus.
        # IBKR status values: PendingSubmit, PendingCancel, PreSubmitted,
        # Submitted, Cancelled, Filled, Inactive
        status_map = {
            "pendingsubmit": "open",
            "pendingcancel": "open",
            "presubmitted": "open",
            "submitted": "open",
            "cancelled": "canceled",
            "apicancelled": "canceled",
            "filled": "closed",
            "inactive": "canceled",
        }
        ibkr_status = (
            getattr(trade.orderStatus, "status", "") or ""
        ).lower().strip()
        ccxt_status = status_map.get(ibkr_status, "open")

        order = trade.order
        order_status = trade.orderStatus

        filled_qty = float(getattr(order_status, "filled", 0) or 0)
        order_qty = float(getattr(order, "totalQuantity", 0) or 0)
        avg_price = (
            float(getattr(order_status, "avgFillPrice", 0) or 0)
            if order_status.avgFillPrice else None
        )
        limit_price = (
            float(getattr(order, "lmtPrice", 0) or 0)
            if getattr(order, "lmtPrice", None) else None
        )

        # IBKR order types: MKT, LMT, STP, STP LMT, etc.
        order_type_map = {"MKT": "market", "LMT": "limit", "STP": "stop"}
        order_type = order_type_map.get(
            getattr(order, "orderType", "") or "", "unknown"
        )

        # Fills timestamps from log
        fill_log = trade.log or []
        last_log_ts = None
        if fill_log:
            try:
                last_log_ts = int(fill_log[-1].time.timestamp() * 1000)
            except Exception:  # noqa: BLE001
                last_log_ts = None

        return {
            "id": str(order.orderId),
            "clientOrderId": (
                str(order.permId) if getattr(order, "permId", None) else None
            ),
            "timestamp": last_log_ts,
            "datetime": (
                datetime.fromtimestamp(last_log_ts / 1000, tz=timezone.utc).isoformat()
                if last_log_ts else None
            ),
            "lastTradeTimestamp": last_log_ts,
            "symbol": pair,
            "type": order_type,
            "timeInForce": getattr(order, "tif", "GTC"),
            "side": getattr(order, "action", "").lower(),
            "price": limit_price,
            "average": avg_price,
            "amount": order_qty,
            "filled": filled_qty,
            "remaining": order_qty - filled_qty,
            "status": ccxt_status,
            "fee": (
                {
                    "cost": float(order_status.commission or 0),
                    "currency": "USD",  # IBKR per-trade currency in info
                }
                if getattr(order_status, "commission", 0) else None
            ),
            "trades": [],
            "info": {
                "ibkr_orderId": order.orderId,
                "ibkr_permId": getattr(order, "permId", None),
                "ibkr_status": ibkr_status,
                "ibkr_whyHeld": getattr(order_status, "whyHeld", "") or "",
            },
        }

    # ---- Validation overrides ------------------------------------------

    def validate_pairs(self, pairs: list[str]) -> None:
        """Override base: ensure pairs resolve via reqContractDetails.

        Lazy variant: each pair's contract is resolved + cached the first
        time it's requested. If reload_markets has run, the markets dict
        is already populated and this is just a dict lookup.
        """
        unknown_pairs: list[str] = []
        for pair in pairs:
            if pair in self._markets:
                continue
            # Try to resolve on the fly
            contract = self._resolve_contract(pair)
            if contract is None:
                unknown_pairs.append(pair)
            else:
                self._markets[pair] = self._contract_to_market_dict(pair, contract)

        if unknown_pairs:
            raise OperationalException(
                f"Pairs not available on IBKR: {unknown_pairs}. "
                f"Verify symbol+venue spelling and that you have data "
                f"subscriptions for the relevant exchanges."
            )

    def close(self) -> None:
        """Close the IBKR connection. Called by Freqtrade on shutdown."""
        if self._ib is not None and self._ib.isConnected():
            try:
                self._ib.disconnect()
                logger.info("IBKR client disconnected.")
            except Exception as e:  # noqa: BLE001
                logger.warning("IBKR disconnect failed: %s", e)



# ---------------------------------------------------------------------------
# Resolver alias: Freqtrade's ExchangeResolver does `.title()` on the
# config exchange name, converting "interactivebrokers" to "Interactivebrokers"
# (only first letter capital, internal caps lowercased). To have the
# resolver find our plugin we expose a second class name that matches
# the title-form, as a thin subclass of the canonical name.
# ---------------------------------------------------------------------------


class Interactivebrokers(InteractiveBrokers):
    """Resolver-friendly alias for InteractiveBrokers.

    Empty subclass; all behaviour is inherited from InteractiveBrokers.
    Tests, type hints, and user code should use InteractiveBrokers.
    Freqtrade's ExchangeResolver finds THIS class via `.title()` lookup.
    """
    pass
