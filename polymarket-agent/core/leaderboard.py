"""
Scraper del leaderboard de Polymarket y tracker de posiciones de top traders.

Consulta la Data API pública de Polymarket para obtener:
1. Top traders por PnL (leaderboard)
2. Posiciones abiertas de cada trader
3. Actividad reciente (trades) de cada trader

Estos datos alimentan la estrategia de copy-trading.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from config import settings
from core.net_utils import retry_async, RETRIABLE_EXCEPTIONS

logger = logging.getLogger(__name__)

# =============================================================================
# Modelos de datos del leaderboard
# =============================================================================


class TopTrader(BaseModel):
    """Un trader del leaderboard de Polymarket."""
    wallet: str = Field(description="Dirección proxy wallet del trader")
    username: str = Field(default="", description="Nombre de usuario o pseudónimo")
    profile_image: str = Field(default="", description="URL de imagen de perfil")
    profit: float = Field(default=0.0, description="PnL total en USD")
    volume: float = Field(default=0.0, description="Volumen total operado en USD")
    rank: int = Field(default=0, description="Posición en el leaderboard")
    markets_traded: int = Field(default=0, description="Cantidad de mercados operados")


class TraderPosition(BaseModel):
    """Posición abierta de un trader en un mercado."""
    market_id: str = Field(description="Condition ID del mercado")
    title: str = Field(default="", description="Título/pregunta del mercado")
    side: str = Field(default="", description="YES o NO")
    size: float = Field(default=0.0, description="Tamaño en shares")
    initial_value: float = Field(default=0.0, description="Valor inicial en USD")
    current_value: float = Field(default=0.0, description="Valor actual en USD")
    price: float = Field(default=0.0, description="Precio actual del token")
    pnl_cash: float = Field(default=0.0, description="PnL realizado en USD")
    pnl_percent: float = Field(default=0.0, description="PnL en porcentaje")
    asset: str = Field(default="", description="Token asset ID")


class TraderTrade(BaseModel):
    """Trade reciente de un trader."""
    market_id: str = Field(default="", description="Condition ID del mercado")
    title: str = Field(default="", description="Título del mercado")
    side: str = Field(default="", description="BUY o SELL")
    outcome: str = Field(default="", description="YES o NO")
    price: float = Field(default=0.0, description="Precio de ejecución")
    size: float = Field(default=0.0, description="Tamaño en shares")
    cash_amount: float = Field(default=0.0, description="Monto en USD")
    timestamp: str | int = Field(default="", description="Timestamp del trade (ISO string o unix int)")
    asset: str = Field(default="", description="Token asset ID")


class TraderSnapshot(BaseModel):
    """Snapshot completo de un top trader: perfil + posiciones + trades."""
    trader: TopTrader
    positions: list[TraderPosition] = Field(default_factory=list)
    recent_trades: list[TraderTrade] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# LeaderboardFetcher
# =============================================================================


class LeaderboardFetcher:
    """
    Consulta el leaderboard y posiciones de los mejores traders de Polymarket.

    Usa la Data API pública (https://data-api.polymarket.com) que no requiere
    autenticación. Los datos son públicos porque viven en blockchain.

    Endpoints utilizados:
    - GET /leaderboard          → top traders por PnL o volumen
    - GET /positions?user=ADDR  → posiciones abiertas de un wallet
    - GET /activity?user=ADDR   → actividad reciente (trades)
    """

    DATA_API_URL = "https://data-api.polymarket.com"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=settings.scanner.http_timeout_seconds,
            headers={"Accept": "application/json"},
        )
        self._top_traders: list[TopTrader] = []
        self._snapshots: list[TraderSnapshot] = []
        self._last_fetch: datetime | None = None

    async def close(self) -> None:
        """Cierra el cliente HTTP."""
        await self._client.aclose()

    # =========================================================================
    # API pública
    # =========================================================================

    async def obtener_top_traders(
        self,
        top_n: int | None = None,
        window: str = "all",
        rank_by: str = "profit",
    ) -> list[TopTrader]:
        """
        Obtiene los top traders del leaderboard de Polymarket.

        Args:
            top_n: Cantidad de traders a retornar (default: config).
            window: Ventana temporal: "1d", "7d", "30d", "all".
            rank_by: Métrica de ranking: "profit" o "volume".

        Returns:
            Lista de TopTrader ordenados por ranking.
        """
        n = top_n or settings.copy_trading.top_n
        logger.info(
            f"Consultando leaderboard: top {n}, window={window}, "
            f"rankBy={rank_by}"
        )

        # Mapear parámetros al formato v1 de la API
        time_period_map = {
            "1d": "DAY", "7d": "WEEK", "30d": "MONTH", "all": "ALL",
            "day": "DAY", "week": "WEEK", "month": "MONTH",
        }
        order_by_map = {
            "profit": "PNL", "volume": "VOL",
            "pnl": "PNL", "vol": "VOL",
        }
        api_period = time_period_map.get(window.lower(), "ALL")
        api_order = order_by_map.get(rank_by.lower(), "PNL")

        data = await self._hacer_request(
            "/v1/leaderboard",
            params={
                "timePeriod": api_period,
                "orderBy": api_order,
                "limit": str(min(n, 50)),
                "offset": "0",
            },
        )

        traders: list[TopTrader] = []
        for i, entry in enumerate(data[:n]):
            trader = self._parsear_trader(entry, rank=i + 1)
            if trader:
                traders.append(trader)

        self._top_traders = traders
        self._last_fetch = datetime.now()

        logger.info(f"Leaderboard obtenido: {len(traders)} traders")
        return traders

    async def obtener_posiciones_trader(
        self,
        wallet: str,
        size_threshold: float = 1.0,
    ) -> list[TraderPosition]:
        """
        Obtiene las posiciones abiertas de un trader por su wallet.

        Args:
            wallet: Dirección proxy wallet del trader.
            size_threshold: Tamaño mínimo de posición en shares.

        Returns:
            Lista de TraderPosition con posiciones abiertas.
        """
        data = await self._hacer_request(
            "/positions",
            params={
                "user": wallet,
                "sizeThreshold": str(size_threshold),
                "limit": "500",
                "sortBy": "CURRENT",
                "sortDirection": "DESC",
            },
        )

        positions: list[TraderPosition] = []
        for entry in data:
            pos = self._parsear_posicion(entry)
            if pos:
                positions.append(pos)

        logger.debug(f"Posiciones de {wallet[:10]}...: {len(positions)}")
        return positions

    async def obtener_trades_trader(
        self,
        wallet: str,
        limit: int = 50,
    ) -> list[TraderTrade]:
        """
        Obtiene los trades recientes de un trader.

        Args:
            wallet: Dirección proxy wallet del trader.
            limit: Máximo de trades a retornar.

        Returns:
            Lista de TraderTrade ordenados por más reciente.
        """
        data = await self._hacer_request(
            "/activity",
            params={
                "user": wallet,
                "limit": str(limit),
                "sortBy": "TIMESTAMP",
                "sortDirection": "DESC",
            },
        )

        trades: list[TraderTrade] = []
        for entry in data:
            trade = self._parsear_trade(entry)
            if trade:
                trades.append(trade)

        logger.debug(f"Trades de {wallet[:10]}...: {len(trades)}")
        return trades

    async def obtener_snapshots_completos(
        self,
        top_n: int | None = None,
        window: str = "all",
    ) -> list[TraderSnapshot]:
        """
        Obtiene snapshots completos de los top traders: perfil + posiciones + trades.

        Consulta el leaderboard primero, luego en paralelo las posiciones
        y trades de cada trader.

        Args:
            top_n: Cantidad de top traders a analizar.
            window: Ventana temporal del leaderboard.

        Returns:
            Lista de TraderSnapshot con toda la información.
        """
        n = top_n or settings.copy_trading.top_n

        # Paso 1: Obtener top traders
        traders = await self.obtener_top_traders(top_n=n, window=window)
        if not traders:
            logger.warning("No se obtuvieron traders del leaderboard")
            return []

        # Paso 2: Obtener posiciones y trades en paralelo para cada trader
        async def _fetch_snapshot(trader: TopTrader) -> TraderSnapshot:
            try:
                positions, trades = await asyncio.gather(
                    self.obtener_posiciones_trader(trader.wallet),
                    self.obtener_trades_trader(trader.wallet, limit=30),
                )
                return TraderSnapshot(
                    trader=trader,
                    positions=positions,
                    recent_trades=trades,
                )
            except Exception as e:
                logger.warning(
                    f"Error obteniendo snapshot de {trader.username or trader.wallet[:10]}: {e}"
                )
                return TraderSnapshot(trader=trader)

        snapshots = await asyncio.gather(
            *[_fetch_snapshot(t) for t in traders]
        )

        self._snapshots = list(snapshots)
        logger.info(
            f"Snapshots completos: {len(self._snapshots)} traders | "
            f"Total posiciones: {sum(len(s.positions) for s in self._snapshots)}"
        )
        return self._snapshots

    # =========================================================================
    # Análisis: identificar mercados populares entre top traders
    # =========================================================================

    def identificar_mercados_consenso(
        self,
        min_traders: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Identifica mercados donde múltiples top traders tienen posiciones.

        Un mercado con consenso de varios traders exitosos tiene mayor
        probabilidad de ser una buena apuesta.

        Args:
            min_traders: Mínimo de traders con posición para considerar consenso.

        Returns:
            Lista de dicts con market_id, title, side, traders_count,
            avg_price, total_size, ordenados por número de traders.
        """
        min_t = min_traders or settings.copy_trading.min_traders_consensus

        # Agrupar posiciones por (market_id, side)
        mercados: dict[tuple[str, str], list[dict]] = {}
        for snapshot in self._snapshots:
            for pos in snapshot.positions:
                key = (pos.market_id, pos.side)
                if key not in mercados:
                    mercados[key] = []
                mercados[key].append({
                    "trader": snapshot.trader,
                    "position": pos,
                })

        # Filtrar por consenso mínimo
        consensos: list[dict[str, Any]] = []
        for (market_id, side), entries in mercados.items():
            if len(entries) < min_t:
                continue

            positions = [e["position"] for e in entries]
            traders = [e["trader"] for e in entries]

            # Ponderar por profit del trader (mejores traders pesan más)
            total_profit = sum(max(t.profit, 0) for t in traders)
            weighted_price = 0.0
            if total_profit > 0:
                for t, p in zip(traders, positions):
                    weight = max(t.profit, 0) / total_profit
                    weighted_price += p.price * weight
            else:
                weighted_price = sum(p.price for p in positions) / len(positions)

            consensos.append({
                "market_id": market_id,
                "title": positions[0].title,
                "side": side,
                "traders_count": len(entries),
                "trader_names": [t.username or t.wallet[:10] for t in traders],
                "avg_price": weighted_price,
                "total_size_usd": sum(p.current_value for p in positions),
                "avg_pnl_pct": sum(p.pnl_percent for p in positions) / len(positions),
            })

        # Ordenar por número de traders (más popular primero)
        consensos.sort(key=lambda c: c["traders_count"], reverse=True)

        logger.info(
            f"Mercados con consenso (>={min_t} traders): {len(consensos)}"
        )
        return consensos

    # =========================================================================
    # Métodos privados
    # =========================================================================

    @retry_async(max_attempts=4, base_delay=2.0, max_delay=16.0)
    async def _hacer_request(
        self, endpoint: str, params: dict | None = None
    ) -> list[dict]:
        """Hace un request GET a la Data API con reintentos."""
        url = f"{self.DATA_API_URL}{endpoint}"
        respuesta = await self._client.get(url, params=params)
        respuesta.raise_for_status()
        data = respuesta.json()

        # Log diagnóstico en primera consulta a cada endpoint
        if not hasattr(self, "_endpoints_vistos"):
            self._endpoints_vistos: set[str] = set()
        if endpoint not in self._endpoints_vistos:
            self._endpoints_vistos.add(endpoint)
            sample = str(data)[:200] if data else "(vacío)"
            logger.info(
                f"Data API {endpoint}: tipo={type(data).__name__}, "
                f"len={len(data) if isinstance(data, (list, dict)) else 'N/A'}, "
                f"sample={sample}"
            )

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Buscar la lista de resultados en keys conocidas
            for key in ("data", "rankings", "leaderboard", "positions", "history"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Dict directo (ej: un solo resultado)
            return [data]
        return []

    @staticmethod
    def _parsear_trader(entry: dict, rank: int = 0) -> TopTrader | None:
        """Parsea un trader del leaderboard."""
        try:
            wallet = (
                entry.get("proxyWallet", "")
                or entry.get("proxy_wallet", "")
                or entry.get("address", "")
                or entry.get("user", "")
            )
            if not wallet:
                return None

            return TopTrader(
                wallet=wallet,
                username=(
                    entry.get("userName", "")
                    or entry.get("name", "")
                    or entry.get("pseudonym", "")
                    or entry.get("username", "")
                ),
                profile_image=(
                    entry.get("profileImage", "")
                    or entry.get("profile_image", "")
                ),
                profit=float(
                    entry.get("pnl", 0)
                    or entry.get("profit", 0)
                    or 0
                ),
                volume=float(
                    entry.get("vol", 0)
                    or entry.get("volume", 0)
                    or 0
                ),
                rank=int(entry.get("rank", 0) or 0) or rank,
                markets_traded=int(entry.get("marketsTraded", 0) or 0),
            )
        except Exception as e:
            logger.warning(f"Error parseando trader: {e}")
            return None

    @staticmethod
    def _parsear_posicion(entry: dict) -> TraderPosition | None:
        """Parsea una posición de la Data API."""
        try:
            market_id = (
                entry.get("conditionId", "")
                or entry.get("market", "")
                or entry.get("condition_id", "")
            )
            if not market_id:
                return None

            # Determinar side desde outcome, direction o asset
            outcome = (
                entry.get("outcome", "")
                or entry.get("direction", "")
                or entry.get("side", "")
            )
            side = ""
            if isinstance(outcome, str):
                if outcome.lower() in ("yes", "no"):
                    side = outcome.upper()
                elif outcome.lower() in ("long", "buy"):
                    side = "YES"
                elif outcome.lower() in ("short", "sell"):
                    side = "NO"

            # Sin side determinado no sirve para consenso
            if not side:
                return None

            return TraderPosition(
                market_id=market_id,
                title=entry.get("title", entry.get("question", "")),
                side=side,
                size=float(entry.get("size", 0) or 0),
                initial_value=float(entry.get("initialValue", 0) or entry.get("initial", 0) or 0),
                current_value=float(entry.get("currentValue", 0) or entry.get("current", 0) or 0),
                price=float(entry.get("curPrice", 0) or entry.get("price", 0) or 0),
                pnl_cash=float(entry.get("cashPnl", 0) or entry.get("pnl", 0) or 0),
                pnl_percent=float(entry.get("percentPnl", 0) or entry.get("pnlPercent", 0) or 0),
                asset=entry.get("asset", ""),
            )
        except Exception as e:
            logger.warning(f"Error parseando posición: {e}")
            return None

    @staticmethod
    def _parsear_trade(entry: dict) -> TraderTrade | None:
        """Parsea un trade de la Data API."""
        try:
            # Filtrar solo trades (no splits, merges, redeems)
            activity_type = entry.get("type", "trade").lower()
            non_trade_types = ("split", "merge", "redeem", "deposit", "withdraw")
            if activity_type in non_trade_types:
                return None

            market_id = (
                entry.get("conditionId", "")
                or entry.get("market", "")
                or entry.get("condition_id", "")
            )
            if not market_id:
                return None

            return TraderTrade(
                market_id=market_id,
                title=entry.get("title", entry.get("question", "")),
                side=entry.get("side", "BUY").upper(),
                outcome=entry.get("outcome", ""),
                price=float(entry.get("price", 0) or 0),
                size=float(entry.get("size", 0) or 0),
                cash_amount=float(entry.get("cashAmount", 0) or entry.get("cash", 0) or 0),
                timestamp=entry.get("timestamp", entry.get("createdAt", "")),
                asset=entry.get("asset", ""),
            )
        except Exception as e:
            logger.warning(f"Error parseando trade: {e}")
            return None

    @property
    def last_fetch(self) -> datetime | None:
        """Última vez que se consultó el leaderboard."""
        return self._last_fetch

    @property
    def cached_snapshots(self) -> list[TraderSnapshot]:
        """Snapshots en cache (última consulta)."""
        return self._snapshots
