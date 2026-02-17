"""
Historial de precios de mercados en Polymarket (async).

Obtiene y analiza datos históricos de precios para identificar
tendencias y movimientos bruscos.

Fase 2 del agente autónomo — refactorizado a asyncio.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum

import httpx
from pydantic import BaseModel, Field

from config import settings
from core.net_utils import retry_async, RETRIABLE_EXCEPTIONS

logger = logging.getLogger(__name__)


class Trend(str, Enum):
    """Tendencia del precio de un mercado."""
    RISING = "rising"          # Subiendo
    FALLING = "falling"        # Bajando
    STABLE = "stable"          # Estable
    VOLATILE = "volatile"      # Volátil sin dirección clara
    INSUFFICIENT = "insufficient"  # Datos insuficientes


class PricePoint(BaseModel):
    """Un punto de precio en el historial."""
    timestamp: datetime = Field(description="Momento del precio")
    price: float = Field(ge=0, le=1, description="Precio del token")


class PriceMovement(BaseModel):
    """Movimiento brusco de precio detectado."""
    timestamp: datetime = Field(description="Cuándo ocurrió")
    price_before: float = Field(description="Precio antes del movimiento")
    price_after: float = Field(description="Precio después del movimiento")
    change_pct: float = Field(description="Cambio porcentual")
    direction: str = Field(description="'up' o 'down'")


class MarketHistoryAnalysis(BaseModel):
    """Resultado del análisis del historial de un mercado."""
    condition_id: str = Field(description="ID del mercado")
    current_price: float = Field(description="Precio actual")
    price_24h_ago: float | None = Field(
        default=None, description="Precio hace 24h"
    )
    price_7d_ago: float | None = Field(
        default=None, description="Precio hace 7 días"
    )
    trend_24h: Trend = Field(
        default=Trend.INSUFFICIENT, description="Tendencia últimas 24h"
    )
    trend_7d: Trend = Field(
        default=Trend.INSUFFICIENT, description="Tendencia últimos 7 días"
    )
    change_24h_pct: float = Field(
        default=0, description="Cambio porcentual 24h"
    )
    change_7d_pct: float = Field(
        default=0, description="Cambio porcentual 7 días"
    )
    volatility: float = Field(
        default=0, ge=0, description="Volatilidad (desviación estándar)"
    )
    sharp_movements: list[PriceMovement] = Field(
        default_factory=list, description="Movimientos bruscos recientes"
    )
    price_history: list[PricePoint] = Field(
        default_factory=list, description="Historial de precios"
    )
    data_points: int = Field(
        default=0, description="Cantidad de puntos de datos"
    )

    def resumen(self) -> str:
        """Resumen legible del análisis."""
        return (
            f"Precio actual: {self.current_price:.2f} | "
            f"24h: {self.change_24h_pct:+.1f}% ({self.trend_24h.value}) | "
            f"7d: {self.change_7d_pct:+.1f}% ({self.trend_7d.value}) | "
            f"Vol: {self.volatility:.3f} | "
            f"Movimientos bruscos: {len(self.sharp_movements)}"
        )


class MarketHistory:
    """
    Obtiene y analiza el historial de precios de mercados en Polymarket (async).

    Usa la API CLOB para obtener datos de precio y calcula tendencias,
    volatilidad y detecta movimientos bruscos.
    """

    # Umbral para considerar un movimiento como "brusco" (5%)
    SHARP_MOVEMENT_THRESHOLD: float = 0.05
    # Umbral para considerar una tendencia "estable" (2%)
    STABLE_THRESHOLD: float = 0.02

    def __init__(self) -> None:
        self._clob_url = settings.polymarket.api_url
        self._gamma_url = settings.polymarket.gamma_api_url
        self._client = httpx.AsyncClient(
            timeout=settings.scanner.http_timeout_seconds,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        """Cierra el cliente HTTP."""
        await self._client.aclose()

    async def __aenter__(self) -> "MarketHistory":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def analizar_mercado(
        self,
        condition_id: str,
        token_id: str,
        current_price: float,
    ) -> MarketHistoryAnalysis:
        """
        Analiza el historial de precios de un mercado.

        Args:
            condition_id: ID de la condición del mercado.
            token_id: ID del token YES para obtener precios.
            current_price: Precio actual del token.

        Returns:
            Análisis completo del historial de precios.
        """
        logger.info(f"Analizando historial de mercado {condition_id[:12]}...")

        # Obtener historial de precios de la API
        history = await self._obtener_historial_precios(condition_id, token_id)

        if not history:
            logger.warning(
                f"Sin datos históricos para {condition_id[:12]}"
            )
            return MarketHistoryAnalysis(
                condition_id=condition_id,
                current_price=current_price,
                data_points=0,
            )

        # Analizar tendencias
        ahora = datetime.now()
        precio_24h = self._precio_en_momento(
            history, ahora - timedelta(hours=24)
        )
        precio_7d = self._precio_en_momento(
            history, ahora - timedelta(days=7)
        )

        # Calcular cambios porcentuales
        cambio_24h = self._calcular_cambio_pct(precio_24h, current_price)
        cambio_7d = self._calcular_cambio_pct(precio_7d, current_price)

        # Determinar tendencias
        tendencia_24h = self._determinar_tendencia(cambio_24h, history[-24:])
        tendencia_7d = self._determinar_tendencia(cambio_7d, history)

        # Calcular volatilidad
        volatilidad = self._calcular_volatilidad(history)

        # Detectar movimientos bruscos
        movimientos = self._detectar_movimientos_bruscos(history)

        return MarketHistoryAnalysis(
            condition_id=condition_id,
            current_price=current_price,
            price_24h_ago=precio_24h,
            price_7d_ago=precio_7d,
            trend_24h=tendencia_24h,
            trend_7d=tendencia_7d,
            change_24h_pct=cambio_24h,
            change_7d_pct=cambio_7d,
            volatility=volatilidad,
            sharp_movements=movimientos,
            price_history=history,
            data_points=len(history),
        )

    # =========================================================================
    # Obtención de datos
    # =========================================================================

    @retry_async(max_attempts=3, base_delay=2.0, max_delay=10.0)
    async def _obtener_historial_precios(
        self, condition_id: str, token_id: str
    ) -> list[PricePoint]:
        """
        Obtiene el historial de precios desde la API de Polymarket.

        Intenta primero la API de Gamma (timeseries), luego CLOB.
        """
        history: list[PricePoint] = []

        # Intentar la API de historial de precios de Gamma
        try:
            url = f"{self._gamma_url}/markets/{condition_id}/timeseries"
            params = {"interval": "1h", "fidelity": 60}  # Intervalos de 1h
            respuesta = await self._client.get(url, params=params)

            if respuesta.status_code == 200:
                data = respuesta.json()
                history = self._parsear_timeseries(data)
        except Exception as e:
            logger.debug(f"Gamma timeseries no disponible: {e}")

        # Si no hay datos, intentar con el historial del CLOB
        if not history:
            try:
                url = f"{self._clob_url}/prices-history"
                params = {
                    "market": condition_id,
                    "interval": "1h",
                    "fidelity": 60,
                }
                respuesta = await self._client.get(url, params=params)

                if respuesta.status_code == 200:
                    data = respuesta.json()
                    history = self._parsear_prices_history(data)
            except Exception as e:
                logger.debug(f"CLOB prices-history no disponible: {e}")

        logger.debug(
            f"Historial obtenido para {condition_id[:12]}: "
            f"{len(history)} puntos"
        )
        return history

    def _parsear_timeseries(
        self, data: list | dict
    ) -> list[PricePoint]:
        """Parsea datos de timeseries de la API Gamma."""
        points: list[PricePoint] = []

        # Puede venir como lista de dicts o como dict con "history"
        items = data if isinstance(data, list) else data.get("history", [])

        for item in items:
            try:
                timestamp = None
                if "t" in item:
                    timestamp = datetime.fromtimestamp(item["t"])
                elif "timestamp" in item:
                    timestamp = datetime.fromisoformat(
                        str(item["timestamp"]).replace("Z", "+00:00")
                    ).replace(tzinfo=None)

                price = float(item.get("p", item.get("price", 0)))

                if timestamp and 0 <= price <= 1:
                    points.append(PricePoint(timestamp=timestamp, price=price))
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Error parseando punto de timeseries: {e}")

        return sorted(points, key=lambda p: p.timestamp)

    def _parsear_prices_history(
        self, data: list | dict
    ) -> list[PricePoint]:
        """Parsea datos de prices-history del CLOB."""
        points: list[PricePoint] = []

        items = data if isinstance(data, list) else data.get("history", [])

        for item in items:
            try:
                timestamp = datetime.fromtimestamp(int(item.get("t", 0)))
                price = float(item.get("p", 0))

                if 0 <= price <= 1:
                    points.append(PricePoint(timestamp=timestamp, price=price))
            except (ValueError, TypeError) as e:
                logger.debug(f"Error parseando prices-history: {e}")

        return sorted(points, key=lambda p: p.timestamp)

    # =========================================================================
    # Análisis
    # =========================================================================

    @staticmethod
    def _precio_en_momento(
        history: list[PricePoint], target_time: datetime
    ) -> float | None:
        """Encuentra el precio más cercano a un momento dado."""
        if not history:
            return None

        closest = min(
            history,
            key=lambda p: abs((p.timestamp - target_time).total_seconds()),
        )

        # Solo considerar si está dentro de un rango razonable (3 horas)
        diff = abs((closest.timestamp - target_time).total_seconds())
        if diff > 3 * 3600:
            return None

        return closest.price

    def _determinar_tendencia(
        self, cambio_pct: float, points: list[PricePoint]
    ) -> Trend:
        """
        Determina la tendencia basada en el cambio porcentual
        y la consistencia de la dirección.
        """
        if not points or len(points) < 2:
            return Trend.INSUFFICIENT

        # Calcular volatilidad del período
        vol = self._calcular_volatilidad(points)

        if abs(cambio_pct) < self.STABLE_THRESHOLD * 100:
            if vol > self.SHARP_MOVEMENT_THRESHOLD:
                return Trend.VOLATILE
            return Trend.STABLE

        if cambio_pct > 0:
            return Trend.RISING
        return Trend.FALLING

    @staticmethod
    def _calcular_volatilidad(points: list[PricePoint]) -> float:
        """Calcula la volatilidad como desviación estándar de retornos."""
        if len(points) < 2:
            return 0.0

        precios = [p.price for p in points]
        retornos = [
            (precios[i] - precios[i - 1]) / max(precios[i - 1], 0.001)
            for i in range(1, len(precios))
        ]

        if not retornos:
            return 0.0

        mean = sum(retornos) / len(retornos)
        variance = sum((r - mean) ** 2 for r in retornos) / len(retornos)
        return variance ** 0.5

    @staticmethod
    def _calcular_cambio_pct(
        precio_anterior: float | None, precio_actual: float
    ) -> float:
        """Calcula el cambio porcentual entre dos precios."""
        if precio_anterior is None or precio_anterior == 0:
            return 0.0
        return ((precio_actual - precio_anterior) / precio_anterior) * 100

    def _detectar_movimientos_bruscos(
        self, points: list[PricePoint]
    ) -> list[PriceMovement]:
        """
        Detecta movimientos bruscos de precio (> 5% en un intervalo).
        """
        movimientos: list[PriceMovement] = []

        if len(points) < 2:
            return movimientos

        for i in range(1, len(points)):
            precio_ant = points[i - 1].price
            precio_act = points[i].price

            if precio_ant == 0:
                continue

            cambio = (precio_act - precio_ant) / precio_ant

            if abs(cambio) >= self.SHARP_MOVEMENT_THRESHOLD:
                movimientos.append(
                    PriceMovement(
                        timestamp=points[i].timestamp,
                        price_before=precio_ant,
                        price_after=precio_act,
                        change_pct=cambio * 100,
                        direction="up" if cambio > 0 else "down",
                    )
                )

        return movimientos


# =============================================================================
# Ejecución directa para pruebas
# =============================================================================

def main() -> None:
    """Prueba el MarketHistory con datos de ejemplo."""
    import asyncio
    from config import configurar_logging
    from rich.console import Console

    configurar_logging()
    console = Console()

    console.print("\n[bold cyan]Market History - Fase 2 (async)[/bold cyan]\n")

    async def _run() -> None:
        async with MarketHistory() as mh:
            # Test con un mercado real (necesita acceso a la API)
            analysis = await mh.analizar_mercado(
                condition_id="0x_example",
                token_id="token_example",
                current_price=0.65,
            )
            console.print(analysis.resumen())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
