"""
Utilidades de red compartidas: retry blindado, caché LRU con TTL.

Proporciona decoradores y helpers reutilizables para:
- Reintentos con backoff exponencial (incluye WinError 10054)
- Caché en memoria con TTL configurable
- Timeout por ciclo para modo Turbo
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# =========================================================================
# Errores de red que merecen reintento (incluye WinError 10054)
# =========================================================================

RETRIABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,
    ConnectionResetError,       # WinError 10054 subyacente
    ConnectionAbortedError,     # Variante Windows
    OSError,                    # Capa baja que envuelve 10054
)


# =========================================================================
# Retry async con backoff exponencial blindado
# =========================================================================

def retry_async(
    max_attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    exceptions: tuple = RETRIABLE_EXCEPTIONS,
) -> Callable:
    """
    Decorador de reintento async con backoff exponencial.

    Diseñado específicamente para manejar WinError 10054
    (ConnectionResetError) y otros errores de red transitorios.

    Delays: 2s, 4s, 8s, 16s (capped en max_delay).
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__}: Fallaron {max_attempts} "
                            f"intentos. Último error: {e}"
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    error_name = type(e).__name__
                    logger.warning(
                        f"{func.__name__}: {error_name} en intento "
                        f"{attempt}/{max_attempts}. Retry en {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


# =========================================================================
# Caché LRU en memoria con TTL (para evaluaciones del LLM)
# =========================================================================

class TTLCache:
    """
    Caché en memoria con Time-To-Live y capacidad máxima.

    Diseñado para almacenar evaluaciones del LLM y evitar
    re-analizar mercados cuyo precio no cambió significativamente.

    Con 32 GB de RAM disponible, puede almacenar miles de entradas.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 2000,
        price_threshold: float = 0.02,
    ) -> None:
        """
        Args:
            ttl_seconds: Tiempo de vida de cada entrada (default 60 min).
            max_size: Máximo de entradas en caché.
            price_threshold: Cambio mínimo de precio para invalidar (2%).
        """
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._price_threshold = price_threshold
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(
        self, market_id: str, current_price: float
    ) -> Any | None:
        """
        Obtiene un valor del caché si existe, no expiró y el precio no
        cambió significativamente.

        Args:
            market_id: ID del mercado.
            current_price: Precio actual del token YES.

        Returns:
            Valor cacheado o None si no hay hit.
        """
        entry = self._cache.get(market_id)
        if entry is None:
            self._misses += 1
            return None

        # Verificar expiración
        if time.monotonic() - entry["timestamp"] > self._ttl:
            del self._cache[market_id]
            self._misses += 1
            return None

        # Verificar si el precio cambió significativamente
        cached_price = entry["price"]
        if cached_price > 0:
            price_change = abs(current_price - cached_price) / cached_price
            if price_change > self._price_threshold:
                del self._cache[market_id]
                self._misses += 1
                logger.debug(
                    f"Cache invalidado para {market_id[:12]}: "
                    f"precio cambió {price_change:.1%}"
                )
                return None

        self._hits += 1
        # Mover al final (LRU)
        self._cache.move_to_end(market_id)
        return entry["value"]

    def put(
        self, market_id: str, value: Any, price: float
    ) -> None:
        """Almacena un valor en el caché."""
        # Evitar overflow
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Eliminar el más viejo

        self._cache[market_id] = {
            "value": value,
            "price": price,
            "timestamp": time.monotonic(),
        }

    def invalidate(self, market_id: str) -> bool:
        """Elimina una entrada específica del caché. Retorna True si existía."""
        if market_id in self._cache:
            del self._cache[market_id]
            return True
        return False

    def clear(self) -> None:
        """Limpia todo el caché."""
        self._cache.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Estadísticas del caché."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate * 100, 1),
        }


# =========================================================================
# Generador de claves de caché
# =========================================================================

def cache_key(market_id: str, *extra: str) -> str:
    """Genera una clave de caché determinística."""
    parts = f"{market_id}:{'|'.join(extra)}"
    return hashlib.md5(parts.encode()).hexdigest()
