"""
Tests de las utilidades de red - Optimización async.

Verifica TTLCache y retry_async decorator.
"""

import asyncio
import time

import pytest

from core.net_utils import TTLCache, retry_async


# =============================================================================
# Tests del TTLCache
# =============================================================================

class TestTTLCache:
    """Tests del caché LRU con TTL."""

    def test_put_y_get(self) -> None:
        """Almacena y recupera un valor."""
        cache = TTLCache(ttl_seconds=60)
        cache.put("m1", {"prob": 0.7}, price=0.50)
        result = cache.get("m1", current_price=0.50)
        assert result == {"prob": 0.7}

    def test_miss_no_existe(self) -> None:
        """Retorna None para clave inexistente."""
        cache = TTLCache(ttl_seconds=60)
        assert cache.get("no_existe", current_price=0.5) is None

    def test_invalidar_por_precio(self) -> None:
        """Invalida si el precio cambió > threshold (2%)."""
        cache = TTLCache(ttl_seconds=60, price_threshold=0.02)
        cache.put("m1", "value", price=0.50)

        # Precio cambió 1% → aún válido
        assert cache.get("m1", current_price=0.505) is not None

        # Precio cambió 5% → invalidado
        assert cache.get("m1", current_price=0.525) is None

    def test_expirar_por_ttl(self) -> None:
        """Expira entradas después del TTL."""
        cache = TTLCache(ttl_seconds=1)  # 1 segundo
        cache.put("m1", "value", price=0.50)

        # Inmediatamente → válido
        assert cache.get("m1", current_price=0.50) is not None

        # Esperar a que expire
        time.sleep(1.1)
        assert cache.get("m1", current_price=0.50) is None

    def test_max_size_eviction(self) -> None:
        """Elimina las entradas más viejas al exceder max_size."""
        cache = TTLCache(ttl_seconds=60, max_size=3)
        cache.put("m1", "v1", price=0.5)
        cache.put("m2", "v2", price=0.5)
        cache.put("m3", "v3", price=0.5)
        cache.put("m4", "v4", price=0.5)  # Debería eliminar m1

        assert cache.get("m1", current_price=0.5) is None
        assert cache.get("m4", current_price=0.5) == "v4"

    def test_lru_ordering(self) -> None:
        """Acceder a una entrada la mueve al final (LRU)."""
        cache = TTLCache(ttl_seconds=60, max_size=3)
        cache.put("m1", "v1", price=0.5)
        cache.put("m2", "v2", price=0.5)
        cache.put("m3", "v3", price=0.5)

        # Acceder a m1 lo mueve al final
        cache.get("m1", current_price=0.5)

        # Ahora m2 es el más viejo → se elimina al insertar m4
        cache.put("m4", "v4", price=0.5)
        assert cache.get("m1", current_price=0.5) == "v1"  # Aún vivo
        assert cache.get("m2", current_price=0.5) is None    # Eliminado

    def test_stats(self) -> None:
        """Verifica las estadísticas del caché."""
        cache = TTLCache(ttl_seconds=60)
        cache.put("m1", "v1", price=0.5)

        cache.get("m1", current_price=0.5)  # hit
        cache.get("m2", current_price=0.5)  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate_pct"] == 50.0

    def test_clear(self) -> None:
        """Limpia todo el caché."""
        cache = TTLCache(ttl_seconds=60)
        cache.put("m1", "v1", price=0.5)
        cache.clear()
        assert cache.get("m1", current_price=0.5) is None
        assert cache.stats["size"] == 0


# =============================================================================
# Tests del retry_async decorator
# =============================================================================

class TestRetryAsync:
    """Tests del decorador de reintentos async."""

    def test_exito_sin_reintentos(self) -> None:
        """Función exitosa no se reintenta."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def func_ok():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = asyncio.get_event_loop().run_until_complete(func_ok())
        assert result == "ok"
        assert call_count == 1

    def test_reintento_tras_error(self) -> None:
        """Reintenta después de error recuperable."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01, exceptions=(ConnectionResetError,))
        async def func_falla_una_vez():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionResetError("10054")
            return "recovered"

        result = asyncio.get_event_loop().run_until_complete(func_falla_una_vez())
        assert result == "recovered"
        assert call_count == 2

    def test_fallo_total_tras_max_intentos(self) -> None:
        """Lanza excepción si excede max_attempts."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01, exceptions=(ConnectionResetError,))
        async def func_siempre_falla():
            nonlocal call_count
            call_count += 1
            raise ConnectionResetError("siempre falla")

        with pytest.raises(ConnectionResetError):
            asyncio.get_event_loop().run_until_complete(func_siempre_falla())
        assert call_count == 3

    def test_no_reintenta_excepciones_no_listadas(self) -> None:
        """No reintenta excepciones que no están en la lista."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01, exceptions=(ConnectionResetError,))
        async def func_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("no retriable")

        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(func_value_error())
        assert call_count == 1


# =============================================================================
# Tests de integración SQLite WAL
# =============================================================================

class TestSQLiteWAL:
    """Tests de la optimización WAL mode en Portfolio."""

    def test_wal_mode_activado(self, tmp_path) -> None:
        """Verifica que el WAL mode se activa correctamente."""
        import sqlite3
        from core.portfolio import Portfolio

        db_path = str(tmp_path / "test.db")
        portfolio = Portfolio(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0] == "wal"

    def test_indices_creados(self, tmp_path) -> None:
        """Verifica que los índices se crean correctamente."""
        import sqlite3
        from core.portfolio import Portfolio

        db_path = str(tmp_path / "test.db")
        portfolio = Portfolio(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            indices = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            nombres = [i[0] for i in indices]

        assert "idx_trades_market" in nombres
        assert "idx_trades_created" in nombres
        assert "idx_trades_status" in nombres
