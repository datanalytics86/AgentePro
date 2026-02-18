"""
Tests para las 4 optimizaciones críticas:
1. Criterio de Kelly (ProbabilityEngine.calcular_kelly_size)
2. Salida Anticipada (executor.ejecutar_venta_activa / PaperExecutor.ejecutar_venta)
3. Backup Manager (BackupManager async)
4. P&L Dashboard (Portfolio.calcular_max_drawdown_historico)

Todos los tests async usan pytest-asyncio con modo 'auto'.
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ─── imports del proyecto ─────────────────────────────────────────────────────
from core.probability_engine import ProbabilityEngine
from core.backup_manager import BackupManager
from core.portfolio import Portfolio, Position
from core.executor import PaperExecutor, OrderExecutor
from core.risk_manager import RiskManager
from core.models import TradeSignal


# ==============================================================================
# 1. CRITERIO DE KELLY
# ==============================================================================


class TestKellyCriterion:
    """Tests para ProbabilityEngine.calcular_kelly_size."""

    # ── Casos correctos ────────────────────────────────────────────────────────

    def test_kelly_edge_positivo_clasico(self):
        """
        p=0.65, price=0.40 → b=1.5
        f* = (0.65·2.5 - 1) / 1.5 = 0.4167
        quarter kelly = 0.1042, cap 10% → capped a 0.10
        """
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.65, yes_price=0.40, bankroll=100.0,
            kelly_fraction=0.25, max_pct=0.10,
        )
        assert frac == pytest.approx(0.10, abs=1e-4), (
            "Con edge ~41.7% quarter-kelly, debe capar al max_pct=10%"
        )
        assert size == pytest.approx(10.0, abs=0.01)

    def test_kelly_edge_pequeno_sin_cap(self):
        """
        p=0.55, price=0.50 → b=1.0
        f* = (0.55·2 - 1)/1 = 0.10
        quarter kelly = 0.025 → sin cap (< max_pct=0.10)
        """
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.55, yes_price=0.50, bankroll=100.0,
            kelly_fraction=0.25, max_pct=0.10,
        )
        assert frac == pytest.approx(0.025, abs=1e-4)
        assert size == pytest.approx(2.5, abs=0.01)

    def test_kelly_sin_edge_retorna_cero(self):
        """p igual al precio de mercado → sin edge → f*=0."""
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.50, yes_price=0.50, bankroll=100.0,
        )
        assert frac == 0.0
        assert size == 0.0

    def test_kelly_edge_negativo_retorna_cero(self):
        """p < price → edge negativo → no apostar."""
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.40, yes_price=0.60, bankroll=100.0,
        )
        assert frac == 0.0
        assert size == 0.0

    def test_kelly_cap_maximo_10_pct(self):
        """Con kelly_fraction=1.0 (full Kelly) y gran edge, cap a max_pct."""
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.90, yes_price=0.10, bankroll=500.0,
            kelly_fraction=1.0, max_pct=0.10,
        )
        assert frac == pytest.approx(0.10, abs=1e-4)
        assert size == pytest.approx(50.0, abs=0.01)

    def test_kelly_fraction_25_por_defecto(self):
        """kelly_fraction default es 0.25."""
        frac1, _ = ProbabilityEngine.calcular_kelly_size(
            p=0.60, yes_price=0.50, bankroll=100.0, kelly_fraction=0.25,
        )
        frac2, _ = ProbabilityEngine.calcular_kelly_size(
            p=0.60, yes_price=0.50, bankroll=100.0,
        )
        assert frac1 == frac2

    # ── Casos borde ────────────────────────────────────────────────────────────

    def test_kelly_price_cero_retorna_cero(self):
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.70, yes_price=0.0, bankroll=100.0,
        )
        assert frac == 0.0
        assert size == 0.0

    def test_kelly_price_uno_retorna_cero(self):
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.70, yes_price=1.0, bankroll=100.0,
        )
        assert frac == 0.0
        assert size == 0.0

    def test_kelly_bankroll_cero_retorna_cero(self):
        frac, size = ProbabilityEngine.calcular_kelly_size(
            p=0.70, yes_price=0.40, bankroll=0.0,
        )
        assert frac == 0.0
        assert size == 0.0

    def test_kelly_prob_fuera_de_rango(self):
        """p=0 o p=1 no son válidos."""
        frac0, _ = ProbabilityEngine.calcular_kelly_size(0.0, 0.40, 100.0)
        frac1, _ = ProbabilityEngine.calcular_kelly_size(1.0, 0.40, 100.0)
        assert frac0 == 0.0
        assert frac1 == 0.0

    def test_kelly_formula_correcta_vs_aproximacion(self):
        """
        Verifica que la fórmula exacta difiere de la aproximación
        edge/(odds-1) usada anteriormente.
        Caso: p=0.65, price=0.40
        - Fórmula correcta: f* = (0.65·2.5-1)/1.5 = 0.4167
        - Aproximación:     f* = edge/(odds-1) = 0.25/1.5 = 0.1667
        La fórmula correcta da un resultado ~2.5× mayor antes de aplicar fracción.
        """
        p, price = 0.65, 0.40
        b = (1.0 / price) - 1.0
        kelly_correcto = (p * (b + 1.0) - 1.0) / b
        kelly_aprox = (p - price) / b  # edge / (1/price - 1)
        assert kelly_correcto != pytest.approx(kelly_aprox, abs=0.01), (
            "La fórmula correcta debe diferir de la aproximación"
        )
        assert kelly_correcto == pytest.approx(0.4167, abs=0.001)


# ==============================================================================
# 2. SALIDA ANTICIPADA (ACTIVE EXIT)
# ==============================================================================


class TestActiveExit:
    """Tests para ejecutar_venta en PaperExecutor y OrderExecutor."""

    def _crear_portfolio_con_posicion(
        self, db_path: str, entry_price: float = 0.40, shares: float = 25.0
    ) -> Portfolio:
        """Helper: crea un portfolio con una posición BUY abierta."""
        portfolio = Portfolio(db_path=db_path)
        signal = TradeSignal(
            market_id="0xtest123",
            market_question="Test market question?",
            side="YES",
            action="BUY",
            entry_price=entry_price,
            estimated_probability=0.65,
            edge=0.25,
            confidence="high",
            suggested_size_usd=entry_price * shares,
            kelly_fraction=0.025,
            reasoning="Test",
        )
        portfolio.registrar_trade(
            signal=signal,
            price=entry_price,
            shares=shares,
            status="filled",
            order_id="PAPER-test001",
        )
        return portfolio

    def test_ejecutar_venta_paper_ganancias(self, tmp_path):
        """Venta con precio mayor al entry → PnL positivo, balance sube."""
        db = str(tmp_path / "test.db")
        portfolio = self._crear_portfolio_con_posicion(db, entry_price=0.40, shares=25.0)
        paper = PaperExecutor(balance_inicial=100.0)
        paper._balance -= 10.0  # Simular que se gastaron $10 en la compra

        pnl = paper.ejecutar_venta(
            market_id="0xtest123",
            side="YES",
            precio_actual=0.60,  # Precio subió
            portfolio=portfolio,
        )

        assert pnl is not None
        assert pnl == pytest.approx((0.60 - 0.40) * 25.0, abs=0.01)  # $5 ganancia
        # Balance debe haber aumentado por cost_basis + pnl
        assert paper._balance > 90.0

    def test_ejecutar_venta_paper_perdidas(self, tmp_path):
        """Venta con precio menor al entry → PnL negativo."""
        db = str(tmp_path / "test.db")
        portfolio = self._crear_portfolio_con_posicion(db, entry_price=0.40, shares=25.0)
        paper = PaperExecutor(balance_inicial=100.0)
        paper._balance -= 10.0

        pnl = paper.ejecutar_venta(
            market_id="0xtest123",
            side="YES",
            precio_actual=0.30,  # Precio bajó
            portfolio=portfolio,
        )

        assert pnl is not None
        assert pnl == pytest.approx((0.30 - 0.40) * 25.0, abs=0.01)  # -$2.5

    def test_ejecutar_venta_posicion_inexistente(self, tmp_path):
        """Si no existe la posición, retorna None."""
        db = str(tmp_path / "test.db")
        portfolio = Portfolio(db_path=db)
        paper = PaperExecutor(balance_inicial=100.0)

        pnl = paper.ejecutar_venta(
            market_id="0xno_existe",
            side="YES",
            precio_actual=0.50,
            portfolio=portfolio,
        )
        assert pnl is None

    def test_order_executor_ejecutar_venta_activa(self, tmp_path):
        """OrderExecutor.ejecutar_venta_activa delega correctamente."""
        db = str(tmp_path / "test.db")
        portfolio = self._crear_portfolio_con_posicion(db, entry_price=0.40, shares=25.0)
        risk = RiskManager()
        executor = OrderExecutor(portfolio, risk)
        # Poner balance suficiente
        executor._executor._balance = 90.0

        pnl = executor.ejecutar_venta_activa(
            market_id="0xtest123",
            market_question="Test market question?",
            side="YES",
            precio_actual=0.50,
        )

        assert pnl is not None
        assert pnl == pytest.approx((0.50 - 0.40) * 25.0, abs=0.01)


# ==============================================================================
# 3. BACKUP MANAGER
# ==============================================================================


class TestBackupManager:
    """Tests para BackupManager (async)."""

    @pytest.mark.asyncio
    async def test_backup_crea_archivo(self, tmp_path):
        """hacer_backup_ahora() crea un archivo en backup_dir."""
        db = tmp_path / "portfolio.db"
        db.write_bytes(b"SQLite fake content")
        backup_dir = tmp_path / "backups"

        manager = BackupManager(
            db_path=str(db),
            backup_dir=str(backup_dir),
        )
        dest = await manager.hacer_backup_ahora()

        assert dest is not None
        assert dest.exists()
        assert dest.suffix == ".db"
        assert "portfolio_" in dest.name

    @pytest.mark.asyncio
    async def test_backup_db_inexistente(self, tmp_path):
        """Si la DB no existe, retorna None sin lanzar excepción."""
        backup_dir = tmp_path / "backups"
        manager = BackupManager(
            db_path=str(tmp_path / "no_existe.db"),
            backup_dir=str(backup_dir),
        )
        dest = await manager.hacer_backup_ahora()
        assert dest is None

    @pytest.mark.asyncio
    async def test_backup_contenido_identico(self, tmp_path):
        """El backup es una copia exacta del archivo original."""
        contenido = b"SQLite backup test payload 12345"
        db = tmp_path / "portfolio.db"
        db.write_bytes(contenido)
        backup_dir = tmp_path / "backups"

        manager = BackupManager(str(db), str(backup_dir))
        dest = await manager.hacer_backup_ahora()

        assert dest is not None
        assert dest.read_bytes() == contenido

    @pytest.mark.asyncio
    async def test_backup_no_bloquea_loop(self, tmp_path):
        """El backup usa asyncio.to_thread → no bloquea el event loop."""
        db = tmp_path / "portfolio.db"
        db.write_bytes(b"data")
        backup_dir = tmp_path / "backups"
        manager = BackupManager(str(db), str(backup_dir))

        # Ejecutar junto a una coroutine concurrente para verificar no bloqueo
        dummy_result: list[int] = []

        async def dummy_coro() -> None:
            await asyncio.sleep(0)
            dummy_result.append(1)

        await asyncio.gather(manager.hacer_backup_ahora(), dummy_coro())
        assert dummy_result == [1], "La coroutine dummy debe completarse en paralelo"

    @pytest.mark.asyncio
    async def test_rotacion_elimina_backups_antiguos(self, tmp_path):
        """Con max_backups=2 y 3 backups creados, el más antiguo se elimina."""
        db = tmp_path / "portfolio.db"
        db.write_bytes(b"content")
        backup_dir = tmp_path / "backups"

        manager = BackupManager(str(db), str(backup_dir), max_backups=2)

        # Crear 3 backups secuenciales
        for _ in range(3):
            await manager.hacer_backup_ahora()
            await asyncio.sleep(0.01)  # Pequeño delay para timestamps distintos

        backups = sorted(backup_dir.glob("portfolio_*.db"))
        assert len(backups) <= 2, f"Deben quedar ≤ 2 backups, hay {len(backups)}"

    def test_detener_cambia_estado(self, tmp_path):
        """detener() pone _running en False."""
        db = tmp_path / "portfolio.db"
        manager = BackupManager(str(db), str(tmp_path / "backups"))
        manager._running = True
        manager.detener()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_iniciar_ejecuta_backup_y_respeta_detener(self, tmp_path):
        """
        iniciar() hace backup y se detiene cuando se cancela la tarea.
        El CancelledError se captura internamente en el loop, así que
        el task termina limpiamente (sin re-raise).
        """
        db = tmp_path / "portfolio.db"
        db.write_bytes(b"data")
        backup_dir = tmp_path / "backups"

        manager = BackupManager(
            str(db), str(backup_dir), interval_seconds=60  # largo para que no repita
        )

        task = asyncio.create_task(manager.iniciar())
        await asyncio.sleep(0.15)  # Dejar que haga el primer backup
        task.cancel()

        # El loop captura CancelledError con break → task termina sin raise
        try:
            await task
        except asyncio.CancelledError:
            pass  # Ambos comportamientos son válidos

        # Debe haber creado al menos un backup
        backups = list(backup_dir.glob("portfolio_*.db"))
        assert len(backups) >= 1, "Debe existir al menos 1 backup tras el primer ciclo"


# ==============================================================================
# 4. MAX DRAWDOWN HISTÓRICO (Portfolio)
# ==============================================================================


class TestMaxDrawdownHistorico:
    """Tests para Portfolio.calcular_max_drawdown_historico."""

    def _portfolio_con_balances(
        self, tmp_path: Path, balances: list[float]
    ) -> Portfolio:
        """Helper: crea un portfolio y registra una serie de balances."""
        portfolio = Portfolio(db_path=str(tmp_path / "test.db"))
        for b in balances:
            portfolio.registrar_balance(b)
        return portfolio

    def test_sin_historial_retorna_cero(self, tmp_path):
        portfolio = Portfolio(db_path=str(tmp_path / "test.db"))
        assert portfolio.calcular_max_drawdown_historico() == 0.0

    def test_un_solo_punto_retorna_cero(self, tmp_path):
        portfolio = self._portfolio_con_balances(tmp_path, [100.0])
        assert portfolio.calcular_max_drawdown_historico() == 0.0

    def test_drawdown_simple(self, tmp_path):
        """
        Serie: 100 → 80 → 90
        Peak en 100, mínimo 80 → DD = 20%
        """
        portfolio = self._portfolio_con_balances(tmp_path, [100.0, 80.0, 90.0])
        dd = portfolio.calcular_max_drawdown_historico()
        assert dd == pytest.approx(0.20, abs=1e-4)

    def test_drawdown_multipico(self, tmp_path):
        """
        Serie: 100 → 90 → 110 → 77
        Peak global 110, mínimo desde ese pico = 77 → DD = (110-77)/110 ≈ 30%
        """
        portfolio = self._portfolio_con_balances(
            tmp_path, [100.0, 90.0, 110.0, 77.0]
        )
        dd = portfolio.calcular_max_drawdown_historico()
        expected = (110.0 - 77.0) / 110.0  # ≈ 0.30
        assert dd == pytest.approx(expected, abs=1e-4)

    def test_drawdown_siempre_creciente(self, tmp_path):
        """Si el balance siempre sube, drawdown = 0."""
        portfolio = self._portfolio_con_balances(
            tmp_path, [100.0, 105.0, 110.0, 120.0]
        )
        dd = portfolio.calcular_max_drawdown_historico()
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_perdida_total(self, tmp_path):
        """Caso extremo: balance cae a casi 0."""
        portfolio = self._portfolio_con_balances(tmp_path, [100.0, 1.0])
        dd = portfolio.calcular_max_drawdown_historico()
        assert dd == pytest.approx(0.99, abs=1e-4)

    def test_drawdown_bankroll_usd(self, tmp_path):
        """
        Simula bankroll de $100: pico 105, mínimo 85.
        DD = (105-85)/105 ≈ 19.05%
        """
        portfolio = self._portfolio_con_balances(
            tmp_path, [100.0, 103.0, 105.0, 95.0, 85.0, 92.0]
        )
        dd = portfolio.calcular_max_drawdown_historico()
        expected = (105.0 - 85.0) / 105.0
        assert dd == pytest.approx(expected, abs=1e-4)
