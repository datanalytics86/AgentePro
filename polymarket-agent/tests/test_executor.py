"""
Tests del Executor y Portfolio - Fase 6.

Verifica paper trading, registro de trades y métricas.
"""

import os
import tempfile

import pytest

from core.executor import PaperExecutor
from core.portfolio import Portfolio
from core.models import TradeSignal


@pytest.fixture
def temp_db() -> str:
    """Crea una DB temporal para tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def portfolio(temp_db: str) -> Portfolio:
    return Portfolio(db_path=temp_db)


@pytest.fixture
def paper_executor() -> PaperExecutor:
    return PaperExecutor(balance_inicial=500.0)


@pytest.fixture
def signal_compra() -> TradeSignal:
    return TradeSignal(
        market_id="0xtest123",
        market_question="Will Bitcoin hit 100k?",
        side="YES",
        action="BUY",
        entry_price=0.50,
        estimated_probability=0.65,
        edge=0.15,
        confidence="high",
        suggested_size_usd=10.0,
        kelly_fraction=0.02,
        reasoning="Positive momentum",
    )


class TestPaperExecutor:
    """Tests del ejecutor de paper trading."""

    def test_ejecutar_orden_exitosa(
        self, paper_executor: PaperExecutor,
        portfolio: Portfolio,
        signal_compra: TradeSignal,
    ) -> None:
        """Ejecuta una orden paper y reduce el balance."""
        record = paper_executor.ejecutar_orden(signal_compra, portfolio)

        assert record is not None
        assert record.status == "filled"
        assert record.mode == "paper"
        assert record.price == 0.50
        assert record.shares == 20.0  # 10 / 0.50
        assert paper_executor.obtener_balance() == 490.0

    def test_ejecutar_orden_balance_insuficiente(
        self, portfolio: Portfolio, signal_compra: TradeSignal,
    ) -> None:
        """Rechaza si no hay balance suficiente."""
        executor = PaperExecutor(balance_inicial=5.0)
        record = executor.ejecutar_orden(signal_compra, portfolio)
        assert record is None

    def test_balance_inicial(self, paper_executor: PaperExecutor) -> None:
        """Balance inicial es el configurado."""
        assert paper_executor.obtener_balance() == 500.0

    def test_multiples_ordenes(
        self, paper_executor: PaperExecutor,
        portfolio: Portfolio,
        signal_compra: TradeSignal,
    ) -> None:
        """Ejecuta múltiples órdenes y el balance se reduce."""
        paper_executor.ejecutar_orden(signal_compra, portfolio)
        paper_executor.ejecutar_orden(signal_compra, portfolio)

        assert paper_executor.obtener_balance() == 480.0


class TestPortfolio:
    """Tests del portfolio con persistencia."""

    def test_registrar_trade(
        self, portfolio: Portfolio, signal_compra: TradeSignal
    ) -> None:
        """Registra un trade correctamente."""
        record = portfolio.registrar_trade(
            signal=signal_compra,
            price=0.50,
            shares=20.0,
            status="filled",
        )
        assert record.trade_id.startswith("TR-")
        assert record.price == 0.50
        assert record.shares == 20.0

    def test_posiciones_abiertas(
        self, portfolio: Portfolio, signal_compra: TradeSignal
    ) -> None:
        """Trackea posiciones abiertas correctamente."""
        portfolio.registrar_trade(
            signal=signal_compra, price=0.50, shares=20.0
        )
        posiciones = portfolio.obtener_posiciones_abiertas()

        assert len(posiciones) == 1
        assert posiciones[0].market_id == "0xtest123"
        assert posiciones[0].side == "YES"

    def test_cerrar_posicion(
        self, portfolio: Portfolio, signal_compra: TradeSignal
    ) -> None:
        """Cierra posición y calcula PnL."""
        portfolio.registrar_trade(
            signal=signal_compra, price=0.50, shares=20.0
        )

        pnl = portfolio.cerrar_posicion("0xtest123", "YES", 0.70)
        # PnL = (0.70 - 0.50) * 20 = 4.0
        assert pnl == pytest.approx(4.0)

    def test_exposicion_total(
        self, portfolio: Portfolio, signal_compra: TradeSignal
    ) -> None:
        """Calcula exposición total correctamente."""
        portfolio.registrar_trade(
            signal=signal_compra, price=0.50, shares=20.0
        )
        exposicion = portfolio.calcular_exposicion_total()
        assert exposicion == pytest.approx(10.0)  # 0.50 * 20

    def test_metricas_vacias(self, portfolio: Portfolio) -> None:
        """Métricas son 0 sin trades."""
        metricas = portfolio.calcular_metricas()
        assert metricas.total_trades == 0
        assert metricas.win_rate == 0.0

    def test_persistencia(
        self, temp_db: str, signal_compra: TradeSignal
    ) -> None:
        """Los datos persisten tras reiniciar el portfolio."""
        # Crear y registrar
        p1 = Portfolio(db_path=temp_db)
        p1.registrar_trade(signal=signal_compra, price=0.50, shares=20.0)

        # Recrear desde el mismo DB
        p2 = Portfolio(db_path=temp_db)
        posiciones = p2.obtener_posiciones_abiertas()
        assert len(posiciones) == 1

    def test_registrar_balance(self, portfolio: Portfolio) -> None:
        """Registra historial de balance."""
        portfolio.registrar_balance(500.0)
        portfolio.registrar_balance(510.0)

        historial = portfolio.obtener_historial_balance()
        assert len(historial) == 2

    def test_trades_recientes(
        self, portfolio: Portfolio, signal_compra: TradeSignal
    ) -> None:
        """Obtiene trades recientes ordenados."""
        portfolio.registrar_trade(
            signal=signal_compra, price=0.50, shares=20.0
        )
        trades = portfolio.obtener_trades_recientes(limit=5)
        assert len(trades) == 1
        assert trades[0]["market_id"] == "0xtest123"
