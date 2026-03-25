"""
Tests del Risk Manager y Strategy - Fases 4-5.

Verifica que las reglas de riesgo se aplican correctamente
y la estrategia genera señales coherentes.
"""

import pytest
from datetime import datetime

from core.risk_manager import RiskManager
from core.strategy import TradingStrategy
from core.models import LLMEvaluation, Market, Token, TradeSignal
from core.data_collector import MarketContext


# =============================================================================
# Tests del RiskManager
# =============================================================================

class TestRiskManager:
    """Tests de las reglas de gestión de riesgo."""

    @pytest.fixture
    def rm(self) -> RiskManager:
        return RiskManager()

    @pytest.fixture
    def signal_basica(self) -> TradeSignal:
        return TradeSignal(
            market_id="0xtest",
            market_question="Test market?",
            side="YES",
            action="BUY",
            entry_price=0.50,
            estimated_probability=0.65,
            edge=0.15,
            confidence="high",
            suggested_size_usd=5.0,
            kelly_fraction=0.02,
            reasoning="test",
        )

    def test_aprobar_trade_valido(
        self, rm: RiskManager, signal_basica: TradeSignal
    ) -> None:
        """Aprueba un trade que cumple todos los límites."""
        aprobado, razon = rm.aprobar_trade(
            signal=signal_basica,
            balance_actual=50.0,
            exposicion_total=5.0,
            exposicion_mercado=0.0,
            exposicion_categoria=0.0,
            perdida_diaria=0.0,
            perdida_semanal=0.0,
            drawdown_actual=0.0,
        )
        assert aprobado is True
        assert "Aprobado" in razon

    def test_rechazar_balance_insuficiente(
        self, rm: RiskManager, signal_basica: TradeSignal
    ) -> None:
        """Rechaza si no hay balance suficiente."""
        aprobado, razon = rm.aprobar_trade(
            signal=signal_basica,
            balance_actual=2.0,  # < 5 USD
            exposicion_total=0.0,
            exposicion_mercado=0.0,
            exposicion_categoria=0.0,
            perdida_diaria=0.0,
            perdida_semanal=0.0,
            drawdown_actual=0.0,
        )
        assert aprobado is False
        assert "Balance insuficiente" in razon

    def test_rechazar_excede_max_por_trade(
        self, rm: RiskManager
    ) -> None:
        """Rechaza si excede el máximo por trade (10% de 60 = $6)."""
        signal = TradeSignal(
            market_id="0xtest", market_question="Test?",
            side="YES", action="BUY", entry_price=0.50,
            estimated_probability=0.65, edge=0.15, confidence="high",
            suggested_size_usd=8.0,  # > $6
            kelly_fraction=0.06, reasoning="test",
        )
        aprobado, _ = rm.aprobar_trade(
            signal=signal, balance_actual=50.0,
            exposicion_total=0.0, exposicion_mercado=0.0,
            exposicion_categoria=0.0, perdida_diaria=0.0,
            perdida_semanal=0.0, drawdown_actual=0.0,
        )
        assert aprobado is False

    def test_rechazar_excede_exposicion_total(
        self, rm: RiskManager, signal_basica: TradeSignal
    ) -> None:
        """Rechaza si excede exposición total (70% de 60 = $42)."""
        aprobado, razon = rm.aprobar_trade(
            signal=signal_basica,
            balance_actual=50.0,
            exposicion_total=39.0,  # 39 + 5 = 44 > 42
            exposicion_mercado=0.0,
            exposicion_categoria=0.0,
            perdida_diaria=0.0,
            perdida_semanal=0.0,
            drawdown_actual=0.0,
        )
        assert aprobado is False
        assert "exposición total" in razon.lower()

    def test_rechazar_drawdown_maximo(
        self, rm: RiskManager, signal_basica: TradeSignal
    ) -> None:
        """Rechaza si el drawdown supera el máximo (25%)."""
        aprobado, razon = rm.aprobar_trade(
            signal=signal_basica,
            balance_actual=50.0,
            exposicion_total=0.0,
            exposicion_mercado=0.0,
            exposicion_categoria=0.0,
            perdida_diaria=0.0,
            perdida_semanal=0.0,
            drawdown_actual=0.30,  # > 0.25
        )
        assert aprobado is False
        assert "Drawdown" in razon

    def test_rechazar_perdida_diaria_maxima(
        self, rm: RiskManager, signal_basica: TradeSignal
    ) -> None:
        """Rechaza si se alcanzó el límite de pérdida diaria."""
        aprobado, _ = rm.aprobar_trade(
            signal=signal_basica,
            balance_actual=50.0,
            exposicion_total=0.0,
            exposicion_mercado=0.0,
            exposicion_categoria=0.0,
            perdida_diaria=7.0,  # > 6 (10% de 60)
            perdida_semanal=0.0,
            drawdown_actual=0.0,
        )
        assert aprobado is False

    def test_rechazar_ordenes_duplicadas(
        self, rm: RiskManager, signal_basica: TradeSignal
    ) -> None:
        """Rechaza si ya hay órdenes pendientes en el mercado."""
        aprobado, razon = rm.aprobar_trade(
            signal=signal_basica,
            balance_actual=50.0,
            exposicion_total=0.0,
            exposicion_mercado=0.0,
            exposicion_categoria=0.0,
            perdida_diaria=0.0,
            perdida_semanal=0.0,
            drawdown_actual=0.0,
            ordenes_pendientes_mercado=1,
        )
        assert aprobado is False
        assert "pendiente" in razon.lower()

    def test_rechazar_tamaño_cero(self, rm: RiskManager) -> None:
        """Rechaza trades con tamaño 0."""
        signal = TradeSignal(
            market_id="0xtest", market_question="Test?",
            side="YES", action="BUY", entry_price=0.50,
            estimated_probability=0.65, edge=0.15, confidence="high",
            suggested_size_usd=0.0, kelly_fraction=0.0, reasoning="test",
        )
        aprobado, _ = rm.aprobar_trade(
            signal=signal, balance_actual=50.0,
            exposicion_total=0.0, exposicion_mercado=0.0,
            exposicion_categoria=0.0, perdida_diaria=0.0,
            perdida_semanal=0.0, drawdown_actual=0.0,
        )
        assert aprobado is False

    def test_cooldown_tras_perdidas(self, rm: RiskManager) -> None:
        """Activa cooldown después de 3 pérdidas consecutivas."""
        rm.registrar_resultado(ganancia=False)
        rm.registrar_resultado(ganancia=False)
        rm.registrar_resultado(ganancia=False)
        assert rm.perdidas_consecutivas == 3
        assert rm._en_cooldown() is True

    def test_reset_perdidas_con_ganancia(self, rm: RiskManager) -> None:
        """Una ganancia resetea el contador de pérdidas."""
        rm.registrar_resultado(ganancia=False)
        rm.registrar_resultado(ganancia=False)
        rm.registrar_resultado(ganancia=True)
        assert rm.perdidas_consecutivas == 0


# =============================================================================
# Tests de la Estrategia
# =============================================================================

class TestTradingStrategy:
    """Tests de la estrategia de trading."""

    @pytest.fixture
    def strategy(self) -> TradingStrategy:
        return TradingStrategy()

    def _hacer_contexto(
        self, yes_price: float = 0.50
    ) -> MarketContext:
        mercado = Market(
            condition_id="0xtest",
            question="Will X happen?",
            yes_price=yes_price,
            no_price=1 - yes_price,
            tokens=[
                Token(token_id="t1", outcome="Yes", price=yes_price),
                Token(token_id="t2", outcome="No", price=1 - yes_price),
            ],
        )
        return MarketContext(market=mercado, data_quality="good")

    def _hacer_evaluacion(
        self, prob: float = 0.65, conf: str = "high"
    ) -> LLMEvaluation:
        return LLMEvaluation(
            probability=prob, confidence=conf,
            reasoning="test", key_factors=["test"],
            risks_to_thesis=["test"],
        )

    def test_genera_buy_con_edge_positivo(
        self, strategy: TradingStrategy
    ) -> None:
        """Genera BUY cuando hay edge > 5% y confianza suficiente."""
        ctx = self._hacer_contexto(yes_price=0.50)
        ev = self._hacer_evaluacion(prob=0.65, conf="high")  # edge = 15%

        señal = strategy.generar_señal(ctx, ev)

        assert señal.action == "BUY"
        assert señal.side == "YES"
        assert señal.edge > 0.05
        assert señal.suggested_size_usd > 0

    def test_genera_hold_sin_edge(
        self, strategy: TradingStrategy
    ) -> None:
        """Genera HOLD cuando no hay edge suficiente."""
        ctx = self._hacer_contexto(yes_price=0.50)
        ev = self._hacer_evaluacion(prob=0.52, conf="high")  # edge = 2%

        señal = strategy.generar_señal(ctx, ev)
        assert señal.action == "HOLD"

    def test_genera_hold_confianza_baja(
        self, strategy: TradingStrategy
    ) -> None:
        """Genera HOLD si la confianza es baja."""
        ctx = self._hacer_contexto(yes_price=0.50)
        ev = self._hacer_evaluacion(prob=0.70, conf="low")

        señal = strategy.generar_señal(ctx, ev)
        assert señal.action == "HOLD"

    def test_kelly_conservador(
        self, strategy: TradingStrategy
    ) -> None:
        """El sizing de Kelly no excede el máximo por trade."""
        ctx = self._hacer_contexto(yes_price=0.40)
        ev = self._hacer_evaluacion(prob=0.80, conf="high")  # Edge enorme

        señal = strategy.generar_señal(ctx, ev)

        # Max per trade = 10% de 60 = $6
        assert señal.suggested_size_usd <= 6.0

    def test_prefiere_mejor_lado(
        self, strategy: TradingStrategy
    ) -> None:
        """Elige el lado con mayor edge."""
        ctx = self._hacer_contexto(yes_price=0.30)
        ev = self._hacer_evaluacion(prob=0.20, conf="high")
        # Edge YES = 0.20 - 0.30 = -0.10
        # Edge NO = 0.80 - 0.70 = 0.10

        señal = strategy.generar_señal(ctx, ev)
        assert señal.side == "NO"

    def test_evaluar_posicion_sell_sin_edge(
        self, strategy: TradingStrategy
    ) -> None:
        """Recomienda SELL si el edge desapareció."""
        ev = self._hacer_evaluacion(prob=0.50, conf="medium")
        resultado = strategy.evaluar_posicion_existente(
            evaluacion_actual=ev,
            precio_entrada=0.40,
            precio_actual=0.50,
            side="YES",
        )
        assert resultado == "SELL"

    def test_evaluar_posicion_hold_con_edge(
        self, strategy: TradingStrategy
    ) -> None:
        """Recomienda HOLD si aún hay edge."""
        ev = self._hacer_evaluacion(prob=0.70, conf="high")
        resultado = strategy.evaluar_posicion_existente(
            evaluacion_actual=ev,
            precio_entrada=0.50,
            precio_actual=0.55,
            side="YES",
        )
        assert resultado in ("HOLD", "ADD")
