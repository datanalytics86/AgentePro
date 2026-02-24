"""
Estrategia de decisión de trading.

Convierte evaluaciones de probabilidad del LLM en señales concretas
de trading con sizing basado en el criterio de Kelly.

Fase 4 del agente autónomo.
"""

import logging
from datetime import datetime

from config import settings
from core.models import LLMEvaluation, Market, TradeSignal
from core.data_collector import MarketContext

logger = logging.getLogger(__name__)


class TradingStrategy:
    """
    Genera señales de trading basadas en evaluaciones del LLM.

    Flujo:
    1. Calcula el edge (diferencia entre probabilidad estimada y precio)
    2. Filtra por edge mínimo y confianza mínima
    3. Calcula sizing con criterio de Kelly fraccional
    4. Genera señales claras BUY/SELL/HOLD
    """

    # Mapeo de confianza a nivel numérico para comparaciones
    _CONFIDENCE_LEVELS = {"low": 0, "medium": 1, "high": 2}

    def __init__(self) -> None:
        self._config = settings.strategy
        self._risk_config = settings.risk

    def generar_señal(
        self,
        contexto: MarketContext,
        evaluacion: LLMEvaluation,
    ) -> TradeSignal:
        """
        Genera una señal de trading para un mercado evaluado.

        Args:
            contexto: Contexto completo del mercado.
            evaluacion: Evaluación de probabilidad del LLM.

        Returns:
            TradeSignal con la recomendación de acción.
        """
        market = contexto.market
        prob = evaluacion.probability
        yes_price = market.yes_price
        no_price = market.no_price

        # Calcular edge para YES y NO
        edge_yes = prob - yes_price
        edge_no = (1 - prob) - no_price

        # Determinar el mejor lado
        if abs(edge_yes) >= abs(edge_no):
            side = "YES"
            edge = edge_yes
            entry_price = yes_price
        else:
            side = "NO"
            edge = edge_no
            entry_price = no_price

        # Resolver token_id para órdenes CLOB
        token_id = ""
        for token in market.tokens:
            if token.outcome.lower() == side.lower():
                token_id = token.token_id
                break

        # Determinar acción
        action = self._determinar_accion(edge, evaluacion.confidence)

        # Calcular sizing con Criterio de Kelly (formula correcta)
        kelly_frac = 0.0
        suggested_size = 0.0
        if action == "BUY":
            kelly_frac, suggested_size = self._calcular_kelly_size(
                prob=prob,
                entry_price=entry_price,
            )

        señal = TradeSignal(
            market_id=market.condition_id,
            token_id=token_id,
            market_question=market.question,
            side=side,
            action=action,
            entry_price=entry_price,
            estimated_probability=prob,
            edge=edge,
            confidence=evaluacion.confidence,
            suggested_size_usd=suggested_size,
            kelly_fraction=round(kelly_frac, 6),
            reasoning=evaluacion.reasoning,
            category=market.category,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Señal generada: {action} {side} @ {entry_price:.2f} | "
            f"edge={edge:+.2f} | size=${suggested_size:.2f} | "
            f"conf={evaluacion.confidence}"
        )

        return señal

    def generar_señales_multiples(
        self,
        evaluaciones: list[tuple[MarketContext, LLMEvaluation | None]],
    ) -> list[TradeSignal]:
        """
        Genera señales para múltiples mercados evaluados.

        Filtra evaluaciones nulas y retorna solo señales válidas.
        """
        señales: list[TradeSignal] = []

        for contexto, evaluacion in evaluaciones:
            if evaluacion is None:
                continue
            señal = self.generar_señal(contexto, evaluacion)
            señales.append(señal)

        # Ordenar por edge descendente (mejores oportunidades primero)
        señales.sort(key=lambda s: abs(s.edge), reverse=True)

        activas = sum(1 for s in señales if s.action == "BUY")
        logger.info(
            f"Señales generadas: {len(señales)} total, {activas} BUY"
        )

        return señales

    def evaluar_posicion_existente(
        self,
        evaluacion_actual: LLMEvaluation,
        precio_entrada: float,
        precio_actual: float,
        side: str,
    ) -> str:
        """
        Evalúa si mantener o cerrar una posición existente.

        Args:
            evaluacion_actual: Evaluación actualizada del LLM.
            precio_entrada: Precio al que se entró.
            precio_actual: Precio actual del mercado.
            side: Lado de la posición ("YES" o "NO").

        Returns:
            "HOLD", "SELL" (cerrar posición) o "ADD" (aumentar).
        """
        prob = evaluacion_actual.probability

        # Calcular edge actual
        if side == "YES":
            edge_actual = prob - precio_actual
        else:
            edge_actual = (1 - prob) - precio_actual

        # Calcular PnL no realizado
        if side == "YES":
            pnl_pct = (precio_actual - precio_entrada) / precio_entrada
        else:
            pnl_pct = (precio_entrada - precio_actual) / precio_entrada

        # Reglas de salida
        # 1. Vender si el edge desapareció
        if abs(edge_actual) < self._config.min_edge / 2:
            logger.info(f"Edge desapareció ({edge_actual:+.3f}) - SELL")
            return "SELL"

        # 2. Vender si el edge se invirtió
        if edge_actual < -self._config.min_edge:
            logger.info(f"Edge invertido ({edge_actual:+.3f}) - SELL")
            return "SELL"

        # 3. Stop loss
        if pnl_pct < -self._risk_config.stop_loss_pct:
            logger.info(
                f"Stop loss activado (PnL: {pnl_pct:.1%}) - SELL"
            )
            return "SELL"

        # 4. Si el edge mejoró mucho, considerar agregar
        if (
            edge_actual > self._config.min_edge * 2
            and evaluacion_actual.confidence == "high"
        ):
            return "ADD"

        return "HOLD"

    # =========================================================================
    # Métodos privados
    # =========================================================================

    def _determinar_accion(self, edge: float, confidence: str) -> str:
        """
        Determina la acción basada en edge y confianza.

        Solo genera BUY si:
        - Edge supera el mínimo configurado
        - Confianza es al menos el nivel mínimo configurado
        """
        # Verificar confianza mínima
        nivel_actual = self._CONFIDENCE_LEVELS.get(confidence, 0)
        nivel_minimo = self._CONFIDENCE_LEVELS.get(
            self._config.min_confidence, 1
        )

        if nivel_actual < nivel_minimo:
            return "HOLD"

        # Verificar edge mínimo
        if edge > self._config.min_edge:
            return "BUY"

        return "HOLD"

    def _calcular_kelly_size(
        self,
        prob: float,
        entry_price: float,
    ) -> tuple[float, float]:
        """
        Calcula fracción de Kelly y tamaño en USD usando la fórmula correcta.

        Fórmula: f* = (p(b+1) - 1) / b, donde b = (1/price) - 1.
        Aplica Fractional Kelly (``kelly_fraction`` de config, default 0.25)
        y un cap duro de ``max_per_trade_pct`` del bankroll (default 5%).

        Args:
            prob:        Probabilidad estimada por el LLM (0-1).
            entry_price: Precio de entrada del token (0-1).

        Returns:
            Tupla ``(kelly_fraction, size_usd)``.
        """
        from core.probability_engine import ProbabilityEngine

        bankroll = self._risk_config.max_bankroll_usd
        kelly_frac, size_usd = ProbabilityEngine.calcular_kelly_size(
            p=prob,
            yes_price=entry_price,
            bankroll=bankroll,
            kelly_fraction=self._config.kelly_fraction,
            max_pct=self._risk_config.max_per_trade_pct,
        )

        # Mínimo viable
        if size_usd < 1.0:
            return 0.0, 0.0

        return kelly_frac, size_usd


# =============================================================================
# Ejecución directa para pruebas
# =============================================================================

def main() -> None:
    """Prueba la estrategia con datos de ejemplo."""
    from config import configurar_logging
    from core.models import Token
    from rich.console import Console
    from rich.table import Table

    configurar_logging()
    console = Console()

    console.print("\n[bold cyan]Trading Strategy - Fase 4[/bold cyan]\n")

    strategy = TradingStrategy()

    # Mercados de prueba con diferentes escenarios
    escenarios = [
        ("Buen edge YES", 0.65, 0.50, "high"),      # Edge +15%
        ("Buen edge NO", 0.30, 0.50, "medium"),      # Edge NO +20%
        ("Edge bajo", 0.53, 0.50, "medium"),          # Edge +3% < 5%
        ("Alta prob baja conf", 0.80, 0.50, "low"),   # Bueno pero baja conf
        ("Sin edge", 0.50, 0.50, "high"),             # Sin edge
    ]

    tabla = Table(title="Señales de Trading", show_lines=True)
    tabla.add_column("Escenario", width=20)
    tabla.add_column("Acción", justify="center", width=8)
    tabla.add_column("Side", justify="center", width=6)
    tabla.add_column("Edge", justify="right", width=8)
    tabla.add_column("Kelly", justify="right", width=8)
    tabla.add_column("Size USD", justify="right", width=10)

    for nombre, prob_est, yes_price, conf in escenarios:
        mercado = Market(
            condition_id="0xtest",
            question=nombre,
            yes_price=yes_price,
            no_price=1 - yes_price,
            tokens=[
                Token(token_id="t1", outcome="Yes", price=yes_price),
                Token(token_id="t2", outcome="No", price=1 - yes_price),
            ],
        )
        ctx = MarketContext(market=mercado, data_quality="partial")
        ev = LLMEvaluation(
            probability=prob_est,
            confidence=conf,
            reasoning="Test",
            key_factors=["test"],
            risks_to_thesis=["test"],
        )

        señal = strategy.generar_señal(ctx, ev)

        style = "green" if señal.action == "BUY" else "dim"
        tabla.add_row(
            nombre,
            f"[{style}]{señal.action}[/{style}]",
            señal.side,
            f"{señal.edge:+.2f}",
            f"{señal.kelly_fraction:.4f}",
            f"${señal.suggested_size_usd:.2f}",
        )

    console.print(tabla)


if __name__ == "__main__":
    main()
