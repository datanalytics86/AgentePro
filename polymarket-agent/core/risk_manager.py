"""
Gestión de riesgo del agente de trading.

Implementa reglas INAMOVIBLES que protegen el capital.
Ningún trade puede ejecutarse sin aprobación del RiskManager.

Fase 5 del agente autónomo.
"""

import logging
from datetime import datetime, timedelta

from config import settings
from core.models import TradeSignal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gestor de riesgo con reglas estrictas que NUNCA pueden violarse.

    Verifica todos los límites antes de aprobar cualquier operación:
    - Límites de posición (por trade, mercado, categoría, total)
    - Límites de pérdida (diaria, semanal, drawdown)
    - Cooldown después de pérdidas consecutivas
    """

    def __init__(self) -> None:
        self._config = settings.risk
        self._perdidas_consecutivas: int = 0
        self._ultimo_trade_perdedor: datetime | None = None

    def aprobar_trade(
        self,
        signal: TradeSignal,
        balance_actual: float,
        exposicion_total: float,
        exposicion_mercado: float,
        exposicion_categoria: float,
        perdida_diaria: float,
        perdida_semanal: float,
        drawdown_actual: float,
        ordenes_pendientes_mercado: int = 0,
    ) -> tuple[bool, str]:
        """
        Evalúa si un trade puede ejecutarse según las reglas de riesgo.

        Args:
            signal: Señal de trading a evaluar.
            balance_actual: Balance disponible en USDC.
            exposicion_total: Capital total ya expuesto (USD).
            exposicion_mercado: Capital ya expuesto en este mercado (USD).
            exposicion_categoria: Capital ya expuesto en esta categoría (USD).
            perdida_diaria: Pérdida acumulada hoy (USD, positivo = pérdida).
            perdida_semanal: Pérdida acumulada esta semana (USD).
            drawdown_actual: Drawdown actual desde máximo (fracción, 0-1).
            ordenes_pendientes_mercado: Órdenes ya pendientes en este mercado.

        Returns:
            Tupla (aprobado, razón).
        """
        bankroll = self._config.max_bankroll_usd
        size = signal.suggested_size_usd

        # =================================================================
        # Check 0: Trade tiene tamaño > 0
        # =================================================================
        if size <= 0:
            return False, "Tamaño de trade es 0 o negativo"

        # =================================================================
        # Check 1: Balance suficiente
        # =================================================================
        if size > balance_actual:
            return False, (
                f"Balance insuficiente: necesita ${size:.2f}, "
                f"disponible ${balance_actual:.2f}"
            )

        # =================================================================
        # Check 2: Máximo por trade individual
        # =================================================================
        max_por_trade = bankroll * self._config.max_per_trade_pct
        if size > max_por_trade:
            return False, (
                f"Excede máximo por trade: ${size:.2f} > "
                f"${max_por_trade:.2f} ({self._config.max_per_trade_pct:.0%} bankroll)"
            )

        # =================================================================
        # Check 3: Máximo en un solo mercado
        # =================================================================
        max_por_mercado = bankroll * self._config.max_per_market_pct
        if exposicion_mercado + size > max_por_mercado:
            return False, (
                f"Excede máximo por mercado: "
                f"${exposicion_mercado + size:.2f} > ${max_por_mercado:.2f}"
            )

        # =================================================================
        # Check 4: Máximo en una categoría
        # =================================================================
        max_por_categoria = bankroll * self._config.max_per_category_pct
        if exposicion_categoria + size > max_por_categoria:
            return False, (
                f"Excede máximo por categoría: "
                f"${exposicion_categoria + size:.2f} > ${max_por_categoria:.2f}"
            )

        # =================================================================
        # Check 5: Máximo total expuesto
        # =================================================================
        max_exposicion = bankroll * self._config.max_exposure_pct
        if exposicion_total + size > max_exposicion:
            return False, (
                f"Excede exposición total máxima: "
                f"${exposicion_total + size:.2f} > ${max_exposicion:.2f}"
            )

        # =================================================================
        # Check 6: Reserva de liquidez
        # =================================================================
        reserva_minima = bankroll * (1 - self._config.max_exposure_pct)
        balance_post_trade = balance_actual - size
        if balance_post_trade < reserva_minima:
            return False, (
                f"Viola reserva de liquidez: post-trade ${balance_post_trade:.2f} "
                f"< mínimo ${reserva_minima:.2f}"
            )

        # =================================================================
        # Check 7: Pérdida diaria máxima
        # =================================================================
        max_perdida_diaria = bankroll * self._config.max_daily_loss_pct
        if perdida_diaria >= max_perdida_diaria:
            return False, (
                f"Límite de pérdida diaria alcanzado: "
                f"${perdida_diaria:.2f} >= ${max_perdida_diaria:.2f}"
            )

        # =================================================================
        # Check 8: Pérdida semanal máxima
        # =================================================================
        max_perdida_semanal = bankroll * self._config.max_weekly_loss_pct
        if perdida_semanal >= max_perdida_semanal:
            return False, (
                f"Límite de pérdida semanal alcanzado: "
                f"${perdida_semanal:.2f} >= ${max_perdida_semanal:.2f}"
            )

        # =================================================================
        # Check 9: Drawdown máximo
        # =================================================================
        if drawdown_actual >= self._config.max_drawdown_pct:
            return False, (
                f"Drawdown máximo alcanzado: {drawdown_actual:.1%} >= "
                f"{self._config.max_drawdown_pct:.0%} - PAUSA AUTOMÁTICA"
            )

        # =================================================================
        # Check 10: Cooldown tras pérdidas consecutivas
        # =================================================================
        if self._en_cooldown():
            horas_restantes = self._horas_cooldown_restantes()
            return False, (
                f"En cooldown por pérdidas consecutivas. "
                f"Faltan {horas_restantes:.1f} horas"
            )

        # =================================================================
        # Check 11: Ordenes duplicadas
        # =================================================================
        if ordenes_pendientes_mercado > 0:
            return False, (
                f"Ya hay {ordenes_pendientes_mercado} orden(es) "
                f"pendiente(s) en este mercado"
            )

        # =================================================================
        # Todos los checks pasaron
        # =================================================================
        logger.info(
            f"Trade APROBADO: {signal.action} {signal.side} "
            f"${size:.2f} en '{signal.market_question[:40]}'"
        )
        return True, "Aprobado - todos los límites de riesgo OK"

    def registrar_resultado(self, ganancia: bool) -> None:
        """
        Registra el resultado de un trade para el tracking de pérdidas
        consecutivas y cooldown.

        Args:
            ganancia: True si el trade fue ganador.
        """
        if ganancia:
            self._perdidas_consecutivas = 0
        else:
            self._perdidas_consecutivas += 1
            self._ultimo_trade_perdedor = datetime.now()

            if self._perdidas_consecutivas >= 3:
                logger.warning(
                    f"¡{self._perdidas_consecutivas} pérdidas consecutivas! "
                    f"Activando cooldown de {self._config.cooldown_hours}h"
                )

    def ajustar_tamaño(
        self,
        signal: TradeSignal,
        balance_actual: float,
        exposicion_total: float,
    ) -> float:
        """
        Ajusta el tamaño del trade para que cumpla con los límites.

        Si el tamaño sugerido es demasiado grande, lo reduce.

        Returns:
            Tamaño ajustado en USDC.
        """
        bankroll = self._config.max_bankroll_usd
        size = signal.suggested_size_usd

        # Límite por trade
        max_trade = bankroll * self._config.max_per_trade_pct
        size = min(size, max_trade)

        # Límite de exposición restante
        max_exp = bankroll * self._config.max_exposure_pct
        espacio_restante = max(0, max_exp - exposicion_total)
        size = min(size, espacio_restante)

        # No usar más del balance disponible
        size = min(size, balance_actual * 0.95)  # 5% margen

        # Mínimo viable
        if size < 1.0:
            return 0.0

        return round(size, 2)

    @property
    def perdidas_consecutivas(self) -> int:
        """Cantidad actual de pérdidas consecutivas."""
        return self._perdidas_consecutivas

    def _en_cooldown(self) -> bool:
        """Verifica si estamos en período de cooldown."""
        if self._perdidas_consecutivas < 3:
            return False
        if self._ultimo_trade_perdedor is None:
            return False

        tiempo_desde_ultima = datetime.now() - self._ultimo_trade_perdedor
        return tiempo_desde_ultima < timedelta(
            hours=self._config.cooldown_hours
        )

    def _horas_cooldown_restantes(self) -> float:
        """Calcula las horas restantes de cooldown."""
        if self._ultimo_trade_perdedor is None:
            return 0.0
        transcurrido = datetime.now() - self._ultimo_trade_perdedor
        total = timedelta(hours=self._config.cooldown_hours)
        restante = total - transcurrido
        return max(0, restante.total_seconds() / 3600)
