"""
Estrategia de Copy-Trading: replica las posiciones de los mejores traders.

Reemplaza la estrategia anterior de edge detection con LLM.
En lugar de estimar probabilidades propias, copia lo que los traders
más exitosos del leaderboard de Polymarket están haciendo.

Flujo:
1. Consulta el leaderboard → top N traders por PnL
2. Obtiene posiciones abiertas de cada trader
3. Identifica mercados con consenso (múltiples top traders posicionados)
4. Genera señales de compra para los mercados con más consenso
5. Sizing proporcional al bankroll (no Kelly, sino % fijo del promedio)
"""

import logging
from datetime import datetime
from typing import Any

from config import settings
from core.models import Market, TradeSignal
from core.leaderboard import (
    LeaderboardFetcher,
    TraderSnapshot,
    TraderPosition,
)

logger = logging.getLogger(__name__)


class CopyTradingStrategy:
    """
    Genera señales de trading copiando las posiciones de los mejores traders.

    Criterios para generar BUY:
    - Al menos N top traders tienen posición en el mismo mercado y lado
    - El precio actual permite un entry razonable (no demasiado caro)
    - No tenemos ya una posición en ese mercado
    - El mercado cumple filtros mínimos de liquidez/volumen

    Sizing:
    - Porcentaje fijo del bankroll por trade (configurable, default 3%)
    - Ajustado por el número de traders en consenso (más traders = más confianza)
    """

    def __init__(self, leaderboard: LeaderboardFetcher) -> None:
        self._leaderboard = leaderboard
        self._config = settings.copy_trading
        self._risk_config = settings.risk

    async def generar_señales_copy(
        self,
        mercados_disponibles: list[Market],
        posiciones_propias: list[str],
    ) -> list[TradeSignal]:
        """
        Genera señales de copy-trading basadas en el leaderboard.

        Args:
            mercados_disponibles: Mercados activos del scanner (para validar).
            posiciones_propias: Lista de market_ids donde ya tenemos posición.

        Returns:
            Lista de TradeSignal con las recomendaciones de copy.
        """
        logger.info("Generando señales de copy-trading...")

        # Paso 1: Obtener snapshots de top traders
        snapshots = await self._leaderboard.obtener_snapshots_completos(
            top_n=self._config.top_n,
            window=self._config.leaderboard_window,
        )

        if not snapshots:
            logger.warning("No se obtuvieron snapshots. Sin señales.")
            return []

        # Paso 2: Identificar mercados con consenso
        consensos = self._leaderboard.identificar_mercados_consenso(
            min_traders=self._config.min_traders_consensus,
        )

        if not consensos:
            logger.info("Sin mercados con consenso suficiente.")
            return []

        logger.info(f"Mercados con consenso: {len(consensos)}")

        # Paso 3: Filtrar y generar señales
        # Crear índice de mercados disponibles para validación
        mercados_index = {m.condition_id: m for m in mercados_disponibles}

        señales: list[TradeSignal] = []

        for consenso in consensos:
            market_id = consenso["market_id"]

            # Skip si ya tenemos posición
            if market_id in posiciones_propias:
                logger.debug(f"Skip {market_id[:12]}: ya tenemos posición")
                continue

            # Validar que el mercado esté en nuestro universo de mercados
            mercado = mercados_index.get(market_id)
            if mercado is None:
                logger.debug(
                    f"Skip {consenso['title'][:40]}: no está en mercados escaneados"
                )
                continue

            # Validar precio de entrada
            side = consenso["side"]
            if not side:
                # Si no pudimos determinar el side, usar YES por default
                side = "YES"

            # Obtener precio real del token (no el default 0.5)
            # Para mercados binarios YES/NO, yes_price/no_price son correctos.
            # Para mercados multi-outcome, los tokens no se llaman "Yes"/"No",
            # así que yes_price queda en 0.5 (default) — usamos el precio
            # del token que coincide con el side, o el consenso avg_price.
            entry_price = (
                mercado.yes_price if side == "YES" else mercado.no_price
            )

            # Si el precio es el default (0.5), verificar con los tokens reales
            # y con el precio promedio del consenso para detectar mercados
            # multi-outcome donde yes_price/no_price no son confiables.
            if abs(entry_price - 0.5) < 0.001:
                # Revisar si realmente hay un token YES/NO
                has_binary_tokens = any(
                    t.outcome.upper() in ("YES", "NO")
                    for t in mercado.tokens
                )
                if not has_binary_tokens:
                    # Mercado multi-outcome: usar precio del consenso
                    consensus_price = consenso.get("avg_price", 0)
                    if consensus_price > 0:
                        entry_price = consensus_price
                        logger.info(
                            f"Multi-outcome {consenso['title'][:35]}: "
                            f"usando consensus price {entry_price:.3f}"
                        )

            # No comprar tokens demasiado caros (>95%) ni demasiado baratos (<5%)
            # Hard floor: nunca comprar a menos de 3% independiente de config
            effective_min = max(self._config.min_entry_price, 0.03)
            if entry_price > self._config.max_entry_price:
                logger.info(
                    f"Skip {consenso['title'][:40]}: precio {entry_price:.3f} "
                    f"> max {self._config.max_entry_price}"
                )
                continue
            if entry_price < effective_min:
                logger.info(
                    f"Skip {consenso['title'][:40]}: precio {entry_price:.3f} "
                    f"< min {effective_min}"
                )
                continue

            # Calcular sizing
            size_usd = self._calcular_size(
                traders_count=consenso["traders_count"],
                entry_price=entry_price,
            )

            if size_usd < 1.0:
                continue

            # Resolver token_id
            token_id = ""
            for token in mercado.tokens:
                if token.outcome.upper() == side:
                    token_id = token.token_id
                    break

            # Crear señal
            señal = TradeSignal(
                market_id=market_id,
                token_id=token_id,
                market_question=mercado.question,
                side=side,
                action="BUY",
                entry_price=entry_price,
                estimated_probability=entry_price,  # Usamos precio como proxy
                edge=0.0,  # No calculamos edge propio en copy-trading
                confidence="high" if consenso["traders_count"] >= 5 else "medium",
                suggested_size_usd=size_usd,
                kelly_fraction=0.0,  # No usamos Kelly en copy-trading
                reasoning=self._generar_razonamiento(consenso),
                category=mercado.category,
                timestamp=datetime.now(),
            )

            señales.append(señal)

            logger.info(
                f"Señal COPY: BUY {side} {mercado.question[:40]} | "
                f"${size_usd:.2f} @ {entry_price:.2f} | "
                f"{consenso['traders_count']} traders"
            )

        # Ordenar por número de traders (más consenso primero)
        señales.sort(
            key=lambda s: int(s.reasoning.split("traders")[0].split()[-1])
            if "traders" in s.reasoning else 0,
            reverse=True,
        )

        # Limitar señales por ciclo
        max_signals = self._config.max_new_positions_per_cycle
        if len(señales) > max_signals:
            señales = señales[:max_signals]

        logger.info(
            f"Señales de copy-trading generadas: {len(señales)} "
            f"(de {len(consensos)} mercados con consenso)"
        )

        return señales

    def evaluar_posicion_copy(
        self,
        market_id: str,
        side: str,
        snapshots: list[TraderSnapshot] | None = None,
    ) -> str:
        """
        Evalúa si mantener o cerrar una posición existente basándose
        en si los top traders aún mantienen su posición.

        Args:
            market_id: ID del mercado.
            side: Lado de nuestra posición.
            snapshots: Snapshots actualizados (usa cached si None).

        Returns:
            "HOLD" si los traders aún tienen posición, "SELL" si salieron.
        """
        snaps = snapshots or self._leaderboard.cached_snapshots

        # Contar cuántos top traders aún mantienen posición
        traders_con_posicion = 0
        for snap in snaps:
            for pos in snap.positions:
                if pos.market_id == market_id and pos.side == side:
                    traders_con_posicion += 1
                    break

        min_hold = max(1, self._config.min_traders_consensus - 1)

        if traders_con_posicion < min_hold:
            logger.info(
                f"Copy EXIT: {market_id[:12]} {side} | "
                f"Solo {traders_con_posicion} traders mantienen "
                f"(min: {min_hold})"
            )
            return "SELL"

        logger.debug(
            f"Copy HOLD: {market_id[:12]} {side} | "
            f"{traders_con_posicion} traders mantienen"
        )
        return "HOLD"

    # =========================================================================
    # Métodos privados
    # =========================================================================

    def _calcular_size(
        self,
        traders_count: int,
        entry_price: float,
    ) -> float:
        """
        Calcula el tamaño del trade basado en consenso y bankroll.

        Sizing fijo como % del bankroll, ajustado por nivel de consenso:
        - Base: copy_size_pct del bankroll (default 3%)
        - Boost: +1% por cada trader adicional sobre el mínimo
        - Cap: max_per_trade_pct del bankroll (default 5%)
        """
        bankroll = self._risk_config.max_bankroll_usd
        base_pct = self._config.copy_size_pct
        min_consensus = self._config.min_traders_consensus

        # Boost por consenso adicional
        extra_traders = max(0, traders_count - min_consensus)
        boost_pct = extra_traders * 0.01  # +1% por trader extra

        total_pct = min(
            base_pct + boost_pct,
            self._risk_config.max_per_trade_pct,
        )

        size = bankroll * total_pct

        # Mínimo viable
        if size < 1.0:
            return 0.0

        return round(size, 2)

    @staticmethod
    def _generar_razonamiento(consenso: dict[str, Any]) -> str:
        """Genera un razonamiento legible para la señal de copy-trading."""
        traders = consenso.get("trader_names", [])
        count = consenso["traders_count"]
        avg_pnl = consenso.get("avg_pnl_pct", 0)
        total_usd = consenso.get("total_size_usd", 0)

        names_str = ", ".join(traders[:5])
        if len(traders) > 5:
            names_str += f" y {len(traders) - 5} más"

        return (
            f"Copy-trading: {count} traders del top leaderboard "
            f"tienen posición {consenso['side']}. "
            f"Traders: {names_str}. "
            f"Exposición combinada: ${total_usd:,.0f}. "
            f"PnL promedio posición: {avg_pnl:+.1f}%."
        )
