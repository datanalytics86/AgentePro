"""
Generador de reportes de performance.

Crea reportes diarios y semanales con métricas del agente.

Fase 7 del agente autónomo.
"""

import logging
from datetime import datetime

from core.portfolio import Portfolio, PerformanceMetrics

logger = logging.getLogger(__name__)


class Reporter:
    """Genera reportes de performance del agente."""

    def __init__(self, portfolio: Portfolio) -> None:
        self._portfolio = portfolio

    def generar_reporte_diario(self) -> str:
        """Genera un reporte diario en formato texto."""
        metricas = self._portfolio.calcular_metricas()
        trades = self._portfolio.obtener_trades_recientes(limit=50)
        posiciones = self._portfolio.obtener_posiciones_abiertas()

        # Filtrar trades de hoy
        hoy = datetime.now().strftime("%Y-%m-%d")
        trades_hoy = [
            t for t in trades
            if t.get("created_at", "").startswith(hoy)
        ]

        lineas = [
            "=" * 60,
            f"  REPORTE DIARIO - {hoy}",
            "=" * 60,
            "",
            "RESUMEN",
            f"  Total trades (histórico): {metricas.total_trades}",
            f"  Trades hoy: {len(trades_hoy)}",
            f"  Win rate: {metricas.win_rate:.1%}",
            f"  PnL realizado: ${metricas.realized_pnl:+.2f}",
            f"  PnL no realizado: ${metricas.unrealized_pnl:+.2f}",
            f"  Drawdown actual: {metricas.current_drawdown:.1%}",
            "",
            f"POSICIONES ABIERTAS ({len(posiciones)})",
        ]

        for pos in posiciones:
            lineas.append(
                f"  - {pos.market_question[:50]} | {pos.side} "
                f"@ {pos.entry_price:.3f} | ${pos.cost_basis:.2f}"
            )

        if not posiciones:
            lineas.append("  (ninguna)")

        lineas.extend([
            "",
            f"TRADES DE HOY ({len(trades_hoy)})",
        ])

        for t in trades_hoy[:10]:
            pnl_str = f"PnL: ${t.get('pnl', 0):+.2f}" if t.get("pnl") else ""
            lineas.append(
                f"  - {t.get('action')} {t.get('side')} "
                f"${t.get('size_usd', 0):.2f} @ {t.get('price', 0):.3f} "
                f"{pnl_str}"
            )

        if not trades_hoy:
            lineas.append("  (ninguno)")

        lineas.extend(["", "=" * 60])

        return "\n".join(lineas)

    def generar_reporte_semanal(self) -> str:
        """Genera un reporte semanal con métricas agregadas."""
        metricas = self._portfolio.calcular_metricas()
        trades = self._portfolio.obtener_trades_recientes(limit=200)

        lineas = [
            "=" * 60,
            f"  REPORTE SEMANAL - {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            "PERFORMANCE",
            f"  Total trades: {metricas.total_trades}",
            f"  Ganadores: {metricas.winning_trades}",
            f"  Perdedores: {metricas.losing_trades}",
            f"  Win rate: {metricas.win_rate:.1%}",
            "",
            "PnL",
            f"  Realizado: ${metricas.realized_pnl:+.2f}",
            f"  No realizado: ${metricas.unrealized_pnl:+.2f}",
            f"  Total: ${metricas.total_pnl:+.2f}",
            "",
            "RIESGO",
            f"  Max drawdown: {metricas.max_drawdown:.1%}",
            f"  Drawdown actual: {metricas.current_drawdown:.1%}",
            "",
            "=" * 60,
        ]

        return "\n".join(lineas)
