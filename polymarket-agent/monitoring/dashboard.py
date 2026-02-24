"""
Dashboard web de monitoreo con Streamlit.

Muestra en tiempo real:
- PnL Realizado: ganancias/pérdidas de trades ya cerrados.
- PnL Flotante: valor de mercado actual de posiciones abiertas vs costo.
- Max Drawdown: mayor caída histórica desde el pico de capital.
- Equity curve, posiciones abiertas y trades recientes.

Ejecutar con: streamlit run monitoring/dashboard.py

Fase 7 del agente autónomo — actualizado con métricas P&L completas.
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _calcular_pnl_flotante(posiciones: list) -> float:
    """
    Calcula el PnL flotante total de las posiciones abiertas.

    En ausencia de precios de mercado en tiempo real, usa el
    ``unrealized_pnl`` almacenado en cada posición (actualizado
    cuando el agente revisa posiciones con precios frescos del CLOB).

    Returns:
        Suma de PnL flotante en USD.
    """
    return sum(p.unrealized_pnl for p in posiciones)


def main() -> None:
    """Punto de entrada del dashboard Streamlit."""
    import streamlit as st
    from core.portfolio import Portfolio
    from monitoring.reporter import Reporter
    from config import settings

    st.set_page_config(
        page_title="Polymarket Agent",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Polymarket Agent — Dashboard")
    st.caption(
        f"Modo: **{settings.agent.mode.upper()}** | "
        f"Bankroll inicial: **${settings.risk.max_bankroll_usd:,.2f} USD**"
    )

    # Inicializar portafolio
    portfolio = Portfolio()
    reporter = Reporter(portfolio)
    metricas = portfolio.calcular_metricas()
    posiciones = portfolio.obtener_posiciones_abiertas()

    # =========================================================================
    # Sección 1: Métricas P&L principales
    # =========================================================================
    st.subheader("💰 Resumen de P&L")

    pnl_flotante = _calcular_pnl_flotante(posiciones)
    max_dd_hist = portfolio.calcular_max_drawdown_historico()
    bankroll = settings.risk.max_bankroll_usd

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if metricas.realized_pnl >= 0 else "inverse"
        st.metric(
            label="PnL Realizado",
            value=f"${metricas.realized_pnl:+.2f}",
            delta=f"{metricas.realized_pnl / bankroll:.1%} del bankroll",
            delta_color=delta_color,
            help="Ganancias/pérdidas de trades ya cerrados (con resolución).",
        )

    with col2:
        delta_color_f = "normal" if pnl_flotante >= 0 else "inverse"
        flotante_label = (
            f"${pnl_flotante:+.2f}"
            if pnl_flotante != 0.0
            else "—"
        )
        st.metric(
            label="PnL Flotante",
            value=flotante_label,
            delta=(
                f"{pnl_flotante / bankroll:.1%} del bankroll"
                if pnl_flotante != 0.0
                else "Actualiza cuando se re-evalúan posiciones"
            ),
            delta_color=delta_color_f,
            help=(
                "Valor actual de posiciones abiertas vs costo de entrada. "
                "Se actualiza cada vez que el agente revisa posiciones con "
                "precios frescos del mercado."
            ),
        )

    with col3:
        st.metric(
            label="Max Drawdown Histórico",
            value=f"{max_dd_hist:.1%}",
            delta=f"Actual: {metricas.current_drawdown:.1%}",
            delta_color="inverse" if max_dd_hist > 0.05 else "off",
            help=(
                "Mayor caída porcentual desde un pico de capital en toda "
                "la historia del agente. Calculado desde el historial de "
                f"balance (bankroll base ${bankroll:,.0f} USD)."
            ),
        )

    with col4:
        pnl_total = metricas.realized_pnl + pnl_flotante
        balance_actual = bankroll + metricas.realized_pnl
        st.metric(
            label="Balance Estimado",
            value=f"${balance_actual:,.2f}",
            delta=f"P&L total: ${pnl_total:+.2f}",
            delta_color="normal" if pnl_total >= 0 else "inverse",
            help="Bankroll inicial + PnL realizado acumulado.",
        )

    st.divider()

    # =========================================================================
    # Sección 2: KPIs de trading
    # =========================================================================
    st.subheader("📈 Métricas de Trading")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Trades", metricas.total_trades)
    with k2:
        st.metric("Ganadores", metricas.winning_trades)
    with k3:
        st.metric("Win Rate", f"{metricas.win_rate:.0%}")
    with k4:
        st.metric("Posiciones Abiertas", len(posiciones))
    with k5:
        exposicion = sum(p.cost_basis for p in posiciones)
        st.metric(
            "Exposición Total",
            f"${exposicion:.2f}",
            delta=f"{exposicion / bankroll:.1%} del bankroll",
            delta_color="off",
        )

    st.divider()

    # =========================================================================
    # Sección 3: Posiciones abiertas con P&L flotante por posición
    # =========================================================================
    st.subheader("📦 Posiciones Abiertas")

    if posiciones:
        data = []
        for p in posiciones:
            pnl_pos = p.unrealized_pnl
            pnl_pct = (pnl_pos / p.cost_basis * 100) if p.cost_basis > 0 else 0.0
            data.append({
                "Mercado": p.market_question[:65],
                "Side": p.side,
                "Entrada": f"${p.entry_price:.3f}",
                "Precio Actual": f"${p.current_price:.3f}",
                "Shares": f"{p.shares:.2f}",
                "Costo": f"${p.cost_basis:.2f}",
                "PnL Flotante": (
                    f"${pnl_pos:+.2f} ({pnl_pct:+.1f}%)"
                    if pnl_pos != 0.0
                    else "—"
                ),
            })
        st.table(data)
    else:
        st.info("Sin posiciones abiertas")

    # =========================================================================
    # Sección 4: Trades recientes
    # =========================================================================
    st.subheader("📋 Trades Recientes")

    trades = portfolio.obtener_trades_recientes(limit=15)
    if trades:
        data_t = []
        for t in trades:
            pnl_val = t.get("pnl", 0) or 0
            data_t.append({
                "ID": t.get("trade_id", "")[:12],
                "Mercado": (t.get("market_question") or "")[:50],
                "Acción": f"{t.get('action', '')} {t.get('side', '')}",
                "Precio": f"${t.get('price', 0):.3f}",
                "Tamaño": f"${t.get('size_usd', 0):.2f}",
                "PnL": f"${pnl_val:+.2f}" if pnl_val else "—",
                "Estado": t.get("status", ""),
                "Modo": t.get("mode", ""),
            })
        st.table(data_t)
    else:
        st.info("Sin trades registrados")

    # =========================================================================
    # Sección 5: Equity curve y Drawdown chart
    # =========================================================================
    st.subheader("📉 Historial de Balance y Drawdown")

    historial = portfolio.obtener_historial_balance(limit=500)
    if historial:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Datos en orden cronológico
        historial_asc = list(reversed(historial))
        timestamps = [h["timestamp"] for h in historial_asc]
        balances = [h["balance"] for h in historial_asc]

        # Calcular drawdown por punto
        peak_running = balances[0] if balances else bankroll
        drawdowns = []
        for b in balances:
            if b > peak_running:
                peak_running = b
            dd = (peak_running - b) / peak_running if peak_running > 0 else 0
            drawdowns.append(dd * 100)  # en %

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            subplot_titles=("Balance (USDC)", "Drawdown (%)"),
            vertical_spacing=0.08,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=balances,
                mode="lines",
                name="Balance",
                line={"color": "#00cc96", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(0,204,150,0.07)",
            ),
            row=1,
            col=1,
        )

        # Línea del bankroll inicial
        fig.add_hline(
            y=bankroll,
            line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text=f"Bankroll ${bankroll:,.0f}",
            annotation_position="top left",
            row=1,
            col=1,
        )

        # Drawdown chart
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=drawdowns,
                mode="lines",
                name="Drawdown %",
                line={"color": "#ef553b", "width": 1.5},
                fill="tozeroy",
                fillcolor="rgba(239,85,59,0.15)",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=500,
            showlegend=False,
            margin={"t": 40, "b": 20},
        )
        fig.update_yaxes(title_text="USDC", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin datos de balance aún. El agente debe registrar al menos un ciclo.")

    # =========================================================================
    # Sección 6: Reporte diario expandible
    # =========================================================================
    with st.expander("📄 Reporte Diario Detallado"):
        st.code(reporter.generar_reporte_diario())

    st.caption(
        "Recarga manual: presiona **R** o actualiza la página. "
        "Los datos se leen directamente de trades.db (WAL mode)."
    )


if __name__ == "__main__":
    main()
