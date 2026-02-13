"""
Dashboard web de monitoreo con Streamlit.

Muestra en tiempo real:
- Estado del portafolio
- Señales recientes
- Métricas de performance
- Gráfico de equity curve

Ejecutar con: streamlit run monitoring/dashboard.py

Fase 7 del agente autónomo.
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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

    st.title("📊 Polymarket Agent - Dashboard")
    st.caption(f"Modo: **{settings.agent.mode.upper()}** | Bankroll: ${settings.risk.max_bankroll_usd:,.2f}")

    # Inicializar portafolio
    portfolio = Portfolio()
    reporter = Reporter(portfolio)
    metricas = portfolio.calcular_metricas()

    # =========================================================================
    # Métricas principales
    # =========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Trades", metricas.total_trades)
    with col2:
        st.metric("Win Rate", f"{metricas.win_rate:.0%}")
    with col3:
        st.metric(
            "PnL Total",
            f"${metricas.total_pnl:+.2f}",
            delta=f"${metricas.realized_pnl:+.2f} realizado",
        )
    with col4:
        st.metric("Drawdown", f"{metricas.current_drawdown:.1%}")
    with col5:
        posiciones = portfolio.obtener_posiciones_abiertas()
        st.metric("Posiciones", len(posiciones))

    st.divider()

    # =========================================================================
    # Posiciones abiertas
    # =========================================================================
    st.subheader("📦 Posiciones Abiertas")

    if posiciones:
        data = []
        for p in posiciones:
            data.append({
                "Mercado": p.market_question[:60],
                "Side": p.side,
                "Entrada": f"${p.entry_price:.3f}",
                "Shares": f"{p.shares:.2f}",
                "Costo": f"${p.cost_basis:.2f}",
            })
        st.table(data)
    else:
        st.info("Sin posiciones abiertas")

    # =========================================================================
    # Trades recientes
    # =========================================================================
    st.subheader("📋 Trades Recientes")

    trades = portfolio.obtener_trades_recientes(limit=15)
    if trades:
        data = []
        for t in trades:
            pnl_val = t.get("pnl", 0)
            data.append({
                "ID": t.get("trade_id", "")[:12],
                "Mercado": (t.get("market_question") or "")[:50],
                "Acción": f"{t.get('action', '')} {t.get('side', '')}",
                "Precio": f"${t.get('price', 0):.3f}",
                "Tamaño": f"${t.get('size_usd', 0):.2f}",
                "PnL": f"${pnl_val:+.2f}" if pnl_val else "-",
                "Estado": t.get("status", ""),
                "Modo": t.get("mode", ""),
            })
        st.table(data)
    else:
        st.info("Sin trades registrados")

    # =========================================================================
    # Equity curve
    # =========================================================================
    st.subheader("📈 Historial de Balance")

    historial = portfolio.obtener_historial_balance(limit=200)
    if historial:
        import plotly.graph_objects as go

        fig = go.Figure()
        timestamps = [h["timestamp"] for h in reversed(historial)]
        balances = [h["balance"] for h in reversed(historial)]

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=balances,
            mode="lines+markers",
            name="Balance",
            line={"color": "#00cc96"},
        ))
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Balance (USDC)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin datos de balance aún")

    # =========================================================================
    # Reporte
    # =========================================================================
    with st.expander("📄 Reporte Diario"):
        st.code(reporter.generar_reporte_diario())

    # Auto-refresh
    st.caption("Actualización manual: presiona R o recarga la página")


if __name__ == "__main__":
    main()
