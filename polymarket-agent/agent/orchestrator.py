"""
Orquestador principal del agente autónomo.

Integra todos los componentes y ejecuta el loop principal
que corre de forma autónoma 24/7.

Fase 8 del agente autónomo.
"""

import logging
import signal
import sys
import time
import traceback
from datetime import datetime

from config import settings, configurar_logging
from core.market_scanner import MarketScanner
from core.data_collector import DataCollector
from core.probability_engine import ProbabilityEngine
from core.strategy import TradingStrategy
from core.risk_manager import RiskManager
from core.portfolio import Portfolio
from core.executor import OrderExecutor
from monitoring.alerts import TelegramAlerts
from monitoring.reporter import Reporter

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orquestador del agente autónomo de trading en Polymarket.

    Loop principal:
    1. Escanear mercados activos
    2. Recopilar datos (noticias, historial, sentimiento)
    3. Evaluar probabilidades con LLM
    4. Generar señales de trading
    5. Validar con risk manager
    6. Ejecutar trades aprobados
    7. Revisar posiciones existentes
    8. Reportar actividad
    9. Dormir hasta próximo ciclo
    """

    def __init__(self) -> None:
        configurar_logging()
        logger.info("Inicializando agente de Polymarket...")

        # Componentes
        self._scanner = MarketScanner()
        self._collector = DataCollector()
        self._engine = ProbabilityEngine()
        self._strategy = TradingStrategy()
        self._risk_manager = RiskManager()
        self._portfolio = Portfolio()
        self._executor = OrderExecutor(self._portfolio, self._risk_manager)
        self._alerts = TelegramAlerts()
        self._reporter = Reporter(self._portfolio)

        # Estado
        self._running = False
        self._ciclo_actual = 0
        self._ultimo_reporte = datetime.now()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._manejar_shutdown)
        signal.signal(signal.SIGTERM, self._manejar_shutdown)

        logger.info(
            f"Agente inicializado en modo {settings.agent.mode.upper()} | "
            f"Bankroll: ${settings.risk.max_bankroll_usd:.2f}"
        )

    def ejecutar(self) -> None:
        """
        Loop principal del agente. Corre hasta recibir señal de parada.
        """
        self._running = True
        intervalo = settings.agent.scan_interval_minutes * 60

        self._alerts.agente_iniciado(settings.agent.mode)
        logger.info(
            f"Agente iniciado. Ciclo cada {settings.agent.scan_interval_minutes} min"
        )

        while self._running:
            self._ciclo_actual += 1
            inicio = time.time()

            try:
                logger.info(f"{'=' * 60}")
                logger.info(f"CICLO #{self._ciclo_actual} - {datetime.now()}")
                logger.info(f"{'=' * 60}")

                self._ejecutar_ciclo()

                # Verificar si toca reporte diario
                self._verificar_reporte_diario()

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error en ciclo #{self._ciclo_actual}: {e}")
                logger.error(traceback.format_exc())
                self._alerts.error_critico(
                    f"Error en ciclo #{self._ciclo_actual}: {str(e)[:300]}"
                )

            # Calcular tiempo de espera
            duracion = time.time() - inicio
            espera = max(0, intervalo - duracion)

            if self._running and espera > 0:
                logger.info(
                    f"Ciclo completado en {duracion:.0f}s. "
                    f"Próximo ciclo en {espera / 60:.1f} min"
                )
                # Dormir en intervalos cortos para permitir shutdown rápido
                for _ in range(int(espera)):
                    if not self._running:
                        break
                    time.sleep(1)

        self._shutdown()

    def ejecutar_ciclo_unico(self) -> None:
        """Ejecuta un solo ciclo del agente (útil para testing)."""
        configurar_logging()
        self._ejecutar_ciclo()

    # =========================================================================
    # Loop principal
    # =========================================================================

    def _ejecutar_ciclo(self) -> None:
        """Ejecuta un ciclo completo del agente."""

        # =====================================================================
        # Paso 1: ESCANEAR mercados activos
        # =====================================================================
        logger.info("Paso 1: Escaneando mercados...")
        mercados = self._scanner.escanear_mercados()

        if not mercados:
            logger.warning("No se encontraron mercados. Saltando ciclo.")
            return

        logger.info(f"  {len(mercados)} mercados encontrados")

        # =====================================================================
        # Paso 2: RECOPILAR datos para cada mercado
        # =====================================================================
        logger.info("Paso 2: Recopilando datos...")
        contextos = self._collector.recopilar_multiples(
            mercados, max_news_per_market=5
        )

        # Filtrar mercados con datos insuficientes
        contextos_validos = [
            c for c in contextos if c.data_quality != "poor"
        ]
        logger.info(
            f"  {len(contextos_validos)}/{len(contextos)} "
            f"con datos suficientes"
        )

        if not contextos_validos:
            logger.warning("Sin datos suficientes. Saltando evaluación.")
            return

        # =====================================================================
        # Paso 3: EVALUAR probabilidades con LLM
        # =====================================================================
        logger.info("Paso 3: Evaluando probabilidades con LLM...")
        evaluaciones = self._engine.evaluar_multiples(contextos_validos)

        exitosas = sum(1 for _, e in evaluaciones if e is not None)
        logger.info(f"  {exitosas}/{len(evaluaciones)} evaluaciones exitosas")

        # =====================================================================
        # Paso 4: GENERAR señales de trading
        # =====================================================================
        logger.info("Paso 4: Generando señales de trading...")
        señales = self._strategy.generar_señales_multiples(evaluaciones)

        señales_compra = [s for s in señales if s.action == "BUY"]
        logger.info(
            f"  {len(señales_compra)} señales de compra de {len(señales)} total"
        )

        # =====================================================================
        # Paso 5 + 6: VALIDAR y EJECUTAR
        # =====================================================================
        logger.info("Pasos 5-6: Validando y ejecutando...")
        trades_ejecutados = 0

        for señal in señales_compra:
            record = self._executor.ejecutar_señal(señal)
            if record:
                trades_ejecutados += 1
                self._alerts.trade_ejecutado(record)
            else:
                logger.info(
                    f"  Señal no ejecutada: {señal.market_question[:40]}"
                )

        logger.info(f"  {trades_ejecutados} trades ejecutados")

        # =====================================================================
        # Paso 7: REVISAR posiciones existentes
        # =====================================================================
        logger.info("Paso 7: Revisando posiciones existentes...")
        self._revisar_posiciones()

        # =====================================================================
        # Paso 8: REPORTAR
        # =====================================================================
        metricas = self._portfolio.calcular_metricas()
        balance = self._executor.obtener_balance()
        self._portfolio.registrar_balance(balance)

        logger.info(
            f"Resumen del ciclo: "
            f"trades={trades_ejecutados}, "
            f"balance=${balance:.2f}, "
            f"pnl=${metricas.realized_pnl:+.2f}, "
            f"posiciones={len(self._portfolio.obtener_posiciones_abiertas())}"
        )

    def _revisar_posiciones(self) -> None:
        """Revisa posiciones abiertas y cierra las que ya no tienen edge."""
        posiciones = self._portfolio.obtener_posiciones_abiertas()

        if not posiciones:
            logger.info("  Sin posiciones abiertas para revisar")
            return

        logger.info(f"  Revisando {len(posiciones)} posiciones abiertas")

        for pos in posiciones:
            # Obtener mercado actualizado
            mercado = self._scanner.obtener_mercado_por_id(pos.market_id)
            if mercado is None:
                continue

            # Verificar si el mercado se resolvió
            if mercado.resolved or mercado.closed:
                logger.info(
                    f"  Mercado resuelto/cerrado: {pos.market_question[:40]}"
                )
                # En paper trading, la resolución se maneja manualmente
                continue

    def _verificar_reporte_diario(self) -> None:
        """Envía reporte diario si es la hora configurada."""
        ahora = datetime.now()
        hora_reporte = settings.agent.daily_report_hour

        if (
            ahora.hour == hora_reporte
            and (ahora - self._ultimo_reporte).total_seconds() > 3600
        ):
            logger.info("Generando reporte diario...")
            metricas = self._portfolio.calcular_metricas()
            balance = self._executor.obtener_balance()
            posiciones = self._portfolio.obtener_posiciones_abiertas()

            self._alerts.resumen_diario(
                balance=balance,
                pnl_dia=metricas.realized_pnl,
                trades_dia=metricas.total_trades,
                win_rate=metricas.win_rate,
                posiciones=len(posiciones),
            )

            reporte = self._reporter.generar_reporte_diario()
            logger.info(f"\n{reporte}")

            self._ultimo_reporte = ahora

    # =========================================================================
    # Shutdown
    # =========================================================================

    def _manejar_shutdown(self, signum: int, frame: object) -> None:
        """Maneja señales de shutdown (Ctrl+C, SIGTERM)."""
        logger.info(f"Señal de shutdown recibida ({signum}). Deteniendo...")
        self._running = False

    def _shutdown(self) -> None:
        """Cierra todos los componentes de forma ordenada."""
        logger.info("Shutdown ordenado del agente...")

        # Guardar estado final
        balance = self._executor.obtener_balance()
        self._portfolio.registrar_balance(balance)

        # Generar reporte final
        reporte = self._reporter.generar_reporte_diario()
        logger.info(f"\nReporte final:\n{reporte}")

        # Cerrar conexiones
        self._scanner.close()
        self._collector.close()

        # Notificar
        self._alerts.agente_detenido("Shutdown ordenado")

        logger.info("Agente detenido correctamente")


# =============================================================================
# Punto de entrada
# =============================================================================

def main() -> None:
    """Inicia el agente autónomo."""
    print("""
    ╔══════════════════════════════════════╗
    ║   Polymarket Trading Agent v1.0      ║
    ║   Agente Autónomo de Inversiones     ║
    ╚══════════════════════════════════════╝
    """)

    agente = AgentOrchestrator()

    # Verificar modo
    if settings.es_modo_live():
        print("⚠️  MODO LIVE - Usando fondos reales")
        print("    Presiona Ctrl+C para detener\n")
    else:
        print("📝 MODO PAPER TRADING - Simulación sin fondos reales")
        print("   Presiona Ctrl+C para detener\n")

    agente.ejecutar()


if __name__ == "__main__":
    main()
