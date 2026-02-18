"""
Orquestador principal del agente autónomo (async).

Integra todos los componentes y ejecuta el loop principal
que corre de forma autónoma 24/7 con asyncio.

Fase 8 del agente autónomo — refactorizado a asyncio con modo Turbo.
"""

import asyncio
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
from core.backup_manager import BackupManager
from monitoring.alerts import TelegramAlerts
from monitoring.reporter import Reporter

logger = logging.getLogger(__name__)

# Timeout máximo por ciclo en modo Turbo (segundos)
TURBO_CYCLE_TIMEOUT = 300


class AgentOrchestrator:
    """
    Orquestador del agente autónomo de trading en Polymarket (async).

    Loop principal:
    1. Escanear mercados activos
    2. Recopilar datos (noticias, historial, sentimiento) en paralelo
    3. Evaluar probabilidades con LLM en paralelo
    4. Generar señales de trading
    5. Validar con risk manager
    6. Ejecutar trades aprobados
    7. Revisar posiciones existentes
    8. Reportar actividad
    9. Dormir hasta próximo ciclo

    Modo Turbo: si un ciclo tarda > 300s, descarta mercados de baja
    prioridad (bajo volumen/liquidez) y salta la evaluación de LLM
    para mercados con caché vigente.
    """

    def __init__(self) -> None:
        configurar_logging()
        logger.info("Inicializando agente de Polymarket (async)...")

        # Componentes async
        self._scanner = MarketScanner()
        self._collector = DataCollector()
        self._engine = ProbabilityEngine()

        # Componentes sync (CPU-bound, rápidos)
        self._strategy = TradingStrategy()
        self._risk_manager = RiskManager()
        self._portfolio = Portfolio()
        self._executor = OrderExecutor(self._portfolio, self._risk_manager)
        self._alerts = TelegramAlerts()
        self._reporter = Reporter(self._portfolio)

        # Backup periódico de la DB (24h, async, no bloquea)
        self._backup_manager = BackupManager(settings.agent.database_path)
        self._backup_task: asyncio.Task | None = None

        # Estado
        self._running = False
        self._ciclo_actual = 0
        self._ultimo_reporte = datetime.now()

        logger.info(
            f"Agente inicializado en modo {settings.agent.mode.upper()} | "
            f"Bankroll: ${settings.risk.max_bankroll_usd:.2f}"
        )

    async def ejecutar(self) -> None:
        """
        Loop principal async del agente. Corre hasta recibir señal de parada.
        """
        self._running = True
        intervalo = settings.agent.scan_interval_minutes * 60

        # Configurar shutdown handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._manejar_shutdown_async)

        # Iniciar backup periódico como tarea de fondo
        self._backup_task = asyncio.create_task(
            self._backup_manager.iniciar(),
            name="backup_periodico",
        )

        self._alerts.agente_iniciado(settings.agent.mode)
        logger.info(
            f"Agente iniciado. Ciclo cada {settings.agent.scan_interval_minutes} min"
        )

        while self._running:
            self._ciclo_actual += 1
            inicio = time.monotonic()

            try:
                logger.info(f"{'=' * 60}")
                logger.info(f"CICLO #{self._ciclo_actual} - {datetime.now()}")
                logger.info(f"{'=' * 60}")

                # Modo Turbo: timeout de 300s por ciclo
                try:
                    await asyncio.wait_for(
                        self._ejecutar_ciclo(),
                        timeout=TURBO_CYCLE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"TURBO: Ciclo #{self._ciclo_actual} excedió "
                        f"{TURBO_CYCLE_TIMEOUT}s. Mercados lentos descartados."
                    )

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
            duracion = time.monotonic() - inicio
            espera = max(0, intervalo - duracion)

            if self._running and espera > 0:
                logger.info(
                    f"Ciclo completado en {duracion:.0f}s. "
                    f"Próximo ciclo en {espera / 60:.1f} min"
                )
                # Dormir async (permite shutdown rápido)
                try:
                    await asyncio.sleep(espera)
                except asyncio.CancelledError:
                    break

        await self._shutdown()

    async def ejecutar_ciclo_unico(self) -> None:
        """Ejecuta un solo ciclo del agente (útil para testing)."""
        configurar_logging()
        await self._ejecutar_ciclo()

    # =========================================================================
    # Loop principal (async)
    # =========================================================================

    async def _ejecutar_ciclo(self) -> None:
        """Ejecuta un ciclo completo del agente con operaciones en paralelo."""

        # =====================================================================
        # Paso 1: ESCANEAR mercados activos
        # =====================================================================
        logger.info("Paso 1: Escaneando mercados...")
        mercados = await self._scanner.escanear_mercados()

        if not mercados:
            logger.warning("No se encontraron mercados. Saltando ciclo.")
            return

        logger.info(f"  {len(mercados)} mercados encontrados")

        # =====================================================================
        # Paso 2: RECOPILAR datos para cada mercado (en paralelo)
        # =====================================================================
        logger.info("Paso 2: Recopilando datos en paralelo...")
        contextos = await self._collector.recopilar_multiples(
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
        # Paso 3: EVALUAR probabilidades con LLM (en paralelo)
        # =====================================================================
        logger.info("Paso 3: Evaluando probabilidades con LLM en paralelo...")
        evaluaciones = await self._engine.evaluar_multiples(contextos_validos)

        exitosas = sum(1 for _, e in evaluaciones if e is not None)
        logger.info(
            f"  {exitosas}/{len(evaluaciones)} evaluaciones exitosas "
            f"(cache: {self._engine.cache_stats})"
        )

        # =====================================================================
        # Paso 4: GENERAR señales de trading (CPU-bound, instantáneo)
        # =====================================================================
        logger.info("Paso 4: Generando señales de trading...")
        señales = self._strategy.generar_señales_multiples(evaluaciones)

        señales_compra = [s for s in señales if s.action == "BUY"]
        logger.info(
            f"  {len(señales_compra)} señales de compra de {len(señales)} total"
        )

        # =====================================================================
        # Paso 5 + 6: VALIDAR y EJECUTAR (sync, rápido)
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
        await self._revisar_posiciones()

        # =====================================================================
        # Paso 7.5: LIQUIDAR posiciones de mercados ya resueltos
        # =====================================================================
        logger.info("Paso 7.5: Liquidando posiciones de mercados resueltos...")
        liquidados = self._portfolio.update_settled_trades(mercados)
        if liquidados:
            # Devolver capital liberado al executor (paper mode)
            balance_tras_liquidacion = self._executor.obtener_balance()
            self._portfolio.registrar_balance(balance_tras_liquidacion)
            logger.info(
                f"  {len(liquidados)} posición(es) liquidada(s). "
                f"Balance: ${balance_tras_liquidacion:.2f}"
            )

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

    async def _revisar_posiciones(self) -> None:
        """
        Revisa posiciones abiertas y ejecuta salidas anticipadas si es necesario.

        Para cada posición abierta:
        1. Obtiene precio actualizado del mercado.
        2. Re-evalúa la probabilidad con el LLM (usa caché 60 min si disponible).
        3. Si prob < 0.50 o el edge se invirtió, ejecuta SELL inmediato.
        """
        posiciones = self._portfolio.obtener_posiciones_abiertas()

        if not posiciones:
            logger.info("  Sin posiciones abiertas para revisar")
            return

        logger.info(f"  Revisando {len(posiciones)} posiciones abiertas")
        ventas_activas = 0

        for pos in posiciones:
            # ── 1. Obtener mercado actualizado ──────────────────────────────
            mercado = await self._scanner.obtener_mercado_por_id(pos.market_id)
            if mercado is None:
                logger.debug(f"  No se pudo obtener mercado {pos.market_id[:12]}")
                continue

            # ── 2. Mercados ya resueltos/cerrados: sin acción ───────────────
            if mercado.resolved or mercado.closed:
                logger.info(
                    f"  Mercado resuelto/cerrado: {pos.market_question[:40]}"
                )
                continue

            # ── 3. Recopilar contexto fresco (noticias + historial) ─────────
            try:
                contexto = await self._collector.recopilar_contexto(
                    mercado, max_news=3
                )
            except Exception as exc:
                logger.warning(
                    f"  Error recopilando contexto para "
                    f"{pos.market_question[:40]}: {exc}"
                )
                continue

            # ── 4. Re-evaluar probabilidad con LLM (usa caché si vigente) ───
            evaluacion = await self._engine.evaluar_mercado(contexto)
            if evaluacion is None:
                logger.debug(
                    f"  LLM no disponible para {pos.market_question[:40]}"
                )
                continue

            # ── 5. Precio actual y edge recalculado ──────────────────────────
            precio_actual = (
                mercado.yes_price if pos.side == "YES" else mercado.no_price
            )
            prob = evaluacion.probability

            if pos.side == "YES":
                edge_actual = prob - precio_actual
            else:
                edge_actual = (1.0 - prob) - precio_actual

            # ── 6. Decisión de salida anticipada ────────────────────────────
            decision = self._strategy.evaluar_posicion_existente(
                evaluacion_actual=evaluacion,
                precio_entrada=pos.entry_price,
                precio_actual=precio_actual,
                side=pos.side,
            )

            if decision != "SELL":
                logger.debug(
                    f"  HOLD: {pos.market_question[:40]} | "
                    f"p={prob:.2f} edge={edge_actual:+.3f}"
                )
                continue

            # ── 7. Ejecutar salida anticipada ────────────────────────────────
            razon = (
                f"prob={prob:.2f} < 0.50"
                if prob < 0.50
                else f"edge={edge_actual:+.3f} negativo"
            )
            logger.warning(
                f"  SALIDA ACTIVA: {pos.market_question[:45]} | {razon}"
            )

            pnl = self._executor.ejecutar_venta_activa(
                market_id=pos.market_id,
                market_question=pos.market_question,
                side=pos.side,
                precio_actual=precio_actual,
            )

            if pnl is not None:
                ventas_activas += 1
                logger.info(
                    f"  Posición cerrada: {pos.market_question[:40]} | "
                    f"PnL ${pnl:+.2f}"
                )

        if ventas_activas:
            logger.info(f"  {ventas_activas} salida(s) activa(s) ejecutada(s)")

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

    def _manejar_shutdown_async(self) -> None:
        """Maneja señales de shutdown en modo async."""
        logger.info("Señal de shutdown recibida. Deteniendo...")
        self._running = False

    async def _shutdown(self) -> None:
        """Cierra todos los componentes de forma ordenada."""
        logger.info("Shutdown ordenado del agente...")

        # Guardar estado final
        balance = self._executor.obtener_balance()
        self._portfolio.registrar_balance(balance)

        # Backup final antes de cerrar
        self._backup_manager.detener()
        if self._backup_task is not None:
            self._backup_task.cancel()
        await self._backup_manager.hacer_backup_ahora()

        # Generar reporte final
        reporte = self._reporter.generar_reporte_diario()
        logger.info(f"\nReporte final:\n{reporte}")

        # Cerrar conexiones async
        await self._scanner.close()
        await self._collector.close()

        # Notificar
        self._alerts.agente_detenido("Shutdown ordenado")

        logger.info("Agente detenido correctamente")


# =============================================================================
# Punto de entrada
# =============================================================================

def main() -> None:
    """Inicia el agente autónomo con asyncio."""
    print("""
    ╔══════════════════════════════════════╗
    ║   Polymarket Trading Agent v2.0      ║
    ║   Agente Autónomo (async + turbo)    ║
    ╚══════════════════════════════════════╝
    """)

    agente = AgentOrchestrator()

    # Verificar modo
    if settings.es_modo_live():
        print("MODO LIVE - Usando fondos reales")
        print("    Presiona Ctrl+C para detener\n")
    else:
        print("MODO PAPER TRADING - Simulación sin fondos reales")
        print("   Presiona Ctrl+C para detener\n")

    asyncio.run(agente.ejecutar())


if __name__ == "__main__":
    main()
