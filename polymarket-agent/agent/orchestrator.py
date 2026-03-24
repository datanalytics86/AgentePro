"""
Orquestador principal del agente autónomo (async).

Integra todos los componentes y ejecuta el loop principal
que corre de forma autónoma 24/7 con asyncio.

Estrategia: Copy-Trading — replica las posiciones de los mejores
traders del leaderboard de Polymarket.

Fase 8 del agente autónomo — refactorizado con copy-trading.
"""

import asyncio
import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from typing import Any

from config import settings, configurar_logging
from core.market_scanner import MarketScanner
from core.leaderboard import LeaderboardFetcher
from core.copy_strategy import CopyTradingStrategy
from core.risk_manager import RiskManager
from core.portfolio import Portfolio
from core.executor import OrderExecutor
from core.backup_manager import BackupManager
from monitoring.alerts import TelegramAlerts
from monitoring.reporter import Reporter

logger = logging.getLogger(__name__)

# Timeout máximo por ciclo en modo Turbo (segundos)
TURBO_CYCLE_TIMEOUT = 300

# Valor placeholder que indica credencial no configurada
_PLACEHOLDER = "PENDIENTE_CONFIGURAR"


def validar_credenciales_live() -> None:
    """
    Valida que las credenciales requeridas para modo LIVE no sean
    placeholders ni estén vacías.

    Se ejecuta al iniciar el agente. Si falta alguna credencial,
    lanza EnvironmentError con la lista de variables pendientes.
    """
    if settings.es_modo_paper():
        return

    faltantes: list[str] = []
    checks = [
        (settings.polymarket.private_key, "POLYMARKET_PRIVATE_KEY"),
        (settings.polymarket.api_key, "POLYMARKET_API_KEY"),
        (settings.polymarket.api_secret, "POLYMARKET_API_SECRET"),
        (settings.polymarket.api_passphrase, "POLYMARKET_API_PASSPHRASE"),
    ]

    for valor, nombre in checks:
        if not valor or valor == _PLACEHOLDER:
            faltantes.append(nombre)

    if faltantes:
        raise EnvironmentError(
            f"Modo LIVE requiere credenciales reales. "
            f"Configura en .env: {', '.join(faltantes)}"
        )


class AgentOrchestrator:
    """
    Orquestador del agente autónomo de copy-trading en Polymarket (async).

    Estrategia principal: COPY-TRADING
    Copia las posiciones de los mejores traders del leaderboard.

    Loop principal:
    1. Escanear mercados activos (filtros de liquidez/volumen)
    2. Consultar leaderboard → top N traders por PnL
    3. Obtener posiciones de cada top trader (en paralelo)
    4. Identificar mercados con consenso (múltiples top traders)
    5. Generar señales de compra para mercados con consenso
    6. Validar con risk manager
    7. Ejecutar trades aprobados
    8. Revisar posiciones existentes (¿siguen los traders posicionados?)
    9. Liquidar mercados resueltos
    10. Reportar actividad
    """

    def __init__(self) -> None:
        configurar_logging()
        logger.info("Inicializando agente de Polymarket (copy-trading)...")

        # Validar credenciales antes de crear componentes costosos
        validar_credenciales_live()

        # Componentes async
        self._scanner = MarketScanner()
        self._leaderboard = LeaderboardFetcher()

        # Estrategia copy-trading
        self._copy_strategy = CopyTradingStrategy(self._leaderboard)

        # Componentes sync (CPU-bound, rápidos)
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
        self._shutdown_event = asyncio.Event()
        self._ciclo_actual = 0
        self._ultimo_reporte = datetime.now()
        self._ultimo_reporte_semanal = datetime.now()

        logger.info(
            f"Agente inicializado en modo {settings.agent.mode.upper()} | "
            f"Estrategia: COPY-TRADING (top {settings.copy_trading.top_n} traders) | "
            f"Bankroll: ${settings.risk.max_bankroll_usd:.2f}"
        )

    async def ejecutar(self) -> None:
        """
        Loop principal async del agente. Corre hasta recibir señal de parada.
        """
        self._running = True
        intervalo = settings.agent.scan_interval_minutes * 60

        # Configurar shutdown handlers
        self._loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._loop.add_signal_handler(sig, self._manejar_shutdown_async)
        else:
            # Windows: signal.signal corre fuera del event loop,
            # por eso usamos call_soon_threadsafe en el handler.
            signal.signal(signal.SIGINT, self._manejar_shutdown_signal)
            signal.signal(signal.SIGTERM, self._manejar_shutdown_signal)

        # Iniciar backup periódico como tarea de fondo
        self._backup_task = asyncio.create_task(
            self._backup_manager.iniciar(),
            name="backup_periodico",
        )

        self._alerts.agente_iniciado(settings.agent.mode)
        logger.info(
            f"Agente iniciado (copy-trading). "
            f"Ciclo cada {settings.agent.scan_interval_minutes} min"
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
                        f"{TURBO_CYCLE_TIMEOUT}s. Continuando."
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        f"Ciclo #{self._ciclo_actual} cancelado. "
                        f"Continuando al siguiente ciclo."
                    )

                # Verificar si toca reporte diario o semanal
                self._verificar_reporte_diario()
                self._verificar_reporte_semanal()

            except KeyboardInterrupt:
                break
            except asyncio.CancelledError:
                logger.info("Agente cancelado. Deteniendo...")
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
                # Esperar con evento cancelable (Ctrl+C lo interrumpe al instante)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=espera
                    )
                    # Si el event se activó, es shutdown
                    break
                except asyncio.TimeoutError:
                    # Timeout normal → siguiente ciclo
                    pass
                except asyncio.CancelledError:
                    break

        await self._shutdown()

    async def ejecutar_ciclo_unico(self) -> None:
        """Ejecuta un solo ciclo del agente (útil para testing)."""
        configurar_logging()
        await self._ejecutar_ciclo()

    # =========================================================================
    # Loop principal (async) — COPY-TRADING
    # =========================================================================

    async def _ejecutar_ciclo(self) -> None:
        """Ejecuta un ciclo completo del agente con copy-trading."""

        # =====================================================================
        # Paso 1: ESCANEAR mercados activos (para validación y precios)
        # =====================================================================
        logger.info("Paso 1: Escaneando mercados activos...")
        mercados = await self._scanner.escanear_mercados()

        if not mercados:
            logger.warning("No se encontraron mercados. Saltando ciclo.")
            return

        logger.info(f"  {len(mercados)} mercados activos")

        # =====================================================================
        # Paso 2-3: CONSULTAR LEADERBOARD + POSICIONES TOP TRADERS
        # =====================================================================
        logger.info(
            f"Pasos 2-3: Consultando leaderboard "
            f"(top {settings.copy_trading.top_n})..."
        )

        snapshots = await self._leaderboard.obtener_snapshots_completos(
            top_n=settings.copy_trading.top_n,
            window=settings.copy_trading.leaderboard_window,
        )

        if not snapshots:
            logger.warning("No se obtuvieron snapshots del leaderboard.")
            return

        # Identificar consensos
        consensos = self._leaderboard.identificar_mercados_consenso(
            min_traders=settings.copy_trading.min_traders_consensus,
        )

        # =====================================================================
        # Paso 4a: ENRIQUECER mercados — buscar los que faltan
        # =====================================================================
        # El scanner solo trae los top N mercados por volumen, pero los top
        # traders pueden estar posicionados en mercados fuera de ese set.
        # Buscamos individualmente los mercados de consenso que faltan.
        mercados_index = {m.condition_id: m for m in mercados}
        consensus_ids_faltantes = [
            c["market_id"]
            for c in consensos
            if c["market_id"] not in mercados_index
        ]

        if consensus_ids_faltantes:
            logger.info(
                f"  Buscando {len(consensus_ids_faltantes)} mercados de "
                f"consenso no incluidos en el escaneo..."
            )
            # Limitar a 40 búsquedas paralelas para no saturar la API
            lote = consensus_ids_faltantes[:40]
            resultados = await asyncio.gather(
                *[self._scanner.obtener_mercado_por_id(mid) for mid in lote],
                return_exceptions=True,
            )
            encontrados = 0
            for r in resultados:
                if isinstance(r, Exception):
                    continue
                if r is not None:
                    mercados.append(r)
                    encontrados += 1
            logger.info(
                f"  {encontrados} mercados adicionales obtenidos "
                f"(total: {len(mercados)})"
            )

        # =====================================================================
        # Paso 4b: GENERAR SEÑALES DE COPY-TRADING
        # =====================================================================
        logger.info("Paso 4: Generando señales de copy-trading...")

        # Obtener IDs de mercados donde ya tenemos posición
        posiciones_propias = [
            p.market_id
            for p in self._portfolio.obtener_posiciones_abiertas()
        ]

        señales = self._copy_strategy.generar_señales_desde_consenso(
            consensos=consensos,
            mercados_disponibles=mercados,
            posiciones_propias=posiciones_propias,
        )

        señales_compra = [s for s in señales if s.action == "BUY"]
        logger.info(
            f"  {len(señales_compra)} señales de compra generadas"
        )

        # =====================================================================
        # Paso 5-6: VALIDAR y EJECUTAR
        # =====================================================================
        logger.info("Pasos 5-6: Validando con risk manager y ejecutando...")
        trades_ejecutados = 0

        for señal in señales_compra:
            # Alerta previa de señal copy (antes de ejecutar)
            if señal.reasoning and "traders" in señal.reasoning.lower():
                try:
                    # Extraer traders_count del reasoning
                    traders_count = int(
                        señal.reasoning.split("traders")[0].strip().split()[-1]
                    )
                    self._alerts.copy_señal_generada(
                        market_question=señal.market_question,
                        side=señal.side,
                        traders_count=traders_count,
                        trader_names=[],
                        size_usd=señal.suggested_size_usd,
                        entry_price=señal.entry_price,
                    )
                except (ValueError, IndexError):
                    pass

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
        # Paso 7: REVISAR posiciones existentes (¿siguen los traders?)
        # =====================================================================
        logger.info("Paso 7: Revisando posiciones existentes...")
        await self._revisar_posiciones_copy()

        # =====================================================================
        # Paso 8: LIQUIDAR posiciones de mercados ya resueltos
        # =====================================================================
        logger.info("Paso 8: Liquidando posiciones de mercados resueltos...")

        # El scanner solo trae mercados activos, así que los resueltos no
        # están en la lista. Buscamos individualmente cada posición abierta
        # que no esté ya en la lista de mercados.
        mercados_index_actual = {m.condition_id: m for m in mercados}
        posiciones_abiertas = self._portfolio.obtener_posiciones_abiertas()
        pos_ids_faltantes = [
            p.market_id
            for p in posiciones_abiertas
            if p.market_id not in mercados_index_actual
        ]

        if pos_ids_faltantes:
            logger.info(
                f"  Buscando estado de {len(pos_ids_faltantes)} mercados "
                f"con posiciones abiertas..."
            )
            pos_resultados = await asyncio.gather(
                *[
                    self._scanner.obtener_mercado_por_id(mid)
                    for mid in pos_ids_faltantes
                ],
                return_exceptions=True,
            )
            for r in pos_resultados:
                if isinstance(r, Exception):
                    continue
                if r is not None:
                    mercados.append(r)

        liquidados = self._portfolio.update_settled_trades(mercados)
        if liquidados:
            balance_tras_liquidacion = self._executor.obtener_balance()
            self._portfolio.registrar_balance(balance_tras_liquidacion)
            logger.info(
                f"  {len(liquidados)} posición(es) liquidada(s). "
                f"Balance: ${balance_tras_liquidacion:.2f}"
            )

        # =====================================================================
        # Paso 9: REPORTAR
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

    async def _revisar_posiciones_copy(self) -> None:
        """
        Revisa posiciones abiertas basándose en si los top traders
        aún mantienen sus posiciones.

        Si los traders del leaderboard salieron de un mercado,
        nosotros también salimos.
        """
        posiciones = self._portfolio.obtener_posiciones_abiertas()

        if not posiciones:
            logger.info("  Sin posiciones abiertas para revisar")
            return

        logger.info(f"  Revisando {len(posiciones)} posiciones abiertas")
        ventas_activas = 0

        for pos in posiciones:
            # Verificar si los top traders aún mantienen posición
            decision = self._copy_strategy.evaluar_posicion_copy(
                market_id=pos.market_id,
                side=pos.side,
            )

            if decision != "SELL":
                logger.debug(
                    f"  HOLD: {pos.market_question[:40]} | "
                    f"Top traders aún posicionados"
                )
                continue

            # Obtener precio actual del mercado
            mercado = await self._scanner.obtener_mercado_por_id(pos.market_id)
            if mercado is None:
                logger.debug(f"  No se pudo obtener mercado {pos.market_id[:12]}")
                continue

            if mercado.resolved or mercado.closed:
                logger.info(
                    f"  Mercado resuelto/cerrado: {pos.market_question[:40]}"
                )
                continue

            precio_actual = (
                mercado.yes_price if pos.side == "YES" else mercado.no_price
            )

            # Resolver token_id para modo live
            from core.executor import resolver_token_id
            token_id = resolver_token_id(mercado.tokens, pos.side)

            logger.warning(
                f"  SALIDA COPY: {pos.market_question[:45]} | "
                f"Top traders salieron de la posición"
            )

            pnl = self._executor.ejecutar_venta_activa(
                market_id=pos.market_id,
                market_question=pos.market_question,
                side=pos.side,
                precio_actual=precio_actual,
                token_id=token_id,
            )

            if pnl is not None:
                ventas_activas += 1
                logger.info(
                    f"  Posición cerrada: {pos.market_question[:40]} | "
                    f"PnL ${pnl:+.2f}"
                )
                self._alerts.copy_salida_ejecutada(
                    market_question=pos.market_question,
                    side=pos.side,
                    pnl=pnl,
                )

        if ventas_activas:
            logger.info(f"  {ventas_activas} salida(s) copy ejecutada(s)")

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

    def _verificar_reporte_semanal(self) -> None:
        """Envía reporte semanal los domingos a la hora del reporte diario."""
        ahora = datetime.now()
        hora_reporte = settings.agent.daily_report_hour

        # Domingo = 6 en weekday()
        if (
            ahora.weekday() == 6
            and ahora.hour == hora_reporte
            and (ahora - self._ultimo_reporte_semanal).total_seconds() > 86400
        ):
            logger.info("Generando reporte semanal...")
            reporte = self._reporter.generar_reporte_semanal()
            logger.info(f"\n{reporte}")
            self._ultimo_reporte_semanal = ahora

    # =========================================================================
    # Shutdown
    # =========================================================================

    def _manejar_shutdown_async(self) -> None:
        """Maneja señales de shutdown en modo async (Unix)."""
        logger.info("Señal de shutdown recibida. Deteniendo...")
        self._running = False
        self._shutdown_event.set()

    def _manejar_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Maneja señales de shutdown via signal.signal (Windows).

        signal.signal() corre fuera del event loop thread, así que
        asyncio.Event.set() no es thread-safe. Usamos call_soon_threadsafe
        para programar el set() dentro del loop.
        """
        logger.info(f"Señal {signum} recibida. Deteniendo...")
        self._running = False
        self._loop.call_soon_threadsafe(self._shutdown_event.set)

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
        await self._leaderboard.close()

        # Notificar
        self._alerts.agente_detenido("Shutdown ordenado")

        logger.info("Agente detenido correctamente")


# =============================================================================
# Punto de entrada
# =============================================================================

def main() -> None:
    """Inicia el agente autónomo con asyncio."""
    print("""
    ╔══════════════════════════════════════════════╗
    ║   Polymarket Copy-Trading Agent v3.0         ║
    ║   Copia las mejores carteras del leaderboard ║
    ╚══════════════════════════════════════════════╝
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
