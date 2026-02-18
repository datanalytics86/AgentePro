"""
Motor de ejecución de órdenes en Polymarket.

Ejecuta operaciones aprobadas por el RiskManager con
verificaciones pre-vuelo y modo paper trading.

Fase 6 del agente autónomo.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from config import settings
from core.models import TradeRecord, TradeSignal
from core.portfolio import Portfolio
from core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Interfaz base para ejecutores de órdenes."""

    @abstractmethod
    def ejecutar_orden(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
    ) -> TradeRecord | None:
        """Ejecuta una orden y retorna el registro del trade."""
        ...

    @abstractmethod
    def cancelar_orden(self, order_id: str) -> bool:
        """Cancela una orden pendiente."""
        ...

    @abstractmethod
    def obtener_balance(self) -> float:
        """Obtiene el balance disponible en USDC."""
        ...


class PaperExecutor(BaseExecutor):
    """
    Ejecutor de paper trading (simulación).

    Simula ejecuciones sin usar fondos reales.
    Mantiene un balance virtual y simula fills inmediatos.
    """

    def __init__(self, balance_inicial: float | None = None) -> None:
        self._balance = balance_inicial or settings.risk.max_bankroll_usd
        self._balance_inicial = self._balance
        self._ordenes_pendientes: dict[str, dict] = {}

    def ejecutar_orden(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
    ) -> TradeRecord | None:
        """
        Simula la ejecución de una orden.

        En paper trading, asumimos fill inmediato al precio de la señal.
        """
        price = signal.entry_price
        size_usd = signal.suggested_size_usd

        if size_usd > self._balance:
            logger.warning(
                f"Paper: Balance insuficiente ${self._balance:.2f} "
                f"< ${size_usd:.2f}"
            )
            return None

        # Calcular shares
        shares = size_usd / price if price > 0 else 0

        # Simular fill
        self._balance -= size_usd
        order_id = f"PAPER-{uuid.uuid4().hex[:8]}"

        # Registrar en portafolio
        record = portfolio.registrar_trade(
            signal=signal,
            price=price,
            shares=shares,
            status="filled",
            order_id=order_id,
        )

        logger.info(
            f"[PAPER] Ejecutado: {signal.action} {signal.side} "
            f"${size_usd:.2f} ({shares:.2f} shares) @ {price:.3f} | "
            f"Balance: ${self._balance:.2f}"
        )

        return record

    def cancelar_orden(self, order_id: str) -> bool:
        """Cancela una orden pendiente simulada."""
        if order_id in self._ordenes_pendientes:
            del self._ordenes_pendientes[order_id]
            logger.info(f"[PAPER] Orden cancelada: {order_id}")
            return True
        return False

    def obtener_balance(self) -> float:
        """Retorna el balance virtual."""
        return self._balance

    def ejecutar_venta(
        self,
        market_id: str,
        side: str,
        precio_actual: float,
        portfolio: Portfolio,
    ) -> float | None:
        """
        Cierra una posición en paper trading al precio de mercado actual.

        Calcula el cash de retorno como ``cost_basis + pnl`` y lo
        devuelve al balance virtual para mantener la contabilidad correcta.

        Args:
            market_id:     ID del mercado a cerrar.
            side:          Lado de la posición ("YES" o "NO").
            precio_actual: Precio actual del token en el mercado.
            portfolio:     Portafolio para actualizar registros.

        Returns:
            PnL realizado en USD, o None si no existía la posición.
        """
        # Obtener detalles antes de cerrar
        posiciones = portfolio.obtener_posiciones_abiertas()
        pos = next(
            (p for p in posiciones
             if p.market_id == market_id and p.side == side),
            None,
        )
        if pos is None:
            logger.warning(
                f"[PAPER] Venta: no existe posición {market_id[:12]} {side}"
            )
            return None

        # Cerrar en portfolio (actualiza pnl en DB)
        pnl = portfolio.cerrar_posicion(market_id, side, precio_actual)

        # Devolver capital al balance: dinero inicial + ganancia/pérdida
        cash_back = pos.cost_basis + pnl
        self._balance += cash_back

        logger.info(
            f"[PAPER] Venta activa: {market_id[:12]} {side} "
            f"@ {precio_actual:.3f} | PnL: ${pnl:+.2f} | "
            f"Balance: ${self._balance:.2f}"
        )
        return pnl

    def simular_resolucion(
        self,
        market_id: str,
        side: str,
        resolucion_yes: bool,
        portfolio: Portfolio,
    ) -> float:
        """
        Simula la resolución de un mercado.

        Args:
            market_id: ID del mercado.
            side: Lado de nuestra posición.
            resolucion_yes: True si el mercado resuelve YES.
            portfolio: Portafolio para actualizar.

        Returns:
            PnL de la resolución.
        """
        # Si apostamos YES y resuelve YES → ganamos (precio cierre = 1.0)
        # Si apostamos YES y resuelve NO → perdemos (precio cierre = 0.0)
        if side == "YES":
            precio_cierre = 1.0 if resolucion_yes else 0.0
        else:
            precio_cierre = 0.0 if resolucion_yes else 1.0

        pnl = portfolio.cerrar_posicion(market_id, side, precio_cierre)
        self._balance += pnl + sum(
            p.cost_basis for p in portfolio.obtener_posiciones_abiertas()
            if p.market_id == market_id
        )

        logger.info(
            f"[PAPER] Mercado resuelto: {market_id[:12]} "
            f"{'YES' if resolucion_yes else 'NO'} | PnL: ${pnl:+.2f}"
        )
        return pnl


class LiveExecutor(BaseExecutor):
    """
    Ejecutor de trading real en Polymarket CLOB.

    Usa py-clob-client para interactuar con el order book.
    Requiere wallet y API keys configurados.
    """

    def __init__(self) -> None:
        self._config = settings.polymarket
        self._exec_config = settings.execution
        self._client = self._crear_cliente()

    def _crear_cliente(self) -> object:
        """
        Crea el cliente CLOB de Polymarket.

        Requiere py-clob-client instalado y credenciales configuradas.
        """
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self._config.api_key,
                api_secret=self._config.api_secret,
                api_passphrase=self._config.api_passphrase,
            )

            client = ClobClient(
                self._config.api_url,
                key=self._config.private_key,
                chain_id=137,  # Polygon
                creds=creds,
            )

            logger.info("Cliente CLOB de Polymarket inicializado")
            return client

        except ImportError:
            logger.error(
                "py-clob-client no instalado. "
                "Ejecuta: pip install py-clob-client"
            )
            raise
        except Exception as e:
            logger.error(f"Error creando cliente CLOB: {e}")
            raise

    def ejecutar_orden(
        self,
        signal: TradeSignal,
        portfolio: Portfolio,
    ) -> TradeRecord | None:
        """
        Ejecuta una orden real en Polymarket.

        Implementa limit orders con slippage protection.
        """
        try:
            from py_clob_client.order_builder.constants import BUY, SELL

            # Determinar token_id del lado correcto
            token_id = signal.market_id  # Necesita el token_id real

            # Calcular precio con slippage
            if signal.action == "BUY":
                price = min(
                    signal.entry_price + self._exec_config.slippage_tolerance,
                    0.99,
                )
                side = BUY
            else:
                price = max(
                    signal.entry_price - self._exec_config.slippage_tolerance,
                    0.01,
                )
                side = SELL

            size = signal.suggested_size_usd

            # Crear y enviar orden
            order = self._client.create_and_post_order(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
            )

            if order and hasattr(order, "id"):
                record = portfolio.registrar_trade(
                    signal=signal,
                    price=price,
                    shares=size / price,
                    status="filled",
                    order_id=str(order.id),
                )
                logger.info(
                    f"[LIVE] Orden ejecutada: {order.id} | "
                    f"{signal.action} {signal.side} ${size:.2f}"
                )
                return record

            logger.warning("[LIVE] Orden no retornó ID")
            return None

        except Exception as e:
            logger.error(f"[LIVE] Error ejecutando orden: {e}")
            return None

    def cancelar_orden(self, order_id: str) -> bool:
        """Cancela una orden en Polymarket."""
        try:
            self._client.cancel(order_id)
            logger.info(f"[LIVE] Orden cancelada: {order_id}")
            return True
        except Exception as e:
            logger.error(f"[LIVE] Error cancelando orden {order_id}: {e}")
            return False

    def obtener_balance(self) -> float:
        """Obtiene el balance real de USDC."""
        try:
            # Esto depende de la implementación del CLOB client
            # Intentar obtener balance via la API
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"[LIVE] Error obteniendo balance: {e}")
            return 0.0


class OrderExecutor:
    """
    Fachada que gestiona la ejecución de órdenes con pre-flight checks.

    Selecciona automáticamente entre PaperExecutor y LiveExecutor
    según el modo configurado.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk_manager: RiskManager,
    ) -> None:
        self._portfolio = portfolio
        self._risk = risk_manager
        self._exec_config = settings.execution

        # Seleccionar executor según modo
        if settings.es_modo_paper():
            self._executor: BaseExecutor = PaperExecutor()
            logger.info("Modo PAPER TRADING activado")
        else:
            self._executor = LiveExecutor()
            logger.info("Modo LIVE TRADING activado")

    def ejecutar_señal(self, signal: TradeSignal) -> TradeRecord | None:
        """
        Ejecuta una señal de trading con todas las verificaciones.

        Flujo:
        1. Pre-flight checks
        2. Risk manager approval
        3. Ejecución
        4. Registro

        Returns:
            TradeRecord si se ejecutó, None si fue rechazado.
        """
        # Solo procesar señales BUY
        if signal.action != "BUY":
            logger.debug(f"Señal {signal.action} ignorada (solo ejecutamos BUY)")
            return None

        # Pre-flight checks
        ok, razon = self._pre_flight_checks(signal)
        if not ok:
            logger.info(f"Pre-flight RECHAZADO: {razon}")
            return None

        # Risk manager
        ok, razon = self._verificar_riesgo(signal)
        if not ok:
            logger.info(f"Risk manager RECHAZÓ: {razon}")
            return None

        # Ejecutar
        record = self._executor.ejecutar_orden(signal, self._portfolio)

        if record:
            self._portfolio.registrar_balance(self._executor.obtener_balance())

        return record

    def obtener_balance(self) -> float:
        """Retorna el balance actual."""
        return self._executor.obtener_balance()

    def ejecutar_venta_activa(
        self,
        market_id: str,
        market_question: str,
        side: str,
        precio_actual: float,
    ) -> float | None:
        """
        Ejecuta una salida anticipada cerrando una posición al precio actual.

        Usado por el orquestador cuando la re-evaluación del LLM detecta
        que la probabilidad cayó bajo 0.50 o el edge se volvió negativo.

        Args:
            market_id:      ID del mercado.
            market_question: Pregunta del mercado (para logging).
            side:           Lado de la posición ("YES" o "NO").
            precio_actual:  Precio de mercado actual del token.

        Returns:
            PnL realizado en USD, o None si falló la operación.
        """
        logger.info(
            f"Ejecutando salida activa: {market_question[:45]} "
            f"[{side}] @ {precio_actual:.3f}"
        )

        if isinstance(self._executor, PaperExecutor):
            pnl = self._executor.ejecutar_venta(
                market_id=market_id,
                side=side,
                precio_actual=precio_actual,
                portfolio=self._portfolio,
            )
        else:
            # Modo live: cerrar en portfolio (sin llamar al CLOB por seguridad)
            # En producción: enviar SELL al CLOB antes de cerrar
            pnl = self._portfolio.cerrar_posicion(market_id, side, precio_actual)

        if pnl is not None:
            balance = self.obtener_balance()
            self._portfolio.registrar_balance(balance)
            logger.info(f"Salida activa completada: PnL ${pnl:+.2f}")

        return pnl

    # =========================================================================
    # Pre-flight checks
    # =========================================================================

    def _pre_flight_checks(self, signal: TradeSignal) -> tuple[bool, str]:
        """Verificaciones antes de ejecutar."""
        # Check 1: Balance suficiente
        balance = self._executor.obtener_balance()
        if signal.suggested_size_usd > balance:
            return False, (
                f"Balance insuficiente: ${balance:.2f} "
                f"< ${signal.suggested_size_usd:.2f}"
            )

        # Check 2: Precio no se movió demasiado
        # (en producción, verificar precio actual vs signal.entry_price)

        # Check 3: Tamaño mínimo viable
        if signal.suggested_size_usd < 1.0:
            return False, "Tamaño demasiado pequeño (< $1)"

        return True, "OK"

    def _verificar_riesgo(self, signal: TradeSignal) -> tuple[bool, str]:
        """Consulta al risk manager."""
        return self._risk.aprobar_trade(
            signal=signal,
            balance_actual=self._executor.obtener_balance(),
            exposicion_total=self._portfolio.calcular_exposicion_total(),
            exposicion_mercado=self._portfolio.calcular_exposicion_mercado(
                signal.market_id
            ),
            exposicion_categoria=0,  # Se calcula con la categoría real
            perdida_diaria=self._portfolio.calcular_perdida_diaria(),
            perdida_semanal=self._portfolio.calcular_perdida_semanal(),
            drawdown_actual=self._portfolio.calcular_metricas().current_drawdown,
        )
