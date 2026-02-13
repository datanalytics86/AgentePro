"""
Sistema de alertas via Telegram.

Envía notificaciones sobre actividad del agente:
trades, rechazos, errores y reportes.

Fase 7 del agente autónomo.
"""

import logging
from datetime import datetime

import httpx

from config import settings
from core.models import TradeRecord, TradeSignal

logger = logging.getLogger(__name__)


class TelegramAlerts:
    """
    Envía alertas al usuario via Telegram Bot API.

    Tipos de alerta:
    - Trade ejecutado
    - Trade rechazado por risk manager
    - Límite de pérdida alcanzado
    - Resumen diario
    - Error crítico
    """

    def __init__(self) -> None:
        self._config = settings.telegram
        self._enabled = self._config.enabled and bool(self._config.bot_token)
        self._base_url = (
            f"https://api.telegram.org/bot{self._config.bot_token}"
            if self._config.bot_token else ""
        )

        if not self._enabled:
            logger.info(
                "Alertas Telegram desactivadas "
                "(configura TELEGRAM_BOT_TOKEN y TELEGRAM_ENABLED=true)"
            )

    def trade_ejecutado(self, record: TradeRecord) -> None:
        """Notifica un trade ejecutado exitosamente."""
        modo = "[PAPER]" if record.mode == "paper" else "[LIVE]"
        mensaje = (
            f"🟢 Trade Ejecutado {modo}\n\n"
            f"📊 {record.market_question[:80]}\n"
            f"➡️ {record.action} {record.side}\n"
            f"💰 ${record.size_usd:.2f} @ {record.price:.3f}\n"
            f"📦 {record.shares:.2f} shares\n"
            f"🔑 {record.trade_id}"
        )
        self._enviar(mensaje)

    def trade_rechazado(self, signal: TradeSignal, razon: str) -> None:
        """Notifica un trade rechazado por el risk manager."""
        mensaje = (
            f"🔴 Trade Rechazado\n\n"
            f"📊 {signal.market_question[:80]}\n"
            f"➡️ {signal.action} {signal.side} ${signal.suggested_size_usd:.2f}\n"
            f"❌ Razón: {razon}"
        )
        self._enviar(mensaje)

    def limite_perdida(self, tipo: str, monto: float, limite: float) -> None:
        """Notifica que se alcanzó un límite de pérdida."""
        mensaje = (
            f"⚠️ Límite de Pérdida Alcanzado\n\n"
            f"📈 Tipo: {tipo}\n"
            f"💸 Pérdida actual: ${monto:.2f}\n"
            f"🚫 Límite: ${limite:.2f}\n"
            f"⏸️ Operaciones pausadas"
        )
        self._enviar(mensaje)

    def resumen_diario(
        self,
        balance: float,
        pnl_dia: float,
        trades_dia: int,
        win_rate: float,
        posiciones: int,
    ) -> None:
        """Envía el resumen diario de performance."""
        emoji_pnl = "📈" if pnl_dia >= 0 else "📉"
        mensaje = (
            f"🔵 Resumen Diario - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            f"💰 Balance: ${balance:.2f}\n"
            f"{emoji_pnl} PnL Día: ${pnl_dia:+.2f}\n"
            f"📊 Trades hoy: {trades_dia}\n"
            f"✅ Win Rate: {win_rate:.0%}\n"
            f"📦 Posiciones abiertas: {posiciones}"
        )
        self._enviar(mensaje)

    def error_critico(self, error: str) -> None:
        """Notifica un error crítico que requiere atención."""
        mensaje = (
            f"🛑 ERROR CRÍTICO\n\n"
            f"⚠️ {error[:500]}\n\n"
            f"🔧 Requiere intervención manual"
        )
        self._enviar(mensaje)

    def agente_iniciado(self, modo: str) -> None:
        """Notifica que el agente se inició."""
        mensaje = (
            f"🚀 Agente Iniciado\n\n"
            f"📊 Modo: {modo.upper()}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💰 Bankroll: ${settings.risk.max_bankroll_usd:.2f}"
        )
        self._enviar(mensaje)

    def agente_detenido(self, razon: str = "Manual") -> None:
        """Notifica que el agente se detuvo."""
        mensaje = (
            f"⏹️ Agente Detenido\n\n"
            f"📝 Razón: {razon}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._enviar(mensaje)

    def _enviar(self, mensaje: str) -> None:
        """Envía un mensaje por Telegram."""
        if not self._enabled:
            logger.debug(f"[Telegram OFF] {mensaje[:80]}...")
            return

        try:
            response = httpx.post(
                f"{self._base_url}/sendMessage",
                json={
                    "chat_id": self._config.chat_id,
                    "text": mensaje,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if response.status_code != 200:
                logger.warning(
                    f"Error enviando Telegram: {response.status_code}"
                )
        except Exception as e:
            logger.warning(f"Error enviando alerta Telegram: {e}")
