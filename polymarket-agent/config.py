"""
Configuración centralizada del agente de Polymarket.

Todas las variables sensibles se cargan desde .env.
Los parámetros de estrategia y riesgo se definen aquí con valores por defecto
conservadores que pueden ser sobreescritos por variables de entorno.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

# Cargar variables de entorno desde .env
# Busca el .env en el directorio raíz del proyecto
_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Obtiene una variable de entorno con validación."""
    valor = os.getenv(key, default)
    if required and not valor:
        raise EnvironmentError(
            f"Variable de entorno requerida '{key}' no está definida. "
            f"Revisa tu archivo .env"
        )
    return valor or ""


# =============================================================================
# Configuración de APIs
# =============================================================================

@dataclass(frozen=True)
class PolymarketConfig:
    """Configuración de conexión a Polymarket CLOB."""
    api_url: str = _get_env("POLYMARKET_API_URL", "https://clob.polymarket.com")
    private_key: str = _get_env("POLYMARKET_PRIVATE_KEY", "")
    api_key: str = _get_env("POLYMARKET_API_KEY", "")
    api_secret: str = _get_env("POLYMARKET_API_SECRET", "")
    api_passphrase: str = _get_env("POLYMARKET_API_PASSPHRASE", "")
    # Endpoints públicos (no requieren autenticación)
    gamma_api_url: str = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class AnthropicConfig:
    """Configuración de la API de Anthropic (Claude)."""
    api_key: str = _get_env("ANTHROPIC_API_KEY", "")
    model: str = _get_env("LLM_MODEL", "claude-sonnet-4-5-20250929")
    max_tokens: int = int(_get_env("LLM_MAX_TOKENS", "1000"))
    temperature: float = float(_get_env("LLM_TEMPERATURE", "0.3"))


@dataclass(frozen=True)
class TelegramConfig:
    """Configuración del bot de Telegram para alertas."""
    bot_token: str = _get_env("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = _get_env("TELEGRAM_CHAT_ID", "")
    enabled: bool = _get_env("TELEGRAM_ENABLED", "false").lower() == "true"


# =============================================================================
# Configuración de Escaneo de Mercados (Fase 1)
# =============================================================================

@dataclass(frozen=True)
class ScannerConfig:
    """Parámetros de filtrado para el escaneo de mercados."""
    # Liquidez mínima en USD para considerar un mercado
    min_liquidity_usd: float = float(_get_env("MIN_LIQUIDITY_USD", "500"))
    # Volumen mínimo en 24h en USD
    min_volume_24h_usd: float = float(_get_env("MIN_VOLUME_24H_USD", "100"))
    # Tiempo mínimo hasta resolución (en días)
    min_days_to_resolution: int = int(_get_env("MIN_DAYS_TO_RESOLUTION", "1"))
    # Tiempo máximo hasta resolución (en días)
    max_days_to_resolution: int = int(_get_env("MAX_DAYS_TO_RESOLUTION", "90"))
    # Máximo de mercados a procesar por ciclo
    max_markets_per_cycle: int = int(_get_env("MAX_MARKETS_PER_CYCLE", "20"))
    # Timeout para requests HTTP en segundos
    http_timeout_seconds: int = int(_get_env("HTTP_TIMEOUT_SECONDS", "30"))


# =============================================================================
# Configuración de Estrategia (Fase 4)
# =============================================================================

@dataclass(frozen=True)
class StrategyConfig:
    """Parámetros de la estrategia de trading."""
    # Edge mínimo para considerar una operación (5% = 0.05)
    min_edge: float = float(_get_env("MIN_EDGE", "0.05"))
    # Confianza mínima del LLM para operar
    min_confidence: Literal["high", "medium", "low"] = _get_env(
        "MIN_CONFIDENCE", "medium"
    )  # type: ignore[assignment]
    # Factor de Kelly fraccional (0.25 = quarter Kelly, conservador)
    kelly_fraction: float = float(_get_env("KELLY_FRACTION", "0.25"))


# =============================================================================
# Configuración de Riesgo (Fase 5)
# =============================================================================

@dataclass(frozen=True)
class RiskConfig:
    """Reglas de gestión de riesgo INAMOVIBLES."""
    # Bankroll total disponible en USDC
    max_bankroll_usd: float = float(_get_env("MAX_BANKROLL_USD", "500"))
    # Máximo por trade individual (% del bankroll)
    max_per_trade_pct: float = float(_get_env("MAX_PER_TRADE_PCT", "0.05"))
    # Máximo en un solo mercado (% del bankroll)
    max_per_market_pct: float = float(_get_env("MAX_PER_MARKET_PCT", "0.10"))
    # Máximo en una categoría (% del bankroll)
    max_per_category_pct: float = float(_get_env("MAX_PER_CATEGORY_PCT", "0.25"))
    # Máximo total expuesto (% del bankroll)
    max_exposure_pct: float = float(_get_env("MAX_EXPOSURE_PCT", "0.60"))
    # Pérdida máxima diaria (% del bankroll)
    max_daily_loss_pct: float = float(_get_env("MAX_DAILY_LOSS_PCT", "0.10"))
    # Pérdida máxima semanal (% del bankroll)
    max_weekly_loss_pct: float = float(_get_env("MAX_WEEKLY_LOSS_PCT", "0.15"))
    # Drawdown máximo antes de pausa (% del bankroll)
    max_drawdown_pct: float = float(_get_env("MAX_DRAWDOWN_PCT", "0.25"))
    # Horas de cooldown tras pérdidas consecutivas
    cooldown_hours: int = int(_get_env("COOLDOWN_HOURS", "6"))
    # Stop-loss: movimiento adverso máximo (%)
    stop_loss_pct: float = float(_get_env("STOP_LOSS_PCT", "0.15"))


# =============================================================================
# Configuración de Ejecución (Fase 6)
# =============================================================================

@dataclass(frozen=True)
class ExecutionConfig:
    """Parámetros de ejecución de órdenes."""
    # Tipo de orden preferido
    order_type: Literal["limit", "market"] = _get_env(
        "ORDER_TYPE", "limit"
    )  # type: ignore[assignment]
    # Tolerancia de slippage (2% = 0.02)
    slippage_tolerance: float = float(_get_env("SLIPPAGE_TOLERANCE", "0.02"))
    # Timeout para órdenes limit no ejecutadas (minutos)
    order_timeout_minutes: int = int(_get_env("ORDER_TIMEOUT_MINUTES", "15"))
    # Máximo de movimiento de precio desde evaluación para ejecutar
    max_price_deviation: float = float(_get_env("MAX_PRICE_DEVIATION", "0.03"))


# =============================================================================
# Configuración del Agente (Fase 8)
# =============================================================================

@dataclass(frozen=True)
class AgentConfig:
    """Parámetros del loop principal del agente."""
    # Intervalo entre ciclos de escaneo (minutos)
    scan_interval_minutes: int = int(_get_env("SCAN_INTERVAL_MINUTES", "30"))
    # Modo de operación
    mode: Literal["paper", "live"] = _get_env(
        "TRADING_MODE", "paper"
    )  # type: ignore[assignment]
    # Ruta de la base de datos SQLite
    database_path: str = _get_env("DATABASE_PATH", "data/portfolio.db")
    # Hora del reporte diario (hora local Chile)
    daily_report_hour: int = int(_get_env("DAILY_REPORT_HOUR", "21"))
    # Puerto del dashboard
    dashboard_port: int = int(_get_env("DASHBOARD_PORT", "8501"))


# =============================================================================
# Configuración de Logging
# =============================================================================

@dataclass(frozen=True)
class LogConfig:
    """Configuración del sistema de logging."""
    level: str = _get_env("LOG_LEVEL", "INFO")
    file: str = _get_env("LOG_FILE", "logs/agent.log")
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# Instancia global de configuración
# =============================================================================

@dataclass(frozen=True)
class Settings:
    """Configuración global del agente. Punto de acceso único."""
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def es_modo_paper(self) -> bool:
        """Retorna True si estamos en modo simulación."""
        return self.agent.mode == "paper"

    def es_modo_live(self) -> bool:
        """Retorna True si estamos en modo real."""
        return self.agent.mode == "live"


# Singleton de configuración - importar esto en todos los módulos
settings = Settings()


def configurar_logging() -> None:
    """Configura el sistema de logging global del agente."""
    log_path = _BASE_DIR / settings.log.file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log.level.upper(), logging.INFO),
        format=settings.log.format,
        datefmt=settings.log.date_format,
        handlers=[
            logging.StreamHandler(),  # Salida a consola
            logging.FileHandler(log_path, encoding="utf-8"),  # Salida a archivo
        ],
    )

    # Reducir ruido de librerías externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


if __name__ == "__main__":
    # Test rápido: mostrar configuración actual
    from rich import print as rprint
    rprint("[bold green]Configuración del Agente Polymarket[/bold green]")
    rprint(settings)
