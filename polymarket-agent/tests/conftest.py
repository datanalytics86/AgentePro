"""
Configuración de entorno para la suite de pruebas.

Establece variables de entorno de prueba ANTES de que pytest importe
los módulos de test (y por tanto antes de que config.py ejecute
``load_dotenv()``).

python-dotenv no sobreescribe variables ya presentes en el proceso
(``override=False`` por defecto), así que estas asignaciones tienen
prioridad sobre el .env de producción, manteniendo los tests aislados
de la configuración de live trading del desarrollador.
"""

import os

# ── Parámetros de riesgo compatibles con los tests existentes ─────────────────
# Los tests crean señales con suggested_size_usd=$10.0; con bankroll=$500
# y max_per_trade=5%, el límite es $25 → los $10 son aprobados.
os.environ.setdefault("MAX_BANKROLL_USD", "500")
os.environ.setdefault("MAX_PER_TRADE_PCT", "0.05")
os.environ.setdefault("MAX_PER_MARKET_PCT", "0.10")
os.environ.setdefault("MAX_PER_CATEGORY_PCT", "0.25")
os.environ.setdefault("MAX_EXPOSURE_PCT", "0.60")
os.environ.setdefault("MAX_DAILY_LOSS_PCT", "0.10")
os.environ.setdefault("MAX_WEEKLY_LOSS_PCT", "0.15")
os.environ.setdefault("MAX_DRAWDOWN_PCT", "0.25")
os.environ.setdefault("STOP_LOSS_PCT", "0.15")

# ── Estrategia ─────────────────────────────────────────────────────────────────
os.environ.setdefault("MIN_EDGE", "0.05")
os.environ.setdefault("MIN_CONFIDENCE", "medium")
os.environ.setdefault("KELLY_FRACTION", "0.25")

# ── Modo de operación: siempre paper en tests ─────────────────────────────────
os.environ.setdefault("TRADING_MODE", "paper")

# ── Noticias ──────────────────────────────────────────────────────────────────
os.environ.setdefault("MIN_NEWS_COUNT", "3")
