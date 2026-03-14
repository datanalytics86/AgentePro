# CLAUDE.md — Documentación del Agente Copy-Trading de Polymarket

## Contexto del Proyecto

Agente autónomo de trading en Polymarket (mercado de predicciones blockchain).
Corre 24/7, escaneando mercados y ejecutando trades automáticamente.

**Estrategia principal: COPY-TRADING** — replica las posiciones de los mejores
traders del leaderboard de Polymarket. No usa LLM para estimar probabilidades;
en su lugar, confía en el consenso de los traders más rentables.

## Arquitectura

```
polymarket-agent/
├── config.py                    # Configuración centralizada (dataclasses + .env)
├── agent/
│   └── orchestrator.py          # Loop principal del agente (async)
├── core/
│   ├── models.py                # Modelos Pydantic (Market, TradeSignal, TradeRecord)
│   ├── market_scanner.py        # Escaneo de mercados via Gamma API
│   ├── leaderboard.py           # ★ Scraper del leaderboard + posiciones de traders
│   ├── copy_strategy.py         # ★ Estrategia de copy-trading
│   ├── strategy.py              # Estrategia legacy (edge detection con LLM)
│   ├── probability_engine.py    # Motor LLM legacy (Claude para probabilidades)
│   ├── data_collector.py        # Recopilación de datos legacy (noticias, historial)
│   ├── risk_manager.py          # Gestión de riesgo (límites INAMOVIBLES)
│   ├── portfolio.py             # Portfolio tracking con SQLite (WAL mode)
│   ├── executor.py              # Ejecución de órdenes (Paper + Live CLOB)
│   ├── net_utils.py             # Retry async, TTLCache
│   └── backup_manager.py        # Backups periódicos de la DB
├── data/
│   ├── news_fetcher.py          # RSS feeds de noticias (legacy)
│   ├── market_history.py        # Historial de precios (legacy)
│   └── sentiment.py             # Análisis de sentimiento (legacy)
├── monitoring/
│   ├── dashboard.py             # Dashboard Streamlit
│   ├── alerts.py                # Alertas Telegram
│   └── reporter.py              # Reportes de performance
├── tests/                       # Tests unitarios (pytest)
└── scripts/                     # Scripts auxiliares
```

## Estrategia Copy-Trading (v3.0)

### Concepto

En vez de intentar ser más inteligentes que el mercado con un LLM,
copiamos lo que los traders más exitosos de Polymarket están haciendo.
Si múltiples top traders tienen la misma posición, es señal fuerte.

### Flujo de Cada Ciclo

```
1. ESCANEAR mercados activos (Gamma API)
   → Filtrar por liquidez, volumen, días hasta resolución

2. CONSULTAR LEADERBOARD (Data API)
   → GET /leaderboard?window=all&rankBy=profit
   → Obtener top N traders (default: 20)

3. OBTENER POSICIONES de cada trader (en paralelo)
   → GET /positions?user={wallet} para cada top trader
   → GET /activity?user={wallet} para trades recientes
   → Todo en asyncio.gather() para velocidad

4. IDENTIFICAR CONSENSO
   → Agrupar posiciones por (market_id, side)
   → Filtrar: mínimo N traders con misma posición (default: 3)
   → Ponderar por profit del trader (mejores pesan más)

5. GENERAR SEÑALES
   → Para cada mercado con consenso:
     - Verificar que no tengamos posición ya
     - Verificar precio de entrada (5% < price < 95%)
     - Calcular sizing: 3% bankroll base + 1% por trader extra
     - Generar TradeSignal

6. VALIDAR CON RISK MANAGER
   → Mismos checks de siempre: límites por trade, mercado,
     categoría, exposición total, pérdida diaria/semanal

7. EJECUTAR trades aprobados
   → Paper: simulado con balance virtual
   → Live: orden al CLOB de Polymarket

8. REVISAR posiciones existentes
   → ¿Los top traders siguen posicionados?
   → Si salieron → ejecutar salida (SELL)

9. LIQUIDAR mercados resueltos
   → Detectar tokens ganadores
   → Registrar PnL realizado

10. REPORTAR
    → Log de métricas
    → Telegram si habilitado
    → Reporte diario/semanal
```

### APIs de Polymarket Utilizadas

| API | Base URL | Auth | Uso |
|-----|----------|------|-----|
| Gamma API | `https://gamma-api.polymarket.com` | No | Mercados, precios, tokens |
| Data API | `https://data-api.polymarket.com` | No | Leaderboard, posiciones, trades |
| CLOB API | `https://clob.polymarket.com` | Sí (API keys) | Ejecución de órdenes (solo live) |

### Endpoints Data API (Copy-Trading)

```
GET /leaderboard
  ?window=all|1d|7d|30d    # Ventana temporal
  &rankBy=profit|volume    # Métrica de ranking
  → Retorna hasta 100 traders con wallet, profit, volume

GET /positions
  ?user={wallet}           # Dirección proxy wallet
  &sizeThreshold=1.0       # Tamaño mínimo
  &limit=500               # Máximo posiciones
  &sortBy=CURRENT          # Ordenar por valor actual
  &sortDirection=DESC
  → Retorna posiciones abiertas con market_id, side, PnL

GET /activity
  ?user={wallet}           # Dirección proxy wallet
  &limit=50                # Últimos N trades
  &sortBy=TIMESTAMP
  &sortDirection=DESC
  → Retorna trades recientes con precio, tamaño, timestamp
```

## Configuración

### Variables de Entorno (Copy-Trading)

```bash
# Activar copy-trading (default: true)
COPY_TRADING_ENABLED=true

# Top N traders a seguir (default: 20)
COPY_TRADING_TOP_N=20

# Ventana del leaderboard: "1d", "7d", "30d", "all"
COPY_TRADING_WINDOW=all

# Mínimo traders con misma posición para generar señal
COPY_MIN_CONSENSUS=3

# Tamaño base por trade como % del bankroll (3% = 0.03)
COPY_SIZE_PCT=0.03

# Máximo nuevas posiciones por ciclo
COPY_MAX_NEW_PER_CYCLE=5

# Rango de precios aceptables (no comprar >95% ni <5%)
COPY_MAX_ENTRY_PRICE=0.95
COPY_MIN_ENTRY_PRICE=0.05

# Intervalo de actualización del leaderboard (minutos)
COPY_UPDATE_MINUTES=60
```

### Variables de Entorno (Risk Management — no cambian)

```bash
MAX_BANKROLL_USD=500       # Capital total
MAX_PER_TRADE_PCT=0.05     # 5% max por trade
MAX_PER_MARKET_PCT=0.10    # 10% max por mercado
MAX_PER_CATEGORY_PCT=0.25  # 25% max por categoría
MAX_EXPOSURE_PCT=0.60      # 60% max total expuesto
MAX_DAILY_LOSS_PCT=0.10    # Stop si -10% en el día
MAX_WEEKLY_LOSS_PCT=0.15   # Stop si -15% en la semana
MAX_DRAWDOWN_PCT=0.25      # Pausa si drawdown > 25%
STOP_LOSS_PCT=0.15         # Exit si posición -15%
```

## Archivos Clave

### Nuevos (Copy-Trading v3.0)

- **`core/leaderboard.py`** — Scraper del leaderboard y tracker de posiciones.
  Modelos: `TopTrader`, `TraderPosition`, `TraderTrade`, `TraderSnapshot`.
  Clase: `LeaderboardFetcher` con métodos async para consultar Data API.
  Método clave: `obtener_snapshots_completos()` obtiene todo en paralelo.
  Método clave: `identificar_mercados_consenso()` agrupa y pondera posiciones.

- **`core/copy_strategy.py`** — Estrategia de copy-trading.
  Clase: `CopyTradingStrategy`.
  Método clave: `generar_señales_copy()` genera TradeSignals desde consenso.
  Método clave: `evaluar_posicion_copy()` decide HOLD/SELL si traders salieron.

### Existentes (modificados)

- **`config.py`** — Agregado `CopyTradingConfig` dataclass con todos los
  parámetros de copy-trading. Acceso: `settings.copy_trading.*`

- **`agent/orchestrator.py`** — Reescrito para usar copy-trading.
  Ya no usa `DataCollector`, `ProbabilityEngine`, ni `TradingStrategy` legacy.
  Usa `LeaderboardFetcher` y `CopyTradingStrategy` en su lugar.

### Existentes (sin cambios)

- **`core/models.py`** — Mismos modelos Pydantic (Market, TradeSignal, etc.)
- **`core/market_scanner.py`** — Mismo scanner de mercados via Gamma API
- **`core/risk_manager.py`** — Mismas reglas de riesgo INAMOVIBLES
- **`core/portfolio.py`** — Mismo tracking SQLite con WAL mode
- **`core/executor.py`** — Mismo executor Paper/Live

### Legacy (aún disponibles pero no usados por el orquestador)

- `core/strategy.py` — Edge detection con Kelly (no se usa con copy-trading)
- `core/probability_engine.py` — Evaluación con Claude (no se usa)
- `core/data_collector.py` — Noticias y sentimiento (no se usa)

## Estado Actual del P&L

- Balance inicial: $500.00
- Balance actual: ~$525.21 (+5.04%)
- PnL realizado: +$9.33
- Posiciones abiertas: 9
- Los últimos ciclos estaban paralizados por bugs (ya corregidos)

## Cómo Ejecutar

### PowerShell (Windows)

```powershell
# Paper trading — TRADING_MODE=paper es el default en scripts/paper_trade.py
python scripts/paper_trade.py

# O explícitamente:
$env:TRADING_MODE="paper"; python scripts/paper_trade.py

# Live trading
$env:TRADING_MODE="live"; python scripts/paper_trade.py

# Un solo ciclo (testing)
python -c "import asyncio; from agent.orchestrator import AgentOrchestrator; asyncio.run(AgentOrchestrator().ejecutar_ciclo_unico())"

# Dashboard
streamlit run monitoring/dashboard.py

# Tests
pytest tests/ -v
```

### Bash / Linux / Mac

```bash
# Paper trading (simulación)
python scripts/paper_trade.py

# Live trading
TRADING_MODE=live python scripts/paper_trade.py

# Un solo ciclo (testing)
python -c "
import asyncio
from agent.orchestrator import AgentOrchestrator
asyncio.run(AgentOrchestrator().ejecutar_ciclo_unico())
"

# Dashboard
streamlit run monitoring/dashboard.py

# Tests
pytest tests/ -v
```

## Decisiones de Diseño

1. **Por qué copy-trading en vez de LLM:** Los mejores traders de Polymarket
   tienen más información, mejor juicio y track record demostrado. Copiar
   su consenso es más robusto que depender de un LLM estimando probabilidades
   de eventos geopolíticos con información limitada de RSS feeds.

2. **Por qué consenso y no un solo trader:** Un solo trader puede tener
   un golpe de suerte. Cuando 3+ de los top 20 coinciden en la misma posición,
   hay señal real.

3. **Por qué mantener el risk manager igual:** Las reglas de riesgo protegen
   el capital independientemente de la estrategia. Son INAMOVIBLES.

4. **Por qué no borrar el código legacy:** Las fases 2-4 (data collector,
   probability engine, strategy) siguen disponibles por si se quiere hacer
   un modo híbrido en el futuro.

## Próximos Pasos Potenciales

- [ ] Filtrar traders del leaderboard por win rate mínimo (no solo PnL)
- [ ] Ponderar sizing por Sharpe del trader (no solo profit total)
- [ ] Alertas Telegram cuando un top trader abre/cierra posición nueva
- [ ] Dashboard: mostrar leaderboard y posiciones de top traders
- [ ] Modo híbrido: usar LLM como filtro adicional sobre las señales de copy
- [ ] Tracking de performance: comparar nuestro P&L vs top traders copiados
