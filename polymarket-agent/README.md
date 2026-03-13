# Polymarket Copy-Trading Agent

Agente autónomo que opera en Polymarket copiando las posiciones de los mejores traders del leaderboard.

## Estrategia

**Copy-Trading:** En lugar de estimar probabilidades propias, el agente consulta el leaderboard de Polymarket, obtiene las posiciones de los top traders, e identifica mercados donde múltiples traders exitosos coinciden. Cuando hay consenso, replica la posición proporcionalmente a nuestro bankroll.

### Flujo por ciclo

1. Escanear mercados activos (filtrar por liquidez/volumen)
2. Consultar leaderboard → top 20 traders por PnL
3. Obtener posiciones abiertas de cada trader (en paralelo)
4. Identificar mercados con consenso (3+ traders con misma posición)
5. Generar señales de compra ponderadas por profit del trader
6. Validar con risk manager (límites estrictos)
7. Ejecutar trades aprobados
8. Revisar posiciones (¿siguen los traders posicionados?)
9. Liquidar mercados resueltos

## Estructura del Proyecto

```
polymarket-agent/
├── config.py                # Configuración centralizada
├── agent/
│   └── orchestrator.py      # Loop principal (async)
├── core/
│   ├── models.py            # Modelos de datos (Pydantic)
│   ├── market_scanner.py    # Escaneo de mercados (Gamma API)
│   ├── leaderboard.py       # Scraper leaderboard + posiciones traders
│   ├── copy_strategy.py     # Estrategia de copy-trading
│   ├── risk_manager.py      # Gestión de riesgo
│   ├── portfolio.py         # Portfolio tracking (SQLite)
│   ├── executor.py          # Ejecución de órdenes (Paper/Live)
│   └── net_utils.py         # Retry async, TTLCache
├── monitoring/
│   ├── dashboard.py         # Dashboard web (Streamlit)
│   ├── alerts.py            # Alertas Telegram
│   └── reporter.py          # Reportes de performance
├── tests/                   # Tests unitarios
└── scripts/                 # Scripts auxiliares
```

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves
```

## Uso Rápido

```bash
# Iniciar agente en modo paper trading
python -m agent.orchestrator

# Dashboard de monitoreo
streamlit run monitoring/dashboard.py

# Ejecutar tests
pytest tests/ -v
```

## Configuración Copy-Trading

| Variable | Default | Descripción |
|----------|---------|-------------|
| `COPY_TRADING_ENABLED` | `true` | Activar copy-trading |
| `COPY_TRADING_TOP_N` | `20` | Top N traders a seguir |
| `COPY_TRADING_WINDOW` | `all` | Ventana: 1d, 7d, 30d, all |
| `COPY_MIN_CONSENSUS` | `3` | Min traders para señal |
| `COPY_SIZE_PCT` | `0.03` | 3% bankroll por trade |
| `COPY_MAX_NEW_PER_CYCLE` | `5` | Max nuevas posiciones/ciclo |

Ver `.env.example` para la lista completa.

## Documentación Técnica

Ver `CLAUDE.md` para documentación completa: arquitectura, APIs, decisiones de diseño, y guía de desarrollo.

## Aviso Legal

Proyecto experimental y educativo. Los mercados de predicción implican riesgo de pérdida total. No existe garantía de rentabilidad.
