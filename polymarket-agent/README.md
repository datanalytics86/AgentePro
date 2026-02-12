# Polymarket Agent - Agente Autónomo de Trading

Agente autónomo de inversiones que opera en Polymarket (mercado de predicciones basado en blockchain) de forma independiente.

## Estructura del Proyecto

```
polymarket-agent/
├── .env.example          # Template de variables de entorno
├── config.py             # Configuración centralizada
├── requirements.txt      # Dependencias Python
├── core/                 # Componentes principales
│   ├── models.py         # Modelos de datos (Market, TradeSignal, etc.)
│   ├── market_scanner.py # Fase 1: Escaneo de mercados
│   ├── data_collector.py # Fase 2: Recopilación de datos
│   ├── probability_engine.py # Fase 3: Motor LLM
│   ├── strategy.py       # Fase 4: Estrategia de decisión
│   ├── risk_manager.py   # Fase 5: Gestión de riesgo
│   ├── executor.py       # Fase 6: Ejecución de órdenes
│   └── portfolio.py      # Fase 5: Tracking de portafolio
├── data/                 # Pipeline de datos
│   ├── news_fetcher.py   # Noticias y contexto
│   ├── market_history.py # Histórico de precios
│   └── sentiment.py      # Análisis de sentimiento
├── monitoring/           # Monitoreo y alertas
│   ├── dashboard.py      # Dashboard web (Streamlit)
│   ├── alerts.py         # Alertas Telegram
│   └── reporter.py       # Reportes de performance
├── agent/                # Orquestación
│   ├── orchestrator.py   # Loop principal
│   └── scheduler.py      # Programación de tareas
├── tests/                # Tests unitarios
└── scripts/              # Scripts auxiliares
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

# 4. Verificar Fase 1
python scripts/verificar_fase1.py
```

## Uso Rápido

```bash
# Escanear mercados activos
python -m core.market_scanner

# Ejecutar tests
pytest tests/ -v
```

## Fases de Desarrollo

- [x] Fase 1: Conexión y escaneo de mercados
- [ ] Fase 2: Recopilación de datos y contexto
- [ ] Fase 3: Motor de evaluación de probabilidades (LLM)
- [ ] Fase 4: Estrategia de decisión
- [ ] Fase 5: Gestión de riesgo y portafolio
- [ ] Fase 6: Motor de ejecución
- [ ] Fase 7: Monitoreo y alertas
- [ ] Fase 8: Orquestador y despliegue

## Aviso Legal

Proyecto experimental y educativo. Los mercados de predicción implican riesgo de pérdida total. No existe garantía de rentabilidad.
