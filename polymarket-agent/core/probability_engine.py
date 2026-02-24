"""
Motor de evaluación de probabilidades usando LLM (Claude) — async.

El cerebro del agente: evalúa la probabilidad real de cada evento
y la compara con el precio del mercado para detectar oportunidades.

Fase 3 del agente autónomo — refactorizado a asyncio con caché TTL.
"""

import asyncio
import json
import logging
import time
from datetime import datetime

from pydantic import ValidationError

from config import settings
from core.models import LLMEvaluation, Market
from core.data_collector import MarketContext
from core.net_utils import retry_async, TTLCache, RETRIABLE_EXCEPTIONS

logger = logging.getLogger(__name__)

# Caché en memoria para evaluaciones del LLM (60 min TTL)
_llm_cache = TTLCache(ttl_seconds=3600, max_size=2000, price_threshold=0.02)

# Prompt del sistema para el LLM evaluador
SYSTEM_PROMPT = """Eres un analista experto en mercados de predicción con formación en \
estadística bayesiana, análisis geopolítico y evaluación de probabilidades.

Tu tarea es evaluar la probabilidad REAL de que un evento ocurra, independientemente \
de lo que el mercado esté priceando actualmente.

REGLAS:
1. Sé calibrado: si dices 70%, debería ocurrir ~70% de las veces
2. Considera la información base (base rates) del tipo de evento
3. Pondera las fuentes por confiabilidad y recencia
4. Identifica sesgos comunes del mercado que podrían crear oportunidad
5. Si no tienes suficiente información, asigna confianza "low"
6. Nunca ancles tu estimación al precio actual del mercado

RESPONDE EXCLUSIVAMENTE en este formato JSON (sin markdown, sin texto adicional):
{
  "probability": 0.XX,
  "confidence": "high|medium|low",
  "reasoning": "Explicación concisa de tu evaluación en 2-3 oraciones",
  "key_factors": ["factor1", "factor2", "factor3"],
  "risks_to_thesis": ["riesgo1", "riesgo2"],
  "information_quality": "sufficient|limited|poor"
}"""


def _construir_prompt_usuario(contexto: MarketContext) -> str:
    """
    Construye el prompt de usuario con toda la información del mercado.

    Incluye datos del mercado, noticias, historial y sentimiento
    pero NO revela el precio actual para evitar anclaje.
    """
    partes: list[str] = []

    partes.append(f"PREGUNTA DEL MERCADO: {contexto.market.question}")

    if contexto.market.description:
        partes.append(f"\nDESCRIPCIÓN: {contexto.market.description[:500]}")

    if contexto.market.category:
        partes.append(f"CATEGORÍA: {contexto.market.category}")

    if contexto.market.days_to_resolution is not None:
        partes.append(
            f"TIEMPO HASTA RESOLUCIÓN: {contexto.market.days_to_resolution} días"
        )

    # Métricas de mercado (sin revelar precios para evitar anclaje)
    partes.append(
        f"\nMÉTRICAS DE MERCADO:"
        f"\n  Volumen 24h: ${contexto.market.volume_24h:,.0f}"
        f"\n  Liquidez: ${contexto.market.liquidity:,.0f}"
    )

    # Historial de precios (tendencia, no precios exactos)
    if contexto.price_history:
        ph = contexto.price_history
        partes.append(f"\nTENDENCIA DE PRECIO:")
        partes.append(f"  24h: {ph.trend_24h.value} ({ph.change_24h_pct:+.1f}%)")
        partes.append(f"  7d: {ph.trend_7d.value} ({ph.change_7d_pct:+.1f}%)")
        partes.append(f"  Volatilidad: {ph.volatility:.3f}")
        if ph.sharp_movements:
            partes.append(f"  Movimientos bruscos recientes: {len(ph.sharp_movements)}")

    # Sentimiento
    if contexto.sentiment:
        partes.append(
            f"\nSENTIMIENTO DE NOTICIAS: {contexto.sentiment.label} "
            f"(score: {contexto.sentiment.score:+.2f}, "
            f"confianza: {contexto.sentiment.confidence:.0%})"
        )
        if contexto.sentiment.positive_signals:
            partes.append(
                f"  Positivo: {', '.join(contexto.sentiment.positive_signals[:3])}"
            )
        if contexto.sentiment.negative_signals:
            partes.append(
                f"  Negativo: {', '.join(contexto.sentiment.negative_signals[:3])}"
            )

    # Noticias relevantes
    if contexto.news_articles:
        partes.append(f"\nNOTICIAS RELEVANTES ({len(contexto.news_articles)}):")
        for i, art in enumerate(contexto.news_articles[:7], 1):
            fecha = (
                art.published_date.strftime("%Y-%m-%d")
                if art.published_date
                else "N/A"
            )
            partes.append(f"  {i}. [{fecha}] {art.title}")
            if art.summary:
                partes.append(f"     {art.summary[:200]}")
    else:
        partes.append("\nNOTICIAS: No se encontraron noticias relevantes recientes.")

    partes.append(
        f"\nCALIDAD DE DATOS DISPONIBLES: {contexto.data_quality}"
    )

    partes.append(
        "\nEvalúa la probabilidad de que esta pregunta se resuelva como YES."
    )

    return "\n".join(partes)


class ProbabilityEngine:
    """
    Motor de evaluación de probabilidades usando Claude API (async).

    Envía contexto estructurado al LLM y parsea la respuesta
    en formato JSON validado. Usa caché TTL de 60 min para evitar
    re-evaluar mercados cuyo precio no cambió significativamente.

    Incluye circuit breaker: tras ``_CB_THRESHOLD`` fallos consecutivos
    de la API, pausa las llamadas por ``_CB_COOLDOWN_SECONDS`` segundos
    y retorna None (HOLD) para todas las evaluaciones.
    """

    # Circuit breaker: 5 fallos consecutivos → pausa 5 minutos
    _CB_THRESHOLD = 5
    _CB_COOLDOWN_SECONDS = 300

    def __init__(self) -> None:
        self._config = settings.anthropic
        self._client = self._crear_cliente()
        self._evaluaciones_log: list[dict] = []
        # Circuit breaker state
        self._cb_failures: int = 0
        self._cb_open_until: float = 0.0  # monotonic timestamp

    def _crear_cliente(self) -> "anthropic.AsyncAnthropic":
        """Crea el cliente async de Anthropic con la API key configurada."""
        import anthropic
        return anthropic.AsyncAnthropic(api_key=self._config.api_key)

    async def evaluar_mercado(
        self, contexto: MarketContext
    ) -> LLMEvaluation | None:
        """
        Evalúa la probabilidad de un mercado usando el LLM.

        Primero consulta el caché TTL. Si no hay hit (o el precio
        cambió > 2%), llama al LLM y almacena el resultado.

        Args:
            contexto: Contexto completo del mercado (datos, noticias, sentimiento).

        Returns:
            LLMEvaluation con la probabilidad estimada, o None si falla.
        """
        market_id = contexto.market.condition_id
        current_price = contexto.market.yes_price

        # Circuit breaker: si está abierto, no llamar al LLM
        if self._cb_is_open():
            logger.warning(
                f"Circuit breaker ABIERTO — saltando LLM para {market_id[:12]}"
            )
            return None

        # Consultar caché
        cached = _llm_cache.get(market_id, current_price)
        if cached is not None:
            logger.info(
                f"Cache HIT para {market_id[:12]} "
                f"(stats: {_llm_cache.stats})"
            )
            return cached

        logger.info(
            f"Evaluando mercado: '{contexto.market.question[:50]}...'"
        )

        prompt_usuario = _construir_prompt_usuario(contexto)

        # Intentar obtener evaluación del LLM con reintentos
        try:
            respuesta_raw = await self._llamar_llm(prompt_usuario)
        except Exception:
            self._cb_record_failure()
            return None

        if respuesta_raw is None:
            self._cb_record_failure()
            logger.error("No se pudo obtener respuesta del LLM")
            return None

        # Éxito: resetear circuit breaker
        self._cb_record_success()

        # Parsear y validar la respuesta JSON
        evaluacion = self._parsear_respuesta(respuesta_raw)

        if evaluacion is None:
            logger.error("No se pudo parsear la respuesta del LLM")
            return None

        # Sanity checks
        evaluacion = self._aplicar_sanity_checks(evaluacion, contexto)

        # Almacenar en caché
        _llm_cache.put(market_id, evaluacion, current_price)

        # Loggear para auditoría
        self._registrar_evaluacion(contexto, evaluacion, prompt_usuario)

        logger.info(
            f"Evaluación: prob={evaluacion.probability:.2f}, "
            f"conf={evaluacion.confidence}, "
            f"info={evaluacion.information_quality}"
        )

        return evaluacion

    async def evaluar_multiples(
        self, contextos: list[MarketContext]
    ) -> list[tuple[MarketContext, LLMEvaluation | None]]:
        """
        Evalúa múltiples mercados en paralelo con asyncio.gather.

        Args:
            contextos: Lista de contextos de mercado.

        Returns:
            Lista de tuplas (contexto, evaluación).
        """
        async def _evaluar_uno(
            ctx: MarketContext,
        ) -> tuple[MarketContext, LLMEvaluation | None]:
            evaluacion = await self.evaluar_mercado(ctx)
            return (ctx, evaluacion)

        resultados = await asyncio.gather(
            *[_evaluar_uno(ctx) for ctx in contextos]
        )

        exitosos = sum(1 for _, e in resultados if e is not None)
        logger.info(
            f"Evaluaciones completadas: {exitosos}/{len(contextos)} exitosas "
            f"(cache stats: {_llm_cache.stats})"
        )

        return list(resultados)

    @property
    def historial_evaluaciones(self) -> list[dict]:
        """Retorna el historial de evaluaciones para auditoría."""
        return self._evaluaciones_log.copy()

    @property
    def cache_stats(self) -> dict[str, int]:
        """Retorna estadísticas del caché."""
        return _llm_cache.stats

    # =========================================================================
    # Criterio de Kelly — Position Sizing
    # =========================================================================

    @staticmethod
    def calcular_kelly_size(
        p: float,
        yes_price: float,
        bankroll: float,
        kelly_fraction: float = 0.25,
        max_pct: float = 0.10,
    ) -> tuple[float, float]:
        """
        Calcula el tamaño óptimo de apuesta con el Criterio de Kelly.

        Fórmula: f* = (p(b+1) - 1) / b
        Donde b = (1/price) - 1 son las odds netas del mercado binario.

        Aplica "Fractional Kelly" (0.25 por defecto) para reducir la
        volatilidad del bankroll, y un cap duro de `max_pct` del capital
        total por trade independientemente del resultado de Kelly.

        Args:
            p:              Probabilidad estimada por el LLM (0-1).
            yes_price:      Precio actual del token YES (0-1, = 1/odds_brutas).
            bankroll:       Capital total disponible en USD.
            kelly_fraction: Factor fraccional de Kelly (default 0.25 = Quarter Kelly).
            max_pct:        Porcentaje máximo del bankroll por trade (default 0.10 = 10%).

        Returns:
            Tupla ``(kelly_frac, size_usd)`` donde ``kelly_frac`` es la
            fracción del bankroll a apostar y ``size_usd`` el monto en USD.
            Ambos son 0.0 si no existe edge o los inputs son inválidos.
        """
        if not (0 < p < 1) or not (0 < yes_price < 1) or bankroll <= 0:
            return 0.0, 0.0

        # Odds netas: cuánto ganamos por cada dólar apostado si acierta
        b = (1.0 / yes_price) - 1.0
        if b <= 0:
            return 0.0, 0.0

        # Criterio de Kelly completo: f* = (p(b+1) - 1) / b
        kelly_full = (p * (b + 1.0) - 1.0) / b

        if kelly_full <= 0:
            # Edge negativo: no apostar
            return 0.0, 0.0

        # Quarter Kelly (reduce varianza del bankroll ~4×)
        kelly_frac = kelly_full * kelly_fraction

        # Cap duro: nunca arriesgar más del max_pct del capital total
        kelly_frac = min(kelly_frac, max_pct)

        size_usd = round(bankroll * kelly_frac, 2)
        return round(kelly_frac, 6), size_usd

    # =========================================================================
    # Métodos privados - LLM
    # =========================================================================

    @retry_async(max_attempts=3, base_delay=2.0, max_delay=15.0)
    async def _llamar_llm(self, prompt_usuario: str) -> str | None:
        """
        Llama a la API de Claude (async) y retorna la respuesta como texto.

        Usa reintentos automáticos con backoff exponencial.
        """
        try:
            message = await self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt_usuario},
                ],
            )

            # Extraer texto de la respuesta
            if message.content and len(message.content) > 0:
                return message.content[0].text

            logger.warning("Respuesta del LLM vacía")
            return None

        except Exception as e:
            logger.error(f"Error llamando al LLM: {e}")
            raise

    # =========================================================================
    # Métodos privados - Parsing
    # =========================================================================

    @staticmethod
    def _parsear_respuesta(respuesta_raw: str) -> LLMEvaluation | None:
        """
        Parsea la respuesta JSON del LLM en un LLMEvaluation.

        Maneja casos donde el LLM incluye markdown o texto adicional.
        """
        # Intentar extraer JSON del texto
        json_str = _extraer_json(respuesta_raw)

        if not json_str:
            logger.warning(
                f"No se encontró JSON en la respuesta: {respuesta_raw[:200]}"
            )
            return None

        try:
            data = json.loads(json_str)
            return LLMEvaluation(**data)
        except json.JSONDecodeError as e:
            logger.warning(f"Error parseando JSON: {e}")
            return None
        except ValidationError as e:
            logger.warning(f"Error validando evaluación: {e}")
            return None

    @staticmethod
    def _aplicar_sanity_checks(
        evaluacion: LLMEvaluation,
        contexto: MarketContext,
    ) -> LLMEvaluation:
        """
        Aplica verificaciones de cordura a la evaluación del LLM.

        Previene evaluaciones absurdas o claramente erradas.
        """
        prob = evaluacion.probability

        # Check 1: Probabilidades extremas (< 0.02 o > 0.98) necesitan
        # alta confianza y buena información
        if (prob < 0.02 or prob > 0.98) and evaluacion.confidence != "high":
            logger.warning(
                f"Probabilidad extrema ({prob}) con confianza "
                f"{evaluacion.confidence} - ajustando confianza a 'low'"
            )
            evaluacion = evaluacion.model_copy(
                update={"confidence": "low"}
            )

        # Check 2: Si la calidad de información es "poor",
        # forzar confianza a "low"
        if evaluacion.information_quality == "poor":
            if evaluacion.confidence != "low":
                logger.warning(
                    "Info quality 'poor' pero confianza no es 'low' - ajustando"
                )
                evaluacion = evaluacion.model_copy(
                    update={"confidence": "low"}
                )

        # Check 3: Si no hay noticias ni historial,
        # la calidad no puede ser "sufficient"
        if (
            contexto.data_quality == "poor"
            and evaluacion.information_quality == "sufficient"
        ):
            logger.warning(
                "Data quality 'poor' pero LLM dice info 'sufficient' - ajustando"
            )
            evaluacion = evaluacion.model_copy(
                update={"information_quality": "limited"}
            )

        return evaluacion

    # =========================================================================
    # Circuit breaker
    # =========================================================================

    def _cb_is_open(self) -> bool:
        """True si el circuit breaker está abierto (API en pausa)."""
        if self._cb_failures < self._CB_THRESHOLD:
            return False
        if time.monotonic() >= self._cb_open_until:
            # Cooldown expiró: half-open, permitir un intento
            logger.info("Circuit breaker: cooldown expiró, intentando reconectar LLM")
            self._cb_failures = self._CB_THRESHOLD - 1
            return False
        return True

    def _cb_record_failure(self) -> None:
        """Registra un fallo de la API."""
        self._cb_failures += 1
        if self._cb_failures >= self._CB_THRESHOLD:
            self._cb_open_until = time.monotonic() + self._CB_COOLDOWN_SECONDS
            logger.error(
                f"Circuit breaker ABIERTO: {self._cb_failures} fallos consecutivos. "
                f"LLM pausado por {self._CB_COOLDOWN_SECONDS}s"
            )

    def _cb_record_success(self) -> None:
        """Registra un éxito de la API, reseteando el circuit breaker."""
        if self._cb_failures > 0:
            logger.info(
                f"Circuit breaker: LLM OK, reseteando contador "
                f"(era {self._cb_failures})"
            )
        self._cb_failures = 0
        self._cb_open_until = 0.0

    # =========================================================================
    # Invalidación de caché
    # =========================================================================

    @staticmethod
    def invalidar_cache_mercado(market_id: str) -> bool:
        """Invalida la caché del LLM para un mercado resuelto."""
        return _llm_cache.invalidate(market_id)

    def _registrar_evaluacion(
        self,
        contexto: MarketContext,
        evaluacion: LLMEvaluation,
        prompt: str,
    ) -> None:
        """Registra la evaluación para auditoría posterior."""
        registro = {
            "timestamp": datetime.now().isoformat(),
            "market_id": contexto.market.condition_id,
            "question": contexto.market.question,
            "market_yes_price": contexto.market.yes_price,
            "estimated_probability": evaluacion.probability,
            "confidence": evaluacion.confidence,
            "edge": evaluacion.probability - contexto.market.yes_price,
            "reasoning": evaluacion.reasoning,
            "data_quality": contexto.data_quality,
            "info_quality": evaluacion.information_quality,
        }
        self._evaluaciones_log.append(registro)

        logger.debug(
            f"Evaluación registrada: mercado={contexto.market.condition_id[:12]}, "
            f"prob={evaluacion.probability:.2f}, "
            f"edge={registro['edge']:+.2f}"
        )


# =============================================================================
# Funciones auxiliares
# =============================================================================

def _extraer_json(texto: str) -> str | None:
    """
    Extrae un bloque JSON de un texto que puede contener markdown u otro contenido.

    El LLM a veces envuelve el JSON en backticks o agrega texto.
    """
    # Intentar el texto completo primero
    texto_limpio = texto.strip()
    if texto_limpio.startswith("{") and texto_limpio.endswith("}"):
        return texto_limpio

    # Buscar JSON entre backticks de markdown
    import re

    # Patrón: ```json ... ``` o ``` ... ```
    patron_markdown = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    match = patron_markdown.search(texto)
    if match:
        return match.group(1)

    # Buscar el primer { y último }
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        return texto[inicio : fin + 1]

    return None
