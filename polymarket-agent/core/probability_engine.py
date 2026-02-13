"""
Motor de evaluación de probabilidades usando LLM (Claude).

El cerebro del agente: evalúa la probabilidad real de cada evento
y la compara con el precio del mercado para detectar oportunidades.

Fase 3 del agente autónomo.
"""

import json
import logging
from datetime import datetime

from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import settings
from core.models import LLMEvaluation, Market
from core.data_collector import MarketContext

logger = logging.getLogger(__name__)

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
    Motor de evaluación de probabilidades usando Claude API.

    Envía contexto estructurado al LLM y parsea la respuesta
    en formato JSON validado.
    """

    def __init__(self) -> None:
        self._config = settings.anthropic
        self._client = self._crear_cliente()
        self._evaluaciones_log: list[dict] = []

    def _crear_cliente(self) -> "anthropic.Anthropic":
        """Crea el cliente de Anthropic con la API key configurada."""
        import anthropic
        return anthropic.Anthropic(api_key=self._config.api_key)

    def evaluar_mercado(
        self, contexto: MarketContext
    ) -> LLMEvaluation | None:
        """
        Evalúa la probabilidad de un mercado usando el LLM.

        Args:
            contexto: Contexto completo del mercado (datos, noticias, sentimiento).

        Returns:
            LLMEvaluation con la probabilidad estimada, o None si falla.
        """
        logger.info(
            f"Evaluando mercado: '{contexto.market.question[:50]}...'"
        )

        prompt_usuario = _construir_prompt_usuario(contexto)

        # Intentar obtener evaluación del LLM con reintentos
        respuesta_raw = self._llamar_llm(prompt_usuario)

        if respuesta_raw is None:
            logger.error("No se pudo obtener respuesta del LLM")
            return None

        # Parsear y validar la respuesta JSON
        evaluacion = self._parsear_respuesta(respuesta_raw)

        if evaluacion is None:
            logger.error("No se pudo parsear la respuesta del LLM")
            return None

        # Sanity checks
        evaluacion = self._aplicar_sanity_checks(evaluacion, contexto)

        # Loggear para auditoría
        self._registrar_evaluacion(contexto, evaluacion, prompt_usuario)

        logger.info(
            f"Evaluación: prob={evaluacion.probability:.2f}, "
            f"conf={evaluacion.confidence}, "
            f"info={evaluacion.information_quality}"
        )

        return evaluacion

    def evaluar_multiples(
        self, contextos: list[MarketContext]
    ) -> list[tuple[MarketContext, LLMEvaluation | None]]:
        """
        Evalúa múltiples mercados secuencialmente.

        Args:
            contextos: Lista de contextos de mercado.

        Returns:
            Lista de tuplas (contexto, evaluación).
        """
        resultados: list[tuple[MarketContext, LLMEvaluation | None]] = []

        for i, ctx in enumerate(contextos, 1):
            logger.info(f"Evaluando mercado {i}/{len(contextos)}...")
            evaluacion = self.evaluar_mercado(ctx)
            resultados.append((ctx, evaluacion))

        exitosos = sum(1 for _, e in resultados if e is not None)
        logger.info(
            f"Evaluaciones completadas: {exitosos}/{len(contextos)} exitosas"
        )

        return resultados

    @property
    def historial_evaluaciones(self) -> list[dict]:
        """Retorna el historial de evaluaciones para auditoría."""
        return self._evaluaciones_log.copy()

    # =========================================================================
    # Métodos privados - LLM
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Reintentando llamada LLM (intento {retry_state.attempt_number})..."
        ),
    )
    def _llamar_llm(self, prompt_usuario: str) -> str | None:
        """
        Llama a la API de Claude y retorna la respuesta como texto.

        Usa reintentos automáticos con backoff exponencial.
        """
        try:
            message = self._client.messages.create(
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


# =============================================================================
# Ejecución directa para pruebas
# =============================================================================

def main() -> None:
    """Prueba el ProbabilityEngine con un mercado de ejemplo."""
    from config import configurar_logging
    from core.models import Token
    from core.data_collector import MarketContext
    from data.news_fetcher import NewsArticle
    from data.sentiment import SentimentResult
    from rich.console import Console
    from rich.panel import Panel

    configurar_logging()
    console = Console()

    console.print("\n[bold cyan]Probability Engine - Fase 3[/bold cyan]\n")

    # Crear contexto de ejemplo
    mercado = Market(
        condition_id="0x_test_123",
        question="Will Bitcoin reach $100,000 by December 2026?",
        description="Market resolves YES if BTC price reaches 100k USD.",
        tokens=[
            Token(token_id="tk_yes", outcome="Yes", price=0.65),
            Token(token_id="tk_no", outcome="No", price=0.35),
        ],
        volume_24h=50000,
        liquidity=100000,
        yes_price=0.65,
        no_price=0.35,
        category="crypto",
    )

    noticias = [
        NewsArticle(
            title="Bitcoin approaching $95k amid institutional demand",
            source="Reuters",
            summary="Bitcoin continues to rally with strong buying pressure.",
            published_date=datetime.now(),
            relevance_score=0.9,
        ),
    ]

    contexto = MarketContext(
        market=mercado,
        news_articles=noticias,
        sentiment=SentimentResult(
            score=0.3, label="positive", confidence=0.6,
        ),
        data_quality="partial",
    )

    engine = ProbabilityEngine()
    evaluacion = engine.evaluar_mercado(contexto)

    if evaluacion:
        edge = evaluacion.probability - mercado.yes_price
        console.print(Panel(
            f"Probabilidad estimada: {evaluacion.probability:.2f}\n"
            f"Confianza: {evaluacion.confidence}\n"
            f"Precio del mercado: {mercado.yes_price:.2f}\n"
            f"Edge: {edge:+.2f}\n"
            f"Razonamiento: {evaluacion.reasoning}\n"
            f"Factores clave: {', '.join(evaluacion.key_factors)}\n"
            f"Riesgos: {', '.join(evaluacion.risks_to_thesis)}",
            title="Evaluación del LLM",
            border_style="green" if edge > 0 else "red",
        ))
    else:
        console.print("[red]No se pudo obtener evaluación del LLM[/red]")


if __name__ == "__main__":
    main()
