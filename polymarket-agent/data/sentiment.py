"""
Analizador de sentimiento para noticias relacionadas a mercados.

Analiza el sentimiento general de noticias recopiladas usando
un enfoque basado en lexicón (sin dependencias externas de ML)
y opcionalmente un LLM para análisis más profundo.

Fase 2 del agente autónomo.
"""

import logging
import re
from datetime import datetime

from pydantic import BaseModel, Field

from data.news_fetcher import NewsArticle

logger = logging.getLogger(__name__)


class SentimentResult(BaseModel):
    """Resultado del análisis de sentimiento."""
    score: float = Field(
        ge=-1, le=1,
        description="Score de sentimiento (-1=muy negativo, 0=neutral, 1=muy positivo)"
    )
    label: str = Field(
        description="Etiqueta: 'very_negative', 'negative', 'neutral', 'positive', 'very_positive'"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Confianza en el análisis (0-1)"
    )
    justification: str = Field(
        default="", description="Justificación del score"
    )
    articles_analyzed: int = Field(
        default=0, description="Número de artículos analizados"
    )
    positive_signals: list[str] = Field(
        default_factory=list, description="Señales positivas encontradas"
    )
    negative_signals: list[str] = Field(
        default_factory=list, description="Señales negativas encontradas"
    )


# =============================================================================
# Lexicones de sentimiento
# =============================================================================

# Palabras positivas en contexto de mercados de predicción
POSITIVE_LEXICON: dict[str, float] = {
    # Certeza / Probabilidad alta
    "confirmed": 0.8, "confirms": 0.8, "officially": 0.7,
    "announced": 0.5, "approved": 0.7, "passed": 0.6,
    "signed": 0.6, "enacted": 0.7, "ratified": 0.7,
    # Progreso
    "progress": 0.4, "advancing": 0.5, "momentum": 0.5,
    "growing": 0.4, "increasing": 0.4, "rising": 0.4,
    "surge": 0.6, "soaring": 0.6, "rally": 0.5,
    # Apoyo
    "support": 0.3, "backs": 0.4, "endorses": 0.5,
    "endorsement": 0.5, "backed": 0.4, "supporting": 0.3,
    # Éxito
    "wins": 0.7, "won": 0.7, "victory": 0.7, "success": 0.6,
    "succeeds": 0.6, "breakthrough": 0.6, "achieved": 0.5,
    # Positivo general
    "likely": 0.4, "expected": 0.3, "positive": 0.4,
    "optimistic": 0.5, "confident": 0.4, "strong": 0.3,
    "leads": 0.4, "leading": 0.4, "ahead": 0.4,
    "boost": 0.4, "boosted": 0.4, "improved": 0.4,
    "agreement": 0.5, "deal": 0.4, "consensus": 0.4,
}

# Palabras negativas en contexto de mercados de predicción
NEGATIVE_LEXICON: dict[str, float] = {
    # Incertidumbre / Probabilidad baja
    "denied": -0.7, "denies": -0.7, "rejected": -0.7,
    "unlikely": -0.5, "doubtful": -0.5, "uncertain": -0.3,
    "failed": -0.7, "fails": -0.7, "failure": -0.7,
    # Retroceso
    "declining": -0.4, "falling": -0.4, "dropped": -0.5,
    "plummeted": -0.7, "crashed": -0.7, "collapse": -0.7,
    "slump": -0.5, "downturn": -0.5, "recession": -0.5,
    # Oposición
    "opposes": -0.4, "opposition": -0.4, "against": -0.3,
    "blocked": -0.5, "vetoed": -0.6, "challenged": -0.3,
    # Conflicto / Crisis
    "crisis": -0.5, "conflict": -0.4, "war": -0.4,
    "threat": -0.4, "threatens": -0.4, "risk": -0.3,
    "scandal": -0.5, "controversy": -0.4, "investigation": -0.3,
    # Negativo general
    "loses": -0.6, "lost": -0.5, "defeat": -0.6,
    "delays": -0.4, "delayed": -0.4, "postponed": -0.4,
    "suspended": -0.5, "cancelled": -0.6, "canceled": -0.6,
    "withdrawn": -0.5, "abandoned": -0.6, "resigned": -0.4,
    "weak": -0.3, "weakened": -0.4, "struggling": -0.4,
    "fears": -0.4, "concern": -0.3, "worried": -0.4,
    "criticism": -0.3, "criticized": -0.3, "warns": -0.3,
}

# Modificadores de intensidad
INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "extremely": 2.0, "highly": 1.5,
    "significantly": 1.5, "dramatically": 1.8,
    "sharply": 1.5, "strongly": 1.5,
    "slightly": 0.5, "marginally": 0.5, "somewhat": 0.7,
}

# Negaciones que invierten el sentimiento
NEGATIONS: set[str] = {
    "not", "no", "never", "neither", "nor", "hardly",
    "barely", "scarcely", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "couldn't", "shouldn't", "isn't",
    "aren't", "wasn't", "weren't",
}


class SentimentAnalyzer:
    """
    Analiza el sentimiento de noticias relacionadas a un mercado.

    Usa un enfoque basado en lexicón para análisis rápido sin
    dependencias externas de ML. El score resultante combina:
    - Análisis de palabras clave positivas/negativas
    - Detección de intensificadores y negaciones
    - Ponderación por relevancia del artículo
    """

    def analizar(
        self,
        articles: list[NewsArticle],
        market_question: str = "",
    ) -> SentimentResult:
        """
        Analiza el sentimiento de una lista de artículos de noticias.

        Args:
            articles: Lista de artículos a analizar.
            market_question: Pregunta del mercado (contexto adicional).

        Returns:
            Resultado del análisis de sentimiento.
        """
        if not articles:
            return SentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.0,
                justification="No hay artículos para analizar",
                articles_analyzed=0,
            )

        logger.info(f"Analizando sentimiento de {len(articles)} artículos...")

        scores: list[float] = []
        pesos: list[float] = []
        señales_positivas: list[str] = []
        señales_negativas: list[str] = []

        for article in articles:
            # Analizar cada artículo
            texto = f"{article.title}. {article.summary}"
            resultado = self._analizar_texto(texto)

            # Ponderar por relevancia del artículo
            peso = max(0.1, article.relevance_score)
            scores.append(resultado["score"])
            pesos.append(peso)

            # Recopilar señales
            if resultado["score"] > 0.1:
                señales_positivas.extend(
                    f"{article.source}: {s}" for s in resultado["positive_words"][:2]
                )
            elif resultado["score"] < -0.1:
                señales_negativas.extend(
                    f"{article.source}: {s}" for s in resultado["negative_words"][:2]
                )

        # Calcular score ponderado
        score_total = sum(s * w for s, w in zip(scores, pesos))
        peso_total = sum(pesos)
        score_final = score_total / peso_total if peso_total > 0 else 0.0

        # Limitar entre -1 y 1
        score_final = max(-1.0, min(1.0, score_final))

        # Determinar etiqueta
        label = self._score_a_label(score_final)

        # Calcular confianza basada en cantidad y consistencia
        confianza = self._calcular_confianza(scores, len(articles))

        # Generar justificación
        justificacion = self._generar_justificacion(
            score_final, label, len(articles),
            señales_positivas[:5], señales_negativas[:5],
        )

        return SentimentResult(
            score=round(score_final, 3),
            label=label,
            confidence=round(confianza, 2),
            justification=justificacion,
            articles_analyzed=len(articles),
            positive_signals=señales_positivas[:5],
            negative_signals=señales_negativas[:5],
        )

    # =========================================================================
    # Análisis de texto
    # =========================================================================

    def _analizar_texto(self, texto: str) -> dict:
        """
        Analiza el sentimiento de un texto individual.

        Retorna dict con score, palabras positivas y negativas encontradas.
        """
        palabras = self._tokenizar(texto)
        score = 0.0
        positivas_encontradas: list[str] = []
        negativas_encontradas: list[str] = []

        i = 0
        while i < len(palabras):
            palabra = palabras[i]

            # Verificar si la palabra anterior era una negación
            is_negated = (
                i > 0 and palabras[i - 1] in NEGATIONS
            )

            # Verificar si hay un intensificador antes
            intensifier = 1.0
            if i > 0 and palabras[i - 1] in INTENSIFIERS:
                intensifier = INTENSIFIERS[palabras[i - 1]]
            if i > 1 and palabras[i - 2] in INTENSIFIERS:
                intensifier = INTENSIFIERS[palabras[i - 2]]

            # Buscar en lexicones
            if palabra in POSITIVE_LEXICON:
                valor = POSITIVE_LEXICON[palabra] * intensifier
                if is_negated:
                    valor = -valor * 0.5  # Negación reduce e invierte
                    negativas_encontradas.append(f"not {palabra}")
                else:
                    positivas_encontradas.append(palabra)
                score += valor

            elif palabra in NEGATIVE_LEXICON:
                valor = NEGATIVE_LEXICON[palabra] * intensifier
                if is_negated:
                    valor = -valor * 0.5  # Negación reduce e invierte
                    positivas_encontradas.append(f"not {palabra}")
                else:
                    negativas_encontradas.append(palabra)
                score += valor

            i += 1

        # Normalizar por longitud del texto
        num_palabras = max(len(palabras), 1)
        score_normalizado = score / (num_palabras ** 0.5)  # Raíz cuadrada para suavizar

        return {
            "score": max(-1.0, min(1.0, score_normalizado)),
            "positive_words": positivas_encontradas,
            "negative_words": negativas_encontradas,
        }

    @staticmethod
    def _tokenizar(texto: str) -> list[str]:
        """Tokeniza y normaliza texto a palabras en minúsculas."""
        texto_limpio = re.sub(r"[^\w\s'-]", " ", texto.lower())
        return texto_limpio.split()

    @staticmethod
    def _score_a_label(score: float) -> str:
        """Convierte un score numérico a una etiqueta de sentimiento."""
        if score >= 0.3:
            return "very_positive"
        if score >= 0.1:
            return "positive"
        if score <= -0.3:
            return "very_negative"
        if score <= -0.1:
            return "negative"
        return "neutral"

    @staticmethod
    def _calcular_confianza(scores: list[float], n_articles: int) -> float:
        """
        Calcula la confianza del análisis.

        Mayor confianza cuando:
        - Más artículos analizados
        - Mayor consistencia entre artículos
        """
        if not scores:
            return 0.0

        # Factor de cantidad (más artículos = más confianza)
        factor_cantidad = min(1.0, n_articles / 10)

        # Factor de consistencia (menor varianza = más confianza)
        mean = sum(scores) / len(scores)
        if len(scores) > 1:
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            factor_consistencia = max(0, 1 - variance * 2)
        else:
            factor_consistencia = 0.5

        return factor_cantidad * 0.5 + factor_consistencia * 0.5

    @staticmethod
    def _generar_justificacion(
        score: float,
        label: str,
        n_articles: int,
        positivas: list[str],
        negativas: list[str],
    ) -> str:
        """Genera una justificación legible del análisis."""
        partes = [
            f"Análisis de {n_articles} artículo(s). "
            f"Sentimiento: {label} (score: {score:+.2f})."
        ]

        if positivas:
            partes.append(f"Señales positivas: {', '.join(positivas[:3])}.")
        if negativas:
            partes.append(f"Señales negativas: {', '.join(negativas[:3])}.")

        if not positivas and not negativas:
            partes.append(
                "No se encontraron señales claras de sentimiento."
            )

        return " ".join(partes)


# =============================================================================
# Ejecución directa para pruebas
# =============================================================================

def main() -> None:
    """Prueba el SentimentAnalyzer con artículos de ejemplo."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    analyzer = SentimentAnalyzer()

    # Artículos de prueba
    articulos_positivos = [
        NewsArticle(
            title="Bitcoin surges past $95,000 as institutional demand grows",
            source="Reuters",
            summary="Bitcoin has rallied significantly, supported by strong institutional buying and positive regulatory signals.",
            relevance_score=0.8,
        ),
        NewsArticle(
            title="Major investment firm confirms Bitcoin ETF approval expected",
            source="Bloomberg",
            summary="Leading financial experts are optimistic about Bitcoin's trajectory, citing growing mainstream adoption.",
            relevance_score=0.9,
        ),
    ]

    articulos_negativos = [
        NewsArticle(
            title="Regulators threaten to block cryptocurrency trading",
            source="WSJ",
            summary="Government officials warn of crisis in crypto markets, investigation launched into major exchanges.",
            relevance_score=0.7,
        ),
    ]

    console.print("\n[bold cyan]Sentiment Analyzer - Fase 2[/bold cyan]\n")

    # Test positivo
    resultado_pos = analyzer.analizar(
        articulos_positivos,
        "Will Bitcoin reach $100k?"
    )
    console.print(Panel(
        f"Score: {resultado_pos.score:+.3f}\n"
        f"Label: {resultado_pos.label}\n"
        f"Confianza: {resultado_pos.confidence:.0%}\n"
        f"Justificación: {resultado_pos.justification}",
        title="Sentimiento Positivo",
        border_style="green",
    ))

    # Test negativo
    resultado_neg = analyzer.analizar(
        articulos_negativos,
        "Will Bitcoin reach $100k?"
    )
    console.print(Panel(
        f"Score: {resultado_neg.score:+.3f}\n"
        f"Label: {resultado_neg.label}\n"
        f"Confianza: {resultado_neg.confidence:.0%}\n"
        f"Justificación: {resultado_neg.justification}",
        title="Sentimiento Negativo",
        border_style="red",
    ))

    # Test vacío
    resultado_vacio = analyzer.analizar([], "Test")
    console.print(Panel(
        f"Score: {resultado_vacio.score:+.3f} | Label: {resultado_vacio.label}",
        title="Sin artículos",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
