"""
Recopilador de datos integrado para mercados de predicción.

Orquesta la recopilación de noticias, historial de precios y
análisis de sentimiento para alimentar al motor de probabilidades.

Fase 2 del agente autónomo.
"""

import logging
from datetime import datetime

from pydantic import BaseModel, Field

from core.models import Market
from data.news_fetcher import NewsFetcher, NewsArticle
from data.market_history import MarketHistory, MarketHistoryAnalysis
from data.sentiment import SentimentAnalyzer, SentimentResult

logger = logging.getLogger(__name__)


class MarketContext(BaseModel):
    """
    Contexto completo recopilado para un mercado.

    Contiene toda la información que el motor de probabilidades
    necesita para evaluar un mercado.
    """
    # Datos del mercado
    market: Market = Field(description="Datos del mercado")

    # Noticias relevantes
    news_articles: list[NewsArticle] = Field(
        default_factory=list, description="Noticias relevantes recopiladas"
    )

    # Historial de precios
    price_history: MarketHistoryAnalysis | None = Field(
        default=None, description="Análisis del historial de precios"
    )

    # Análisis de sentimiento
    sentiment: SentimentResult | None = Field(
        default=None, description="Resultado del análisis de sentimiento"
    )

    # Metadatos
    collected_at: datetime = Field(
        default_factory=datetime.now, description="Momento de recopilación"
    )
    data_quality: str = Field(
        default="unknown", description="Calidad de los datos: 'good', 'partial', 'poor'"
    )

    def resumen(self) -> str:
        """Retorna un resumen del contexto recopilado."""
        noticias = len(self.news_articles)
        sent_score = self.sentiment.score if self.sentiment else 0
        sent_label = self.sentiment.label if self.sentiment else "N/A"
        trend = self.price_history.trend_24h.value if self.price_history else "N/A"

        return (
            f"Mercado: {self.market.question[:60]}\n"
            f"  Noticias: {noticias} | Sentimiento: {sent_label} ({sent_score:+.2f})\n"
            f"  Tendencia 24h: {trend} | Calidad datos: {self.data_quality}"
        )

    def generar_resumen_para_llm(self) -> str:
        """
        Genera un resumen estructurado para enviar al LLM.

        Este texto será parte del prompt del motor de probabilidades.
        """
        partes: list[str] = []

        # Datos del mercado
        partes.append(f"PREGUNTA DEL MERCADO: {self.market.question}")
        if self.market.description:
            partes.append(f"DESCRIPCIÓN: {self.market.description[:300]}")
        partes.append(
            f"PRECIO ACTUAL: YES=${self.market.yes_price:.2f}, "
            f"NO=${self.market.no_price:.2f}"
        )
        partes.append(
            f"VOLUMEN 24H: ${self.market.volume_24h:,.0f} | "
            f"LIQUIDEZ: ${self.market.liquidity:,.0f}"
        )
        if self.market.days_to_resolution is not None:
            partes.append(
                f"RESUELVE EN: {self.market.days_to_resolution} días"
            )

        # Historial de precios
        if self.price_history:
            ph = self.price_history
            partes.append(f"\nTENDENCIA DE PRECIO:")
            partes.append(f"  - 24h: {ph.change_24h_pct:+.1f}% ({ph.trend_24h.value})")
            partes.append(f"  - 7d: {ph.change_7d_pct:+.1f}% ({ph.trend_7d.value})")
            partes.append(f"  - Volatilidad: {ph.volatility:.3f}")
            if ph.sharp_movements:
                partes.append(
                    f"  - Movimientos bruscos recientes: {len(ph.sharp_movements)}"
                )
                for mov in ph.sharp_movements[-3:]:
                    partes.append(
                        f"    {mov.direction} {mov.change_pct:+.1f}% en {mov.timestamp}"
                    )

        # Sentimiento
        if self.sentiment:
            partes.append(f"\nSENTIMIENTO DE NOTICIAS:")
            partes.append(
                f"  Score: {self.sentiment.score:+.2f} ({self.sentiment.label})"
            )
            partes.append(
                f"  Confianza: {self.sentiment.confidence:.0%}"
            )
            if self.sentiment.positive_signals:
                partes.append(
                    f"  Señales positivas: {', '.join(self.sentiment.positive_signals[:3])}"
                )
            if self.sentiment.negative_signals:
                partes.append(
                    f"  Señales negativas: {', '.join(self.sentiment.negative_signals[:3])}"
                )

        # Noticias relevantes
        if self.news_articles:
            partes.append(f"\nNOTICIAS RELEVANTES ({len(self.news_articles)}):")
            for i, art in enumerate(self.news_articles[:5], 1):
                fecha = art.published_date.strftime("%Y-%m-%d") if art.published_date else "N/A"
                partes.append(f"  {i}. [{fecha}] {art.title}")
                if art.summary:
                    partes.append(f"     {art.summary[:150]}")
        else:
            partes.append("\nNOTICIAS: No se encontraron noticias relevantes.")

        return "\n".join(partes)


class DataCollector:
    """
    Orquesta la recopilación de datos para mercados de predicción.

    Integra NewsFetcher, MarketHistory y SentimentAnalyzer para
    generar un contexto completo que alimentará al motor de
    probabilidades (Fase 3).
    """

    def __init__(self) -> None:
        self._news_fetcher = NewsFetcher()
        self._market_history = MarketHistory()
        self._sentiment_analyzer = SentimentAnalyzer()

    def close(self) -> None:
        """Cierra todas las conexiones."""
        self._news_fetcher.close()
        self._market_history.close()

    def __enter__(self) -> "DataCollector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def recopilar_contexto(
        self,
        market: Market,
        max_news: int = 10,
    ) -> MarketContext:
        """
        Recopila todo el contexto disponible para un mercado.

        Args:
            market: Mercado para el cual recopilar datos.
            max_news: Máximo de noticias a buscar.

        Returns:
            MarketContext con toda la información recopilada.
        """
        logger.info(
            f"Recopilando contexto para: '{market.question[:50]}...'"
        )

        # 1. Buscar noticias
        articles = self._recopilar_noticias(market, max_news)

        # 2. Obtener historial de precios
        price_history = self._recopilar_historial(market)

        # 3. Analizar sentimiento de las noticias
        sentiment = self._analizar_sentimiento(articles, market.question)

        # 4. Evaluar calidad de datos
        data_quality = self._evaluar_calidad(articles, price_history, sentiment)

        contexto = MarketContext(
            market=market,
            news_articles=articles,
            price_history=price_history,
            sentiment=sentiment,
            data_quality=data_quality,
        )

        logger.info(
            f"Contexto recopilado: {len(articles)} noticias, "
            f"calidad={data_quality}"
        )

        return contexto

    def recopilar_multiples(
        self,
        markets: list[Market],
        max_news_per_market: int = 5,
    ) -> list[MarketContext]:
        """
        Recopila contexto para múltiples mercados.

        Args:
            markets: Lista de mercados a investigar.
            max_news_per_market: Noticias máximas por mercado.

        Returns:
            Lista de contextos, uno por mercado.
        """
        logger.info(f"Recopilando contexto para {len(markets)} mercados...")
        contextos: list[MarketContext] = []

        for i, market in enumerate(markets, 1):
            logger.info(f"Mercado {i}/{len(markets)}: {market.question[:40]}...")
            try:
                contexto = self.recopilar_contexto(
                    market, max_news=max_news_per_market
                )
                contextos.append(contexto)
            except Exception as e:
                logger.error(
                    f"Error recopilando datos para '{market.question[:40]}': {e}"
                )
                # Crear contexto mínimo con lo que tenemos
                contextos.append(
                    MarketContext(
                        market=market,
                        data_quality="poor",
                    )
                )

        logger.info(f"Contexto recopilado para {len(contextos)} mercados")
        return contextos

    # =========================================================================
    # Métodos privados
    # =========================================================================

    def _recopilar_noticias(
        self, market: Market, max_news: int
    ) -> list[NewsArticle]:
        """Busca noticias relevantes para el mercado."""
        try:
            return self._news_fetcher.buscar_noticias(
                question=market.question,
                category=market.category,
                max_results=max_news,
                days_back=7,
            )
        except Exception as e:
            logger.warning(f"Error buscando noticias: {e}")
            return []

    def _recopilar_historial(
        self, market: Market
    ) -> MarketHistoryAnalysis | None:
        """Obtiene y analiza el historial de precios."""
        try:
            # Buscar el token YES para el historial
            token_id = ""
            for token in market.tokens:
                if token.outcome.lower() == "yes":
                    token_id = token.token_id
                    break

            if not token_id:
                return None

            return self._market_history.analizar_mercado(
                condition_id=market.condition_id,
                token_id=token_id,
                current_price=market.yes_price,
            )
        except Exception as e:
            logger.warning(f"Error obteniendo historial: {e}")
            return None

    def _analizar_sentimiento(
        self,
        articles: list[NewsArticle],
        question: str,
    ) -> SentimentResult | None:
        """Analiza el sentimiento de las noticias."""
        try:
            if not articles:
                return None
            return self._sentiment_analyzer.analizar(articles, question)
        except Exception as e:
            logger.warning(f"Error en análisis de sentimiento: {e}")
            return None

    @staticmethod
    def _evaluar_calidad(
        articles: list[NewsArticle],
        history: MarketHistoryAnalysis | None,
        sentiment: SentimentResult | None,
    ) -> str:
        """
        Evalúa la calidad general de los datos recopilados.

        Returns:
            'good': Noticias + historial + sentimiento disponibles
            'partial': Al menos noticias disponibles
            'poor': Datos insuficientes
        """
        tiene_noticias = len(articles) >= 3
        tiene_historial = history is not None and history.data_points > 0
        tiene_sentimiento = sentiment is not None and sentiment.confidence > 0.2

        if tiene_noticias and tiene_historial and tiene_sentimiento:
            return "good"
        if tiene_noticias or tiene_historial:
            return "partial"
        return "poor"


# =============================================================================
# Ejecución directa para pruebas
# =============================================================================

def main() -> None:
    """Prueba el DataCollector con un mercado de ejemplo."""
    from config import configurar_logging
    from core.models import Token
    from rich.console import Console
    from rich.panel import Panel

    configurar_logging()
    console = Console()

    console.print("\n[bold cyan]Data Collector - Fase 2[/bold cyan]\n")

    # Crear un mercado de ejemplo
    mercado = Market(
        condition_id="0x_test_123",
        question="Will Bitcoin reach $100,000 by December 2026?",
        description="This market resolves to YES if Bitcoin reaches $100k.",
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

    with DataCollector() as collector:
        contexto = collector.recopilar_contexto(mercado)

        console.print(Panel(
            contexto.resumen(),
            title="Resumen del Contexto",
        ))

        console.print("\n[bold]Texto para el LLM:[/bold]")
        console.print(contexto.generar_resumen_para_llm())


if __name__ == "__main__":
    main()
