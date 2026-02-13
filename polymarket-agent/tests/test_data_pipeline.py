"""
Tests de la Fase 2 - Recopilación de datos y contexto.

Verifica que los componentes de datos funcionan correctamente:
- NewsFetcher: extracción de keywords, cálculo de relevancia, deduplicación
- MarketHistory: cálculo de tendencias, volatilidad, movimientos bruscos
- SentimentAnalyzer: análisis de sentimiento basado en lexicón
- DataCollector: integración de los componentes
"""

import json
from datetime import datetime, timedelta

import pytest

from core.models import Market, Token
from data.news_fetcher import NewsFetcher, NewsArticle
from data.market_history import (
    MarketHistory,
    MarketHistoryAnalysis,
    PricePoint,
    PriceMovement,
    Trend,
)
from data.sentiment import SentimentAnalyzer, SentimentResult
from core.data_collector import DataCollector, MarketContext


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def news_fetcher() -> NewsFetcher:
    return NewsFetcher()


@pytest.fixture
def market_history() -> MarketHistory:
    return MarketHistory()


@pytest.fixture
def sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@pytest.fixture
def mercado_ejemplo() -> Market:
    return Market(
        condition_id="0x_test_btc",
        question="Will Bitcoin reach $100,000 by December 2026?",
        description="Resolves YES if BTC hits 100k.",
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


@pytest.fixture
def articulos_positivos() -> list[NewsArticle]:
    return [
        NewsArticle(
            title="Bitcoin surges past $95,000 as institutional demand grows",
            source="Reuters",
            summary="Bitcoin rallied significantly with strong buying momentum.",
            published_date=datetime.now() - timedelta(hours=6),
            relevance_score=0.8,
        ),
        NewsArticle(
            title="Major investment firm confirms Bitcoin ETF approval likely",
            source="Bloomberg",
            summary="Experts are optimistic and confident about Bitcoin success.",
            published_date=datetime.now() - timedelta(hours=12),
            relevance_score=0.9,
        ),
    ]


@pytest.fixture
def articulos_negativos() -> list[NewsArticle]:
    return [
        NewsArticle(
            title="Regulators threaten to block cryptocurrency trading",
            source="WSJ",
            summary="Government warns of crisis, investigation into exchanges.",
            published_date=datetime.now() - timedelta(hours=3),
            relevance_score=0.7,
        ),
        NewsArticle(
            title="Bitcoin crashes as market fears grow",
            source="CNBC",
            summary="Bitcoin plummeted after failed regulatory support.",
            published_date=datetime.now() - timedelta(hours=8),
            relevance_score=0.8,
        ),
    ]


@pytest.fixture
def historial_subiendo() -> list[PricePoint]:
    """Historial de precios con tendencia alcista."""
    ahora = datetime.now()
    return [
        PricePoint(timestamp=ahora - timedelta(hours=i), price=0.50 + i * 0.02)
        for i in range(10, 0, -1)
    ]


@pytest.fixture
def historial_bajando() -> list[PricePoint]:
    """Historial de precios con tendencia bajista."""
    ahora = datetime.now()
    return [
        PricePoint(timestamp=ahora - timedelta(hours=i), price=0.80 - i * 0.02)
        for i in range(10, 0, -1)
    ]


# =============================================================================
# Tests del NewsFetcher
# =============================================================================

class TestNewsFetcher:
    """Tests del componente de búsqueda de noticias."""

    def test_extraer_keywords(self) -> None:
        """Extrae keywords relevantes eliminando stop words."""
        kw = NewsFetcher._extraer_keywords(
            "Will Bitcoin reach $100,000 by December 2026?"
        )
        assert "bitcoin" in kw
        assert "100" in kw or "000" in kw
        assert "december" in kw
        assert "2026" in kw
        # Stop words eliminadas
        assert "will" not in kw
        assert "by" not in kw

    def test_extraer_keywords_vacio(self) -> None:
        """String vacío retorna lista vacía."""
        kw = NewsFetcher._extraer_keywords("")
        assert kw == []

    def test_calcular_relevancia_alta(self) -> None:
        """Artículo con muchas keywords tiene alta relevancia."""
        articulo = NewsArticle(
            title="Bitcoin reaches new high at $100,000",
            source="Test",
            summary="Bitcoin surged to reach the milestone of 100k in December 2026.",
        )
        keywords = ["bitcoin", "reach", "100", "december", "2026"]
        score = NewsFetcher._calcular_relevancia(articulo, keywords)
        assert score > 0.3

    def test_calcular_relevancia_baja(self) -> None:
        """Artículo sin keywords tiene baja relevancia."""
        articulo = NewsArticle(
            title="Weather forecast for tomorrow",
            source="Test",
            summary="Rain expected in the afternoon.",
        )
        keywords = ["bitcoin", "reach", "100k", "december"]
        score = NewsFetcher._calcular_relevancia(articulo, keywords)
        assert score < 0.1

    def test_calcular_relevancia_sin_keywords(self) -> None:
        """Sin keywords retorna 0."""
        articulo = NewsArticle(title="Test", source="Test")
        assert NewsFetcher._calcular_relevancia(articulo, []) == 0.0

    def test_deduplicar(self) -> None:
        """Elimina artículos con títulos similares (mismas primeras 8 palabras)."""
        articulos = [
            NewsArticle(
                title="Bitcoin surges to new record high today say analysts",
                source="A",
            ),
            NewsArticle(
                title="Bitcoin surges to new record high today say market observers",
                source="B",
            ),
            NewsArticle(title="Ethereum reaches new all-time high", source="C"),
        ]
        resultado = NewsFetcher._deduplicar(articulos)
        # Los dos primeros comparten las primeras 8 palabras normalizadas
        assert len(resultado) == 2

    def test_parsear_fecha_rss_rfc822(self) -> None:
        """Parsea fecha en formato RFC 822."""
        fecha = NewsFetcher._parsear_fecha_rss("Tue, 15 Jan 2025 10:30:00 GMT")
        assert fecha is not None
        assert fecha.year == 2025

    def test_parsear_fecha_rss_iso(self) -> None:
        """Parsea fecha ISO 8601."""
        fecha = NewsFetcher._parsear_fecha_rss("2025-06-15T12:00:00Z")
        assert fecha is not None
        assert fecha.month == 6

    def test_parsear_fecha_rss_invalida(self) -> None:
        """Fecha inválida retorna None."""
        assert NewsFetcher._parsear_fecha_rss("not a date") is None


# =============================================================================
# Tests del MarketHistory
# =============================================================================

class TestMarketHistory:
    """Tests del componente de historial de precios."""

    def test_calcular_volatilidad_estable(self) -> None:
        """Precios estables dan volatilidad baja."""
        points = [
            PricePoint(timestamp=datetime.now() - timedelta(hours=i), price=0.50)
            for i in range(10)
        ]
        vol = MarketHistory._calcular_volatilidad(points)
        assert vol == 0.0

    def test_calcular_volatilidad_alta(self) -> None:
        """Precios oscilantes dan volatilidad alta."""
        ahora = datetime.now()
        points = [
            PricePoint(
                timestamp=ahora - timedelta(hours=i),
                price=0.5 + (0.1 if i % 2 == 0 else -0.1),
            )
            for i in range(10)
        ]
        vol = MarketHistory._calcular_volatilidad(points)
        assert vol > 0.1

    def test_calcular_volatilidad_insuficiente(self) -> None:
        """Con 0 o 1 punto retorna 0."""
        assert MarketHistory._calcular_volatilidad([]) == 0.0
        assert MarketHistory._calcular_volatilidad(
            [PricePoint(timestamp=datetime.now(), price=0.5)]
        ) == 0.0

    def test_precio_en_momento(self) -> None:
        """Encuentra el precio más cercano a un momento."""
        ahora = datetime.now()
        points = [
            PricePoint(timestamp=ahora - timedelta(hours=2), price=0.60),
            PricePoint(timestamp=ahora - timedelta(hours=1), price=0.65),
            PricePoint(timestamp=ahora, price=0.70),
        ]
        # El precio hace ~1 hora debería ser ~0.65
        precio = MarketHistory._precio_en_momento(
            points, ahora - timedelta(hours=1)
        )
        assert precio == 0.65

    def test_precio_en_momento_vacio(self) -> None:
        """Sin datos retorna None."""
        assert MarketHistory._precio_en_momento([], datetime.now()) is None

    def test_cambio_porcentual(self) -> None:
        """Calcula correctamente el cambio porcentual."""
        assert MarketHistory._calcular_cambio_pct(0.50, 0.60) == pytest.approx(20.0)
        assert MarketHistory._calcular_cambio_pct(0.60, 0.50) == pytest.approx(-16.67, abs=0.01)
        assert MarketHistory._calcular_cambio_pct(None, 0.50) == 0.0
        assert MarketHistory._calcular_cambio_pct(0, 0.50) == 0.0

    def test_detectar_movimientos_bruscos(
        self, market_history: MarketHistory
    ) -> None:
        """Detecta movimientos de precio mayores al umbral."""
        ahora = datetime.now()
        points = [
            PricePoint(timestamp=ahora - timedelta(hours=3), price=0.50),
            PricePoint(timestamp=ahora - timedelta(hours=2), price=0.51),  # +2% - no brusco
            PricePoint(timestamp=ahora - timedelta(hours=1), price=0.60),  # +17.6% - brusco
            PricePoint(timestamp=ahora, price=0.55),                       # -8.3% - brusco
        ]
        movimientos = market_history._detectar_movimientos_bruscos(points)
        assert len(movimientos) == 2
        assert movimientos[0].direction == "up"
        assert movimientos[1].direction == "down"

    def test_determinar_tendencia_subiendo(
        self, market_history: MarketHistory, historial_subiendo: list[PricePoint]
    ) -> None:
        """Cambio positivo se clasifica como rising."""
        tendencia = market_history._determinar_tendencia(10.0, historial_subiendo)
        assert tendencia == Trend.RISING

    def test_determinar_tendencia_bajando(
        self, market_history: MarketHistory, historial_bajando: list[PricePoint]
    ) -> None:
        """Cambio negativo se clasifica como falling."""
        tendencia = market_history._determinar_tendencia(-10.0, historial_bajando)
        assert tendencia == Trend.FALLING

    def test_determinar_tendencia_insuficiente(
        self, market_history: MarketHistory
    ) -> None:
        """Con datos insuficientes retorna INSUFFICIENT."""
        tendencia = market_history._determinar_tendencia(0.0, [])
        assert tendencia == Trend.INSUFFICIENT


# =============================================================================
# Tests del SentimentAnalyzer
# =============================================================================

class TestSentimentAnalyzer:
    """Tests del analizador de sentimiento."""

    def test_sentimiento_positivo(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        articulos_positivos: list[NewsArticle],
    ) -> None:
        """Artículos positivos dan score > 0."""
        resultado = sentiment_analyzer.analizar(articulos_positivos)
        assert resultado.score > 0
        assert resultado.label in ("positive", "very_positive")
        assert resultado.articles_analyzed == 2

    def test_sentimiento_negativo(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        articulos_negativos: list[NewsArticle],
    ) -> None:
        """Artículos negativos dan score < 0."""
        resultado = sentiment_analyzer.analizar(articulos_negativos)
        assert resultado.score < 0
        assert resultado.label in ("negative", "very_negative")

    def test_sentimiento_vacio(
        self, sentiment_analyzer: SentimentAnalyzer
    ) -> None:
        """Sin artículos retorna neutral con confianza 0."""
        resultado = sentiment_analyzer.analizar([])
        assert resultado.score == 0.0
        assert resultado.label == "neutral"
        assert resultado.confidence == 0.0

    def test_score_en_rango(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        articulos_positivos: list[NewsArticle],
    ) -> None:
        """El score siempre está entre -1 y 1."""
        resultado = sentiment_analyzer.analizar(articulos_positivos)
        assert -1 <= resultado.score <= 1

    def test_justificacion_no_vacia(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        articulos_positivos: list[NewsArticle],
    ) -> None:
        """La justificación nunca está vacía."""
        resultado = sentiment_analyzer.analizar(articulos_positivos)
        assert len(resultado.justification) > 0

    def test_label_mapping(self, sentiment_analyzer: SentimentAnalyzer) -> None:
        """Verifica la conversión score -> label."""
        assert SentimentAnalyzer._score_a_label(0.5) == "very_positive"
        assert SentimentAnalyzer._score_a_label(0.15) == "positive"
        assert SentimentAnalyzer._score_a_label(0.0) == "neutral"
        assert SentimentAnalyzer._score_a_label(-0.15) == "negative"
        assert SentimentAnalyzer._score_a_label(-0.5) == "very_negative"


# =============================================================================
# Tests del DataCollector
# =============================================================================

class TestDataCollector:
    """Tests del collector integrado."""

    def test_evaluar_calidad_buena(self) -> None:
        """Calidad 'good' cuando hay datos de todas las fuentes."""
        articulos = [NewsArticle(title=f"News {i}", source="Test") for i in range(5)]
        historial = MarketHistoryAnalysis(
            condition_id="test", current_price=0.5, data_points=10
        )
        sentimiento = SentimentResult(
            score=0.3, label="positive", confidence=0.5
        )
        calidad = DataCollector._evaluar_calidad(articulos, historial, sentimiento)
        assert calidad == "good"

    def test_evaluar_calidad_parcial(self) -> None:
        """Calidad 'partial' cuando faltan datos."""
        articulos = [NewsArticle(title=f"News {i}", source="Test") for i in range(5)]
        calidad = DataCollector._evaluar_calidad(articulos, None, None)
        assert calidad == "partial"

    def test_evaluar_calidad_pobre(self) -> None:
        """Calidad 'poor' sin datos."""
        calidad = DataCollector._evaluar_calidad([], None, None)
        assert calidad == "poor"

    def test_market_context_resumen(self, mercado_ejemplo: Market) -> None:
        """El contexto genera un resumen legible."""
        contexto = MarketContext(
            market=mercado_ejemplo,
            data_quality="partial",
        )
        resumen = contexto.resumen()
        assert "Bitcoin" in resumen
        assert "partial" in resumen

    def test_market_context_resumen_llm(self, mercado_ejemplo: Market) -> None:
        """El resumen para LLM contiene los datos clave."""
        articulos = [
            NewsArticle(
                title="Bitcoin news update",
                source="Reuters",
                summary="Important BTC development.",
                published_date=datetime.now(),
                relevance_score=0.8,
            ),
        ]
        sentimiento = SentimentResult(
            score=0.3, label="positive", confidence=0.6
        )
        contexto = MarketContext(
            market=mercado_ejemplo,
            news_articles=articulos,
            sentiment=sentimiento,
            data_quality="partial",
        )
        texto_llm = contexto.generar_resumen_para_llm()

        assert "PREGUNTA DEL MERCADO" in texto_llm
        assert "Bitcoin" in texto_llm
        assert "PRECIO ACTUAL" in texto_llm
        assert "YES=$0.65" in texto_llm
        assert "NOTICIAS RELEVANTES" in texto_llm
        assert "SENTIMIENTO" in texto_llm
