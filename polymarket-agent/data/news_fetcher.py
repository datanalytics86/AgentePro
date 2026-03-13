"""
Recopilador de noticias relevantes para mercados de predicción (async).

Busca noticias recientes relacionadas con el tema de cada mercado
usando RSS feeds de medios internacionales y búsqueda web.

Fase 2 del agente autónomo — refactorizado a asyncio con feeds en paralelo.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel, Field

from config import settings
from core.net_utils import retry_async, RETRIABLE_EXCEPTIONS

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    """Artículo de noticias relevante para un mercado."""
    title: str = Field(description="Título del artículo")
    source: str = Field(description="Fuente (medio de comunicación)")
    url: str = Field(default="", description="URL del artículo")
    published_date: datetime | None = Field(
        default=None, description="Fecha de publicación"
    )
    summary: str = Field(default="", description="Resumen del contenido")
    relevance_score: float = Field(
        default=0.0, ge=0, le=1, description="Score de relevancia (0-1)"
    )


# Feeds RSS de medios internacionales accesibles desde Chile
# Nota: Reuters cerró feeds gratuitos; rsshub.app/apnews da 403.
# Feeds ordenados por fiabilidad — los primeros se usan en búsquedas generales.
RSS_FEEDS: dict[str, str] = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "Google News": "https://news.google.com/rss",
    "ABC News": "https://abcnews.go.com/abcnews/internationalheadlines",
    "CBS News": "https://www.cbsnews.com/latest/rss/world",
    "France24 EN": "https://www.france24.com/en/rss",
}

# Feeds temáticos para categorías específicas
CATEGORY_FEEDS: dict[str, list[str]] = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ],
    "politics": [
        "https://rss.politico.com/politics-news.xml",
        "https://thehill.com/feed/",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/news",
    ],
    "technology": [
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
    ],
}


class NewsFetcher:
    """
    Recopila noticias relevantes para un mercado de predicción (async).

    Estrategia de búsqueda:
    1. Busca noticias en Google News RSS con la pregunta del mercado
    2. Busca en feeds RSS temáticos según la categoría del mercado
    3. Filtra y ordena por relevancia

    No requiere API keys - usa solo feeds RSS públicos.
    Todos los feeds se descargan en paralelo con asyncio.gather.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=settings.scanner.http_timeout_seconds,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            },
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Cierra el cliente HTTP."""
        await self._client.aclose()

    async def __aenter__(self) -> "NewsFetcher":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def buscar_noticias(
        self,
        question: str,
        category: str = "",
        max_results: int = 10,
        days_back: int = 7,
        min_results: int | None = None,
    ) -> list[NewsArticle]:
        """
        Busca noticias relevantes para una pregunta de mercado.

        Estrategia bilingüe con fallback automático en español:

        1. Busca en inglés en paralelo (Google News EN + feeds RSS).
        2. Si hay menos de ``min_results`` artículos únicos, lanza
           automáticamente una segunda búsqueda en español (Google News CL)
           para mercados globales con mejor cobertura en medios hispanos
           (tenis, críquet, política latinoamericana, etc.).
        3. Fusiona y deduplica los resultados de ambas búsquedas.

        Esto elimina los avisos de calidad=poor en mercados globales y
        mejora la evaluación del LLM al proveer más contexto.

        Args:
            question:    Pregunta del mercado (en inglés, típicamente).
            category:    Categoría del mercado (e.g., "crypto", "sports").
            max_results: Máximo de artículos a retornar.
            days_back:   Cuántos días hacia atrás buscar.
            min_results: Umbral para activar fallback español. Si es None,
                         usa ``settings.scanner.min_news_count`` (default 3).

        Returns:
            Lista de artículos ordenados por relevancia (bilingüe si aplica).
        """
        if min_results is None:
            min_results = settings.scanner.min_news_count

        logger.info(f"Buscando noticias [EN] para: '{question[:60]}'")
        keywords = self._extraer_keywords(question)

        # ── Fase 1: Búsqueda en inglés (paralelo) ──────────────────────────
        noticias_en = await self._buscar_en_idioma(
            keywords=keywords,
            category=category,
            days_back=days_back,
            lang="en",
            gl="US",
            ceid="US:en",
        )

        # ── Fase 2: Fallback español si hay pocos resultados ────────────────
        if len(noticias_en) < min_results:
            logger.info(
                f"  Solo {len(noticias_en)} artículo(s) en inglés "
                f"(umbral={min_results}). "
                f"Activando búsqueda de respaldo en español [ES-CL]..."
            )
            noticias_es = await self._buscar_en_idioma(
                keywords=keywords,
                category=category,
                days_back=days_back,
                lang="es",
                gl="CL",
                ceid="CL:es",
            )
            todas: list[NewsArticle] = noticias_en + noticias_es
            idiomas_log = f"EN={len(noticias_en)}, ES={len(noticias_es)}"
        else:
            todas = noticias_en
            idiomas_log = f"EN={len(noticias_en)}"

        # ── Fase 3: Calcular relevancia, ordenar y deduplicar ───────────────
        for noticia in todas:
            noticia.relevance_score = self._calcular_relevancia(
                noticia, keywords
            )

        todas.sort(key=lambda n: n.relevance_score, reverse=True)
        resultado_final = self._deduplicar(todas)[:max_results]

        logger.info(
            f"Noticias encontradas: {len(resultado_final)} "
            f"({idiomas_log})"
        )
        return resultado_final

    async def _buscar_en_idioma(
        self,
        keywords: list[str],
        category: str,
        days_back: int,
        lang: str,
        gl: str,
        ceid: str,
    ) -> list[NewsArticle]:
        """
        Ejecuta una búsqueda RSS completa en el idioma indicado.

        Descarga Google News + feeds de categoría + feeds generales
        en paralelo con asyncio.gather y filtra por fecha.

        Args:
            keywords: Palabras clave extraídas de la pregunta del mercado.
            category: Categoría del mercado.
            days_back: Horizonte temporal de búsqueda (días).
            lang:     Código de idioma para Google News (``en`` / ``es``).
            gl:       Código de país para Google News (``US`` / ``CL``).
            ceid:     CEID de Google News (``US:en`` / ``CL:es``).

        Returns:
            Lista de artículos filtrados por fecha (sin deduplicar aún).
        """
        tareas: list[asyncio.Task] = []
        query = "+".join(keywords[:5])
        idioma_label = lang.upper()

        # 1. Google News RSS en el idioma solicitado
        google_url = (
            f"https://news.google.com/rss/search"
            f"?q={query}&hl={lang}&gl={gl}&ceid={ceid}"
        )
        tareas.append(asyncio.ensure_future(
            self._parsear_feed(google_url, f"Google News ({idioma_label})")
        ))

        # 2. Feeds de categoría (solo en inglés; no duplicar en fallback ES)
        if category and lang == "en":
            for feed_url in CATEGORY_FEEDS.get(category.lower(), []):
                tareas.append(asyncio.ensure_future(
                    self._parsear_feed(feed_url, f"RSS-{category}")
                ))

        # 3. Feeds generales (solo en inglés para no duplicar fuentes)
        if lang == "en":
            for nombre, url in list(RSS_FEEDS.items())[:5]:
                tareas.append(asyncio.ensure_future(
                    self._parsear_feed(url, nombre)
                ))

        # Ejecutar en paralelo
        resultados = await asyncio.gather(*tareas, return_exceptions=True)

        todas: list[NewsArticle] = []
        for resultado in resultados:
            if isinstance(resultado, list):
                todas.extend(resultado)
            elif isinstance(resultado, Exception):
                logger.debug(f"Feed falló ({idioma_label}): {resultado}")

        # Filtrar por fecha
        fecha_limite = datetime.now() - timedelta(days=days_back)
        return [
            n for n in todas
            if n.published_date is None or n.published_date >= fecha_limite
        ]

    # =========================================================================
    # Métodos de búsqueda
    # =========================================================================

    @retry_async(max_attempts=2, base_delay=1.0, max_delay=5.0)
    async def _parsear_feed(
        self, feed_url: str, source_name: str
    ) -> list[NewsArticle]:
        """
        Parsea un feed RSS y extrae artículos.

        Usa parsing XML básico con lxml en lugar de feedparser
        para evitar la dependencia problemática sgmllib3k.
        """
        noticias: list[NewsArticle] = []

        try:
            respuesta = await self._client.get(feed_url)
            respuesta.raise_for_status()

            noticias = self._parsear_xml_rss(
                respuesta.text, source_name
            )
            logger.debug(
                f"Feed '{source_name}': {len(noticias)} artículos"
            )

        except httpx.HTTPStatusError as e:
            logger.warning(f"Error HTTP en feed '{source_name}': {e}")
        except Exception as e:
            logger.warning(f"Error parseando feed '{source_name}': {e}")

        return noticias

    def _parsear_xml_rss(
        self, xml_text: str, source_name: str
    ) -> list[NewsArticle]:
        """Parsea XML de RSS usando BeautifulSoup con lxml."""
        from bs4 import BeautifulSoup

        noticias: list[NewsArticle] = []

        try:
            soup = BeautifulSoup(xml_text, "xml")
            items = soup.find_all("item") or soup.find_all("entry")

            for item in items[:20]:  # Máximo 20 por feed
                title_tag = item.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""
                if not title:
                    continue

                # Buscar link
                link_tag = item.find("link")
                url = ""
                if link_tag:
                    url = link_tag.get("href", "") or link_tag.get_text(strip=True)

                # Buscar descripción/resumen
                desc_tag = (
                    item.find("description")
                    or item.find("summary")
                    or item.find("content")
                )
                summary = ""
                if desc_tag:
                    # Limpiar HTML del resumen
                    summary_soup = BeautifulSoup(
                        desc_tag.get_text(), "html.parser"
                    )
                    summary = summary_soup.get_text(strip=True)[:500]

                # Buscar fecha
                date_tag = (
                    item.find("pubDate")
                    or item.find("published")
                    or item.find("updated")
                )
                pub_date = None
                if date_tag:
                    pub_date = self._parsear_fecha_rss(
                        date_tag.get_text(strip=True)
                    )

                noticias.append(
                    NewsArticle(
                        title=title,
                        source=source_name,
                        url=url,
                        published_date=pub_date,
                        summary=summary[:500],
                    )
                )

        except Exception as e:
            logger.warning(f"Error parsing XML de '{source_name}': {e}")

        return noticias

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================

    @staticmethod
    def _extraer_keywords(question: str) -> list[str]:
        """
        Extrae keywords relevantes de una pregunta de mercado.

        Elimina stop words y caracteres especiales.
        """
        # Stop words en inglés (las preguntas de Polymarket son en inglés)
        stop_words = {
            "will", "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "or", "and", "not", "no", "but", "if", "then", "than",
            "this", "that", "it", "its", "his", "her", "he", "she",
            "they", "them", "their", "what", "which", "who", "whom",
            "before", "after", "during", "above", "below", "between",
            "about", "into", "through", "any", "more", "most",
        }

        # Limpiar y tokenizar
        texto = re.sub(r"[^\w\s]", " ", question.lower())
        palabras = texto.split()

        # Filtrar stop words y palabras cortas
        keywords = [
            p for p in palabras
            if p not in stop_words and len(p) > 2
        ]

        return keywords

    @staticmethod
    def _calcular_relevancia(
        noticia: NewsArticle, keywords: list[str]
    ) -> float:
        """
        Calcula un score de relevancia basado en coincidencia de keywords.

        Score entre 0 y 1 basado en cuántas keywords aparecen en el
        título y resumen del artículo.
        """
        if not keywords:
            return 0.0

        texto = f"{noticia.title} {noticia.summary}".lower()
        coincidencias = sum(1 for kw in keywords if kw in texto)

        # Ponderación: título tiene más peso
        titulo_lower = noticia.title.lower()
        coincidencias_titulo = sum(1 for kw in keywords if kw in titulo_lower)
        score = (coincidencias + coincidencias_titulo * 2) / (len(keywords) * 3)

        return min(1.0, score)

    @staticmethod
    def _deduplicar(noticias: list[NewsArticle]) -> list[NewsArticle]:
        """Elimina artículos con títulos muy similares."""
        vistos: set[str] = set()
        unicas: list[NewsArticle] = []

        for noticia in noticias:
            # Normalizar título para comparación
            titulo_norm = re.sub(r"\W+", " ", noticia.title.lower()).strip()
            # Usar primeras 8 palabras como clave
            clave = " ".join(titulo_norm.split()[:8])

            if clave not in vistos:
                vistos.add(clave)
                unicas.append(noticia)

        return unicas

    @staticmethod
    def _parsear_fecha_rss(fecha_str: str) -> datetime | None:
        """Parsea fechas en formatos comunes de RSS."""
        formatos = [
            "%a, %d %b %Y %H:%M:%S %Z",     # RFC 822
            "%a, %d %b %Y %H:%M:%S %z",     # RFC 822 con offset
            "%Y-%m-%dT%H:%M:%S%z",           # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",            # ISO 8601 UTC
            "%Y-%m-%dT%H:%M:%S.%fZ",         # ISO 8601 con ms
            "%Y-%m-%d %H:%M:%S",             # Formato simple
            "%Y-%m-%d",                       # Solo fecha
        ]
        for fmt in formatos:
            try:
                dt = datetime.strptime(fecha_str.strip(), fmt)
                return dt.replace(tzinfo=None)
            except ValueError:
                continue
        return None


# =============================================================================
# Ejecución directa para pruebas
# =============================================================================

def main() -> None:
    """Prueba el NewsFetcher con una pregunta de ejemplo."""
    import asyncio
    from config import configurar_logging
    from rich.console import Console
    from rich.table import Table

    configurar_logging()
    console = Console()

    console.print("\n[bold cyan]News Fetcher - Fase 2 (async)[/bold cyan]\n")

    async def _run() -> None:
        async with NewsFetcher() as fetcher:
            noticias = await fetcher.buscar_noticias(
                question="Will Bitcoin reach $100k by end of 2026?",
                category="crypto",
                max_results=10,
            )

            if not noticias:
                console.print("[red]No se encontraron noticias.[/red]")
                return

            tabla = Table(title=f"Noticias ({len(noticias)})", show_lines=True)
            tabla.add_column("Relevancia", justify="center", width=10)
            tabla.add_column("Fuente", width=15)
            tabla.add_column("Título", max_width=60)
            tabla.add_column("Fecha", width=12)

            for n in noticias:
                fecha = n.published_date.strftime("%Y-%m-%d") if n.published_date else "N/A"
                score_bar = "=" * int(n.relevance_score * 10)
                tabla.add_row(
                    f"{n.relevance_score:.1%} {score_bar}",
                    n.source,
                    n.title[:60],
                    fecha,
                )

            console.print(tabla)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
