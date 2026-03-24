"""
Escáner de mercados de Polymarket (async).

Se conecta a la API Gamma de Polymarket para descubrir mercados activos
y los filtra según criterios configurables de liquidez, volumen y tiempo.

Fase 1 del agente autónomo — refactorizado a asyncio para rendimiento.
"""

import json
import logging
from datetime import datetime, timezone

import httpx

from config import settings
from core.models import Market, Token
from core.net_utils import retry_async, RETRIABLE_EXCEPTIONS

logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Escáner de mercados activos en Polymarket (async).

    Usa la API Gamma (pública, sin autenticación) para descubrir mercados
    y los filtra según los criterios definidos en ScannerConfig.
    """

    def __init__(self) -> None:
        self._gamma_url = settings.polymarket.gamma_api_url
        self._config = settings.scanner
        self._client = httpx.AsyncClient(
            timeout=self._config.http_timeout_seconds,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        """Cierra el cliente HTTP."""
        await self._client.aclose()

    async def __aenter__(self) -> "MarketScanner":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # =========================================================================
    # Métodos públicos
    # =========================================================================

    async def escanear_mercados(self) -> list[Market]:
        """
        Escanea y retorna mercados activos filtrados.

        Flujo:
        1. Obtiene todos los mercados activos de la API Gamma
        2. Parsea los datos crudos a nuestro modelo Market
        3. Aplica filtros de liquidez, volumen y tiempo
        4. Ordena por volumen 24h descendente
        5. Limita al máximo configurado por ciclo

        Returns:
            Lista de mercados que cumplen todos los criterios.
        """
        logger.info("Iniciando escaneo de mercados en Polymarket...")

        # Paso 1: Obtener mercados crudos de la API
        mercados_crudos = await self._obtener_mercados_activos()
        logger.info(f"Mercados crudos obtenidos: {len(mercados_crudos)}")

        # Paso 2: Parsear a nuestro modelo
        mercados = self._parsear_mercados(mercados_crudos)
        logger.info(f"Mercados parseados correctamente: {len(mercados)}")

        # Paso 3: Filtrar
        mercados_filtrados = self._filtrar_mercados(mercados)
        logger.info(f"Mercados después de filtros: {len(mercados_filtrados)}")

        # Paso 4: Ordenar por volumen 24h (mayor primero)
        mercados_filtrados.sort(key=lambda m: m.volume_24h, reverse=True)

        # Paso 5: Limitar cantidad
        resultado = mercados_filtrados[: self._config.max_markets_per_cycle]
        logger.info(
            f"Mercados seleccionados para este ciclo: {len(resultado)}"
        )

        return resultado

    async def obtener_mercado_por_id(self, condition_id: str) -> Market | None:
        """
        Obtiene un mercado específico por su condition_id.

        Args:
            condition_id: ID de la condición del mercado.

        Returns:
            Market si se encuentra, None si no existe.
        """
        try:
            respuesta = await self._hacer_request(
                "/markets",
                params={"condition_ids": condition_id},
            )
            if respuesta:
                mercados = self._parsear_mercados(respuesta)
                return mercados[0] if mercados else None
        except Exception as e:
            logger.error(f"Error obteniendo mercado {condition_id}: {e}")
        return None

    # =========================================================================
    # Métodos privados - Obtención de datos
    # =========================================================================

    async def _obtener_mercados_activos(self) -> list[dict]:
        """
        Obtiene todos los mercados activos de la API Gamma con paginación.

        Usa paginación offset-based hasta agotar resultados.
        """
        todos_los_mercados: list[dict] = []
        offset = 0
        limit = 100  # Máximo permitido por la API

        while True:
            params = {
                "active": "true",
                "closed": "false",
                "archived": "false",
                "order": "volume24hr",
                "ascending": "false",
                "limit": limit,
                "offset": offset,
            }

            mercados = await self._hacer_request("/markets", params=params)

            if not mercados:
                break

            todos_los_mercados.extend(mercados)
            logger.debug(
                f"Página obtenida: offset={offset}, "
                f"mercados={len(mercados)}, total={len(todos_los_mercados)}"
            )

            # Si obtenemos menos del límite, ya no hay más páginas
            if len(mercados) < limit:
                break

            offset += limit

        return todos_los_mercados

    @retry_async(max_attempts=4, base_delay=2.0, max_delay=16.0)
    async def _hacer_request(
        self, endpoint: str, params: dict | None = None
    ) -> list[dict]:
        """
        Hace un request GET a la API Gamma con reintentos automáticos.

        Args:
            endpoint: Path del endpoint (e.g., "/markets").
            params: Query parameters.

        Returns:
            Lista de diccionarios con los datos de la respuesta.

        Raises:
            httpx.HTTPStatusError: Si la API responde con error después de reintentos.
        """
        url = f"{self._gamma_url}{endpoint}"
        respuesta = await self._client.get(url, params=params)
        respuesta.raise_for_status()
        data = respuesta.json()

        # La API puede retornar una lista directa o un dict con data
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return []

    # =========================================================================
    # Métodos privados - Parsing (sin cambios, son CPU-bound)
    # =========================================================================

    def _parsear_mercados(self, mercados_crudos: list[dict]) -> list[Market]:
        """
        Convierte datos crudos de la API a nuestro modelo Market.

        Los campos outcomes, outcomePrices y clobTokenIds vienen como
        strings JSON que necesitan ser parseados.
        """
        mercados: list[Market] = []

        for raw in mercados_crudos:
            try:
                mercado = self._parsear_mercado_individual(raw)
                if mercado is not None:
                    mercados.append(mercado)
            except Exception as e:
                question = raw.get("question", "desconocido")
                logger.warning(
                    f"Error parseando mercado '{question}': {e}"
                )

        return mercados

    def _parsear_mercado_individual(self, raw: dict) -> Market | None:
        """Parsea un mercado individual desde datos crudos de la API."""
        # Parsear campos que vienen como strings JSON
        outcomes = self._parsear_json_string(raw.get("outcomes", "[]"))
        outcome_prices = self._parsear_json_string(
            raw.get("outcomePrices", "[]")
        )
        clob_token_ids = self._parsear_json_string(
            raw.get("clobTokenIds", "[]")
        )

        # Necesitamos al menos outcomes y precios para un mercado válido
        if not outcomes or not outcome_prices:
            return None

        # Determinar si el mercado está resuelto/cerrado para parsear winners.
        # La Gamma API puede tener active=True + closed=True en mercados
        # recién resueltos, así que basta con closed=True para inferir.
        is_closed = bool(raw.get("closed", False))

        # Construir tokens
        tokens: list[Token] = []
        for i, outcome in enumerate(outcomes):
            token_id = clob_token_ids[i] if i < len(clob_token_ids) else ""
            price = float(outcome_prices[i]) if i < len(outcome_prices) else 0.5

            # Determinar winner: campo explícito o inferido del precio final
            winner = None
            if is_closed:
                # Precio ~1.0 = ganador, ~0.0 = perdedor
                if price >= 0.95:
                    winner = True
                elif price <= 0.05:
                    winner = False

            tokens.append(
                Token(
                    token_id=token_id,
                    outcome=outcome,
                    price=price,
                    winner=winner,
                )
            )

        # Extraer precios YES/NO
        yes_price = 0.5
        no_price = 0.5
        for token in tokens:
            if token.outcome.lower() == "yes":
                yes_price = token.price
            elif token.outcome.lower() == "no":
                no_price = token.price

        # Parsear fechas
        end_date = self._parsear_fecha(raw.get("endDate"))
        created_at = self._parsear_fecha(raw.get("createdAt"))

        return Market(
            condition_id=raw.get("conditionId", raw.get("condition_id", "")),
            question_id=raw.get("questionID", ""),
            question=raw.get("question", "Sin pregunta"),
            description=raw.get("description", ""),
            market_slug=raw.get("slug", ""),
            tokens=tokens,
            volume=float(raw.get("volumeNum", 0) or 0),
            volume_24h=float(raw.get("volume24hr", 0) or 0),
            liquidity=float(raw.get("liquidityNum", 0) or 0),
            spread=float(raw.get("spread", 0) or 0),
            end_date=end_date,
            created_at=created_at,
            active=bool(raw.get("active", True)),
            closed=bool(raw.get("closed", False)),
            resolved=not bool(raw.get("active", True)) and bool(
                raw.get("closed", False)
            ),
            category=raw.get("category", ""),
            yes_price=yes_price,
            no_price=no_price,
        )

    @staticmethod
    def _parsear_json_string(valor: str | list | None) -> list:
        """
        Parsea campos que la API Gamma retorna como strings JSON.

        La API retorna outcomes, outcomePrices y clobTokenIds como
        strings JSON serializados, e.g.: '["Yes","No"]'
        """
        if valor is None:
            return []
        if isinstance(valor, list):
            return valor
        if isinstance(valor, str) and valor:
            try:
                parsed = json.loads(valor)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _parsear_fecha(valor: str | None) -> datetime | None:
        """Parsea una fecha ISO 8601 de la API."""
        if not valor:
            return None
        try:
            # Manejar distintos formatos de fecha
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(valor, fmt)
                except ValueError:
                    continue
            # Último intento con fromisoformat
            return datetime.fromisoformat(valor.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
        except (ValueError, TypeError):
            return None

    # =========================================================================
    # Métodos privados - Filtrado
    # =========================================================================

    def _filtrar_mercados(self, mercados: list[Market]) -> list[Market]:
        """
        Aplica todos los filtros configurados a la lista de mercados.

        Filtros:
        1. No resueltos / activos
        2. Liquidez mínima
        3. Volumen 24h mínimo
        4. Tiempo hasta resolución (entre min y max días)
        """
        filtrados: list[Market] = []

        for mercado in mercados:
            # Filtro 1: Solo mercados activos y no cerrados
            if not mercado.active or mercado.closed or mercado.resolved:
                continue

            # Filtro 2: Liquidez mínima
            if mercado.liquidity < self._config.min_liquidity_usd:
                continue

            # Filtro 3: Volumen 24h mínimo
            if mercado.volume_24h < self._config.min_volume_24h_usd:
                continue

            # Filtro 4: Tiempo hasta resolución
            dias = mercado.days_to_resolution
            if dias is not None:
                if dias < self._config.min_days_to_resolution:
                    continue
                if dias > self._config.max_days_to_resolution:
                    continue

            filtrados.append(mercado)

        return filtrados


# =============================================================================
# Ejecución directa para pruebas rápidas
# =============================================================================

def main() -> None:
    """Ejecuta el scanner y muestra los mercados encontrados."""
    import asyncio
    from config import configurar_logging
    from rich.console import Console
    from rich.table import Table

    configurar_logging()
    console = Console()

    console.print(
        "\n[bold cyan]Polymarket Market Scanner - Fase 1 (async)[/bold cyan]\n"
    )

    async def _run() -> None:
        async with MarketScanner() as scanner:
            mercados = await scanner.escanear_mercados()

            if not mercados:
                console.print("[bold red]No se encontraron mercados activos.[/bold red]")
                return

            # Crear tabla bonita
            tabla = Table(
                title=f"Mercados Activos ({len(mercados)})",
                show_lines=True,
            )
            tabla.add_column("ID", style="dim", width=10)
            tabla.add_column("Pregunta", style="bold", max_width=50)
            tabla.add_column("YES", justify="center", style="green")
            tabla.add_column("NO", justify="center", style="red")
            tabla.add_column("Vol 24h", justify="right", style="cyan")
            tabla.add_column("Liquidez", justify="right", style="blue")
            tabla.add_column("Días", justify="center")
            tabla.add_column("Categoría", style="magenta")

            for m in mercados:
                dias = str(m.days_to_resolution) if m.days_to_resolution else "N/A"
                tabla.add_row(
                    m.condition_id[:8] + "...",
                    m.question[:50],
                    f"${m.yes_price:.2f}",
                    f"${m.no_price:.2f}",
                    f"${m.volume_24h:,.0f}",
                    f"${m.liquidity:,.0f}",
                    dias,
                    m.category or "-",
                )

            console.print(tabla)

            # Resumen
            console.print(f"\n[bold green]Total mercados encontrados: {len(mercados)}")
            vol_total = sum(m.volume_24h for m in mercados)
            liq_total = sum(m.liquidity for m in mercados)
            console.print(f"Volumen 24h total: ${vol_total:,.0f}")
            console.print(f"Liquidez total: ${liq_total:,.0f}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
