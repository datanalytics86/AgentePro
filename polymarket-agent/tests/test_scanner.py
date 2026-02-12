"""
Tests del Market Scanner - Fase 1.

Verifica que el escáner de mercados funciona correctamente:
- Parsing de datos crudos de la API
- Filtrado de mercados según criterios
- Manejo de errores y datos malformados
"""

import json
from datetime import datetime, timedelta

import pytest

from core.market_scanner import MarketScanner
from core.models import Market, Token


# =============================================================================
# Fixtures - Datos de prueba
# =============================================================================

def _crear_mercado_crudo(
    condition_id: str = "0xabc123def456",
    question: str = "Will Bitcoin reach $100k by end of 2026?",
    yes_price: float = 0.65,
    no_price: float = 0.35,
    volume_24h: float = 5000,
    liquidity: float = 10000,
    days_ahead: int = 30,
    active: bool = True,
    closed: bool = False,
) -> dict:
    """Crea un mercado crudo simulando la respuesta de la API Gamma."""
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "conditionId": condition_id,
        "questionID": "qid_123",
        "question": question,
        "description": "Mercado de prueba",
        "slug": "bitcoin-100k-2026",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps([str(yes_price), str(no_price)]),
        "clobTokenIds": json.dumps(["token_yes_123", "token_no_456"]),
        "volumeNum": volume_24h * 10,
        "volume24hr": volume_24h,
        "liquidityNum": liquidity,
        "spread": 0.02,
        "endDate": end_date,
        "createdAt": "2024-01-01T00:00:00Z",
        "active": active,
        "closed": closed,
        "category": "crypto",
    }


@pytest.fixture
def scanner() -> MarketScanner:
    """Crea una instancia del scanner para tests."""
    return MarketScanner()


@pytest.fixture
def mercado_crudo_valido() -> dict:
    """Retorna un mercado crudo válido de prueba."""
    return _crear_mercado_crudo()


@pytest.fixture
def mercados_crudos_variados() -> list[dict]:
    """Retorna una lista variada de mercados para testear filtros."""
    return [
        # Mercado bueno: alta liquidez, buen volumen, 30 días
        _crear_mercado_crudo(
            condition_id="0x001",
            question="Mercado bueno",
            volume_24h=5000,
            liquidity=10000,
            days_ahead=30,
        ),
        # Mercado con baja liquidez (debe ser filtrado)
        _crear_mercado_crudo(
            condition_id="0x002",
            question="Baja liquidez",
            volume_24h=5000,
            liquidity=100,  # < 500 USD mínimo
            days_ahead=30,
        ),
        # Mercado con bajo volumen (debe ser filtrado)
        _crear_mercado_crudo(
            condition_id="0x003",
            question="Bajo volumen",
            volume_24h=50,  # < 100 USD mínimo
            liquidity=10000,
            days_ahead=30,
        ),
        # Mercado que resuelve muy pronto (< 1 día)
        _crear_mercado_crudo(
            condition_id="0x004",
            question="Resuelve mañana",
            volume_24h=5000,
            liquidity=10000,
            days_ahead=0,  # Hoy
        ),
        # Mercado muy lejano (> 90 días)
        _crear_mercado_crudo(
            condition_id="0x005",
            question="Mercado lejano",
            volume_24h=5000,
            liquidity=10000,
            days_ahead=120,  # > 90 días
        ),
        # Mercado cerrado (debe ser filtrado)
        _crear_mercado_crudo(
            condition_id="0x006",
            question="Mercado cerrado",
            volume_24h=5000,
            liquidity=10000,
            days_ahead=30,
            active=False,
            closed=True,
        ),
        # Otro mercado bueno
        _crear_mercado_crudo(
            condition_id="0x007",
            question="Otro mercado bueno",
            volume_24h=3000,
            liquidity=8000,
            days_ahead=15,
        ),
    ]


# =============================================================================
# Tests de parsing
# =============================================================================

class TestParsing:
    """Tests de parsing de datos de la API."""

    def test_parsear_json_string_valido(self) -> None:
        """Parsea correctamente un string JSON de la API."""
        resultado = MarketScanner._parsear_json_string('["Yes","No"]')
        assert resultado == ["Yes", "No"]

    def test_parsear_json_string_ya_lista(self) -> None:
        """Si ya es una lista, la retorna directamente."""
        resultado = MarketScanner._parsear_json_string(["Yes", "No"])
        assert resultado == ["Yes", "No"]

    def test_parsear_json_string_none(self) -> None:
        """None retorna lista vacía."""
        resultado = MarketScanner._parsear_json_string(None)
        assert resultado == []

    def test_parsear_json_string_vacio(self) -> None:
        """String vacío retorna lista vacía."""
        resultado = MarketScanner._parsear_json_string("")
        assert resultado == []

    def test_parsear_json_string_malformado(self) -> None:
        """String JSON malformado retorna lista vacía sin error."""
        resultado = MarketScanner._parsear_json_string("{invalido")
        assert resultado == []

    def test_parsear_fecha_iso(self) -> None:
        """Parsea correctamente una fecha ISO 8601."""
        fecha = MarketScanner._parsear_fecha("2024-06-15T12:00:00Z")
        assert fecha is not None
        assert fecha.year == 2024
        assert fecha.month == 6
        assert fecha.day == 15

    def test_parsear_fecha_none(self) -> None:
        """None retorna None."""
        assert MarketScanner._parsear_fecha(None) is None

    def test_parsear_fecha_vacia(self) -> None:
        """String vacío retorna None."""
        assert MarketScanner._parsear_fecha("") is None

    def test_parsear_mercado_individual(
        self, scanner: MarketScanner, mercado_crudo_valido: dict
    ) -> None:
        """Parsea correctamente un mercado individual."""
        mercado = scanner._parsear_mercado_individual(mercado_crudo_valido)

        assert mercado is not None
        assert mercado.condition_id == "0xabc123def456"
        assert "Bitcoin" in mercado.question
        assert mercado.yes_price == 0.65
        assert mercado.no_price == 0.35
        assert len(mercado.tokens) == 2
        assert mercado.tokens[0].outcome == "Yes"
        assert mercado.tokens[1].outcome == "No"
        assert mercado.volume_24h == 5000
        assert mercado.liquidity == 10000
        assert mercado.active is True
        assert mercado.closed is False
        assert mercado.category == "crypto"

    def test_parsear_mercado_sin_precios(self, scanner: MarketScanner) -> None:
        """Mercado sin precios retorna None."""
        raw = {"conditionId": "0x123", "question": "Test?"}
        mercado = scanner._parsear_mercado_individual(raw)
        assert mercado is None

    def test_parsear_lista_mercados(
        self, scanner: MarketScanner, mercados_crudos_variados: list[dict]
    ) -> None:
        """Parsea correctamente una lista de mercados."""
        mercados = scanner._parsear_mercados(mercados_crudos_variados)
        # Todos deberían parsearse correctamente
        assert len(mercados) == 7


# =============================================================================
# Tests de filtrado
# =============================================================================

class TestFiltrado:
    """Tests del sistema de filtrado de mercados."""

    def test_filtrar_mercados_completo(
        self, scanner: MarketScanner, mercados_crudos_variados: list[dict]
    ) -> None:
        """El filtrado completo deja solo los mercados válidos."""
        mercados = scanner._parsear_mercados(mercados_crudos_variados)
        filtrados = scanner._filtrar_mercados(mercados)

        # Solo deberían quedar 0x001 y 0x007 (los "buenos")
        assert len(filtrados) == 2
        ids = {m.condition_id for m in filtrados}
        assert "0x001" in ids
        assert "0x007" in ids

    def test_filtrar_excluye_baja_liquidez(
        self, scanner: MarketScanner
    ) -> None:
        """Excluye mercados con liquidez por debajo del mínimo."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            liquidity=100,  # < 500
            volume_24h=5000,
            end_date=datetime.now() + timedelta(days=30),
        )
        filtrados = scanner._filtrar_mercados([mercado])
        assert len(filtrados) == 0

    def test_filtrar_excluye_bajo_volumen(
        self, scanner: MarketScanner
    ) -> None:
        """Excluye mercados con volumen 24h bajo."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            liquidity=5000,
            volume_24h=50,  # < 100
            end_date=datetime.now() + timedelta(days=30),
        )
        filtrados = scanner._filtrar_mercados([mercado])
        assert len(filtrados) == 0

    def test_filtrar_excluye_cerrado(self, scanner: MarketScanner) -> None:
        """Excluye mercados cerrados."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            liquidity=5000,
            volume_24h=5000,
            active=False,
            closed=True,
            end_date=datetime.now() + timedelta(days=30),
        )
        filtrados = scanner._filtrar_mercados([mercado])
        assert len(filtrados) == 0

    def test_filtrar_acepta_mercado_sin_fecha(
        self, scanner: MarketScanner
    ) -> None:
        """Mercados sin fecha de resolución pasan el filtro de tiempo."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            liquidity=5000,
            volume_24h=5000,
            end_date=None,  # Sin fecha
        )
        filtrados = scanner._filtrar_mercados([mercado])
        assert len(filtrados) == 1


# =============================================================================
# Tests del modelo Market
# =============================================================================

class TestModeloMarket:
    """Tests del modelo de datos Market."""

    def test_days_to_resolution(self) -> None:
        """Calcula correctamente los días hasta resolución."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            end_date=datetime.now() + timedelta(days=15, hours=1),
        )
        assert mercado.days_to_resolution == 15

    def test_days_to_resolution_sin_fecha(self) -> None:
        """Retorna None si no hay fecha de resolución."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            end_date=None,
        )
        assert mercado.days_to_resolution is None

    def test_probabilidad_implicita(self) -> None:
        """Las probabilidades implícitas coinciden con los precios."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            yes_price=0.70,
            no_price=0.30,
        )
        assert mercado.implied_probability_yes == 0.70
        assert mercado.implied_probability_no == 0.30

    def test_resumen_legible(self) -> None:
        """El resumen contiene la información clave."""
        mercado = Market(
            condition_id="0xabc123def456",
            question="Will it rain tomorrow?",
            yes_price=0.65,
            no_price=0.35,
            volume_24h=5000,
            liquidity=10000,
            end_date=datetime.now() + timedelta(days=10),
        )
        resumen = mercado.resumen()
        assert "0xabc123" in resumen
        assert "Will it rain" in resumen
        assert "0.65" in resumen

    def test_parsear_precio_string(self) -> None:
        """Parsea correctamente precios que vienen como string."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            yes_price="0.72",  # type: ignore[arg-type]
            no_price="0.28",  # type: ignore[arg-type]
        )
        assert mercado.yes_price == 0.72
        assert mercado.no_price == 0.28

    def test_parsear_precio_none(self) -> None:
        """Precio None se convierte a 0.5 (default)."""
        mercado = Market(
            condition_id="test",
            question="Test?",
            yes_price=None,  # type: ignore[arg-type]
        )
        assert mercado.yes_price == 0.5
