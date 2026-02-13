"""
Tests del Motor de Probabilidades - Fase 3.

Verifica parsing de respuestas del LLM y sanity checks.
"""

import pytest

from core.probability_engine import ProbabilityEngine, _extraer_json
from core.models import LLMEvaluation, Market, Token
from core.data_collector import MarketContext


class TestExtraerJson:
    """Tests de extracción de JSON de respuestas del LLM."""

    def test_json_puro(self) -> None:
        """Extrae JSON cuando es el texto completo."""
        texto = '{"probability": 0.7, "confidence": "high"}'
        assert _extraer_json(texto) == texto

    def test_json_con_markdown(self) -> None:
        """Extrae JSON envuelto en backticks markdown."""
        texto = '```json\n{"probability": 0.7}\n```'
        result = _extraer_json(texto)
        assert result is not None
        assert '"probability": 0.7' in result

    def test_json_con_texto_extra(self) -> None:
        """Extrae JSON con texto adicional alrededor."""
        texto = 'Aquí está mi análisis:\n{"probability": 0.65}\nEso es todo.'
        result = _extraer_json(texto)
        assert result is not None
        assert '"probability": 0.65' in result

    def test_sin_json(self) -> None:
        """Retorna None si no hay JSON."""
        assert _extraer_json("No hay JSON aquí") is None

    def test_json_vacio(self) -> None:
        """String vacío retorna None."""
        assert _extraer_json("") is None


class TestParsearRespuesta:
    """Tests del parsing de respuestas del LLM."""

    def test_respuesta_valida(self) -> None:
        """Parsea correctamente una respuesta JSON válida."""
        respuesta = '''{
            "probability": 0.72,
            "confidence": "high",
            "reasoning": "Based on analysis",
            "key_factors": ["factor1"],
            "risks_to_thesis": ["risk1"],
            "information_quality": "sufficient"
        }'''
        result = ProbabilityEngine._parsear_respuesta(respuesta)
        assert result is not None
        assert result.probability == 0.72
        assert result.confidence == "high"

    def test_respuesta_invalida(self) -> None:
        """Retorna None para JSON inválido."""
        assert ProbabilityEngine._parsear_respuesta("not json") is None

    def test_respuesta_campos_faltantes(self) -> None:
        """Retorna None si faltan campos requeridos."""
        respuesta = '{"probability": 0.5}'
        assert ProbabilityEngine._parsear_respuesta(respuesta) is None


class TestSanityChecks:
    """Tests de los sanity checks de evaluaciones."""

    def _hacer_contexto(self, data_quality: str = "good") -> MarketContext:
        mercado = Market(
            condition_id="test", question="Test?",
            yes_price=0.5, no_price=0.5,
        )
        return MarketContext(market=mercado, data_quality=data_quality)

    def _hacer_evaluacion(
        self, prob: float = 0.5, conf: str = "medium", info: str = "sufficient"
    ) -> LLMEvaluation:
        return LLMEvaluation(
            probability=prob, confidence=conf,
            reasoning="test", key_factors=["test"],
            risks_to_thesis=["test"], information_quality=info,
        )

    def test_probabilidad_extrema_baja_confianza(self) -> None:
        """Prob extrema con baja confianza queda en low."""
        ev = self._hacer_evaluacion(prob=0.99, conf="medium")
        ctx = self._hacer_contexto()
        result = ProbabilityEngine._aplicar_sanity_checks(ev, ctx)
        assert result.confidence == "low"

    def test_info_poor_fuerza_confianza_low(self) -> None:
        """Info quality 'poor' fuerza confianza a 'low'."""
        ev = self._hacer_evaluacion(prob=0.6, conf="high", info="poor")
        ctx = self._hacer_contexto()
        result = ProbabilityEngine._aplicar_sanity_checks(ev, ctx)
        assert result.confidence == "low"

    def test_data_poor_ajusta_info_quality(self) -> None:
        """Data quality 'poor' ajusta info si LLM dice 'sufficient'."""
        ev = self._hacer_evaluacion(prob=0.6, info="sufficient")
        ctx = self._hacer_contexto(data_quality="poor")
        result = ProbabilityEngine._aplicar_sanity_checks(ev, ctx)
        assert result.information_quality == "limited"

    def test_evaluacion_normal_no_cambia(self) -> None:
        """Evaluación normal no se modifica."""
        ev = self._hacer_evaluacion(prob=0.6, conf="high", info="sufficient")
        ctx = self._hacer_contexto(data_quality="good")
        result = ProbabilityEngine._aplicar_sanity_checks(ev, ctx)
        assert result.confidence == "high"
        assert result.information_quality == "sufficient"
