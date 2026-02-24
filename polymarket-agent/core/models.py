"""
Modelos de datos del agente de Polymarket.

Define las estructuras de datos principales usadas en todo el sistema.
Usa Pydantic para validación automática y serialización.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Token(BaseModel):
    """Representa un token de un mercado de Polymarket (YES o NO)."""
    token_id: str = Field(description="ID único del token en el CLOB")
    outcome: str = Field(description="Resultado que representa: 'Yes' o 'No'")
    price: float = Field(ge=0, le=1, description="Precio actual (0 a 1)")
    winner: bool | None = Field(
        default=None, description="Si ya ganó (True/False/None si no resuelto)"
    )


class Market(BaseModel):
    """
    Representa un mercado de predicción en Polymarket.

    Cada mercado tiene una pregunta, tokens asociados (YES/NO),
    y métricas de liquidez y volumen.
    """
    # Identificación
    condition_id: str = Field(description="ID de la condición en Polymarket")
    question_id: str = Field(default="", description="ID de la pregunta")
    question: str = Field(description="Pregunta del mercado")
    description: str = Field(default="", description="Descripción detallada")
    market_slug: str = Field(default="", description="Slug URL del mercado")

    # Tokens
    tokens: list[Token] = Field(
        default_factory=list, description="Tokens del mercado (YES/NO)"
    )

    # Métricas de mercado
    volume: float = Field(default=0, ge=0, description="Volumen total en USD")
    volume_24h: float = Field(default=0, ge=0, description="Volumen 24h en USD")
    liquidity: float = Field(default=0, ge=0, description="Liquidez en USD")
    spread: float = Field(default=0, ge=0, description="Spread bid-ask")

    # Temporal
    end_date: datetime | None = Field(
        default=None, description="Fecha de resolución del mercado"
    )
    created_at: datetime | None = Field(
        default=None, description="Fecha de creación"
    )

    # Estado
    active: bool = Field(default=True, description="Si el mercado está activo")
    closed: bool = Field(default=False, description="Si el mercado está cerrado")
    resolved: bool = Field(default=False, description="Si ya se resolvió")
    category: str = Field(default="", description="Categoría del mercado")

    # Precios rápidos
    yes_price: float = Field(
        default=0.5, ge=0, le=1, description="Precio del token YES"
    )
    no_price: float = Field(
        default=0.5, ge=0, le=1, description="Precio del token NO"
    )

    @field_validator("yes_price", "no_price", mode="before")
    @classmethod
    def parsear_precio(cls, v: float | str | None) -> float:
        """Convierte precios string a float, manejando valores nulos."""
        if v is None:
            return 0.5
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.5

    @property
    def days_to_resolution(self) -> int | None:
        """Calcula los días restantes hasta la resolución."""
        if self.end_date is None:
            return None
        delta = self.end_date - datetime.now()
        return max(0, delta.days)

    @property
    def implied_probability_yes(self) -> float:
        """Probabilidad implícita del mercado para YES."""
        return self.yes_price

    @property
    def implied_probability_no(self) -> float:
        """Probabilidad implícita del mercado para NO."""
        return self.no_price

    def resumen(self) -> str:
        """Retorna un resumen legible del mercado."""
        dias = self.days_to_resolution
        dias_str = f"{dias}d" if dias is not None else "N/A"
        return (
            f"[{self.condition_id[:8]}] {self.question}\n"
            f"  YES: ${self.yes_price:.2f} | NO: ${self.no_price:.2f}\n"
            f"  Vol24h: ${self.volume_24h:,.0f} | Liq: ${self.liquidity:,.0f} | "
            f"Resuelve en: {dias_str}"
        )


class LLMEvaluation(BaseModel):
    """Resultado de la evaluación de probabilidad por el LLM."""
    probability: float = Field(
        ge=0, le=1, description="Probabilidad estimada del evento"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Nivel de confianza en la evaluación"
    )
    reasoning: str = Field(description="Razonamiento del LLM")
    key_factors: list[str] = Field(
        default_factory=list, description="Factores clave considerados"
    )
    risks_to_thesis: list[str] = Field(
        default_factory=list, description="Riesgos a la tesis"
    )
    information_quality: Literal["sufficient", "limited", "poor"] = Field(
        default="limited", description="Calidad de la información disponible"
    )


class TradeSignal(BaseModel):
    """
    Señal de trading generada por la estrategia.

    Contiene toda la información necesaria para decidir y ejecutar
    una operación.
    """
    market_id: str = Field(description="ID del mercado (condition_id)")
    token_id: str = Field(default="", description="Token ID para órdenes CLOB")
    market_question: str = Field(description="Pregunta del mercado")
    side: Literal["YES", "NO"] = Field(description="Lado de la apuesta")
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        default="HOLD", description="Acción a tomar"
    )
    entry_price: float = Field(
        ge=0, le=1, description="Precio actual del token"
    )
    estimated_probability: float = Field(
        ge=0, le=1, description="Nuestra estimación de probabilidad"
    )
    edge: float = Field(description="Diferencia vs. mercado")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confianza del LLM"
    )
    suggested_size_usd: float = Field(
        ge=0, description="Tamaño sugerido en USDC"
    )
    kelly_fraction: float = Field(
        ge=0, le=1, description="Fracción de Kelly utilizada"
    )
    reasoning: str = Field(description="Justificación")
    category: str = Field(default="", description="Categoría del mercado")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Momento de la señal"
    )


class TradeRecord(BaseModel):
    """Registro de un trade ejecutado (o simulado)."""
    trade_id: str = Field(description="ID único del trade")
    market_id: str = Field(description="ID del mercado")
    market_question: str = Field(description="Pregunta del mercado")
    side: Literal["YES", "NO"] = Field(description="Lado de la apuesta")
    action: Literal["BUY", "SELL"] = Field(description="Acción ejecutada")
    price: float = Field(ge=0, le=1, description="Precio de ejecución")
    size_usd: float = Field(ge=0, description="Tamaño en USDC")
    shares: float = Field(ge=0, description="Cantidad de shares")
    status: Literal["pending", "filled", "cancelled", "failed"] = Field(
        default="pending", description="Estado de la orden"
    )
    mode: Literal["paper", "live"] = Field(
        default="paper", description="Modo de ejecución"
    )
    order_id: str = Field(default="", description="ID de la orden en Polymarket")
    reasoning: str = Field(default="", description="Razón del trade")
    pnl: float = Field(default=0, description="PnL realizado")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Momento de creación"
    )
    resolved_at: datetime | None = Field(
        default=None, description="Momento de resolución"
    )
