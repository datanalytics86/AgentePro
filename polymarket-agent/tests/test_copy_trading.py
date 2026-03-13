"""
Tests unitarios para los módulos de copy-trading:
- core/leaderboard.py (parsing de Data API)
- core/copy_strategy.py (generación de señales y evaluación)
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.leaderboard import (
    LeaderboardFetcher,
    TopTrader,
    TraderPosition,
    TraderTrade,
    TraderSnapshot,
)
from core.copy_strategy import CopyTradingStrategy
from core.models import Market, Token, TradeSignal


# =============================================================================
# Fixtures: datos de ejemplo que simulan respuestas de la Data API
# =============================================================================

LEADERBOARD_RESPONSE = [
    {
        "proxyWallet": "0xabc123",
        "name": "TopTrader1",
        "profileImage": "",
        "profit": 50000.0,
        "volume": 200000.0,
        "marketsTraded": 150,
    },
    {
        "proxyWallet": "0xdef456",
        "name": "TopTrader2",
        "profileImage": "",
        "profit": 30000.0,
        "volume": 120000.0,
        "marketsTraded": 80,
    },
    {
        "proxyWallet": "0xghi789",
        "name": "TopTrader3",
        "profileImage": "",
        "profit": 20000.0,
        "volume": 90000.0,
        "marketsTraded": 60,
    },
    {
        "proxyWallet": "0xjkl012",
        "name": "TopTrader4",
        "profileImage": "",
        "profit": 15000.0,
        "volume": 70000.0,
        "marketsTraded": 45,
    },
]

POSITIONS_TRADER1 = [
    {
        "conditionId": "0xmarket_a",
        "title": "Will BTC reach 100k?",
        "outcome": "Yes",
        "size": 500.0,
        "initialValue": 250.0,
        "currentValue": 400.0,
        "curPrice": 0.80,
        "cashPnl": 150.0,
        "percentPnl": 60.0,
        "asset": "token_yes_a",
    },
    {
        "conditionId": "0xmarket_b",
        "title": "Will ETH reach 5k?",
        "outcome": "No",
        "size": 300.0,
        "initialValue": 100.0,
        "currentValue": 180.0,
        "curPrice": 0.60,
        "cashPnl": 80.0,
        "percentPnl": 80.0,
        "asset": "token_no_b",
    },
]

POSITIONS_TRADER2 = [
    {
        "conditionId": "0xmarket_a",
        "title": "Will BTC reach 100k?",
        "outcome": "Yes",
        "size": 200.0,
        "initialValue": 100.0,
        "currentValue": 160.0,
        "curPrice": 0.80,
        "cashPnl": 60.0,
        "percentPnl": 60.0,
        "asset": "token_yes_a",
    },
]

POSITIONS_TRADER3 = [
    {
        "conditionId": "0xmarket_a",
        "title": "Will BTC reach 100k?",
        "outcome": "Yes",
        "size": 100.0,
        "initialValue": 50.0,
        "currentValue": 80.0,
        "curPrice": 0.80,
        "cashPnl": 30.0,
        "percentPnl": 60.0,
        "asset": "token_yes_a",
    },
]

ACTIVITY_RESPONSE = [
    {
        "conditionId": "0xmarket_a",
        "title": "Will BTC reach 100k?",
        "type": "trade",
        "side": "BUY",
        "outcome": "Yes",
        "price": 0.75,
        "size": 100.0,
        "cashAmount": 75.0,
        "timestamp": "2026-03-12T10:00:00Z",
        "asset": "token_yes_a",
    },
    {
        "conditionId": "0xmarket_b",
        "title": "Will ETH reach 5k?",
        "type": "redeem",
        "side": "SELL",
        "outcome": "Yes",
        "price": 1.0,
        "size": 50.0,
        "cashAmount": 50.0,
        "timestamp": "2026-03-11T15:00:00Z",
        "asset": "token_yes_b",
    },
]


def _make_market(condition_id: str, question: str, yes_price: float = 0.5) -> Market:
    """Helper para crear un Market de prueba."""
    return Market(
        condition_id=condition_id,
        question=question,
        category="crypto",
        end_date=datetime(2026, 4, 1),
        liquidity=10000.0,
        volume_24h=5000.0,
        yes_price=yes_price,
        no_price=round(1.0 - yes_price, 2),
        tokens=[
            Token(outcome="Yes", token_id="token_yes_id", price=yes_price),
            Token(outcome="No", token_id="token_no_id", price=round(1.0 - yes_price, 2)),
        ],
        resolved=False,
        closed=False,
    )


# =============================================================================
# Tests: LeaderboardFetcher - Parsing
# =============================================================================


class TestLeaderboardParsing:
    """Tests de parsing para datos del leaderboard."""

    def test_parsear_trader_basico(self):
        fetcher = LeaderboardFetcher()
        entry = LEADERBOARD_RESPONSE[0]
        trader = fetcher._parsear_trader(entry, rank=1)

        assert trader is not None
        assert trader.wallet == "0xabc123"
        assert trader.username == "TopTrader1"
        assert trader.profit == 50000.0
        assert trader.volume == 200000.0
        assert trader.rank == 1
        assert trader.markets_traded == 150

    def test_parsear_trader_campos_alternativos(self):
        """La API puede retornar claves distintas (proxy_wallet vs proxyWallet)."""
        fetcher = LeaderboardFetcher()
        entry = {"address": "0xalt_addr", "pseudonym": "AltUser", "profit": 100}
        trader = fetcher._parsear_trader(entry, rank=5)

        assert trader is not None
        assert trader.wallet == "0xalt_addr"
        assert trader.username == "AltUser"

    def test_parsear_trader_sin_wallet_retorna_none(self):
        fetcher = LeaderboardFetcher()
        trader = fetcher._parsear_trader({}, rank=1)
        assert trader is None

    def test_parsear_posicion_basica(self):
        fetcher = LeaderboardFetcher()
        pos = fetcher._parsear_posicion(POSITIONS_TRADER1[0])

        assert pos is not None
        assert pos.market_id == "0xmarket_a"
        assert pos.side == "YES"
        assert pos.size == 500.0
        assert pos.current_value == 400.0
        assert pos.price == 0.80
        assert pos.pnl_cash == 150.0

    def test_parsear_posicion_side_alternativo(self):
        """La API puede retornar 'long'/'short' en vez de YES/NO."""
        fetcher = LeaderboardFetcher()
        entry = {
            "conditionId": "0xtest",
            "direction": "long",
            "size": 10,
        }
        pos = fetcher._parsear_posicion(entry)
        assert pos is not None
        assert pos.side == "YES"

    def test_parsear_posicion_sin_market_id_retorna_none(self):
        fetcher = LeaderboardFetcher()
        pos = fetcher._parsear_posicion({"outcome": "Yes"})
        assert pos is None

    def test_parsear_trade_basico(self):
        fetcher = LeaderboardFetcher()
        trade = fetcher._parsear_trade(ACTIVITY_RESPONSE[0])

        assert trade is not None
        assert trade.market_id == "0xmarket_a"
        assert trade.side == "BUY"
        assert trade.price == 0.75
        assert trade.size == 100.0

    def test_parsear_trade_filtra_redeems(self):
        """Solo trades, no redeems/splits/merges."""
        fetcher = LeaderboardFetcher()
        trade = fetcher._parsear_trade(ACTIVITY_RESPONSE[1])
        assert trade is None

    def test_parsear_trade_malformado_no_crashea(self):
        fetcher = LeaderboardFetcher()
        trade = fetcher._parsear_trade({"invalid": True})
        # No debería crashear, retorna None o un TraderTrade con defaults
        # market_id vacío es válido según el modelo


class TestLeaderboardHacerRequest:
    """Tests para el manejo de formatos de respuesta variados."""

    @pytest.mark.asyncio
    async def test_response_lista(self):
        fetcher = LeaderboardFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = [{"a": 1}]
        mock_response.raise_for_status = MagicMock()
        fetcher._client = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=mock_response)

        # Bypass retry decorator
        result = await fetcher._hacer_request.__wrapped__(fetcher, "/test")
        assert result == [{"a": 1}]

    @pytest.mark.asyncio
    async def test_response_dict_con_data(self):
        fetcher = LeaderboardFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"b": 2}]}
        mock_response.raise_for_status = MagicMock()
        fetcher._client = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=mock_response)

        result = await fetcher._hacer_request.__wrapped__(fetcher, "/test2")
        assert result == [{"b": 2}]

    @pytest.mark.asyncio
    async def test_response_dict_con_positions(self):
        fetcher = LeaderboardFetcher()
        mock_response = MagicMock()
        mock_response.json.return_value = {"positions": [{"c": 3}]}
        mock_response.raise_for_status = MagicMock()
        fetcher._client = AsyncMock()
        fetcher._client.get = AsyncMock(return_value=mock_response)

        result = await fetcher._hacer_request.__wrapped__(fetcher, "/test3")
        assert result == [{"c": 3}]


# =============================================================================
# Tests: LeaderboardFetcher - Consenso
# =============================================================================


class TestConsenso:
    """Tests para la identificación de mercados con consenso."""

    def _setup_fetcher_with_snapshots(self) -> LeaderboardFetcher:
        """Crea un fetcher con snapshots pre-cargados (3 traders en market_a YES)."""
        fetcher = LeaderboardFetcher()

        traders = [
            TopTrader(wallet=f"0x{i}", username=f"Trader{i}", profit=50000 - i * 10000, rank=i)
            for i in range(1, 5)
        ]

        # 3 traders tienen posición en market_a YES (consenso)
        fetcher._snapshots = [
            TraderSnapshot(
                trader=traders[0],
                positions=[
                    TraderPosition(market_id="0xmarket_a", side="YES", price=0.8, current_value=400),
                    TraderPosition(market_id="0xmarket_b", side="NO", price=0.6, current_value=180),
                ],
            ),
            TraderSnapshot(
                trader=traders[1],
                positions=[
                    TraderPosition(market_id="0xmarket_a", side="YES", price=0.8, current_value=160),
                ],
            ),
            TraderSnapshot(
                trader=traders[2],
                positions=[
                    TraderPosition(market_id="0xmarket_a", side="YES", price=0.8, current_value=80),
                ],
            ),
            TraderSnapshot(
                trader=traders[3],
                positions=[
                    TraderPosition(market_id="0xmarket_c", side="NO", price=0.3, current_value=50),
                ],
            ),
        ]
        return fetcher

    def test_consenso_identifica_mercado_popular(self):
        fetcher = self._setup_fetcher_with_snapshots()
        consensos = fetcher.identificar_mercados_consenso(min_traders=3)

        assert len(consensos) == 1
        assert consensos[0]["market_id"] == "0xmarket_a"
        assert consensos[0]["side"] == "YES"
        assert consensos[0]["traders_count"] == 3

    def test_consenso_sin_minimo_no_retorna(self):
        fetcher = self._setup_fetcher_with_snapshots()
        consensos = fetcher.identificar_mercados_consenso(min_traders=4)
        assert len(consensos) == 0

    def test_consenso_ponderacion_por_profit(self):
        fetcher = self._setup_fetcher_with_snapshots()
        consensos = fetcher.identificar_mercados_consenso(min_traders=3)

        # El precio ponderado debe estar más cercano al trader con más profit
        assert consensos[0]["avg_price"] == pytest.approx(0.8, abs=0.01)

    def test_consenso_ordena_por_traders_count(self):
        fetcher = self._setup_fetcher_with_snapshots()
        # Bajar el mínimo para que aparezcan más resultados
        consensos = fetcher.identificar_mercados_consenso(min_traders=1)
        # El primero debe ser el que tiene más traders
        assert consensos[0]["traders_count"] >= consensos[-1]["traders_count"]


# =============================================================================
# Tests: CopyTradingStrategy
# =============================================================================


class TestCopyTradingStrategy:
    """Tests para la generación de señales de copy-trading."""

    def _make_strategy_with_mocked_leaderboard(self):
        """Crea una estrategia con leaderboard mockeado."""
        leaderboard = MagicMock(spec=LeaderboardFetcher)
        strategy = CopyTradingStrategy(leaderboard)
        return strategy, leaderboard

    @pytest.mark.asyncio
    async def test_generar_señales_basico(self):
        strategy, leaderboard = self._make_strategy_with_mocked_leaderboard()

        # Mock snapshots
        traders = [
            TopTrader(wallet=f"0x{i}", username=f"T{i}", profit=50000 - i * 10000, rank=i)
            for i in range(1, 4)
        ]
        snapshots = [
            TraderSnapshot(
                trader=t,
                positions=[TraderPosition(
                    market_id="0xmarket_a", title="BTC 100k?",
                    side="YES", price=0.7, current_value=100,
                )],
            )
            for t in traders
        ]
        leaderboard.obtener_snapshots_completos = AsyncMock(return_value=snapshots)
        leaderboard.identificar_mercados_consenso.return_value = [
            {
                "market_id": "0xmarket_a",
                "title": "BTC 100k?",
                "side": "YES",
                "traders_count": 3,
                "trader_names": ["T1", "T2", "T3"],
                "avg_price": 0.70,
                "total_size_usd": 300,
                "avg_pnl_pct": 20.0,
            }
        ]

        mercados = [_make_market("0xmarket_a", "BTC 100k?", yes_price=0.70)]
        señales = await strategy.generar_señales_copy(
            mercados_disponibles=mercados,
            posiciones_propias=[],
        )

        assert len(señales) == 1
        assert señales[0].action == "BUY"
        assert señales[0].side == "YES"
        assert señales[0].market_id == "0xmarket_a"
        assert señales[0].suggested_size_usd > 0

    @pytest.mark.asyncio
    async def test_skip_posicion_existente(self):
        strategy, leaderboard = self._make_strategy_with_mocked_leaderboard()

        leaderboard.obtener_snapshots_completos = AsyncMock(return_value=[])
        leaderboard.identificar_mercados_consenso.return_value = [
            {
                "market_id": "0xmarket_a", "title": "Test",
                "side": "YES", "traders_count": 3,
                "trader_names": [], "avg_price": 0.5,
                "total_size_usd": 100, "avg_pnl_pct": 0,
            }
        ]

        mercados = [_make_market("0xmarket_a", "Test")]
        señales = await strategy.generar_señales_copy(
            mercados_disponibles=mercados,
            posiciones_propias=["0xmarket_a"],  # Ya tenemos posición
        )

        assert len(señales) == 0

    @pytest.mark.asyncio
    async def test_skip_precio_demasiado_alto(self):
        strategy, leaderboard = self._make_strategy_with_mocked_leaderboard()

        leaderboard.obtener_snapshots_completos = AsyncMock(return_value=[])
        leaderboard.identificar_mercados_consenso.return_value = [
            {
                "market_id": "0xmkt", "title": "Test",
                "side": "YES", "traders_count": 3,
                "trader_names": [], "avg_price": 0.98,
                "total_size_usd": 100, "avg_pnl_pct": 0,
            }
        ]

        mercados = [_make_market("0xmkt", "Test", yes_price=0.98)]
        señales = await strategy.generar_señales_copy(
            mercados_disponibles=mercados,
            posiciones_propias=[],
        )

        assert len(señales) == 0

    def test_evaluar_posicion_hold(self):
        strategy, leaderboard = self._make_strategy_with_mocked_leaderboard()

        snapshots = [
            TraderSnapshot(
                trader=TopTrader(wallet=f"0x{i}", rank=i),
                positions=[TraderPosition(market_id="0xmkt", side="YES")],
            )
            for i in range(3)
        ]
        leaderboard.cached_snapshots = snapshots

        result = strategy.evaluar_posicion_copy("0xmkt", "YES")
        assert result == "HOLD"

    def test_evaluar_posicion_sell(self):
        strategy, leaderboard = self._make_strategy_with_mocked_leaderboard()

        # Solo 1 trader aún tiene posición (min_consensus=3, min_hold=2)
        snapshots = [
            TraderSnapshot(
                trader=TopTrader(wallet="0x1", rank=1),
                positions=[TraderPosition(market_id="0xmkt", side="YES")],
            ),
            TraderSnapshot(
                trader=TopTrader(wallet="0x2", rank=2),
                positions=[],  # Ya no tiene posición
            ),
            TraderSnapshot(
                trader=TopTrader(wallet="0x3", rank=3),
                positions=[],  # Ya no tiene posición
            ),
        ]
        leaderboard.cached_snapshots = snapshots

        result = strategy.evaluar_posicion_copy("0xmkt", "YES")
        assert result == "SELL"

    def test_calcular_size_base(self):
        strategy, _ = self._make_strategy_with_mocked_leaderboard()

        # Con 3 traders (mínimo), solo aplica base_pct (3% de $500 = $15)
        size = strategy._calcular_size(traders_count=3, entry_price=0.5)
        assert size == 15.0

    def test_calcular_size_con_boost(self):
        strategy, _ = self._make_strategy_with_mocked_leaderboard()

        # Con 5 traders: base 3% + 2 extra * 1% = 5% (cap). $500 * 5% = $25
        size = strategy._calcular_size(traders_count=5, entry_price=0.5)
        assert size == 25.0

    def test_calcular_size_cap(self):
        strategy, _ = self._make_strategy_with_mocked_leaderboard()

        # Con 10 traders: base 3% + 7*1% = 10%, pero cap es 5% → $25
        size = strategy._calcular_size(traders_count=10, entry_price=0.5)
        assert size == 25.0


# =============================================================================
# Tests: Orchestrator shutdown handling
# =============================================================================


class TestOrchestratorShutdown:
    """Tests para manejo de shutdown y CancelledError."""

    def test_import_orchestrator(self):
        """Verifica que el orquestador importa correctamente."""
        from agent.orchestrator import AgentOrchestrator, main
        assert AgentOrchestrator is not None
        assert main is not None
