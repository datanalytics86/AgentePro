"""
Tracking de portafolio con persistencia en SQLite.

Mantiene registro de todas las posiciones, trades y métricas
de performance. Sobrevive a reinicios del agente.

Fase 5 del agente autónomo — optimizado con WAL mode.
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import settings
from core.models import TradeRecord, TradeSignal

logger = logging.getLogger(__name__)


class Position(BaseModel):
    """Posición abierta en un mercado."""
    market_id: str
    market_question: str
    side: Literal["YES", "NO"]
    entry_price: float
    current_price: float
    shares: float
    cost_basis: float  # Total invertido en USDC
    unrealized_pnl: float = 0.0
    category: str = ""
    opened_at: datetime = Field(default_factory=datetime.now)


class PerformanceMetrics(BaseModel):
    """Métricas de performance del portafolio."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    avg_edge: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    sharpe_ratio: float = 0.0


class Portfolio:
    """
    Gestor de portafolio con persistencia en SQLite.

    Optimizado con WAL mode para lecturas no-bloqueantes
    que permiten al Dashboard consultar sin frenar al agente.

    Funcionalidades:
    - Registrar trades ejecutados
    - Trackear posiciones abiertas
    - Calcular PnL realizado y no realizado
    - Persistir estado en disco
    - Calcular métricas de performance
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.agent.database_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Inicializa las tablas de la base de datos con WAL mode."""
        with sqlite3.connect(self._db_path) as conn:
            # Optimización: WAL mode para lecturas no-bloqueantes
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
            conn.execute("PRAGMA temp_store=MEMORY")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    market_question TEXT,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    size_usd REAL NOT NULL,
                    shares REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    mode TEXT DEFAULT 'paper',
                    order_id TEXT DEFAULT '',
                    reasoning TEXT DEFAULT '',
                    pnl REAL DEFAULT 0,
                    category TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS balance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    question TEXT,
                    market_price REAL,
                    estimated_prob REAL,
                    confidence TEXT,
                    edge REAL,
                    reasoning TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            # Índices para consultas rápidas del Dashboard
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_market
                ON trades(market_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_created
                ON trades(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON trades(status, resolved_at)
            """)

            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Crea una conexión con WAL mode habilitado."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # =========================================================================
    # Registro de trades
    # =========================================================================

    def registrar_trade(
        self,
        signal: TradeSignal,
        price: float,
        shares: float,
        status: str = "filled",
        order_id: str = "",
        category: str = "",
    ) -> TradeRecord:
        """
        Registra un trade ejecutado en el portafolio.

        Args:
            signal: Señal original que generó el trade.
            price: Precio real de ejecución.
            shares: Cantidad de shares compradas.
            status: Estado de la orden.
            order_id: ID de la orden en Polymarket.
            category: Categoría del mercado.

        Returns:
            TradeRecord del trade registrado.
        """
        trade_id = f"TR-{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        record = TradeRecord(
            trade_id=trade_id,
            market_id=signal.market_id,
            market_question=signal.market_question,
            side=signal.side,
            action=signal.action if signal.action in ("BUY", "SELL") else "BUY",
            price=price,
            size_usd=price * shares,
            shares=shares,
            status=status,
            mode=settings.agent.mode,
            order_id=order_id,
            reasoning=signal.reasoning,
            created_at=now,
        )

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO trades
                (trade_id, market_id, market_question, side, action,
                 price, size_usd, shares, status, mode, order_id,
                 reasoning, pnl, category, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.trade_id, record.market_id,
                    record.market_question, record.side, record.action,
                    record.price, record.size_usd, record.shares,
                    record.status, record.mode, record.order_id,
                    record.reasoning, record.pnl, category,
                    now.isoformat(), None,
                ),
            )
            conn.commit()

        logger.info(
            f"Trade registrado: {trade_id} | {record.action} {record.side} "
            f"${record.size_usd:.2f} @ {record.price:.3f}"
        )
        return record

    def cerrar_posicion(
        self,
        market_id: str,
        side: str,
        precio_cierre: float,
    ) -> float:
        """
        Cierra una posición y calcula PnL realizado.

        Returns:
            PnL realizado en USDC.
        """
        trades = self.obtener_trades_por_mercado(market_id)
        buys = [
            t for t in trades
            if t["action"] == "BUY" and t["side"] == side
            and t["status"] == "filled" and t["pnl"] == 0
        ]

        pnl_total = 0.0
        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            for trade in buys:
                # PnL = (precio_cierre - precio_entrada) * shares
                if side == "YES":
                    pnl = (precio_cierre - trade["price"]) * trade["shares"]
                else:
                    pnl = (trade["price"] - precio_cierre) * trade["shares"]

                conn.execute(
                    "UPDATE trades SET pnl = ?, resolved_at = ? WHERE trade_id = ?",
                    (pnl, now, trade["trade_id"]),
                )
                pnl_total += pnl

            conn.commit()

        logger.info(
            f"Posición cerrada: {market_id[:12]} {side} | "
            f"PnL: ${pnl_total:+.2f}"
        )
        return pnl_total

    # =========================================================================
    # Consultas
    # =========================================================================

    def obtener_posiciones_abiertas(self) -> list[Position]:
        """Retorna todas las posiciones abiertas (compras sin cerrar)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT market_id, market_question, side, price,
                          SUM(shares) as total_shares,
                          SUM(size_usd) as total_cost,
                          category
                   FROM trades
                   WHERE action = 'BUY' AND status = 'filled'
                     AND pnl = 0 AND resolved_at IS NULL
                   GROUP BY market_id, side""",
            ).fetchall()

        posiciones: list[Position] = []
        for row in rows:
            posiciones.append(
                Position(
                    market_id=row["market_id"],
                    market_question=row["market_question"] or "",
                    side=row["side"],
                    entry_price=row["price"],
                    current_price=row["price"],  # Se actualiza externamente
                    shares=row["total_shares"],
                    cost_basis=row["total_cost"],
                    category=row["category"] or "",
                )
            )

        return posiciones

    def obtener_trades_por_mercado(self, market_id: str) -> list[dict]:
        """Retorna todos los trades de un mercado."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE market_id = ? ORDER BY created_at",
                (market_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def obtener_trades_recientes(self, limit: int = 20) -> list[dict]:
        """Retorna los trades más recientes."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Métricas de riesgo
    # =========================================================================

    def calcular_exposicion_total(self) -> float:
        """Capital total expuesto en posiciones abiertas (USD)."""
        posiciones = self.obtener_posiciones_abiertas()
        return sum(p.cost_basis for p in posiciones)

    def calcular_exposicion_mercado(self, market_id: str) -> float:
        """Capital expuesto en un mercado específico."""
        posiciones = self.obtener_posiciones_abiertas()
        return sum(
            p.cost_basis for p in posiciones
            if p.market_id == market_id
        )

    def calcular_exposicion_categoria(self, category: str) -> float:
        """Capital expuesto en una categoría."""
        posiciones = self.obtener_posiciones_abiertas()
        return sum(
            p.cost_basis for p in posiciones
            if p.category == category
        )

    def calcular_perdida_diaria(self) -> float:
        """Pérdida acumulada hoy (positivo = pérdida)."""
        hoy = datetime.now().strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            result = conn.execute(
                """SELECT COALESCE(SUM(pnl), 0) FROM trades
                   WHERE resolved_at LIKE ? AND pnl < 0""",
                (f"{hoy}%",),
            ).fetchone()
        return abs(result[0]) if result else 0.0

    def calcular_perdida_semanal(self) -> float:
        """Pérdida acumulada esta semana."""
        hace_7d = (datetime.now() - timedelta(days=7)).isoformat()
        with self._get_conn() as conn:
            result = conn.execute(
                """SELECT COALESCE(SUM(pnl), 0) FROM trades
                   WHERE resolved_at > ? AND pnl < 0""",
                (hace_7d,),
            ).fetchone()
        return abs(result[0]) if result else 0.0

    # =========================================================================
    # Performance
    # =========================================================================

    def calcular_metricas(self) -> PerformanceMetrics:
        """Calcula las métricas de performance del portafolio."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row

            # Trades cerrados
            trades = conn.execute(
                """SELECT * FROM trades
                   WHERE status = 'filled' AND resolved_at IS NOT NULL"""
            ).fetchall()

        total = len(trades)
        ganadores = sum(1 for t in trades if t["pnl"] > 0)
        perdedores = sum(1 for t in trades if t["pnl"] < 0)
        pnl_realizado = sum(t["pnl"] for t in trades)

        # PnL no realizado
        posiciones = self.obtener_posiciones_abiertas()
        pnl_no_realizado = sum(p.unrealized_pnl for p in posiciones)

        # Win rate
        win_rate = ganadores / total if total > 0 else 0.0

        # Drawdown (simplificado)
        bankroll = settings.risk.max_bankroll_usd
        balance_actual = bankroll + pnl_realizado
        peak = max(bankroll, balance_actual)
        drawdown = (peak - balance_actual) / peak if peak > 0 else 0.0

        return PerformanceMetrics(
            total_trades=total,
            winning_trades=ganadores,
            losing_trades=perdedores,
            win_rate=win_rate,
            total_pnl=pnl_realizado + pnl_no_realizado,
            realized_pnl=pnl_realizado,
            unrealized_pnl=pnl_no_realizado,
            max_drawdown=drawdown,
            current_drawdown=drawdown,
            peak_balance=peak,
        )

    def registrar_balance(self, balance: float) -> None:
        """Registra el balance actual para tracking histórico."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO balance_history (balance, timestamp) VALUES (?, ?)",
                (balance, datetime.now().isoformat()),
            )
            conn.commit()

    def obtener_historial_balance(self, limit: int = 100) -> list[dict]:
        """Retorna el historial de balance."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM balance_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
