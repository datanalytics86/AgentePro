"""
Script para iniciar el agente en modo paper trading.

Ejecuta el agente con fondos simulados para validar el sistema
antes de usar dinero real.

Uso:
    python scripts/paper_trade.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("TRADING_MODE", "paper")

from agent.orchestrator import main

if __name__ == "__main__":
    main()
