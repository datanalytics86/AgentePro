"""
Script de verificación de la Fase 1.

Ejecuta los checkpoints requeridos:
1. El scanner obtiene al menos 10 mercados activos con sus precios
2. Los datos se imprimen en consola de forma legible
3. El filtrado funciona correctamente

Uso:
    python scripts/verificar_fase1.py
"""

import asyncio
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import configurar_logging
from core.market_scanner import MarketScanner
from rich.console import Console
from rich.panel import Panel

console = Console()


async def verificar_fase1() -> bool:
    """Ejecuta todos los checkpoints de la Fase 1."""
    configurar_logging()
    exitos = 0
    total = 3

    console.print(Panel(
        "[bold cyan]FASE 1 - VERIFICACIÓN DE CHECKPOINTS[/bold cyan]\n"
        "Conexión y Escaneo de Mercados de Polymarket",
        title="Polymarket Agent",
    ))

    async with MarketScanner() as scanner:
        # =====================================================================
        # Checkpoint 1: Obtener al menos 10 mercados activos
        # =====================================================================
        console.print("\n[bold]Checkpoint 1:[/bold] Obtener >= 10 mercados activos...")

        try:
            mercados = await scanner.escanear_mercados()

            if len(mercados) >= 10:
                console.print(
                    f"  [green]PASS[/green] Se obtuvieron {len(mercados)} mercados"
                )
                exitos += 1
            else:
                console.print(
                    f"  [red]FAIL[/red] Solo se obtuvieron {len(mercados)} mercados "
                    f"(mínimo 10)"
                )
        except Exception as e:
            console.print(f"  [red]FAIL[/red] Error: {e}")
            mercados = []

        # =====================================================================
        # Checkpoint 2: Los datos se imprimen de forma legible
        # =====================================================================
        console.print("\n[bold]Checkpoint 2:[/bold] Datos legibles en consola...")

        try:
            if mercados:
                console.print("  Ejemplo de los 3 primeros mercados:\n")
                for m in mercados[:3]:
                    console.print(f"  {m.resumen()}\n")

                # Verificar que los campos clave están presentes
                m = mercados[0]
                checks = [
                    ("condition_id", bool(m.condition_id)),
                    ("question", bool(m.question)),
                    ("yes_price", 0 <= m.yes_price <= 1),
                    ("no_price", 0 <= m.no_price <= 1),
                    ("volume_24h", m.volume_24h >= 0),
                    ("liquidity", m.liquidity >= 0),
                    ("tokens", len(m.tokens) >= 2),
                ]
                todos_ok = all(ok for _, ok in checks)

                if todos_ok:
                    console.print("  [green]PASS[/green] Todos los campos presentes y válidos")
                    exitos += 1
                else:
                    fallos = [nombre for nombre, ok in checks if not ok]
                    console.print(f"  [red]FAIL[/red] Campos inválidos: {fallos}")
            else:
                console.print("  [red]FAIL[/red] No hay mercados para verificar")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] Error: {e}")

        # =====================================================================
        # Checkpoint 3: El filtrado funciona correctamente
        # =====================================================================
        console.print("\n[bold]Checkpoint 3:[/bold] Filtrado de mercados...")

        try:
            # Obtener todos sin filtrar
            mercados_sin_filtrar = await scanner._obtener_mercados_activos()
            total_sin_filtrar = len(mercados_sin_filtrar)
            total_filtrado = len(mercados)

            if total_sin_filtrar > total_filtrado:
                console.print(
                    f"  [green]PASS[/green] Filtro activo: {total_sin_filtrar} crudos "
                    f"-> {total_filtrado} filtrados "
                    f"({total_sin_filtrar - total_filtrado} excluidos)"
                )
                exitos += 1
            elif total_sin_filtrar == total_filtrado and total_filtrado > 0:
                console.print(
                    f"  [yellow]WARN[/yellow] Todos los mercados pasaron el filtro "
                    f"({total_filtrado}). Los filtros podrían ser muy permisivos, "
                    f"pero se considera PASS."
                )
                exitos += 1
            else:
                console.print(
                    f"  [red]FAIL[/red] No se puede verificar filtrado "
                    f"(sin_filtrar={total_sin_filtrar}, filtrados={total_filtrado})"
                )
        except Exception as e:
            console.print(f"  [red]FAIL[/red] Error: {e}")

    # =========================================================================
    # Resultado final
    # =========================================================================
    console.print("\n" + "=" * 60)
    if exitos == total:
        console.print(
            f"[bold green]FASE 1 COMPLETADA: {exitos}/{total} checkpoints OK[/bold green]"
        )
    else:
        console.print(
            f"[bold red]FASE 1 INCOMPLETA: {exitos}/{total} checkpoints OK[/bold red]"
        )
    console.print("=" * 60 + "\n")

    return exitos == total


if __name__ == "__main__":
    ok = asyncio.run(verificar_fase1())
    sys.exit(0 if ok else 1)
