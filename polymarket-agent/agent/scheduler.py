"""
Programador de tareas del agente.

Gestiona la programación de tareas periódicas como
escaneos, reportes y mantenimiento.

Fase 8 del agente autónomo.
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Programador simple de tareas periódicas.

    Mantiene un registro de cuándo se ejecutó cada tarea
    y determina si toca ejecutarla de nuevo.
    """

    def __init__(self) -> None:
        self._last_run: dict[str, datetime] = {}

    def debe_ejecutar(
        self, task_name: str, intervalo_minutos: int
    ) -> bool:
        """
        Determina si una tarea debe ejecutarse.

        Args:
            task_name: Nombre de la tarea.
            intervalo_minutos: Minutos entre ejecuciones.

        Returns:
            True si toca ejecutar la tarea.
        """
        ahora = datetime.now()
        ultima = self._last_run.get(task_name)

        if ultima is None:
            return True

        delta = ahora - ultima
        return delta >= timedelta(minutes=intervalo_minutos)

    def marcar_ejecutada(self, task_name: str) -> None:
        """Marca una tarea como ejecutada ahora."""
        self._last_run[task_name] = datetime.now()
        logger.debug(f"Tarea '{task_name}' marcada como ejecutada")

    def tiempo_hasta_proxima(
        self, task_name: str, intervalo_minutos: int
    ) -> float:
        """
        Calcula minutos restantes hasta la próxima ejecución.

        Returns:
            Minutos restantes (0 si ya toca ejecutar).
        """
        ultima = self._last_run.get(task_name)
        if ultima is None:
            return 0.0

        transcurrido = (datetime.now() - ultima).total_seconds() / 60
        restante = intervalo_minutos - transcurrido
        return max(0, restante)
