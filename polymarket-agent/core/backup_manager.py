"""
Módulo de respaldo asíncrono de la base de datos SQLite.

Realiza copias de seguridad periódicas de trades.db cada 24 horas
en la carpeta /backups/ sin bloquear el loop principal del agente.

Usa ``asyncio.to_thread`` para delegar la operación de copia al
pool de hilos del sistema operativo, manteniendo el loop async libre.
"""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Intervalo entre backups (segundos)
BACKUP_INTERVAL_SECONDS: int = 24 * 3600  # 24 horas
# Máximo de backups a conservar (los más antiguos se eliminan)
MAX_BACKUPS: int = 7


class BackupManager:
    """
    Gestor de respaldos periódicos de la base de datos SQLite.

    Ejecuta un loop async que cada 24 horas copia ``trades.db``
    al directorio de backups usando un hilo separado para no
    bloquear las operaciones de escaneo y evaluación del agente.

    Uso típico::

        manager = BackupManager(settings.agent.database_path)
        backup_task = asyncio.create_task(manager.iniciar())
        # ... agente corriendo ...
        manager.detener()
        backup_task.cancel()
    """

    def __init__(
        self,
        db_path: str,
        backup_dir: str = "backups",
        interval_seconds: int = BACKUP_INTERVAL_SECONDS,
        max_backups: int = MAX_BACKUPS,
    ) -> None:
        self._db_path = Path(db_path)
        self._backup_dir = Path(backup_dir)
        self._interval = interval_seconds
        self._max_backups = max_backups
        self._running = False

        # Crear directorio de backups si no existe
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    async def iniciar(self) -> None:
        """
        Inicia el loop de backup periódico.

        Hace un backup inmediatamente al arrancar, luego espera
        ``interval_seconds`` entre cada copia.  El loop continúa
        hasta que se llame a ``detener()``.
        """
        self._running = True
        logger.info(
            f"BackupManager iniciado: cada {self._interval // 3600}h → "
            f"{self._backup_dir.resolve()}"
        )

        while self._running:
            try:
                dest = await self._hacer_backup_async()
                if dest:
                    await self._limpiar_backups_antiguos()
            except Exception as exc:
                logger.error(f"BackupManager: error durante backup: {exc}")

            # Esperar hasta el próximo ciclo (cancelable)
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    def detener(self) -> None:
        """Señala al loop de backup que se detenga en el próximo ciclo."""
        self._running = False
        logger.info("BackupManager: señal de detención recibida")

    async def hacer_backup_ahora(self) -> Path | None:
        """
        Fuerza un backup inmediato (útil para pruebas o shutdown).

        Returns:
            Path del archivo de backup creado, o None si falló.
        """
        return await self._hacer_backup_async()

    # =========================================================================
    # Privados
    # =========================================================================

    async def _hacer_backup_async(self) -> Path | None:
        """
        Delega la copia de archivo a un hilo del SO (no bloquea el loop).

        ``asyncio.to_thread`` envía ``_copiar_db`` al ThreadPoolExecutor
        del loop para que la operación de I/O de disco no congele las
        coroutines del agente.
        """
        return await asyncio.to_thread(self._copiar_db)

    def _copiar_db(self) -> Path | None:
        """
        Copia ``trades.db`` al directorio de backups (ejecuta en hilo).

        Usa ``shutil.copy2`` para preservar metadatos del archivo.
        El nombre del backup incluye timestamp ISO para ordenamiento
        cronológico directo.
        """
        if not self._db_path.exists():
            logger.warning(
                f"BackupManager: DB no encontrada en {self._db_path}"
            )
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = self._db_path.stem  # "portfolio" si es "portfolio.db"
        dest = self._backup_dir / f"{stem}_{timestamp}.db"

        try:
            shutil.copy2(self._db_path, dest)
            size_kb = dest.stat().st_size / 1024
            logger.info(
                f"BackupManager: backup creado → {dest.name} "
                f"({size_kb:.1f} KB)"
            )
            return dest
        except OSError as exc:
            logger.error(f"BackupManager: error copiando DB: {exc}")
            return None

    async def _limpiar_backups_antiguos(self) -> None:
        """Elimina en hilo los backups más antiguos si supera max_backups."""
        await asyncio.to_thread(self._rotar_backups)

    def _rotar_backups(self) -> None:
        """
        Elimina los backups más antiguos si hay más de ``max_backups``.

        Ordena por nombre (que incluye timestamp) y borra los primeros.
        """
        stem = self._db_path.stem
        backups = sorted(self._backup_dir.glob(f"{stem}_*.db"))
        exceso = len(backups) - self._max_backups
        for old in backups[:exceso]:
            try:
                old.unlink()
                logger.info(f"BackupManager: backup antiguo eliminado → {old.name}")
            except OSError as exc:
                logger.warning(f"BackupManager: no se pudo eliminar {old.name}: {exc}")
