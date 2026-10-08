"""Прогон: идентификатор, папка результатов, манифест.

Требование пропозала (п. 2 «ОТЧЁТ ≠ РЕЗУЛЬТАТ»): из сохранённых входов и кода
метрики должны пересчитываться. Значит каждый запуск обязан записать, на каком
коммите и на каких файлах он сделан.

Ничего не перезаписывается: у каждого запуска своя папка runs/<city>/<run_id>/.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git(*args: str) -> str | None:
    """Вызов git; None, если git недоступен или это не репозиторий."""
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True,
                             timeout=15, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict:
    """Коммит и чистота рабочего дерева.

    Грязное дерево — не ошибка сама по себе, но запуск на городах проверки
    при нём запрещён (п. 13), поэтому флаг пишется всегда."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    """Хеш файла. Читаем кусками — входные файлы бывают по гигабайту."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def make_run_id(command: str) -> str:
    """Идентификатор вида 20261008T143012Z-prepare-city."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{command}"


class Run:
    """Папка прогона и его манифест.

    Использование:
        with Run.start("moscow", "prepare-city", cfg) as run:
            run.record_input("raw/moscow/2026-10-08/city.osm.pbf")
            ...
            run.record_output(run.dir / "grid.parquet")
    """

    def __init__(self, city: str, command: str, cfg: dict, root: Path) -> None:
        self.city = city
        self.command = command
        self.cfg = cfg
        self.run_id = make_run_id(command)
        self.dir = root / cfg["paths"]["runs"] / city / self.run_id
        self._inputs: dict[str, str] = {}
        self._outputs: dict[str, str] = {}
        self._started = datetime.now(timezone.utc)

    @classmethod
    def start(cls, city: str, command: str, cfg: dict,
              root: str | Path = ".") -> "Run":
        run = cls(city, command, cfg, Path(root))
        run.dir.mkdir(parents=True, exist_ok=False)
        # копия эффективного конфига — чтобы прогон читался без доступа к репозиторию
        with (run.dir / "effective_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return run

    def record_input(self, path: str | Path) -> None:
        """Запомнить входной файл и его хеш."""
        self._inputs[str(path)] = sha256_of(path)

    def record_output(self, path: str | Path) -> None:
        """Запомнить выходной файл и его хеш."""
        self._outputs[str(path)] = sha256_of(path)

    def manifest(self) -> dict:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "city": self.city,
            "started_at": self._started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "git": git_state(),
            "python": sys.version.split()[0],
            "packages": _frozen_packages(),
            "snapshot_date": self.cfg.get("snapshot_date"),
            "inputs": self._inputs,
            "outputs": self._outputs,
        }

    def finish(self) -> Path:
        """Записать manifest.json. Возвращает путь к нему."""
        path = self.dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.manifest(), f, ensure_ascii=False, indent=2)
        return path

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # манифест пишем и при падении: нужно знать, на чём прогон оборвался
        self.finish()


def _frozen_packages() -> list[str]:
    """pip freeze окружения. Пустой список, если pip недоступен."""
    try:
        out = subprocess.run((sys.executable, "-m", "pip", "freeze"),
                             capture_output=True, text=True, timeout=60,
                             encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.splitlines() if out.returncode == 0 else []
