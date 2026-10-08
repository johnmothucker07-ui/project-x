"""Загрузка сырых файлов в raw/ с фиксацией версии.

Принцип п. 4: всё скачивается ОДИН раз в raw/<city>/<snapshot_date>/, хешируется
и дальше читается только оттуда. Повторная загрузка — только новой командой
с новой датой. Живые запросы во время расчётов запрещены (п. 4.3).
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from ..run import sha256_of

_HEADERS = {"User-Agent": "seto-research/0.1 (clinic siting experiment)"}
_CHUNK = 1 << 20


@dataclass
class Source:
    """Описание скачанного файла для манифеста данных (п. 4.8)."""
    name: str
    url: str
    path: str
    sha256: str
    size_bytes: int
    downloaded_at: str
    last_modified: str | None = None
    license: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class DownloadError(RuntimeError):
    """Не удалось скачать файл."""


def _remote_info(url: str) -> tuple[int | None, str | None]:
    """Размер и дата файла на сервере, если сервер их сообщает."""
    request = urllib.request.Request(url, headers=_HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return (int(length) if length else None,
                    response.headers.get("Last-Modified"))
    except (urllib.error.URLError, ValueError):
        # не все зеркала отвечают на HEAD — это не повод падать
        return None, None


def fetch(url: str, target: Path, name: str, license_note: str | None = None,
          overwrite: bool = False, progress: bool = True) -> Source:
    """Скачать файл в target и описать его для манифеста.

    Если файл уже есть и overwrite=False, он не перекачивается: raw/ после
    фиксации считается неизменяемым, а выгрузки весят гигабайты."""
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size, last_modified = _remote_info(url)

    if target.exists() and not overwrite:
        return _describe(name, url, target, last_modified, license_note)

    # качаем во временный файл: обрыв не должен оставить битый файл под нужным именем
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, \
                partial.open("wb") as out:
            done = 0
            while block := response.read(_CHUNK):
                out.write(block)
                done += len(block)
                if progress:
                    _print_progress(name, done, expected_size)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"не удалось скачать {url}: {error}") from error
    if progress:
        print()

    if expected_size is not None and partial.stat().st_size != expected_size:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"{name}: скачано {partial.stat().st_size} байт вместо {expected_size}")

    shutil.move(str(partial), str(target))
    return _describe(name, url, target, last_modified, license_note)


def _describe(name: str, url: str, path: Path, last_modified: str | None,
              license_note: str | None) -> Source:
    return Source(
        name=name,
        url=url,
        path=str(path),
        sha256=sha256_of(path),
        size_bytes=path.stat().st_size,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        last_modified=last_modified,
        license=license_note,
    )


def _print_progress(name: str, done: int, total: int | None) -> None:
    mb = done / (1 << 20)
    if total:
        print(f"\r  {name}: {mb:,.0f} / {total / (1 << 20):,.0f} МБ "
              f"({100 * done / total:.0f} %)", end="", flush=True)
    else:
        print(f"\r  {name}: {mb:,.0f} МБ", end="", flush=True)


def write_data_manifest(sources: list[Source], path: Path,
                        snapshot_date: str, notes: dict | None = None) -> Path:
    """Манифест данных (п. 4.8): по каждому источнику версия, дата, хеш, лицензия.

    Отдельно пишутся расхождения дат источников со snapshot_date — пропозал
    требует указывать их явно (например, снимки 2023 против OSM 2026)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_date": snapshot_date,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "sources": [s.as_dict() for s in sources],
        "notes": notes or {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
