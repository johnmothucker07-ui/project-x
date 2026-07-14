"""
Ввод-вывод: чтение/запись таблиц клеток и инкрементальное сохранение.

Формат: GeoParquet для рабочих данных (быстро, хранит геометрию и типы),
CSV — для финальных человекочитаемых выгрузок.

Инкрементальное сохранение критично для этапа Estimate: если прогон
через API оборвётся, уже посчитанные клетки не должны теряться.
"""
from __future__ import annotations
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Загрузить config.yaml в словарь."""
    raise NotImplementedError


def save_cells(gdf, path: str) -> None:
    """Сохранить GeoDataFrame клеток в GeoParquet."""
    raise NotImplementedError


def load_cells(path: str):
    """Загрузить GeoDataFrame клеток из GeoParquet."""
    raise NotImplementedError


def export_csv(df, path: str) -> None:
    """Выгрузить таблицу в CSV (для отчёта/просмотра, без геометрии)."""
    raise NotImplementedError


def cache_get(key: str, cache_dir: str):
    """Достать закэшированный ответ модели по ключу (хеш входа). None, если нет."""
    raise NotImplementedError


def cache_set(key: str, value: dict, cache_dir: str) -> None:
    """Сохранить ответ модели в кэш."""
    raise NotImplementedError
