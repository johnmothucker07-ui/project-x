"""
Ввод-вывод: чтение/запись таблиц клеток и инкрементальное сохранение.

Формат: GeoParquet для рабочих данных (быстро, хранит геометрию и типы),
CSV — для финальных человекочитаемых выгрузок.

Инкрементальное сохранение критично для этапа Estimate: если прогон
через API оборвётся, уже посчитанные клетки не должны теряться.
"""
from __future__ import annotations
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Прочитать config.yaml в словарь."""
    config_path = Path(path)
    if not config_path.is_file():
        # частая причина — запуск не из корня проекта, поэтому показываем полный путь
        raise FileNotFoundError(f"config не найден: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
