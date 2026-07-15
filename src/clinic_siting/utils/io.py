"""
Ввод-вывод: чтение/запись таблиц клеток и инкрементальное сохранение.

Формат: GeoParquet для рабочих данных (быстро, хранит геометрию и типы),
CSV — для финальных человекочитаемых выгрузок.

Инкрементальное сохранение критично для этапа Estimate: если прогон
через API оборвётся, уже посчитанные клетки не должны теряться.
"""
from __future__ import annotations
import json
from pathlib import Path

import geopandas as gpd
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)


def load_cells(path: str):
    """Загрузить GeoDataFrame клеток из GeoParquet."""
    return gpd.read_parquet(path)


def export_csv(df, path: str) -> None:
    """Выгрузить таблицу в CSV (для отчёта/просмотра, без геометрии)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # геометрию в CSV не пишем — она нечитаема; убираем её, если это GeoDataFrame
    if isinstance(df, gpd.GeoDataFrame):
        df = df.drop(columns=df.geometry.name)
    df.to_csv(path, index=False, encoding="utf-8")


def cache_get(key: str, cache_dir: str):
    """Достать закэшированный ответ модели по ключу (хеш входа). None, если нет."""
    path = Path(cache_dir) / f"{key}.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cache_set(key: str, value: dict, cache_dir: str) -> None:
    """Сохранить ответ модели в кэш."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{key}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
