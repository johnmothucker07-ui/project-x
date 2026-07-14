"""
ЭТАП 1 · OpenStreetMap.
Вход: клетки сетки.
Выход: по каждой клетке — число конкурентов (клиник рядом) и текстовая
выжимка окружения для промпта ("рядом 3 аптеки, 1 клиника, жилая застройка").

ВАЖНО: сырой OSM в модель НЕ отдаём — только компактную выжимку (её делает код).
"""
from __future__ import annotations


def fetch_pois(bbox: list[float], tags: list[str], overpass_url: str):
    """Запросить POI из Overpass по bbox и тегам. Возвращает GeoDataFrame точек."""
    raise NotImplementedError


def count_competitors(cells, clinics):
    """Посчитать для каждой клетки число медучреждений-конкурентов рядом."""
    raise NotImplementedError


def build_context_text(cells, pois) -> dict:
    """Собрать по каждой клетке текстовую выжимку окружения для промпта VLM."""
    raise NotImplementedError
