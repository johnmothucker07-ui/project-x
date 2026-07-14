"""
Геометрические утилиты: проекции, bbox, расстояния.
Используется всеми этапами. Держим гео-логику в одном месте,
чтобы не дублировать перевод координат и не путать lon/lat.
"""
from __future__ import annotations


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Расстояние по прямой между двумя точками (метры), формула гаверсинуса.
    Используется как запасной вариант, если OSRM недоступен."""
    raise NotImplementedError


def bbox_to_cells(bbox: list[float], cell_size_m: int) -> list[dict]:
    """Разбить bbox [min_lon, min_lat, max_lon, max_lat] на квадратные клетки.
    Возвращает список клеток: {cell_id, min_lon, min_lat, max_lon, max_lat, center_lon, center_lat}."""
    raise NotImplementedError


def to_metric(gdf, crs_metric: str = "EPSG:3857"):
    """Перепроецировать GeoDataFrame в метрическую систему (для расчёта метров/площадей)."""
    raise NotImplementedError


def to_geographic(gdf, crs_geographic: str = "EPSG:4326"):
    """Перепроецировать обратно в WGS84 (для хранения/отображения)."""
    raise NotImplementedError
