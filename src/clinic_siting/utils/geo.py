"""
Геометрические утилиты: проекции, bbox, расстояния.
Используется всеми этапами. Держим гео-логику в одном месте,
чтобы не дублировать перевод координат и не путать lon/lat.
"""
from __future__ import annotations

import math

# средняя длина дуги в 1° широты, метров (Земля ≈ сфера) — для перевода метров в градусы
_METERS_PER_DEG_LAT = 111_320.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Расстояние по прямой между двумя точками (метры), формула гаверсинуса.
    Используется как запасной вариант, если OSRM недоступен."""
    earth_radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * earth_radius_m * math.asin(math.sqrt(a))


def bbox_to_cells(bbox: list[float], cell_size_m: int, id_prefix: str = "cell_") -> list[dict]:
    """Разбить bbox [min_lon, min_lat, max_lon, max_lat] на квадратные клетки.
    Возвращает список клеток: {cell_id, min_lon, min_lat, max_lon, max_lat, center_lon, center_lat}."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # шаг в градусах: по широте постоянный, по долготе — уже к полюсам (зависит от широты).
    # берём среднюю широту района, чтобы клетки были примерно квадратными в метрах.
    center_lat = (min_lat + max_lat) / 2
    lat_step = cell_size_m / _METERS_PER_DEG_LAT
    lon_step = cell_size_m / (_METERS_PER_DEG_LAT * math.cos(math.radians(center_lat)))

    cells: list[dict] = []
    index = 0
    lat0 = min_lat
    while lat0 < max_lat:
        lon0 = min_lon
        while lon0 < max_lon:
            cells.append({
                "cell_id": f"{id_prefix}{index:06d}",
                "min_lon": lon0,
                "min_lat": lat0,
                "max_lon": lon0 + lon_step,
                "max_lat": lat0 + lat_step,
                "center_lon": lon0 + lon_step / 2,
                "center_lat": lat0 + lat_step / 2,
            })
            index += 1
            lon0 += lon_step
        lat0 += lat_step
    return cells


def to_metric(gdf, crs_metric: str = "EPSG:3857"):
    """Перепроецировать GeoDataFrame в метрическую систему (для расчёта метров/площадей)."""
    raise NotImplementedError


def to_geographic(gdf, crs_geographic: str = "EPSG:4326"):
    """Перепроецировать обратно в WGS84 (для хранения/отображения)."""
    raise NotImplementedError
