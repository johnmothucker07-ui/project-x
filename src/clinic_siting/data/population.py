"""
ЭТАП 1 · Население (Kontur Population).
Вход: клетки сетки + файл Kontur.
Выход: население по каждой клетке.

Kontur отдаёт данные гексагонами H3 (на базе WorldPop). Нужно агрегировать
их на нашу прямоугольную сетку через пространственное соединение.
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

# Kontur хранит геометрию в метрах (Web Mercator); в ней же считаем площади
_METRIC_CRS = "EPSG:3857"


def load_kontur(path: str, bbox: list[float] | None = None):
    """Загрузить данные Kontur Population (GeoPackage с population по гексагонам).
    bbox [min_lon, min_lat, max_lon, max_lat] в градусах — опциональный фильтр области,
    чтобы не читать все ~900 тыс. гексагонов России."""
    if bbox is None:
        return gpd.read_file(path)

    # bbox приходит в градусах (4326), а файл в метрах (3857) — переводим рамку под CRS файла
    min_lon, min_lat, max_lon, max_lat = bbox
    frame = gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")
    bounds_metric = tuple(frame.to_crs(_METRIC_CRS).total_bounds)
    return gpd.read_file(path, bbox=bounds_metric)


def population_per_cell(cells, kontur):
    """Агрегировать население Kontur на клетки сетки (spatial join + сумма с учётом площади).
    Возвращает копию сетки с колонкой population."""
    cells_m = cells[["cell_id", "geometry"]].to_crs(_METRIC_CRS)
    kontur_m = kontur[["population", "geometry"]].to_crs(_METRIC_CRS).copy()
    kontur_m["hex_area"] = kontur_m.geometry.area

    # режем гексагоны по границам клеток; доля населения = доля площади куска в гексагоне
    pieces = gpd.overlay(kontur_m, cells_m, how="intersection")
    pieces["pop_share"] = pieces["population"] * (pieces.geometry.area / pieces["hex_area"])
    per_cell = pieces.groupby("cell_id")["pop_share"].sum()

    result = cells.copy()
    result["population"] = result["cell_id"].map(per_cell).fillna(0.0).round().astype(int)
    return result
