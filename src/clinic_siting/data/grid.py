"""
ЭТАП 1 · Сетка.
Вход: bbox района + размер клетки (из config).
Выход: GeoDataFrame клеток с cell_id и геометрией-полигоном.
Это первый шаг — cell_id отсюда протаскивается через ВЕСЬ пайплайн.
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from ..utils.geo import bbox_to_cells


def build_grid(bbox: list[float], cell_size_m: int, id_prefix: str = "cell_"):
    """Построить сетку клеток по bbox. Возвращает GeoDataFrame [cell_id, geometry, center_lon, center_lat]."""
    cells = bbox_to_cells(bbox, cell_size_m, id_prefix)

    records = []
    geometries = []
    for cell in cells:
        records.append({
            "cell_id": cell["cell_id"],
            "center_lon": cell["center_lon"],
            "center_lat": cell["center_lat"],
        })
        geometries.append(box(cell["min_lon"], cell["min_lat"], cell["max_lon"], cell["max_lat"]))

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    return gdf[["cell_id", "geometry", "center_lon", "center_lat"]]
