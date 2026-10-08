"""Регулярная сетка клеток в UTM (п. 4.4).

Решение Д5: проекция метрическая, начало координат округляется вниз до кратного
размеру клетки. Привязка к проекции, а не к границе города — иначе правка
границы на десяток метров сдвинула бы все cell_id, а на них завязаны кэш
ответов модели и имена файлов со снимками.

Отличие от MVP зафиксировано в BASELINE_AUDIT.md: там сетка строилась
в градусах и была квадратной лишь приближённо.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from ..geo import grid_origin


def build_grid(boundary: gpd.GeoDataFrame, size_m: float,
               id_format: str = "cell_{:06d}") -> gpd.GeoDataFrame:
    """Сетка, накрывающая границу города.

    Клетки, пересекающие границу, остаются — хранится доля площади внутри
    (п. 4.4). Население считается по всем клеткам, включая непригодные
    для размещения, поэтому обрезать сетку нельзя."""
    if boundary.crs is None or boundary.crs.is_geographic:
        raise ValueError("граница должна быть в метрической проекции (UTM)")

    shape = boundary.union_all()
    min_x, min_y, max_x, max_y = shape.bounds
    origin_x, origin_y = grid_origin(min_x, min_y, size_m)

    columns = int(np.ceil((max_x - origin_x) / size_m))
    rows = int(np.ceil((max_y - origin_y) / size_m))
    if columns <= 0 or rows <= 0:
        raise ValueError("граница пустая: сетку не построить")

    records = []
    geometries = []
    for row in range(rows):
        for column in range(columns):
            x0 = origin_x + column * size_m
            y0 = origin_y + row * size_m
            cell = box(x0, y0, x0 + size_m, y0 + size_m)
            if not cell.intersects(shape):
                continue   # клетки, не касающиеся города, не нужны вовсе
            records.append({
                "cell_id": id_format.format(row * columns + column),
                "col": column,
                "row": row,
            })
            geometries.append(cell)

    grid = gpd.GeoDataFrame(records, geometry=geometries, crs=boundary.crs)
    grid["area_m2"] = grid.geometry.area
    grid["inside_share"] = grid.geometry.intersection(shape).area / grid["area_m2"]
    centroids = grid.geometry.centroid
    grid["x"] = centroids.x
    grid["y"] = centroids.y
    geographic = centroids.to_crs("EPSG:4326")
    grid["lon"] = geographic.x
    grid["lat"] = geographic.y
    return grid.reset_index(drop=True)


def grid_summary(grid: gpd.GeoDataFrame) -> dict:
    """Короткая сводка для манифеста и отчёта."""
    return {
        "cells": int(len(grid)),
        "fully_inside": int((grid["inside_share"] >= 0.999).sum()),
        "on_border": int(((grid["inside_share"] > 0) & (grid["inside_share"] < 0.999)).sum()),
        "cell_size_m": float(np.sqrt(grid["area_m2"].iloc[0])) if len(grid) else None,
    }
