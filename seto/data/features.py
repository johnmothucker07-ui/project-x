"""Признаки клетки по её полигону (п. 4.7).

Считаются по ПОЛИГОНУ клетки, а не по bbox и не по центроиду: именно доли
площади нужны правилу допустимых площадок (п. 8), и именно их нельзя было
получить в MVP, где Overpass отдавал полигоны центроидами.

Один файл cell_features.parquet используется трижды: для допустимости (п. 8),
для текста варианта C (п. 7.3) и для передачи команде AHP (п. 10.5).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

# какие building=* считаем жилыми и общественными (п. 3.1: «жилая или
# общественная застройка»)
RESIDENTIAL = ("apartments", "house", "detached", "residential", "dormitory",
               "terrace", "semidetached_house")
PUBLIC = ("commercial", "office", "retail", "public", "civic", "school",
          "hospital", "university", "kindergarten")


def _area_share(cells: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame,
                name: str) -> pd.Series:
    """Доля площади клетки, накрытая полигонами слоя.

    Полигоны слоя объединяются перед пересечением: иначе наложения
    (здание внутри landuse) посчитались бы дважды и доля превысила бы 1."""
    empty = pd.Series(0.0, index=cells.index, name=name)
    if polygons is None or polygons.empty:
        return empty

    # Сначала режем по клеткам, и только потом объединяем — внутри клетки
    # полигонов десятки, а на весь город их сотни тысяч, и один общий
    # union_all на городском масштабе считается минутами.
    pieces = gpd.overlay(cells[["cell_id", "geometry"]],
                         polygons[["geometry"]], how="intersection",
                         keep_geom_type=True)
    if pieces.empty:
        return empty

    merged = pieces.dissolve(by="cell_id")      # unary_union внутри каждой клетки
    share = (cells["cell_id"].map(merged.geometry.area).fillna(0.0)
             / cells.geometry.area)
    return share.clip(upper=1.0).rename(name)


def _count_within(cells: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
                  name: str) -> pd.Series:
    """Сколько точек слоя попало в клетку."""
    empty = pd.Series(0, index=cells.index, name=name, dtype="int64")
    if points is None or points.empty:
        return empty
    joined = gpd.sjoin(points[["geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    counts = joined.groupby("cell_id").size()
    return cells["cell_id"].map(counts).fillna(0).astype("int64").rename(name)


def _all_poi(points: gpd.GeoDataFrame,
             poi_polygons: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    """Точечные и полигональные объекты в одном наборе.

    Школы, детсады и супермаркеты в OSM чаще нарисованы полигоном, а не точкой:
    если считать только точки, школ в районе находится 9 вместо нескольких
    десятков. Полигон представляем внутренней точкой."""
    if poi_polygons is None or poi_polygons.empty:
        return points
    as_points = poi_polygons.assign(
        geometry=poi_polygons.geometry.representative_point())
    if points is None or points.empty:
        return as_points
    # объединяем по union колонок, а не по пересечению: у полигонов нет
    # highway и public_transport, и пересечение молча обнулило бы остановки
    columns = list(dict.fromkeys([*points.columns, *as_points.columns]))
    joined = pd.concat([points.reindex(columns=columns),
                        as_points.reindex(columns=columns)], ignore_index=True)
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=points.crs)


def build(cells: gpd.GeoDataFrame, layers: dict, roads: gpd.GeoDataFrame,
          points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Таблица признаков: одна строка на клетку.

    cells и все слои должны быть в одной метрической проекции."""
    if cells.crs is None or cells.crs.is_geographic:
        raise ValueError("клетки должны быть в метрической проекции (UTM)")
    points = _all_poi(points, layers.get("poi_polygons"))

    buildings = layers.get("buildings")
    features = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()})

    # --- застройка ---
    if buildings is not None and not buildings.empty:
        residential = buildings[buildings["building"].isin(RESIDENTIAL)]
        public = buildings[buildings["building"].isin(PUBLIC)]
        features["built_share"] = _area_share(cells, buildings, "built_share").to_numpy()
        features["residential_share"] = _area_share(
            cells, residential, "residential_share").to_numpy()
        features["public_share"] = _area_share(cells, public, "public_share").to_numpy()
        features["buildings_count"] = _count_within(
            cells, buildings.assign(geometry=buildings.geometry.representative_point()),
            "buildings_count").to_numpy()
        features["mean_levels"] = _mean_levels(cells, buildings).to_numpy()
        features["levels_known_share"] = _levels_known(cells, buildings).to_numpy()
    else:
        for column in ("built_share", "residential_share", "public_share"):
            features[column] = 0.0
        features["buildings_count"] = 0
        features["mean_levels"] = np.nan
        features["levels_known_share"] = 0.0

    # --- препятствия: именно они отсекают клетку (п. 8) ---
    features["water_share"] = _area_share(cells, layers.get("water"),
                                          "water_share").to_numpy()
    features["green_share"] = _area_share(cells, layers.get("green"),
                                          "green_share").to_numpy()
    landuse = layers.get("landuse")
    for value in ("industrial", "railway", "cemetery", "military"):
        subset = None if landuse is None else landuse[landuse["landuse"] == value]
        features[f"landuse_{value}_share"] = _area_share(
            cells, subset, f"landuse_{value}_share").to_numpy()

    # --- точечные объекты для текста варианта C (п. 7.3) ---
    if points is not None and not points.empty:
        for name, column, values in (
                ("schools", "amenity", ("school",)),
                ("kindergartens", "amenity", ("kindergarten",)),
                ("pharmacies", "amenity", ("pharmacy",)),
                ("shops", "shop", ("supermarket", "convenience")),
                ("transport_stops", "public_transport", ("platform",)),
        ):
            subset = points[points[column].isin(values)] if column in points else None
            features[name] = _count_within(cells, subset, name).to_numpy()
        bus = points[points["highway"] == "bus_stop"] if "highway" in points else None
        features["transport_stops"] += _count_within(cells, bus, "bus").to_numpy()

    # --- плотность пешеходной сети ---
    features["street_density_m_per_km2"] = _street_density(cells, roads).to_numpy()
    return features


def _mean_levels(cells: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.Series:
    """Средняя этажность среди зданий С ТЕГОМ. Пропуск остаётся пропуском."""
    known = buildings[buildings["levels"].notna()]
    if known.empty:
        return pd.Series(np.nan, index=cells.index)
    points = known.assign(geometry=known.geometry.representative_point())
    joined = gpd.sjoin(points[["levels", "geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    mean = joined.groupby("cell_id")["levels"].mean()
    return cells["cell_id"].map(mean)


def _levels_known(cells: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> pd.Series:
    """Доля зданий клетки с известной этажностью — честность текста для C."""
    points = buildings.assign(geometry=buildings.geometry.representative_point())
    joined = gpd.sjoin(points[["levels", "geometry"]], cells[["cell_id", "geometry"]],
                       how="inner", predicate="within")
    if joined.empty:
        return pd.Series(0.0, index=cells.index)
    share = joined.groupby("cell_id")["levels"].apply(lambda s: s.notna().mean())
    return cells["cell_id"].map(share).fillna(0.0)


def _street_density(cells: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Метры улиц на квадратный километр клетки."""
    if roads is None or roads.empty:
        return pd.Series(0.0, index=cells.index)
    pieces = gpd.overlay(cells[["cell_id", "geometry"]],
                         roads[["geometry"]], how="intersection",
                         keep_geom_type=False)
    if pieces.empty:
        return pd.Series(0.0, index=cells.index)
    length = pieces.groupby("cell_id").geometry.apply(lambda g: g.length.sum())
    area_km2 = cells.geometry.area / 1e6
    return (cells["cell_id"].map(length).fillna(0.0) / area_km2).rename(
        "street_density_m_per_km2")
