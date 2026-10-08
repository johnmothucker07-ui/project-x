"""Выделение рабочих слоёв из снимка OSM (п. 4.3).

Каждый слой — свой parquet. Геометрия настоящая: полигоны остаются полигонами,
а не центроидами, как было в MVP. Именно это нужно для допустимых площадок —
доля площади клетки под водой, промзоной или парком по центроиду не считается.

Невалидные кольца и самопересечения в OSM обычны, поэтому каждая геометрия
проходит через make_valid: GDAL при чтении предупреждает о них, но не чинит.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid

from ..acquire.osm import read_layer

# значения landuse, которые нас интересуют (остальные просто не нужны)
LANDUSE_KEEP = {
    "residential", "commercial", "retail", "industrial", "railway",
    "cemetery", "military", "farmland", "construction", "grass", "forest",
}

# слои-препятствия: клетка под ними не годится под клинику
BARRIER_LAYERS = ("water", "park", "industrial", "railway", "cemetery", "military")


def _valid(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Починить геометрию и выбросить пустую.

    Пустая и отсутствующая геометрия — разные вещи, и notna() ловит только
    вторую, поэтому проверяем обе явно."""
    keep = frame.geometry.notna() & ~frame.geometry.is_empty
    frame = frame[keep].copy()
    broken = ~frame.geometry.is_valid
    if broken.any():
        frame.loc[broken, frame.geometry.name] = frame.loc[broken].geometry.apply(make_valid)
    # make_valid иногда схлопывает вырожденный полигон в пустую геометрию
    return frame[~frame.geometry.is_empty].reset_index(drop=True)


def _dedupe(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Один объект OSM может прийти и как way, и как relation."""
    keys = [c for c in ("osm_id", "osm_way_id") if c in frame.columns]
    return frame.drop_duplicates(subset=keys).reset_index(drop=True) if keys else frame


def extract_polygons(pbf: Path, bbox: tuple[float, float, float, float]):
    """Полигональные слои одним чтением: здания, землепользование, вода, зелень.

    Читаем multipolygons один раз — чтение 85 МБ занимает секунды десятки,
    и делать его по разу на слой незачем."""
    raw = _dedupe(_valid(read_layer(pbf, "multipolygons", bbox=bbox)))

    buildings = raw[raw["building"].notna()].copy()
    buildings["levels"] = _parse_levels(buildings)

    landuse = raw[raw["landuse"].isin(LANDUSE_KEEP)].copy()
    water = raw[(raw["natural"] == "water")
                | (raw.get("waterway", pd.Series(dtype=object)) == "riverbank")].copy()
    green = raw[raw["leisure"].isin(["park", "garden"])
                | raw["landuse"].isin(["grass", "forest"])].copy()
    poi = raw[raw["amenity"].notna() | raw["shop"].notna()
              | raw["healthcare"].notna()].copy()

    return {"buildings": buildings, "landuse": landuse, "water": water,
            "green": green, "poi_polygons": poi}


def _parse_levels(frame: gpd.GeoDataFrame) -> pd.Series:
    """building:levels приходит строкой и бывает мусорным ("2;3", "нет").

    Пропуск остаётся пропуском: выдумывать этажность нельзя (п. 22)."""
    column = "building_levels" if "building_levels" in frame.columns else None
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    raw = frame[column].astype("string").str.split(";").str[0].str.strip()
    return pd.to_numeric(raw, errors="coerce").astype("Float64")


def extract_roads(pbf: Path, bbox: tuple[float, float, float, float]):
    """Линейные дороги по классам. Пешеходный граф строится отдельно (Ш3)."""
    lines = _valid(read_layer(pbf, "lines", bbox=bbox))
    return lines[lines["highway"].notna()].reset_index(drop=True)


def extract_points(pbf: Path, bbox: tuple[float, float, float, float]):
    """Точечные объекты: медучреждения, школы, магазины, остановки."""
    points = _valid(read_layer(pbf, "points", bbox=bbox))
    keep = (points["amenity"].notna() | points["shop"].notna()
            | points["healthcare"].notna() | points["office"].notna()
            | points["public_transport"].notna()
            | (points["highway"] == "bus_stop"))
    return points[keep].reset_index(drop=True)


def medical_facilities(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame,
                       clinic_tags: dict) -> gpd.GeoDataFrame:
    """Кандидаты в медучреждения из обоих слоёв, полигоны — центроидом.

    Это ещё НЕ сеть государственных поликлиник: классификация требует реестра
    и ручной проверки (п. 4.5). Здесь только собираем кандидатов и сохраняем
    теги, по которым потом будем сопоставлять."""
    amenity = set(clinic_tags.get("amenity", []))
    healthcare = set(clinic_tags.get("healthcare", []))

    def pick(frame: gpd.GeoDataFrame, kind: str) -> gpd.GeoDataFrame:
        mask = frame["amenity"].isin(amenity) | frame["healthcare"].isin(healthcare)
        sub = frame[mask].copy()
        sub["source_geometry"] = kind
        if kind == "polygon":
            # точка нужна для привязки к графу; площадь здания не теряем
            sub["geometry"] = sub.geometry.representative_point()
        return sub

    columns = ["osm_id", "name", "amenity", "healthcare", "healthcare_speciality",
               "operator", "operator_type", "source_geometry", "geometry"]
    parts = [pick(points, "point"), pick(polygons, "polygon")]
    joined = pd.concat(parts, ignore_index=True)
    present = [c for c in columns if c in joined.columns]
    return gpd.GeoDataFrame(joined[present], geometry="geometry", crs=points.crs)


def save_layers(layers: dict[str, gpd.GeoDataFrame], out_dir: Path) -> dict[str, Path]:
    """Разложить слои по parquet. Возвращает пути для манифеста прогона."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in layers.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path)
        paths[name] = path
    return paths


def completeness_report(buildings: gpd.GeoDataFrame, roads: gpd.GeoDataFrame,
                        points: gpd.GeoDataFrame) -> dict:
    """Полнота OSM по городу (п. 4.3).

    Нужна дважды: при отборе городов и чтобы честно описать, из чего собран
    текст варианта C."""
    return {
        "buildings_total": int(len(buildings)),
        "buildings_with_type": int((buildings["building"] != "yes").sum()),
        "buildings_with_levels": int(buildings["levels"].notna().sum()),
        "buildings_levels_share": round(
            float(buildings["levels"].notna().mean()) if len(buildings) else 0.0, 4),
        "roads_total": int(len(roads)),
        "roads_by_class": roads["highway"].value_counts().head(12).to_dict(),
        "points_total": int(len(points)),
    }
