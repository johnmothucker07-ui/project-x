"""Граница города (п. 4.2).

Берётся из того же снимка OSM, что и всё остальное. Городские выгрузки
обрезаны по рамке, и отношение самого города в них может не попасть целиком:
в выгрузке BBBike по Москве есть 111 районов (admin_level 8), но нет
отношения города (admin_level 4). Поэтому предусмотрен запасной путь —
объединение единиц уровнем ниже, и он ВСЕГДА отмечается в результате,
чтобы подмена не прошла молча.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely import make_valid

from ..acquire.osm import read_layer
from ..geo import utm_epsg


@dataclass
class Boundary:
    """Граница города и то, откуда она взялась."""
    frame: gpd.GeoDataFrame      # одна строка, в UTM
    utm_epsg: int
    source: str                  # relation | admin_level | union_of_level
    parts: int                   # из скольких объектов собрана
    note: str | None = None

    @property
    def area_km2(self) -> float:
        return float(self.frame.geometry.area.iloc[0] / 1e6)

    def as_dict(self) -> dict:
        return {"source": self.source, "parts": self.parts,
                "utm_epsg": self.utm_epsg,
                "area_km2": round(self.area_km2, 1), "note": self.note}


def load(pbf: Path, relation_id: int | None, admin_level: int,
         fallback_level: int | None = 8) -> Boundary:
    """Загрузить границу: по relation id, по admin_level или объединением."""
    admin = read_layer(pbf, "multipolygons",
                       columns=["osm_id", "name", "boundary", "admin_level"])
    admin = admin[admin["boundary"] == "administrative"].copy()
    if admin.empty:
        raise ValueError("в снимке нет ни одной административной границы")
    admin["admin_level"] = admin["admin_level"].astype("string")

    exact = admin[admin["osm_id"].astype("string") == str(relation_id)]
    if len(exact):
        return _finish(exact, "relation", note=None)

    same_level = admin[admin["admin_level"] == str(admin_level)]
    if len(same_level):
        return _finish(same_level, "admin_level", note=None)

    if fallback_level is None:
        raise ValueError(
            f"в снимке нет ни relation {relation_id}, ни admin_level {admin_level}")

    lower = admin[admin["admin_level"] == str(fallback_level)]
    if lower.empty:
        raise ValueError(
            f"в снимке нет ни relation {relation_id}, ни уровней "
            f"{admin_level} и {fallback_level}")
    note = (f"relation {relation_id} и admin_level {admin_level} в снимке "
            f"отсутствуют; граница собрана объединением {len(lower)} единиц "
            f"уровня {fallback_level} — выгрузка обрезана по рамке")
    return _finish(lower, "union_of_level", note=note)


def _finish(parts: gpd.GeoDataFrame, source: str, note: str | None) -> Boundary:
    """Собрать один полигон, починить геометрию и перевести в UTM."""
    parts = parts.copy()
    broken = ~parts.geometry.is_valid
    if broken.any():
        parts.loc[broken, parts.geometry.name] = parts.loc[broken].geometry.apply(make_valid)

    shape = parts.union_all()
    geographic = gpd.GeoDataFrame(geometry=[shape], crs=parts.crs)
    centroid = geographic.geometry.iloc[0].centroid
    epsg = utm_epsg(centroid.x, centroid.y)
    return Boundary(frame=geographic.to_crs(epsg=epsg), utm_epsg=epsg,
                    source=source, parts=len(parts), note=note)
