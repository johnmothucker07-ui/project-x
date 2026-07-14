"""
ЭТАП 1 · OpenStreetMap.
Вход: клетки сетки.
Выход: по каждой клетке — число конкурентов (клиник рядом) и текстовая
выжимка окружения для промпта ("рядом 3 аптеки, 1 клиника, жилая застройка").

ВАЖНО: сырой OSM в модель НЕ отдаём — только компактную выжимку (её делает код).
"""
from __future__ import annotations

import geopandas as gpd
import requests
from shapely.geometry import Point

# публичный Overpass просит User-Agent, иначе может отвечать ошибкой
_HEADERS = {"User-Agent": "clinic-siting-student-project/0.1"}


def _build_overpass_query(bbox: list[float], tags: list[str]) -> str:
    """Собрать Overpass QL: объединение node+way по всем тегам в пределах bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    # ВНИМАНИЕ: Overpass ждёт bbox как (south, west, north, east) = (min_lat, min_lon, max_lat, max_lon)
    bb = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    parts = []
    for tag in tags:
        key, value = tag.split("=")
        parts.append(f'node["{key}"="{value}"]({bb});')
        parts.append(f'way["{key}"="{value}"]({bb});')
    return "[out:json][timeout:60];(" + "".join(parts) + ");out center;"


def fetch_pois(bbox: list[float], tags: list[str], overpass_url: str,
               exclude_healthcare: list[str] | None = None):
    """Запросить POI из Overpass по bbox и тегам. Возвращает GeoDataFrame точек.
    exclude_healthcare — значения тега healthcare, которые не считаем конкурентами
    (лаборатории, стоматология и т.п.); такие объекты отсеиваем."""
    query = _build_overpass_query(bbox, tags)
    # запрос шлём form-полем data={"data": ...}, иначе Overpass отвечает 406
    response = requests.post(overpass_url, data={"data": query}, headers=_HEADERS, timeout=90)
    response.raise_for_status()
    elements = response.json().get("elements", [])

    records = []
    geometries = []
    for element in elements:
        # у node координаты прямо в элементе, у way/relation — в center (мы просили out center)
        if element["type"] == "node":
            lon, lat = element["lon"], element["lat"]
        else:
            center = element.get("center")
            if center is None:
                continue  # без координат точку не поставить — пропускаем
            lon, lat = center["lon"], center["lat"]

        element_tags = element.get("tags", {})
        records.append({
            "osm_id": element["id"],
            "osm_type": element["type"],
            "name": element_tags.get("name"),
            "amenity": element_tags.get("amenity"),
            "healthcare": element_tags.get("healthcare"),
            "lon": lon,
            "lat": lat,
        })
        geometries.append(Point(lon, lat))

    pois = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")

    # отсеиваем узкие специализации/не-конкурентов по тегу healthcare
    if exclude_healthcare:
        pois = pois[~pois["healthcare"].isin(exclude_healthcare)].reset_index(drop=True)

    return pois


def count_competitors(cells, clinics):
    """Посчитать для каждой клетки число медучреждений-конкурентов рядом.
    Возвращает копию сетки с колонкой existing_clinics."""
    # каждую клинику относим к клетке, в которую попадает её точка (point within polygon)
    joined = gpd.sjoin(
        clinics[["geometry"]],
        cells[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    counts = joined.groupby("cell_id").size()

    result = cells.copy()
    # клетки без клиник получают 0 (map вернёт NaN → заполняем нулём)
    result["existing_clinics"] = result["cell_id"].map(counts).fillna(0).astype(int)
    return result


# человекочитаемые названия типов медучреждений (по тегу amenity)
_AMENITY_LABELS = {
    "clinic": "клиники",
    "hospital": "больницы",
    "doctors": "кабинеты врачей",
    "dentist": "стоматологии",
}


def build_context_text(cells, pois) -> dict:
    """Собрать по каждой клетке текстовую выжимку окружения для промпта VLM.
    Вариант A: описываем только медучреждения рядом (из pois). Возвращает {cell_id: text}."""
    # по умолчанию рядом ничего — потом перезапишем те клетки, где клиники есть
    context = {cell_id: "Медучреждений рядом не обнаружено." for cell_id in cells["cell_id"]}

    joined = gpd.sjoin(
        pois[["amenity", "geometry"]],
        cells[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    for cell_id, group in joined.groupby("cell_id"):
        total = len(group)
        parts = []
        for amenity, count in group["amenity"].value_counts().items():
            label = _AMENITY_LABELS.get(amenity, amenity or "прочее")
            parts.append(f"{label}: {count}")
        context[cell_id] = f"Рядом медучреждений: {total} ({', '.join(parts)})."
    return context
