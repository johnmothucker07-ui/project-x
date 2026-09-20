"""
ЭТАП 1 · OpenStreetMap.
Вход: клетки сетки.
Выход: по каждой клетке — число конкурентов (клиник рядом) и текстовая
выжимка окружения для промпта ("рядом 3 аптеки, 1 клиника, жилая застройка").

ВАЖНО: сырой OSM в модель НЕ отдаём — только компактную выжимку (её делает код).
"""
from __future__ import annotations

import hashlib
import time

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from ..utils.io import cache_get, cache_set

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
               exclude_healthcare: list[str] | None = None,
               exclude_name_keywords: list[str] | None = None):
    """Запросить POI из Overpass по bbox и тегам. Возвращает GeoDataFrame точек.
    exclude_healthcare — значения тега healthcare, которые не считаем конкурентами
    (лаборатории, стоматология и т.п.); такие объекты отсеиваем.
    exclude_name_keywords — стоп-слова в названии (космет/стоматолог/…): OSM часто метит
    их как clinic, а по тегу не отличить. Грубая эвристика по имени."""
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

    # отсеиваем косметологию/стоматологию/пластику по стоп-словам в названии
    if exclude_name_keywords:
        name_lower = pois["name"].fillna("").str.lower()
        pattern = "|".join(exclude_name_keywords)
        pois = pois[~name_lower.str.contains(pattern, regex=True)].reset_index(drop=True)

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


# ============================================================
# Окружение (не медицина): сырые признаки района.
# Нужны для двух вещей сразу:
#   1) аналитический baseline (композиционный индекс POI) — конкурент VLM;
#   2) богатая выжимка окружения для промпта (чинит вырожденный area_character).
# ЗДЕСЬ НЕТ ВЕСОВ И ИНДЕКСОВ — только счётчики. Веса вводятся отдельно,
# иначе невозможно построить обучаемый baseline (подгонка под цены).
# ============================================================

# landuse тянем, но в признаки не берём: Overpass отдаёт полигон центроидом (out center),
# и крупный полигон residential, накрывающий десяток клеток, засчитается ровно в одну.
# Тот же сигнал корректно даёт состав building=* — здания мелкие, центроид попадает куда надо.
_EXCLUDED_FROM_FEATURES = {"landuse"}


def _build_environment_query(bbox: list[float], vocabulary: dict, timeout_s: int) -> str:
    """Собрать Overpass QL по словарю признаков {ключ: [значения]}.
    nwr = node+way+relation, out center — полигоны приходят центроидом."""
    min_lon, min_lat, max_lon, max_lat = bbox
    # Overpass ждёт bbox как (south, west, north, east)
    bb = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    parts = []
    for key, values in vocabulary.items():
        for value in values:
            parts.append(f'nwr["{key}"="{value}"]({bb});')
    return f"[out:json][timeout:{timeout_s}];(" + "".join(parts) + ");out center tags;"


def _vocabulary_from_config(env_cfg: dict) -> dict:
    """Выделить из config-секции environment только словарь тегов (без настроек вроде таймаута)."""
    return {key: values for key, values in env_cfg.items() if isinstance(values, list)}


def _split_bbox(bbox: list[float], tile_deg: float) -> list[list[float]]:
    """Разрезать bbox на тайлы. Целиком большой район Overpass не отдаёт (504)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tiles.append([lon, lat, min(lon + tile_deg, max_lon), min(lat + tile_deg, max_lat)])
            lon += tile_deg
        lat += tile_deg
    return tiles


def fetch_environment(bbox: list[float], env_cfg: dict, overpass_url: str):
    """Запросить объекты окружения по фиксированному словарю из config.

    Возвращает GeoDataFrame [osm_id, osm_type, key, value, levels, geometry].
    Один объект может дать НЕСКОЛЬКО строк: здание с магазином внутри попадёт
    и как building=*, и как shop=* — это осознанно, счётчики категорий независимы.
    Запрос идёт тайлами: на полном районе Overpass отвечает 504."""
    vocabulary = _vocabulary_from_config(env_cfg)
    timeout_s = env_cfg.get("overpass_timeout_s", 180)
    tiles = _split_bbox(bbox, env_cfg.get("tile_deg", 0.02))
    pause_s = env_cfg.get("tile_pause_s", 1.0)

    retries = env_cfg.get("tile_retries", 4)
    cache_dir = env_cfg.get("cache_dir", "data/cache/overpass")

    frames = []
    for number, tile in enumerate(tiles, start=1):
        frame, from_cache = _fetch_environment_tile(tile, vocabulary, timeout_s,
                                                    overpass_url, retries, cache_dir)
        frames.append(frame)
        print(f"  тайл {number}/{len(tiles)}: {len(frame)} строк"
              f"{' (из кэша)' if from_cache else ''}", flush=True)
        # пауза нужна только после реального запроса к API
        if not from_cache and number < len(tiles):
            time.sleep(pause_s)

    environment = pd.concat(frames, ignore_index=True)
    # полигон на границе тайлов возвращается несколько раз — центроид тот же, строка дубль
    environment = environment.drop_duplicates(subset=["osm_id", "osm_type", "key", "value"])
    return gpd.GeoDataFrame(environment, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)


def _fetch_environment_tile(bbox: list[float], vocabulary: dict, timeout_s: int,
                            overpass_url: str, retries: int, cache_dir: str):
    """Один тайл: взять из кэша или запросить Overpass с ретраями.
    Возвращает (GeoDataFrame, из_кэша_ли).

    Кэш на диске обязателен: публичный Overpass под нагрузкой отдаёт 429/504, SSL-обрывы
    или HTML-страницу лимита вместо JSON, и без кэша падение одного тайла теряет все
    предыдущие — то же правило, что для ответов модели."""
    query = _build_environment_query(bbox, vocabulary, timeout_s)
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()

    cached = cache_get(key, cache_dir)
    if cached is not None:
        return _elements_to_frame(cached["elements"], vocabulary), True

    elements = None
    last_error = None
    for attempt in range(retries):
        try:
            # form-поле data=..., иначе Overpass отвечает 406 (как и в fetch_pois)
            response = requests.post(overpass_url, data={"data": query}, headers=_HEADERS,
                                     timeout=timeout_s + 30)
            response.raise_for_status()
            elements = response.json().get("elements", [])
            break
        except (requests.RequestException, ValueError) as error:
            last_error = error
            # экспоненциальный backoff: при лимите короткая пауза не помогает
            time.sleep(10 * 2 ** attempt)
    if elements is None:
        raise RuntimeError(f"Overpass не ответил по тайлу {bbox} за {retries} попыток: {last_error}")

    cache_set(key, {"elements": elements}, cache_dir)
    return _elements_to_frame(elements, vocabulary), False


def _elements_to_frame(elements: list, vocabulary: dict):
    """Разобрать ответ Overpass в GeoDataFrame [osm_id, osm_type, key, value, levels, geometry]."""

    records = []
    geometries = []
    for element in elements:
        if element["type"] == "node":
            lon, lat = element["lon"], element["lat"]
        else:
            center = element.get("center")
            if center is None:
                continue  # без координат объект не разместить
            lon, lat = center["lon"], center["lat"]

        tags = element.get("tags", {})
        levels = _parse_levels(tags.get("building:levels"))

        # по строке на каждый тег из словаря, который есть у объекта
        for key, allowed in vocabulary.items():
            value = tags.get(key)
            if value in allowed:
                records.append({
                    "osm_id": element["id"],
                    "osm_type": element["type"],
                    "key": key,
                    "value": value,
                    "levels": levels,
                })
                geometries.append(Point(lon, lat))

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")


def _parse_levels(raw) -> float | None:
    """building:levels приходит строкой и бывает мусорным ("2;3", "нет") — не доверяем вслепую."""
    if raw is None:
        return None
    try:
        return float(str(raw).split(";")[0].strip())
    except ValueError:
        return None


def environment_features(cells, environment, env_cfg: dict):
    """По каждой клетке — счётчики объектов окружения по категориям + средняя этажность.

    Возвращает DataFrame [cell_id, env_<key>_<value>..., mean_building_levels].
    Колонки фиксированы словарём из config (одинаковы для любого города).
    mean_building_levels = NaN там, где зданий с этажностью не нашлось."""
    vocabulary = _vocabulary_from_config(env_cfg)
    # колонки строим из словаря, а не из того, что вернулось — иначе города не сравнить
    columns = [f"env_{key}_{value}"
               for key, values in vocabulary.items() if key not in _EXCLUDED_FROM_FEATURES
               for value in values]

    features = pd.DataFrame(0, index=cells["cell_id"], columns=columns, dtype="int64")
    features["mean_building_levels"] = pd.NA

    joined = gpd.sjoin(
        environment[["key", "value", "levels", "geometry"]],
        cells[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined[~joined["key"].isin(_EXCLUDED_FROM_FEATURES)]

    if len(joined):
        joined["category"] = "env_" + joined["key"] + "_" + joined["value"]
        counts = joined.groupby(["cell_id", "category"]).size().unstack(fill_value=0)
        # оставляем только колонки словаря: если в OSM всплыло что-то вне него — игнорируем
        counts = counts.reindex(columns=columns, fill_value=0)
        features.loc[counts.index, columns] = counts

        levels = joined.dropna(subset=["levels"]).groupby("cell_id")["levels"].mean()
        features.loc[levels.index, "mean_building_levels"] = levels

    return features.reset_index()
