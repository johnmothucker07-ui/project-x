"""Загрузка снимка OSM и чтение его слоёв.

Почему снимок, а не живой Overpass (как было в MVP):
  1) воспроизводимость — Overpass отвечает по текущему состоянию карты, и через
     неделю те же метрики получатся другими; п. 4.3 запрещает живые запросы
     во время расчётов;
  2) геометрия — MVP просил `out center`, и полигоны приходили центроидами.
     Парк на двадцать гектаров становился точкой, а для допустимых площадок
     нужна доля площади клетки под водой, промзоной, парком.

Читаем через GDAL (драйвер OSM, доступен в pyogrio) — pyrosm на Windows не
собирается: его зависимость cykhash требует компилятор и колёс под Windows нет.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from .download import Source, fetch

# слои, которые отдаёт драйвер OSM
LAYERS = ("points", "lines", "multilinestrings", "multipolygons", "other_relations")

# Свой конфиг драйвера: по умолчанию GDAL не выносит в колонки amenity, operator
# и healthcare:speciality — они оседают в other_tags. Без них не выделить сеть
# государственных поликлиник (п. 4.5), а разбирать hstore на сотнях тысяч точек
# и дороже, и легче ошибиться.
OSM_CONFIG = Path(__file__).parent / "osmconf.ini"


def _apply_gdal_config() -> None:
    """Указать GDAL на наш osmconf.ini. Идемпотентно."""
    import pyogrio

    pyogrio.set_gdal_config_options({"OSM_CONFIG_FILE": str(OSM_CONFIG)})


def download_pbf(url: str, raw_dir: Path, license_note: str | None = None,
                 overwrite: bool = False) -> Source:
    """Скачать выгрузку OSM в raw/<city>/<snapshot_date>/."""
    return fetch(url, raw_dir / "osm.pbf", name="osm_pbf",
                 license_note=license_note, overwrite=overwrite)


def read_layer(pbf: Path, layer: str, columns: list[str] | None = None,
               bbox: tuple[float, float, float, float] | None = None):
    """Прочитать слой PBF в GeoDataFrame.

    bbox в WGS84 (min_lon, min_lat, max_lon, max_lat) заметно ускоряет чтение:
    GDAL отбрасывает объекты вне рамки, не разбирая их."""
    if layer not in LAYERS:
        raise ValueError(f"неизвестный слой {layer!r}, ожидался один из {LAYERS}")
    _apply_gdal_config()
    return gpd.read_file(pbf, layer=layer, columns=columns, bbox=bbox,
                         engine="pyogrio")


def parse_other_tags(value: str | None) -> dict[str, str]:
    """Разобрать поле other_tags драйвера OSM (формат hstore).

    Выглядит так: "amenity"=>"pharmacy","opening_hours"=>"24/7"
    Драйвер продвигает в колонки только часть тегов, остальные складывает сюда.
    Значения могут содержать запятые и экранированные кавычки, поэтому разбор
    посимвольный, а не split по запятой."""
    if not value:
        return {}

    tags: dict[str, str] = {}
    key: list[str] = []
    val: list[str] = []
    buf = key
    inside = False
    escaped = False

    for char in value:
        if escaped:
            buf.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            inside = not inside
            if not inside and buf is val:
                # закрылась кавычка значения — пара готова
                tags["".join(key)] = "".join(val)
                key, val = [], []
                buf = key
        elif inside:
            buf.append(char)
        elif char == ">" and buf is key:
            buf = val
    return tags


def describe_layers(pbf: Path) -> dict[str, dict]:
    """Что реально лежит в файле: по каждому слою число объектов и колонки.

    Нужно до любого парсинга (правило проекта: сначала посмотреть, что пришло)."""
    import pyogrio

    _apply_gdal_config()
    summary: dict[str, dict] = {}
    for layer in LAYERS:
        try:
            info = pyogrio.read_info(pbf, layer=layer)
            summary[layer] = {
                "features": info.get("features"),
                "geometry_type": info.get("geometry_type"),
                "fields": list(info.get("fields", [])),
            }
        except Exception as error:   # слоя может не быть в конкретной выгрузке
            summary[layer] = {"error": f"{type(error).__name__}: {error}"}
    return summary
