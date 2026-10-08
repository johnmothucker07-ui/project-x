"""Тесты Ш2: разбор тегов, починка геометрии, этажность, загрузка."""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.acquire.osm import OSM_CONFIG, parse_other_tags  # noqa: E402
from seto.data.osm_layers import _parse_levels, _valid  # noqa: E402


# --- разбор other_tags (hstore) ---

def test_parses_simple_pair():
    assert parse_other_tags('"amenity"=>"pharmacy"') == {"amenity": "pharmacy"}


def test_parses_several_pairs():
    raw = '"amenity"=>"cafe","cuisine"=>"coffee_shop","outdoor_seating"=>"yes"'
    assert parse_other_tags(raw) == {
        "amenity": "cafe", "cuisine": "coffee_shop", "outdoor_seating": "yes"}


def test_comma_inside_value_does_not_split():
    """Поэтому разбор посимвольный, а не split(',')."""
    raw = '"addr:street"=>"ул. Садовая, 5","amenity"=>"clinic"'
    assert parse_other_tags(raw) == {
        "addr:street": "ул. Садовая, 5", "amenity": "clinic"}


def test_escaped_quote_inside_value():
    raw = '"name"=>"Кафе \\"Уют\\"","amenity"=>"cafe"'
    assert parse_other_tags(raw)["name"] == 'Кафе "Уют"'


def test_empty_and_none():
    assert parse_other_tags(None) == {}
    assert parse_other_tags("") == {}


def test_cyrillic_values():
    raw = '"operator"=>"ГУП «Московский метрополитен»"'
    assert parse_other_tags(raw) == {"operator": "ГУП «Московский метрополитен»"}


# --- конфиг драйвера ---

def test_osmconf_promotes_tags_needed_for_state_network():
    """Без operator и healthcare:speciality не выделить гос-сеть (п. 4.5)."""
    text = OSM_CONFIG.read_text(encoding="utf-8")
    points = next(block for block in text.split("[") if block.startswith("points]"))
    for tag in ("amenity", "operator", "healthcare", "healthcare:speciality"):
        assert tag in points, f"тег {tag} не продвигается в колонки для points"


# --- починка геометрии ---

def test_invalid_polygon_is_repaired():
    """Самопересечение (бабочка) — обычное дело в OSM; GDAL о нём только предупреждает."""
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    frame = gpd.GeoDataFrame(geometry=[bowtie], crs="EPSG:4326")
    assert _valid(frame).geometry.is_valid.all()


def test_self_intersection_does_not_leave_dangling_lines():
    """make_valid на «бабочке» возвращает коллекцию с повисшими линиями.

    На реальных данных так вышло у 6 зданий из 15948. Линия в слое зданий
    ломает расчёт доли площади клетки под застройкой."""
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    frame = gpd.GeoDataFrame(geometry=[bowtie], crs="EPSG:4326")
    result = _valid(frame, polygonal=True)
    assert result.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    assert (result.geometry.area > 0).all()


def test_empty_geometry_is_dropped():
    frame = gpd.GeoDataFrame(
        geometry=[Polygon(), Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326")
    assert len(_valid(frame)) == 1


# --- этажность ---

def test_levels_parsed_and_garbage_becomes_na():
    frame = pd.DataFrame({"building_levels": ["5", "2;3", "нет", None, "12", "0"]})
    levels = _parse_levels(frame)
    assert levels.tolist()[:2] == [5.0, 2.0]      # "2;3" -> берём первое
    assert pd.isna(levels[2]) and pd.isna(levels[3])   # мусор и пропуск -> NA
    assert levels[4] == 12.0
    # ноль этажей — мусор в данных, а не «нет данных»; на районе таких 4
    assert pd.isna(levels[5])


def test_missing_levels_column_gives_all_na():
    """Пропуск остаётся пропуском: выдумывать этажность нельзя (п. 22)."""
    levels = _parse_levels(pd.DataFrame({"building": ["yes"]}))
    assert levels.isna().all()


# --- загрузка ---

def test_fetch_does_not_redownload(tmp_path):
    """raw/ после фиксации неизменяем, а выгрузки весят гигабайты."""
    from seto.acquire.download import fetch

    payload = "уже скачано".encode("utf-8")
    target = tmp_path / "osm.pbf"
    target.write_bytes(payload)
    source = fetch("http://example.invalid/osm.pbf", target, name="osm_pbf")
    assert source.size_bytes == len(payload)
    assert target.read_bytes() == payload


def test_download_error_on_bad_url(tmp_path):
    from seto.acquire.download import DownloadError, fetch

    with pytest.raises(DownloadError):
        fetch("http://nonexistent.invalid/x.pbf", tmp_path / "x.pbf",
              name="x", progress=False)


def test_data_manifest_has_hashes(tmp_path):
    from seto.acquire.download import Source, write_data_manifest
    import json

    src = Source(name="osm_pbf", url="http://x/y.pbf", path="y.pbf",
                 sha256="deadbeef", size_bytes=10, downloaded_at="now")
    path = write_data_manifest([src], tmp_path / "m.json", "2026-10-08")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["snapshot_date"] == "2026-10-08"
    assert data["sources"][0]["sha256"] == "deadbeef"
