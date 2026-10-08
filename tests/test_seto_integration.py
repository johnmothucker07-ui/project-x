"""Сквозная проверка цепочки Ш1-Ш3 на настоящем снимке OSM.

Пропускается, если снимок не скачан: `python -m seto acquire moscow`.
Работает на небольшом bbox (район MVP), чтобы проверка шла секунды, а не минуты.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto import settings  # noqa: E402
from seto.data import grid as G, osm_layers as L  # noqa: E402
from seto.routing import graph as GR, matrix as M  # noqa: E402

# тот же участок, на котором работал MVP — есть с чем сравнивать
BBOX = (37.55, 55.72, 37.65, 55.78)


def _snapshot() -> Path | None:
    cfg = settings.load("moscow")
    root = settings.data_root(cfg) / cfg["paths"]["raw"] / "moscow"
    if not root.is_dir():
        return None
    files = sorted(root.glob("*/osm.pbf"))
    return files[-1] if files else None


pbf = _snapshot()
needs_snapshot = pytest.mark.skipif(
    pbf is None, reason="нет снимка OSM: запусти python -m seto acquire moscow")


@pytest.fixture(scope="module")
def cfg():
    return settings.load("moscow")


@pytest.fixture(scope="module")
def area():
    """Рамка как граница — сквозную цепочку это проверяет не хуже города."""
    frame = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt([
            f"POLYGON(({BBOX[0]} {BBOX[1]}, {BBOX[2]} {BBOX[1]}, "
            f"{BBOX[2]} {BBOX[3]}, {BBOX[0]} {BBOX[3]}, {BBOX[0]} {BBOX[1]}))"]),
        crs="EPSG:4326")
    return frame.to_crs(32637)


@pytest.fixture(scope="module")
def roads(cfg):
    return L.extract_roads(pbf, BBOX).to_crs(32637)


@pytest.fixture(scope="module")
def walk_graph(roads, cfg):
    return GR.build(roads, cfg["walk_speed_m_per_min"], cfg["osm"]["walk_network"])


# --- сетка ---

@needs_snapshot
def test_grid_cells_are_square_and_unique(area, cfg):
    grid = G.build_grid(area, cfg["grid"]["size_m"], cfg["grid"]["id_format"])
    assert len(grid) > 100
    assert grid["cell_id"].is_unique
    assert np.allclose(grid["area_m2"], cfg["grid"]["size_m"] ** 2)
    assert (grid["inside_share"] > 0).all() and (grid["inside_share"] <= 1.0 + 1e-9).all()


@needs_snapshot
def test_grid_is_deterministic(area, cfg):
    """Два построения дают те же cell_id — на них завязан кэш модели."""
    first = G.build_grid(area, cfg["grid"]["size_m"], cfg["grid"]["id_format"])
    second = G.build_grid(area, cfg["grid"]["size_m"], cfg["grid"]["id_format"])
    assert first["cell_id"].tolist() == second["cell_id"].tolist()
    assert np.allclose(first["x"], second["x"])


# --- слои OSM ---

@needs_snapshot
def test_layers_have_real_polygons():
    """Главное отличие от MVP: полигоны остались полигонами, а не центроидами."""
    layers = L.extract_polygons(pbf, BBOX)
    assert len(layers["buildings"]) > 1000
    assert layers["buildings"].geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]).all()
    assert (layers["buildings"].geometry.area > 0).all()
    assert layers["buildings"].geometry.is_valid.all()


@needs_snapshot
def test_levels_are_numeric_or_missing():
    buildings = L.extract_polygons(pbf, BBOX)["buildings"]
    levels = buildings["levels"].dropna()
    assert len(levels) > 0
    assert (levels > 0).all() and (levels <= 150).all()


@needs_snapshot
def test_medical_candidates_carry_tags_needed_for_state_network(cfg):
    layers = L.extract_polygons(pbf, BBOX)
    points = L.extract_points(pbf, BBOX)
    med = L.medical_facilities(points, layers["poi_polygons"],
                               cfg["osm"]["clinic_tags"])
    assert len(med) > 100
    for column in ("operator", "healthcare_speciality"):
        assert column in med.columns


# --- граф ---

@needs_snapshot
def test_graph_is_connected_and_large(walk_graph):
    assert walk_graph.node_count > 10_000
    assert walk_graph.edge_count > walk_graph.node_count / 2


@needs_snapshot
def test_no_motorways_in_walk_graph(roads, cfg):
    kept = GR.walkable(roads, cfg["osm"]["walk_network"])
    assert not kept["highway"].isin(["motorway", "trunk"]).any()


# --- времена ---

@needs_snapshot
def test_network_time_never_beats_straight_line(walk_graph, area, cfg):
    """Та самая проверка, ради которой менялась маршрутизация.

    Прямая — нижняя граница: по улицам короче быть не может."""
    grid = G.build_grid(area, cfg["grid"]["size_m"], cfg["grid"]["id_format"])
    xy = grid[["x", "y"]].to_numpy()
    rng = np.random.default_rng(0)
    sample = xy[rng.choice(len(xy), size=12, replace=False)]

    snapped = M.snap(walk_graph, sample, cfg["snap_max_m"],
                     cfg["walk_speed_m_per_min"])
    times = M.travel_times(walk_graph, snapped, snapped, cfg["t_max_minutes"])

    straight = (np.hypot(sample[:, None, 0] - sample[None, :, 0],
                         sample[:, None, 1] - sample[None, :, 1])
                / cfg["walk_speed_m_per_min"])
    reachable = times < cfg["t_max_minutes"]
    assert reachable.sum() > 10
    assert (times[reachable] >= straight[reachable] - 1e-6).all()


@needs_snapshot
def test_times_are_bounded_and_finite(walk_graph, area, cfg):
    grid = G.build_grid(area, cfg["grid"]["size_m"], cfg["grid"]["id_format"])
    cells = M.snap(walk_graph, grid[["x", "y"]].to_numpy(), cfg["snap_max_m"],
                   cfg["walk_speed_m_per_min"])
    sources = M.snap(walk_graph, grid[["x", "y"]].to_numpy()[:5],
                     cfg["snap_max_m"], cfg["walk_speed_m_per_min"])
    times = M.travel_times(walk_graph, sources, cells, cfg["t_max_minutes"])

    assert np.isfinite(times).all()
    assert (times >= 0).all() and (times <= cfg["t_max_minutes"]).all()
    # большая часть клеток в плотном районе должна быть привязана к графу
    assert cells.unattached_count / len(grid) < 0.05


@needs_snapshot
def test_distance_to_self_is_zero(walk_graph, area, cfg):
    grid = G.build_grid(area, cfg["grid"]["size_m"], cfg["grid"]["id_format"])
    xy = grid[["x", "y"]].to_numpy()[:6]
    snapped = M.snap(walk_graph, xy, cfg["snap_max_m"], cfg["walk_speed_m_per_min"])
    times = M.travel_times(walk_graph, snapped, snapped, cfg["t_max_minutes"])
    # путь «сам к себе» — это два подхода до своего узла и обратно
    assert np.allclose(np.diag(times), 2 * snapped.approach_min, atol=1e-6)
