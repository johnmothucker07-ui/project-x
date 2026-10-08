"""Тесты Ш3: сетка в UTM, пешеходный граф, привязка, матрица времён.

Везде игрушечные примеры с известным заранее ответом — иначе ошибку
в формулировке не заметишь: граф вернёт «оптимум» неправильной задачи,
и выглядеть это будет правдоподобно.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.data.grid import build_grid, grid_summary  # noqa: E402
from seto.routing import graph as GR, matrix as M  # noqa: E402

UTM = "EPSG:32637"
SPEED = 75.0            # м/мин
WALK = {"include": ["footway", "residential"], "exclude": ["motorway"]}


def _lines(*geoms, highway="footway", **columns):
    data = {"highway": [highway] * len(geoms), **columns}
    return gpd.GeoDataFrame(data, geometry=list(geoms), crs=UTM)


# --- сетка ---

def test_grid_covers_boundary_and_snaps_origin():
    boundary = gpd.GeoDataFrame(geometry=[box(412_340, 6_181_770, 413_360, 6_182_790)],
                                crs=UTM)
    grid = build_grid(boundary, 500)
    # origin округляется вниз до кратного 500 (решение Д5)
    assert grid["x"].min() - 250 == 412_000
    assert len(grid) == 9          # 3x3 клетки накрывают рамку

def test_grid_records_share_inside_city():
    """Клетки на границе остаются: население считается по всем (п. 3.1)."""
    boundary = gpd.GeoDataFrame(geometry=[box(412_000, 6_181_500, 412_250, 6_181_750)],
                                crs=UTM)
    grid = build_grid(boundary, 500)
    assert len(grid) == 1
    assert grid["inside_share"].iloc[0] == pytest.approx(0.25)   # четверть клетки


def test_grid_rejects_geographic_crs():
    boundary = gpd.GeoDataFrame(geometry=[box(37.5, 55.7, 37.6, 55.8)], crs="EPSG:4326")
    with pytest.raises(ValueError):
        build_grid(boundary, 500)


def test_grid_summary_counts_border_cells():
    boundary = gpd.GeoDataFrame(geometry=[box(412_000, 6_181_500, 412_900, 6_182_400)],
                                crs=UTM)
    summary = grid_summary(build_grid(boundary, 500))
    assert summary["cells"] == 4 and summary["cell_size_m"] == pytest.approx(500)


# --- граф ---

def test_graph_edge_weight_is_minutes():
    """Ребро 750 м при 75 м/мин — ровно 10 минут."""
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    assert g.node_count == 2 and g.edge_count == 1
    assert g.matrix[0, 1] == pytest.approx(10.0)


def test_graph_is_undirected():
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    assert g.matrix[0, 1] == pytest.approx(g.matrix[1, 0])


def test_shared_vertex_joins_ways():
    """Пути с общей точкой образуют перекрёсток."""
    g = GR.build(_lines(LineString([(0, 0), (100, 0)]),
                        LineString([(100, 0), (100, 100)])), SPEED, WALK)
    assert g.node_count == 3 and g.edge_count == 2


def test_crossing_without_shared_vertex_stays_separate():
    """Мост над дорогой общей точки не имеет — связи быть не должно.

    Отсюда же берутся физические барьеры: река без моста рядом разрывает граф.
    Пути геометрически пересекаются в (50, 50), но общей вершины там нет,
    поэтому получаются два независимых компонента, и меньший отбрасывается —
    сам факт отбрасывания и доказывает, что связь не возникла."""
    long_way = LineString([(0, 50), (100, 50), (200, 50)])   # 3 вершины
    crossing = LineString([(50, 0), (50, 100)])              # 2 вершины
    g = GR.build(_lines(long_way, crossing), SPEED, WALK)
    assert g.node_count == 3        # осталась только длинная линия
    assert g.dropped_nodes == 2     # пересекающая ушла отдельным компонентом


def test_motorway_is_excluded():
    lines = gpd.GeoDataFrame(
        {"highway": ["motorway", "footway"]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)])],
        crs=UTM)
    assert len(GR.walkable(lines, WALK)) == 1


def test_foot_no_is_excluded():
    lines = _lines(LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)]))
    lines["foot"] = ["no", None]
    assert len(GR.walkable(lines, WALK)) == 1


def test_largest_component_kept_and_counted():
    """Висячие куски в OSM обычны; терять их молча нельзя (п. 6)."""
    g = GR.build(_lines(LineString([(0, 0), (100, 0)]),
                        LineString([(100, 0), (200, 0)]),
                        LineString([(9000, 9000), (9100, 9000)])), SPEED, WALK)
    assert g.node_count == 3            # крупнейшая компонента
    assert g.dropped_nodes == 2


def test_parallel_edges_keep_the_fastest():
    """coo складывает дубликаты, а сумма времён бессмысленна."""
    g = GR.build(_lines(LineString([(0, 0), (300, 0)]),
                        LineString([(0, 0), (150, 10), (300, 0)])), SPEED, WALK)
    assert g.matrix[0, g.node_count - 1] == pytest.approx(300 / SPEED, rel=1e-6) or \
           g.matrix[0, 1] == pytest.approx(300 / SPEED, rel=1e-6)


def test_graph_rejects_geographic_crs():
    lines = gpd.GeoDataFrame({"highway": ["footway"]},
                             geometry=[LineString([(37.5, 55.7), (37.6, 55.8)])],
                             crs="EPSG:4326")
    with pytest.raises(ValueError):
        GR.build(lines, SPEED, WALK)


# --- привязка ---

def test_snap_adds_approach_time():
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    s = M.snap(g, np.array([[0.0, 150.0]]), max_distance_m=300, walk_speed_m_per_min=SPEED)
    assert s.node[0] == 0
    assert s.approach_min[0] == pytest.approx(2.0)      # 150 м / 75


def test_snap_marks_far_points_unattached():
    """Дальше 300 м от графа — «недостижимо» (п. 6)."""
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    s = M.snap(g, np.array([[0.0, 500.0]]), max_distance_m=300, walk_speed_m_per_min=SPEED)
    assert s.node[0] == -1 and s.unattached_count == 1


# --- матрица времён ---

def test_matrix_sums_route_and_both_approaches():
    """750 м по ребру + 150 м подхода у источника + 75 м у цели = 13 минут."""
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    src = M.snap(g, np.array([[0.0, 150.0]]), 300, SPEED)
    dst = M.snap(g, np.array([[750.0, 75.0]]), 300, SPEED)
    times = M.travel_times(g, src, dst, t_max_minutes=60)
    assert times[0, 0] == pytest.approx(10.0 + 2.0 + 1.0)


def test_unreachable_is_t_max():
    g = GR.build(_lines(LineString([(0, 0), (100, 0)]),
                        LineString([(9000, 9000), (9100, 9000)])), SPEED, WALK)
    src = M.snap(g, np.array([[0.0, 0.0]]), 300, SPEED)
    dst = M.snap(g, np.array([[9000.0, 9000.0]]), 300, SPEED)
    assert M.travel_times(g, src, dst, t_max_minutes=60)[0, 0] == 60.0


def test_unattached_cell_gets_t_max():
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    src = M.snap(g, np.array([[0.0, 0.0]]), 300, SPEED)
    dst = M.snap(g, np.array([[0.0, 5000.0]]), 300, SPEED)
    assert M.travel_times(g, src, dst, t_max_minutes=60)[0, 0] == 60.0


def test_nearest_time_takes_minimum_over_sources():
    g = GR.build(_lines(LineString([(0, 0), (750, 0)])), SPEED, WALK)
    src = M.snap(g, np.array([[0.0, 0.0], [750.0, 0.0]]), 300, SPEED)
    dst = M.snap(g, np.array([[750.0, 0.0]]), 300, SPEED)
    times = M.travel_times(g, src, dst, t_max_minutes=60)
    assert M.nearest_time(times, 60)[0] == pytest.approx(0.0)


def test_network_time_is_never_shorter_than_straight_line():
    """Главная проверка замены прямой на сеть: по улицам всегда не ближе."""
    g = GR.build(_lines(LineString([(0, 0), (0, 400)]),
                        LineString([(0, 400), (300, 400)])), SPEED, WALK)
    src = M.snap(g, np.array([[0.0, 0.0]]), 300, SPEED)
    dst = M.snap(g, np.array([[300.0, 400.0]]), 300, SPEED)
    network = M.travel_times(g, src, dst, 60)[0, 0]
    straight = np.hypot(300, 400) / SPEED
    assert network == pytest.approx(700 / SPEED)
    assert network > straight
