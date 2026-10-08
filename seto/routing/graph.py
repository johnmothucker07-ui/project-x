"""Пешеходный граф из снимка OSM (п. 6).

Строим сами из слоя линий, а не скачиваем через OSMnx: расчёт обязан идти
из зафиксированного снимка (п. 4.3), а OSMnx ходит в Overpass.

Узлы опознаются по совпадению координат. В корректно размеченном OSM пути
делят общую точку на перекрёстке, а мост и дорога под ним общей точки не имеют —
значит пересечение без узла само собой не станет связью. Это ровно то поведение,
которое нужно: физические барьеры (река без моста, ж/д) возникают из геометрии,
а не из отдельного правила.

Считаем в scipy.sparse, а не в networkx: Дейкстра из csgraph берёт сразу много
источников и умеет отсечку по времени, и на сотнях тысяч рёбер это важно.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# координаты в метрах; округление до сантиметра склеивает общие вершины
_SNAP_PRECISION = 2


@dataclass
class WalkGraph:
    """Пешеходный граф: координаты узлов и матрица времён по рёбрам (минуты)."""
    nodes: np.ndarray          # (N, 2) в метрической проекции
    matrix: csr_matrix         # (N, N), вес = минуты пешком
    crs: object
    dropped_nodes: int         # узлов, выброшенных вне крупнейшей компоненты
    dropped_edges: int

    def __post_init__(self) -> None:
        self._tree = cKDTree(self.nodes)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return int(self.matrix.nnz // 2)

    def node_of(self, snapped, positions: np.ndarray) -> np.ndarray:
        """Индексы узлов графа для выбранных позиций привязанного набора."""
        return snapped.node[positions]

    def snap(self, points: np.ndarray, max_distance_m: float):
        """Привязать точки к ближайшим узлам.

        Возвращает (индексы узлов, расстояния). Точка дальше max_distance_m
        считается непривязанной: индекс -1, расстояние inf. Для клетки спроса
        это «недостижимо», для площадки — исключение из допустимых (п. 6)."""
        distances, indices = self._tree.query(points, k=1)
        too_far = distances > max_distance_m
        indices = np.asarray(indices, dtype=np.int64).copy()
        distances = np.asarray(distances, dtype=float).copy()
        indices[too_far] = -1
        distances[too_far] = np.inf
        return indices, distances


def walkable(lines: gpd.GeoDataFrame, config: dict) -> gpd.GeoDataFrame:
    """Отобрать линии, по которым можно идти пешком."""
    allowed = set(config["include"])
    excluded = set(config.get("exclude", []))

    keep = lines["highway"].isin(allowed) & ~lines["highway"].isin(excluded)
    # явные запреты прохода уважаем; отсутствие тега запретом не считается
    if "foot" in lines.columns:
        keep &= ~lines["foot"].isin(["no", "private"])
    if "access" in lines.columns:
        keep &= ~lines["access"].isin(["no", "private"])
    return lines[keep]


def build(lines: gpd.GeoDataFrame, walk_speed_m_per_min: float,
          config: dict) -> WalkGraph:
    """Собрать граф из линий. lines должны быть в метрической проекции."""
    if lines.crs is None or lines.crs.is_geographic:
        raise ValueError("линии должны быть в метрической проекции (UTM)")
    if walk_speed_m_per_min <= 0:
        raise ValueError("скорость должна быть больше нуля")

    roads = walkable(lines, config)
    if roads.empty:
        raise ValueError("после фильтра не осталось ни одной пешеходной линии")

    node_id: dict[tuple[float, float], int] = {}
    coordinates: list[tuple[float, float]] = []
    starts: list[int] = []
    ends: list[int] = []
    weights: list[float] = []

    def node_of(point: tuple[float, float]) -> int:
        key = (round(point[0], _SNAP_PRECISION), round(point[1], _SNAP_PRECISION))
        index = node_id.get(key)
        if index is None:
            index = len(coordinates)
            node_id[key] = index
            coordinates.append(key)
        return index

    for geometry in roads.geometry:
        if geometry is None or geometry.is_empty:
            continue
        parts = geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
        for part in parts:
            coords = list(part.coords)
            previous = node_of(coords[0])
            for current_xy in coords[1:]:
                current = node_of(current_xy)
                if current == previous:
                    continue      # нулевое ребро: дубль вершины
                length = float(np.hypot(coordinates[current][0] - coordinates[previous][0],
                                        coordinates[current][1] - coordinates[previous][1]))
                starts.append(previous)
                ends.append(current)
                weights.append(length / walk_speed_m_per_min)
                previous = current

    nodes = np.asarray(coordinates, dtype=float)
    size = len(nodes)
    # граф неориентированный: идти можно в обе стороны
    rows = np.concatenate([starts, ends])
    cols = np.concatenate([ends, starts])
    data = np.concatenate([weights, weights])
    matrix = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    # на параллельных рёбрах coo суммирует веса — оставляем минимальное
    matrix = _keep_minimum(matrix)

    return _largest_component(nodes, matrix, lines.crs)


def _keep_minimum(matrix: csr_matrix) -> csr_matrix:
    """Между парой узлов может быть несколько линий — берём быстрейшую.

    coo_matrix при сборке складывает дубликаты, а сумма времён бессмысленна."""
    coo = matrix.tocoo()
    order = np.lexsort((coo.col, coo.row))
    rows, cols, data = coo.row[order], coo.col[order], coo.data[order]
    keep = np.ones(len(rows), dtype=bool)
    same = (rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])
    keep[1:] = ~same
    groups = np.cumsum(keep) - 1
    minimal = np.full(groups[-1] + 1 if len(groups) else 0, np.inf)
    np.minimum.at(minimal, groups, data)
    return coo_matrix((minimal, (rows[keep], cols[keep])),
                      shape=matrix.shape).tocsr()


def _largest_component(nodes: np.ndarray, matrix: csr_matrix, crs) -> WalkGraph:
    """Оставить крупнейшую связную компоненту.

    Висячие куски в OSM обычны; молча терять их нельзя, поэтому считаем,
    сколько выброшено (п. 6 требует отчёта)."""
    count, labels = connected_components(matrix, directed=False)
    if count == 1:
        return WalkGraph(nodes, matrix, crs, dropped_nodes=0, dropped_edges=0)

    biggest = np.bincount(labels).argmax()
    keep = labels == biggest
    kept_matrix = matrix[keep][:, keep]
    return WalkGraph(
        nodes=nodes[keep],
        matrix=kept_matrix,
        crs=crs,
        dropped_nodes=int((~keep).sum()),
        dropped_edges=int((matrix.nnz - kept_matrix.nnz) // 2),
    )
