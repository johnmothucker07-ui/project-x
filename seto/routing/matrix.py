"""Матрица времён пешком: от объектов до клеток спроса (п. 6).

Дейкстра запускается ОТ ОБЪЕКТОВ, а не от клеток: источников меньше
(существующие поликлиники плюс кандидатные площадки), а клеток в городе
многие тысячи.

Отсечка t_max нужна не только по смыслу: без неё матрица плотная, с ней —
разреженная, и всё, чего в ней нет, читается как t_max.

Время подхода прибавляется с обеих сторон: от центроида клетки до узла графа
и от узла до объекта (п. 6). Без этого клетка в 250 м от ближайшей улицы
выглядела бы стоящей прямо на ней.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .graph import WalkGraph


@dataclass
class Snapped:
    """Результат привязки набора точек к графу."""
    node: np.ndarray        # индекс узла, -1 если не привязана
    approach_min: np.ndarray    # время подхода до узла, минуты (inf у непривязанных)

    @property
    def attached(self) -> np.ndarray:
        return self.node >= 0

    @property
    def unattached_count(self) -> int:
        return int((~self.attached).sum())


def snap(graph: WalkGraph, xy: np.ndarray, max_distance_m: float,
         walk_speed_m_per_min: float) -> Snapped:
    """Привязать точки к графу и посчитать время подхода."""
    nodes, distances = graph.snap(np.asarray(xy, dtype=float), max_distance_m)
    approach = distances / walk_speed_m_per_min
    return Snapped(node=nodes, approach_min=approach)


# Дейкстра возвращает массив [источники x ВСЕ узлы графа]. На городском графе
# это 1,3 млн узлов, то есть 10 МБ на источник: сотня площадок — уже гигабайт.
# Поэтому источники обрабатываются порциями, и каждая сразу сжимается до целей.
_CHUNK_BYTES = 256 << 20


def _chunk_size(node_count: int) -> int:
    return max(1, int(_CHUNK_BYTES / (node_count * 8)))


def travel_times(graph: WalkGraph, sources: Snapped, targets: Snapped,
                 t_max_minutes: float) -> np.ndarray:
    """Матрица времён [источники x цели], минуты.

    Недостижимое и то, что дальше t_max, равно t_max — так требует п. 3.2
    пропозала (время ограничивается сверху), и такие клетки считаются
    непокрытыми при любом разумном T."""
    result = np.full((len(sources.node), len(targets.node)), t_max_minutes,
                     dtype=float)

    live_sources = np.flatnonzero(sources.attached)
    live_targets = np.flatnonzero(targets.attached)
    if live_sources.size == 0 or live_targets.size == 0:
        return result

    target_nodes = targets.node[live_targets]
    target_approach = targets.approach_min[live_targets]
    step = _chunk_size(graph.node_count)

    for start in range(0, live_sources.size, step):
        batch = live_sources[start:start + step]
        distances = dijkstra(graph.matrix, directed=False,
                             indices=graph.node_of(sources, batch),
                             limit=float(t_max_minutes))
        block = (distances[:, target_nodes]
                 + sources.approach_min[batch][:, None]
                 + target_approach[None, :])
        np.minimum(block, t_max_minutes, out=block)
        block[~np.isfinite(block)] = t_max_minutes
        result[np.ix_(batch, live_targets)] = block
    return result


def nearest_time(matrix: np.ndarray, t_max_minutes: float) -> np.ndarray:
    """Время до ближайшего источника по каждой цели (минимум по столбцу)."""
    if matrix.size == 0:
        return np.array([], dtype=float)
    return np.minimum(matrix.min(axis=0), t_max_minutes)


def network_distance(graph: WalkGraph, a: Snapped, b: Snapped,
                     walk_speed_m_per_min: float, limit_m: float) -> np.ndarray:
    """Расстояния по сети в метрах между двумя наборами точек.

    Нужны для ограничений «не ближе 500 м» (п. 3.1): пропозал требует мерить
    их по пешеходной сети, а не по прямой. Дальше limit_m возвращается inf —
    для проверки «не ближе» этого достаточно."""
    result = np.full((len(a.node), len(b.node)), np.inf, dtype=float)
    rows = np.flatnonzero(a.attached)
    cols = np.flatnonzero(b.attached)
    if rows.size == 0 or cols.size == 0:
        return result

    limit_min = limit_m / walk_speed_m_per_min
    target_nodes = b.node[cols]
    step = _chunk_size(graph.node_count)

    for start in range(0, rows.size, step):
        batch = rows[start:start + step]
        distances = dijkstra(graph.matrix, directed=False,
                             indices=graph.node_of(a, batch), limit=limit_min)
        block = distances[:, target_nodes] * walk_speed_m_per_min
        result[np.ix_(batch, cols)] = block
    return result
