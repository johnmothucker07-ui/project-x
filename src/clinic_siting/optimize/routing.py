"""
ЭТАП 3 · Маршрутизация через OSRM (локальный сервер в Docker).

Матрица времени в пути: от каждой точки спроса до каждой клиники.

ГРАБЛИ: OSRM принимает координаты как lon,lat (НЕ lat,lon). Легко перепутать.
Эндпоинт /table отдаёт матрицу; на больших наборах бить на части.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.geo import haversine_m

# грубая средняя городская скорость для перевода расстояния в время (мин).
# ВЕРСИЯ MVP: реальное время по дорогам даст OSRM (см. TODO «ОТЛОЖЕНО»).
_AVG_SPEED_KMH = 25.0


def _distance_to_minutes(distance_m: float) -> float:
    """Перевести расстояние (м) в минуты по средней скорости."""
    return distance_m / 1000.0 / _AVG_SPEED_KMH * 60.0


def travel_time_matrix(sources, destinations, osrm_url: str | None = None, profile: str = "driving"):
    """Матрица времени в пути [len(sources) x len(destinations)], минуты.
    sources/destinations — списки координат (lon, lat).
    ВЕРСИЯ MVP: по прямой (haversine → время). osrm_url пока не используется — OSRM см. TODO."""
    matrix = np.zeros((len(sources), len(destinations)))
    for i, (s_lon, s_lat) in enumerate(sources):
        for j, (d_lon, d_lat) in enumerate(destinations):
            matrix[i, j] = _distance_to_minutes(haversine_m(s_lon, s_lat, d_lon, d_lat))
    return matrix


def nearest_clinic_time(cells, clinics, osrm_url: str | None = None):
    """Для каждой клетки — время до ближайшей клиники (минуты). Series по индексу cells.
    ВЕРСИЯ MVP: по прямой (haversine)."""
    sources = list(zip(cells["center_lon"], cells["center_lat"]))
    destinations = list(zip(clinics["lon"], clinics["lat"]))
    matrix = travel_time_matrix(sources, destinations, osrm_url)
    return pd.Series(matrix.min(axis=1), index=cells.index)
