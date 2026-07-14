"""
ЭТАП 3 · Маршрутизация через OSRM (локальный сервер в Docker).

Матрица времени в пути: от каждой точки спроса до каждой клиники.

ГРАБЛИ: OSRM принимает координаты как lon,lat (НЕ lat,lon). Легко перепутать.
Эндпоинт /table отдаёт матрицу; на больших наборах бить на части.
"""
from __future__ import annotations


def travel_time_matrix(sources, destinations, osrm_url: str, profile: str = "driving"):
    """Матрица времени в пути [len(sources) x len(destinations)] через OSRM /table."""
    raise NotImplementedError


def nearest_clinic_time(cells, clinics, osrm_url: str):
    """Для каждой клетки — время до ближайшей клиники (для метрик доступности)."""
    raise NotImplementedError
