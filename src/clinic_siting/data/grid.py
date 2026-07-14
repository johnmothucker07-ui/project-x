"""
ЭТАП 1 · Сетка.
Вход: bbox района + размер клетки (из config).
Выход: GeoDataFrame клеток с cell_id и геометрией-полигоном.
Это первый шаг — cell_id отсюда протаскивается через ВЕСЬ пайплайн.
"""
from __future__ import annotations


def build_grid(bbox: list[float], cell_size_m: int, id_prefix: str = "cell_"):
    """Построить сетку клеток по bbox. Возвращает GeoDataFrame [cell_id, geometry, center_lon, center_lat]."""
    raise NotImplementedError
