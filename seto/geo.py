"""Проекции и сетка.

Правило п. 2: всё геометрическое считается в метрической проекции города (UTM),
WGS84 — только на входе и выходе. Web Mercator (EPSG:3857), который использовал MVP,
не годится: он искажает площади тем сильнее, чем севернее, а мы переносим население
по доле площади пересечения.
"""
from __future__ import annotations

import math


def utm_epsg(lon: float, lat: float) -> int:
    """Код EPSG зоны UTM по координатам центроида города.

    32601-32660 — северное полушарие, 32701-32760 — южное."""
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError(f"координаты вне диапазона: lon={lon}, lat={lat}")
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def grid_origin(min_x: float, min_y: float, size_m: float) -> tuple[float, float]:
    """Начало сетки в UTM, округлённое вниз до кратного размеру клетки.

    Решение Д5: привязываем сетку к проекции, а не к границе города. Иначе правка
    границы сдвинет все клетки, а на cell_id завязаны кэш ответов модели и имена
    файлов со снимками."""
    if size_m <= 0:
        raise ValueError("размер клетки должен быть больше нуля")
    return math.floor(min_x / size_m) * size_m, math.floor(min_y / size_m) * size_m


def cell_index(x: float, y: float, origin: tuple[float, float],
               size_m: float) -> tuple[int, int]:
    """Номер клетки (столбец, строка) для точки в UTM."""
    return (int((x - origin[0]) // size_m), int((y - origin[1]) // size_m))


def cell_id(column: int, row: int, columns: int, id_format: str) -> str:
    """Идентификатор клетки.

    Нумерация построчно снизу вверх, слева направо — как в MVP, чтобы формат
    остался прежним даже при смене проекции."""
    if column < 0 or row < 0:
        raise ValueError(f"отрицательный индекс клетки: {column}, {row}")
    if column >= columns:
        raise ValueError(f"столбец {column} вне сетки шириной {columns}")
    return id_format.format(row * columns + column)
