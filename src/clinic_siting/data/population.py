"""
ЭТАП 1 · Население (Kontur Population).
Вход: клетки сетки + файл Kontur.
Выход: население по каждой клетке.

Kontur отдаёт данные гексагонами H3 (на базе WorldPop). Нужно агрегировать
их на нашу прямоугольную сетку через пространственное соединение.
"""
from __future__ import annotations


def load_kontur(path: str):
    """Загрузить данные Kontur Population (GeoPackage с population по гексагонам)."""
    raise NotImplementedError


def population_per_cell(cells, kontur):
    """Агрегировать население Kontur на клетки сетки (spatial join + сумма с учётом площади)."""
    raise NotImplementedError
