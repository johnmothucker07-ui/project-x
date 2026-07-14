"""
ЭТАП 3 · Оптимизатор (P-Median / MCLP на PuLP).

Совместно выбирает ЛОКАЦИЮ и ТИП: для каждой кандидатной клетки —
открывать ли и если да, то гос или частную (по argmax из двух скоров),
под ограничениями бюджета и мин. расстояния 1000м между объектами.

area_character используется как фильтр: промзона/вода отсекаются из кандидатов.
"""
from __future__ import annotations


def filter_candidates(cells):
    """Отсеять непригодные клетки (по area_character: промзона, вода, парк)."""
    raise NotImplementedError


def solve(cells, time_matrix, cfg: dict):
    """Решить задачу размещения. Вернуть выбранные точки с типом (public/private) и вкладом в цель."""
    raise NotImplementedError
