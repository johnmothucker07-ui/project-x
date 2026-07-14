"""
ЭТАП 4 · Доступность — ГЛАВНАЯ метрика (не зависит от реестра).

Считает среднее время до ближайшей клиники ДО и ПОСЛЕ добавления
рекомендованных точек, и сравнивает с baseline (случайно / по плотности).
Это основное доказательство "локация выбрана грамотно".
"""
from __future__ import annotations


def mean_access_time(cells, clinics, osrm_url: str) -> float:
    """Среднее по населению время до ближайшей клиники."""
    raise NotImplementedError


def before_after(cells, existing_clinics, recommended, osrm_url: str) -> dict:
    """Время до/после добавления рекомендованных точек. Вернуть {before, after, improvement}."""
    raise NotImplementedError


def baseline_placement(cells, n: int, method: str = "population_density"):
    """Baseline-расстановка для сравнения (случайно или по плотности населения)."""
    raise NotImplementedError
