"""Главная метрика: покрытие населения в пределах T минут пешком (п. 11.1).

    Cov_T(S) = 100 · Σ_i P_i · 1[t_i(S) ≤ T] / Σ_i P_i      проценты
    ΔCov_T   = Cov_T(S) − Cov_T(∅)                          процентные пункты
    ΔN_T     = Σ_i P_i · (1[t_i(S) ≤ T] − 1[t_i^E ≤ T])     жители

Знаменатель — ВСЁ население города, включая жителей непривязанных и
недостижимых клеток. Уменьшать его на «проблемные» клетки нельзя: так
покрытие вырастет само собой, без единой новой поликлиники.

Множитель 100 обязателен: метрика в процентах, разницы в процентных пунктах.
"""
from __future__ import annotations

import numpy as np


def _check(times: np.ndarray, population: np.ndarray) -> None:
    if times.shape != population.shape:
        raise ValueError(f"времена {times.shape} и население {population.shape} "
                         "разной длины")
    if population.sum() <= 0:
        raise ValueError("суммарное население равно нулю")


def coverage(times: np.ndarray, population: np.ndarray, threshold: float) -> float:
    """Cov_T в процентах."""
    times = np.asarray(times, dtype=float)
    population = np.asarray(population, dtype=float)
    _check(times, population)
    return float(100.0 * population[times <= threshold].sum() / population.sum())


def delta_coverage(times_after: np.ndarray, times_before: np.ndarray,
                   population: np.ndarray, threshold: float) -> float:
    """ΔCov_T в процентных пунктах."""
    return (coverage(times_after, population, threshold)
            - coverage(times_before, population, threshold))


def delta_people(times_after: np.ndarray, times_before: np.ndarray,
                 population: np.ndarray, threshold: float) -> float:
    """ΔN_T — сколько жителей получили поликлинику в пешей доступности."""
    after = np.asarray(times_after, dtype=float) <= threshold
    before = np.asarray(times_before, dtype=float) <= threshold
    population = np.asarray(population, dtype=float)
    _check(after, population)
    return float((population * (after.astype(int) - before.astype(int))).sum())


def times_with(existing: np.ndarray, added: np.ndarray | None,
               t_max: float) -> np.ndarray:
    """t_i(S) = min(t_i^E, min по выбранным площадкам), с потолком t_max.

    added — матрица [площадки x клетки] только по выбранным площадкам."""
    base = np.minimum(np.asarray(existing, dtype=float), t_max)
    if added is None or len(added) == 0:
        return base
    return np.minimum(base, np.asarray(added, dtype=float).min(axis=0))


def worst_decile_time(times: np.ndarray, population: np.ndarray,
                      share: float = 0.10) -> float:
    """Среднее время для доли населения с наибольшим временем (п. 11.2).

    Клетка, попавшая на границу отсечки, учитывается ДОЛЕЙ своего населения —
    иначе результат прыгал бы от размера клетки.

    Зачем метрика: низкая неравномерность бывает и тогда, когда всем одинаково
    плохо, а покрытие может выглядеть прилично при тяжёлом хвосте."""
    times = np.asarray(times, dtype=float)
    population = np.asarray(population, dtype=float)
    _check(times, population)
    if not 0 < share <= 1:
        raise ValueError("доля должна быть в (0, 1]")

    order = np.argsort(-times)
    sorted_times = times[order]
    sorted_pop = population[order]

    target = share * population.sum()
    taken = np.minimum(np.cumsum(sorted_pop), target)
    weights = np.diff(np.concatenate([[0.0], taken]))
    return float((sorted_times * weights).sum() / target)


def mean_time(times: np.ndarray, population: np.ndarray) -> float:
    """Среднее взвешенное по населению время (п. 11.2).

    ВНИМАНИЕ: совпадает с целевой функцией P-median, поэтому даёт вариантам
    B, C и D встроенное преимущество. Приводится с оговоркой, выводы
    о качестве на ней не строятся."""
    times = np.asarray(times, dtype=float)
    population = np.asarray(population, dtype=float)
    _check(times, population)
    return float((times * population).sum() / population.sum())


def unreachable_people(times: np.ndarray, population: np.ndarray,
                       t_max: float) -> float:
    """Жители клеток без маршрута или дальше потолка (п. 11.2)."""
    times = np.asarray(times, dtype=float)
    population = np.asarray(population, dtype=float)
    _check(times, population)
    return float(population[times >= t_max].sum())
