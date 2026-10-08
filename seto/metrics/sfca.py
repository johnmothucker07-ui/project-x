"""2SFCA с гауссовым затуханием и единой зоной T (п. 11.3).

Покрытие отвечает «есть ли поликлиника рядом». 2SFCA отвечает «сколько
поликлиники достаётся жителю», с учётом того, что её делят все, кто может
до неё дойти.

    G(t) = (exp(−½·(t/T)²) − exp(−½)) / (1 − exp(−½))   при t ≤ T, иначе 0
    Шаг 1:  R_j = S_j / Σ_{i: t_ij ≤ T} P_i · G(t_ij)
    Шаг 2:  A_i = Σ_{j: t_ij ≤ T} R_j · G(t_ij)

Мощность S_j = 1 у всех учреждений — допущение пропозала, данных о мощностях нет.

СРЕДНЕЕ A_i НЕ ВЫВОДИТСЯ. При такой схеме взвешенное населением среднее равно
общей мощности учреждений (у которых в зоне есть население), делённой на всё
население города, и при одинаковом k не различает размещения. Это свойство
проверяется тестом как инвариант: оно же и доказывает, что считаем правильно.
"""
from __future__ import annotations

import numpy as np


def decay(times: np.ndarray, threshold: float) -> np.ndarray:
    """Гауссово затухание G(t), ноль за порогом."""
    if threshold <= 0:
        raise ValueError("порог должен быть больше нуля")
    times = np.asarray(times, dtype=float)
    edge = np.exp(-0.5)
    value = (np.exp(-0.5 * (times / threshold) ** 2) - edge) / (1 - edge)
    return np.where(times <= threshold, value, 0.0)


def accessibility(times: np.ndarray, population: np.ndarray,
                  threshold: float, capacity: np.ndarray | None = None):
    """A_i по клеткам. times — матрица [учреждения x клетки].

    Возвращает (A_i, R_j). Если в зоне учреждения нет населения, R_j = 0."""
    times = np.asarray(times, dtype=float)
    population = np.asarray(population, dtype=float)
    if times.shape[1] != len(population):
        raise ValueError(f"матрица {times.shape} не совпадает с населением "
                         f"({len(population)})")
    supply = (np.ones(times.shape[0]) if capacity is None
              else np.asarray(capacity, dtype=float))

    weights = decay(times, threshold)
    demand = weights @ population                     # знаменатель шага 1
    ratio = np.divide(supply, demand, out=np.zeros_like(demand, dtype=float),
                      where=demand > 0)
    return weights.T @ ratio, ratio


def share_below(access: np.ndarray, population: np.ndarray,
                floor: float) -> float:
    """Доля населения с A_i ниже порога, в процентах."""
    access = np.asarray(access, dtype=float)
    population = np.asarray(population, dtype=float)
    return float(100.0 * population[access < floor].sum() / population.sum())


def floor_value(facility_count: int, population: np.ndarray) -> float:
    """A_0 = 0,5 · N_E / ΣP — половина средней обеспеченности исходной сети.

    Считается ОДИН раз по исходной сети и не равен нулю, даже если исходная
    медиана нулевая (п. 11.3)."""
    total = float(np.asarray(population, dtype=float).sum())
    if total <= 0:
        raise ValueError("суммарное население равно нулю")
    return 0.5 * facility_count / total


def weighted_percentile(values: np.ndarray, population: np.ndarray,
                        quantile: float) -> float:
    """Процентиль, взвешенный по населению."""
    values = np.asarray(values, dtype=float)
    population = np.asarray(population, dtype=float)
    order = np.argsort(values)
    cumulative = np.cumsum(population[order]) / population.sum()
    return float(np.interp(quantile, cumulative, values[order]))


def summary(times: np.ndarray, population: np.ndarray, threshold: float,
            floor: float) -> dict:
    """Показатели 2SFCA для отчёта. Среднего здесь нет намеренно."""
    access, _ = accessibility(times, population, threshold)
    return {
        "share_below_floor_pct": share_below(access, population, floor),
        "p10": weighted_percentile(access, population, 0.10),
        "p25": weighted_percentile(access, population, 0.25),
    }
