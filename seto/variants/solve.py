"""Варианты размещения A, B, M и решатель (п. 10).

При k ≤ 2 перебираются ВСЕ допустимые пары, поэтому B, C, D и M получают
точный оптимум и результат не зависит от решателя (п. 10.3). Это важно для
вывода: разница между вариантами тогда объясняется весами, а не качеством
эвристики.

Ничьи разрешаются детерминированным правилом (п. 10.2): сначала меньшее
суммарное время, затем меньший индекс площадки. Без него два прогона могли бы
дать разные точки при одинаковой метрике — на сухом прогоне ничьи встретились
сразу же.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sites.feasible import validate


@dataclass
class Solution:
    """Выбранный набор и то, чем он получен."""
    sites: tuple[int, ...]        # позиции в списке допустимых площадок
    objective: float
    variant: str
    evaluated: int                # сколько наборов рассмотрено
    tie_count: int = 0            # сколько наборов дали тот же оптимум

    def as_dict(self) -> dict:
        return {"variant": self.variant, "sites": list(self.sites),
                "objective": self.objective, "evaluated": self.evaluated,
                "tie_count": self.tie_count}


def _improvement(site_times: np.ndarray, base_times: np.ndarray) -> np.ndarray:
    """Насколько площадка сокращает время в каждой клетке, но не меньше нуля.

    min(base, t) = base − max(0, base − t). В таком виде задача раскладывается
    на «вклад площадки», и перебор пар сводится к поэлементному максимуму."""
    return np.maximum(0.0, base_times[None, :] - site_times)


# Перебор пар держит в памяти массив (чанк x площадки x клетки). На городе это
# 1497 площадок и 3958 клеток, то есть 47 МБ на каждую строку чанка: без лимита
# размер улетает в гигабайты.
_CHUNK_BYTES = 256 << 20


def chunk_for(count: int, cells: int) -> int:
    """Сколько строк брать за раз, чтобы уложиться в бюджет памяти."""
    per_row = max(1, count * cells * 8)
    return max(1, int(_CHUNK_BYTES / per_row))


def _best_pair(score_pair, count: int, allowed: np.ndarray,
               chunk: int = 64) -> tuple[tuple[int, int], float, int, int]:
    """Перебор всех допустимых пар с детерминированным разрешением ничьих.

    score_pair(a_rows, b_all) возвращает матрицу значений цели: чем больше,
    тем лучше. Пары, нарушающие разнесение, отбрасываются ДО расчёта."""
    best_value = -np.inf
    best_pair = None
    evaluated = 0
    ties = 0

    for start in range(0, count, chunk):
        rows = np.arange(start, min(start + chunk, count))
        values = score_pair(rows)                       # (len(rows), count)
        mask = allowed[rows]
        # верхний треугольник: пара (a, b) рассматривается один раз
        mask &= np.arange(count)[None, :] > rows[:, None]
        if not mask.any():
            continue

        evaluated += int(mask.sum())
        candidates = np.where(mask, values, -np.inf)
        local = candidates.max()
        if local < best_value:
            continue
        hits = np.argwhere(candidates == local)
        ties += len(hits) if local == best_value else 0
        if local > best_value:
            best_value = local
            ties = len(hits) - 1
            first = hits[0]
            best_pair = (int(rows[first[0]]), int(first[1]))
        else:
            first = hits[0]
            candidate = (int(rows[first[0]]), int(first[1]))
            best_pair = min(best_pair, candidate)       # меньший индекс
    if best_pair is None:
        raise ValueError("нет ни одной допустимой пары площадок")
    return best_pair, float(best_value), evaluated, ties


def solve_pmedian(site_times: np.ndarray, base_times: np.ndarray,
                  weights: np.ndarray, allowed: np.ndarray, k: int,
                  variant: str = "B") -> Solution:
    """S* = argmin Σ w_i · t_i(S). Веса задают вариант: B, C или D (п. 10.1)."""
    gain = _improvement(site_times, base_times) * weights[None, :]

    if k == 1:
        totals = gain.sum(axis=1)
        best = int(np.argmax(totals))
        ties = int((totals == totals[best]).sum()) - 1
        return Solution((best,), float(totals[best]), variant, len(totals), ties)

    def score(rows: np.ndarray) -> np.ndarray:
        # выигрыш пары — поэлементный максимум вкладов, а не их сумма
        return np.maximum(gain[rows][:, None, :], gain[None, :, :]).sum(axis=2)

    pair, value, evaluated, ties = _best_pair(
        score, len(site_times), allowed,
        chunk=chunk_for(len(site_times), site_times.shape[1]))
    return Solution(tuple(sorted(pair)), float(value), variant, evaluated, ties)


def solve_max_coverage(site_times: np.ndarray, base_times: np.ndarray,
                       population: np.ndarray, allowed: np.ndarray, k: int,
                       threshold: float) -> Solution:
    """S* = argmax Σ P_i · 1[t_i(S) ≤ T] (п. 10.2).

    M прямо оптимизирует оцениваемую метрику, поэтому служит верхней границей
    прироста покрытия, а не рядовым конкурентом."""
    base_covered = base_times <= threshold
    # клетки, которые площадка покрывает ДОПОЛНИТЕЛЬНО к существующей сети
    new = (site_times <= threshold) & ~base_covered[None, :]
    weighted = new * population[None, :]

    if k == 1:
        totals = weighted.sum(axis=1)
        best = int(np.argmax(totals))
        ties = int((totals == totals[best]).sum()) - 1
        return Solution((best,), float(totals[best]), "M", len(totals), ties)

    def score(rows: np.ndarray) -> np.ndarray:
        # объединение покрытий: сумма минус пересечение
        union = (new[rows][:, None, :] | new[None, :, :])
        return (union * population[None, None, :]).sum(axis=2)

    pair, value, evaluated, ties = _best_pair(
        score, len(site_times), allowed,
        chunk=chunk_for(len(site_times), site_times.shape[1]))
    return Solution(tuple(sorted(pair)), float(value), "M", evaluated, ties)


def random_sets(allowed: np.ndarray, k: int, draws: int, seed: int,
                max_attempts_factor: int = 200) -> list[tuple[int, ...]]:
    """Случайные ДОПУСТИМЫЕ наборы для варианта A (п. 10.4).

    Выборка с отказом; если за отведённые попытки не набралось нужного числа,
    перечисляются все допустимые наборы и берутся из них — молча вернуть
    меньше наборов нельзя, это исказило бы распределение."""
    rng = np.random.default_rng(seed)
    count = len(allowed)
    if k == 1:
        picks = rng.choice(count, size=min(draws, count), replace=False)
        return [(int(p),) for p in picks]

    found: set[tuple[int, ...]] = set()
    for _ in range(draws * max_attempts_factor):
        if len(found) >= draws:
            break
        a, b = rng.choice(count, size=2, replace=False)
        if allowed[a, b]:
            found.add(tuple(sorted((int(a), int(b)))))

    if len(found) < draws:
        every = [(int(a), int(b)) for a, b in zip(*np.triu_indices(count, k=1))
                 if allowed[a, b]]
        if len(every) <= draws:
            return sorted(every)
        picks = rng.choice(len(every), size=draws, replace=False)
        return sorted(every[int(i)] for i in picks)
    return sorted(found)


def solve_random(allowed: np.ndarray, k: int, draws: int, seed: int) -> list:
    """Набор решений варианта A: каждое проходит общий валидатор."""
    sets = random_sets(allowed, k, draws, seed)
    for selection in sets:
        validate(selection, allowed, expected_k=k)
    return sets


def estimate_pairs(count: int, allowed: np.ndarray) -> dict:
    """Оценка объёма перебора ДО запуска (п. 10.3)."""
    total = count * (count - 1) // 2
    return {"sites": int(count), "pairs_total": int(total),
            "pairs_allowed": int(np.triu(allowed, k=1).sum())}
