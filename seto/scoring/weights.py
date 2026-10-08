"""Скор модели → вес спроса (п. 9).

    w_i^B = P_i
    w_i^C = P_i · (1 + α · ŝ_i^текст)
    w_i^D = P_i · (1 + α · ŝ_i^изобр)

Это ЕДИНСТВЕННЫЙ канал, через который скор влияет на решение. Площадки он
не фильтрует и не штрафует (п. 22), поэтому изображение может изменить выбор,
не меняя ни набор допустимых точек, ни ограничения, и вклад виден по разнице
весов.

Нормализация идёт по ВСЕМ клеткам города, включая пустые: иначе шкала ŝ
зависела бы от того, какие клетки мы решили считать населёнными.
"""
from __future__ import annotations

import numpy as np


def normalize(scores: np.ndarray) -> tuple[np.ndarray, bool]:
    """ŝ = (s − min s) / (max s − min s) по всем клеткам города.

    Если max = min, вернуть нули и пометить: модель не различила клетки,
    и вариант сведётся к B (п. 7.1)."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return values, True
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if high <= low:
        return np.zeros_like(values), True
    return (values - low) / (high - low), False


def demand(population: np.ndarray, scores: np.ndarray | None = None,
           alpha: float = 1.0) -> np.ndarray:
    """Вес спроса клетки. scores=None даёт вариант B (w = P)."""
    people = np.asarray(population, dtype=float)
    if scores is None:
        return people.copy()

    normalized, _ = normalize(scores)
    if alpha < 0:
        raise ValueError("alpha не может быть отрицательной")
    # клетки без жителей получают нулевой вес независимо от скора (п. 7.1):
    # скор для них считается, чтобы не искажать нормализацию, но спроса нет
    return people * (1.0 + alpha * normalized)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Насколько веса варианта отличаются от базовых (п. 9).

    Нужно для правила «C ≈ B»: если веса почти совпали и решение не изменилось,
    дублирующий запуск не делается."""
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape:
        raise ValueError("наборы весов разной длины")

    nonzero = a > 0
    relative = np.zeros_like(a)
    relative[nonzero] = np.abs(b[nonzero] - a[nonzero]) / a[nonzero]
    correlation = (float(np.corrcoef(a, b)[0, 1])
                   if a.size > 1 and a.std() > 0 and b.std() > 0 else 1.0)
    return {"correlation": correlation,
            "max_relative_diff": float(relative.max()) if relative.size else 0.0,
            "mean_relative_diff": float(relative.mean()) if relative.size else 0.0}
