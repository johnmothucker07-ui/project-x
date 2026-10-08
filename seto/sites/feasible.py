"""Допустимые площадки и общие ограничения (п. 8).

Правила одинаковы для ВСЕХ вариантов — A, B, M, C, D и внешнего E. Поэтому
модуль один, и решение E при импорте проходит через тот же валидатор.
Ограничение, которое есть только у одного варианта, сделало бы разницу
следствием правила, а не метода (п. 22).

Нарушение ограничений — исключение, а не предупреждение: молча принятое
недопустимое решение испортит все метрики города.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ConstraintViolation(ValueError):
    """Набор площадок нарушает общие ограничения."""


@dataclass
class FeasibleSites:
    """Допустимые площадки и причины отсева остальных клеток."""
    index: np.ndarray                 # позиции допустимых клеток в сетке
    reasons: dict[str, int] = field(default_factory=dict)
    total_cells: int = 0

    def __len__(self) -> int:
        return len(self.index)

    def as_dict(self) -> dict:
        return {"feasible": len(self), "total_cells": self.total_cells,
                "excluded_by": dict(self.reasons)}


def select(features: pd.DataFrame, rule: dict, attached: np.ndarray,
           distance_to_existing_m: np.ndarray,
           min_dist_to_existing_m: float) -> FeasibleSites:
    """Отобрать клетки, пригодные под размещение.

    Считает, сколько клеток отсеяло каждое правило по отдельности — иначе
    непонятно, почему допустимых мало (п. 8 требует счётчиков по причинам)."""
    size = len(features)
    built = (features["residential_share"].to_numpy()
             + features["public_share"].to_numpy())

    checks = {
        "мало жилой и общественной застройки": built >= rule["min_built_area_share"],
        "вода": features["water_share"].to_numpy() < rule.get("max_water_share", 0.5),
        "парк или зелень": (features["green_share"].to_numpy()
                            < rule.get("max_green_share", 0.5)),
        "промзона": (features["landuse_industrial_share"].to_numpy()
                     < rule.get("max_industrial_share", 0.5)),
        "железная дорога": (features["landuse_railway_share"].to_numpy()
                            < rule.get("max_railway_share", 0.5)),
        "кладбище": (features["landuse_cemetery_share"].to_numpy()
                     < rule.get("max_cemetery_share", 0.5)),
        "военная территория": (features["landuse_military_share"].to_numpy()
                               < rule.get("max_military_share", 0.5)),
        "не привязана к графу": np.asarray(attached, dtype=bool),
        "ближе минимума к существующей поликлинике":
            np.asarray(distance_to_existing_m, dtype=float) >= min_dist_to_existing_m,
    }

    keep = np.ones(size, dtype=bool)
    reasons: dict[str, int] = {}
    for name, passed in checks.items():
        # считаем вклад каждого правила среди ещё не отсеянных
        reasons[name] = int((keep & ~passed).sum())
        keep &= passed

    return FeasibleSites(index=np.flatnonzero(keep), reasons=reasons,
                         total_cells=size)


def pair_allowed(distance_matrix: np.ndarray, min_dist_m: float) -> np.ndarray:
    """Булева матрица «пару можно выбрать вместе» (разнесение по сети).

    Диагональ ложна: одну площадку нельзя выбрать дважды."""
    allowed = np.asarray(distance_matrix, dtype=float) >= min_dist_m
    np.fill_diagonal(allowed, False)
    return allowed


def max_feasible_k(allowed: np.ndarray, k: int) -> int:
    """Наибольшее число площадок, которые можно разнести (п. 8).

    Если набора размера k нет, город помечается `k_reduced`, и ВСЕ варианты
    решают задачу с этим наибольшим k — правило одинаково для всех.

    Для k ≤ 2 ответ точный: достаточно существования хотя бы одной
    допустимой пары. Для k > 2 это задача о наибольшей клике, поэтому
    возвращается жадная нижняя оценка — она честно помечена."""
    size = len(allowed)
    if size == 0:
        return 0
    if k <= 1:
        return min(k, size)
    if not allowed.any():
        return 1
    if k == 2:
        return 2
    return _greedy_independent(allowed, k)


def _greedy_independent(allowed: np.ndarray, k: int) -> int:
    """Жадно набрать попарно разнесённый набор."""
    chosen: list[int] = []
    for candidate in range(len(allowed)):
        if all(allowed[candidate, other] for other in chosen):
            chosen.append(candidate)
            if len(chosen) == k:
                break
    return len(chosen)


def validate(selection, allowed: np.ndarray, expected_k: int) -> None:
    """Проверить набор. Нарушение — исключение (п. 8).

    Через эту функцию проходит КАЖДОЕ решение, включая импортированное
    решение варианта E."""
    chosen = list(selection)
    if len(chosen) != expected_k:
        raise ConstraintViolation(
            f"в наборе {len(chosen)} площадок вместо {expected_k}")
    if len(set(chosen)) != len(chosen):
        raise ConstraintViolation("площадка повторяется в наборе")
    # границы проверяем ДО обращения к матрице: иначе чужой индекс из
    # импортированного решения E даст IndexError вместо внятной причины
    for site in chosen:
        if not 0 <= site < len(allowed):
            raise ConstraintViolation(
                f"площадка {site} вне списка допустимых (их {len(allowed)})")

    for position, site in enumerate(chosen):
        for other in chosen[position + 1:]:
            if not allowed[site, other]:
                raise ConstraintViolation(
                    f"площадки {site} и {other} разнесены меньше минимума")
