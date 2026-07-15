"""
ЭТАП 3 · Оптимизатор (P-Median / MCLP на PuLP).

Совместно выбирает ЛОКАЦИЮ и ТИП: для каждой кандидатной клетки —
открывать ли и если да, то гос или частную (по argmax из двух скоров),
под ограничениями бюджета и мин. расстояния 1000м между объектами.

area_character используется как фильтр: промзона/вода отсекаются из кандидатов.
"""
from __future__ import annotations

import pulp

from ..utils.geo import haversine_m
from .scores import equity_score, profit_score

# area_character, непригодные под медучреждение (промзона и т.п.)
_EXCLUDED_AREA = {"industrial"}


def filter_candidates(cells):
    """Отсеять непригодные клетки (по area_character: промзона, вода, парк)."""
    return cells[~cells["area_character"].isin(_EXCLUDED_AREA)].copy()


def solve(cells, time_matrix, cfg: dict):
    """Решить задачу размещения. Вернуть выбранные точки с типом (public/private) и вкладом в цель.

    MILP на PuLP: максимизируем суммарную ценность выбранных клеток при бюджете и мин. расстоянии.
    time_matrix зарезервирован для P-median-покрытия (улучшение); в MVP расстояния для 1000 м
    считаем внутри по haversine."""
    opt = cfg["optimize"]
    budget = opt["budget_clinics"]
    min_dist = opt["min_distance_m"]
    w_equity = opt["weights"]["equity"]
    w_profit = opt["weights"]["profit"]

    cand = cells.reset_index(drop=True).copy()
    equity = equity_score(cand) * w_equity
    profit = profit_score(cand) * w_profit

    # тип клетки и её «ценность» = что больше из двух взвешенных скоров
    cand["facility_type"] = ["public" if equity[i] >= profit[i] else "private" for i in cand.index]
    cand["best_score"] = [max(equity[i], profit[i]) for i in cand.index]

    n = len(cand)
    problem = pulp.LpProblem("clinic_siting", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # цель: максимум суммарной ценности выбранных клеток
    problem += pulp.lpSum(cand["best_score"].iloc[i] * x[i] for i in range(n))

    # бюджет: не больше budget объектов
    problem += pulp.lpSum(x) <= budget

    # мин. расстояние: пара клеток ближе min_dist не может быть выбрана одновременно
    lons = cand["center_lon"].to_numpy()
    lats = cand["center_lat"].to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if haversine_m(lons[i], lats[i], lons[j], lats[j]) < min_dist:
                problem += x[i] + x[j] <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen = [i for i in range(n) if x[i].value() == 1]
    result = cand.iloc[chosen][["cell_id", "center_lon", "center_lat", "facility_type", "best_score"]]
    return result.sort_values("best_score", ascending=False).reset_index(drop=True)
