"""Чувствительность выводов к параметрам (п. 12).

Результаты идут в приложение отчёта, а не в главную таблицу: это проверка
устойчивости, а не отдельный результат. Если вывод переворачивается при
разумном изменении параметра, это надо знать до публикации, а не после.

α проверяется ТОЛЬКО на городах разработки (п. 7.4): подбирать его на городах
проверки значило бы настраивать систему на данных, по которым её оценивают.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import coverage as cov
from ..scoring import weights as weights_mod
from ..variants import solve as solver


def over_thresholds(data, solutions: dict[str, tuple], cfg: dict) -> pd.DataFrame:
    """Метрики при T = 10, 15, 20 (п. 12). Решение M пересчитывается заново:
    оно зависит от порога, в отличие от p-median."""
    rows = []
    t_max = cfg["t_max_minutes"]
    for threshold in sorted({cfg["T_minutes"], *cfg["T_sensitivity"]}):
        remade = solver.solve_max_coverage(
            data.site_times, data.base_times, data.population,
            data.allowed_pairs, cfg["k"], threshold)
        all_solutions = {**solutions, "M": remade.sites}
        for name, sites in all_solutions.items():
            if not sites:
                continue
            times = cov.times_with(data.base_times,
                                   data.site_times[list(sites)], t_max)
            rows.append({
                "T": threshold, "variant": name, "sites": list(sites),
                "dCov": cov.delta_coverage(times, data.base_times,
                                           data.population, threshold),
                "dN": cov.delta_people(times, data.base_times,
                                       data.population, threshold),
            })
    return pd.DataFrame(rows)


def over_k(data, cfg: dict, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Метрики при k = 1 и k = 2 (п. 12)."""
    rows = []
    t_max = cfg["t_max_minutes"]
    threshold = cfg["T_minutes"]
    for k in sorted({cfg["k"], *cfg["k_sensitivity"]}):
        variants = {
            "B": solver.solve_pmedian(
                data.site_times, data.base_times,
                weights_mod.demand(data.population), data.allowed_pairs, k, "B"),
            "M": solver.solve_max_coverage(
                data.site_times, data.base_times, data.population,
                data.allowed_pairs, k, threshold),
        }
        for name, key in (("C", "text"), ("D", "image")):
            if key in scores:
                variants[name] = solver.solve_pmedian(
                    data.site_times, data.base_times,
                    weights_mod.demand(data.population, scores[key], cfg["alpha"]),
                    data.allowed_pairs, k, name)
        for name, solution in variants.items():
            times = cov.times_with(data.base_times,
                                   data.site_times[list(solution.sites)], t_max)
            rows.append({
                "k": k, "variant": name, "sites": list(solution.sites),
                "dCov": cov.delta_coverage(times, data.base_times,
                                           data.population, threshold),
                "dN": cov.delta_people(times, data.base_times,
                                       data.population, threshold),
            })
    return pd.DataFrame(rows)


def over_alpha(data, cfg: dict, scores: dict[str, np.ndarray],
               is_dev_city: bool) -> pd.DataFrame:
    """Чувствительность к α — только на городах разработки (п. 7.4)."""
    if not is_dev_city:
        return pd.DataFrame([{"status": "пропущено",
                              "reason": "α проверяется только на городах "
                                        "разработки (п. 7.4)"}])
    rows = []
    t_max = cfg["t_max_minutes"]
    threshold = cfg["T_minutes"]
    for name, key in (("C", "text"), ("D", "image")):
        if key not in scores:
            continue
        for alpha in sorted({cfg["alpha"], *cfg["alpha_sensitivity"]}):
            weights = weights_mod.demand(data.population, scores[key], alpha)
            solution = solver.solve_pmedian(data.site_times, data.base_times,
                                            weights, data.allowed_pairs,
                                            cfg["k"], name)
            times = cov.times_with(data.base_times,
                                   data.site_times[list(solution.sites)], t_max)
            rows.append({
                "variant": name, "alpha": alpha, "sites": list(solution.sites),
                "dCov": cov.delta_coverage(times, data.base_times,
                                           data.population, threshold),
            })
    return pd.DataFrame(rows)


def verdict(thresholds: pd.DataFrame, k_table: pd.DataFrame) -> dict:
    """Переворачивается ли порядок вариантов при смене параметров.

    Если да, вывод о качестве держится на конкретном значении параметра,
    и это обязано попасть в отчёт."""
    def ranking(frame: pd.DataFrame, group: str) -> dict:
        order = {}
        for value, subset in frame.groupby(group):
            order[value] = tuple(subset.sort_values("dCov", ascending=False)
                                 ["variant"].tolist())
        return order

    by_threshold = ranking(thresholds, "T") if len(thresholds) else {}
    by_k = ranking(k_table, "k") if len(k_table) else {}
    stable_t = len(set(by_threshold.values())) <= 1
    stable_k = len(set(by_k.values())) <= 1
    return {"ranking_by_threshold": {str(k): v for k, v in by_threshold.items()},
            "ranking_by_k": {str(k): v for k, v in by_k.items()},
            "stable_over_thresholds": stable_t,
            "stable_over_k": stable_k,
            "note": ("порядок вариантов устойчив" if stable_t and stable_k else
                     "порядок вариантов МЕНЯЕТСЯ — вывод зависит от параметра")}
