"""Решение вариантов и расчёт метрик (п. 10-12, `solve` и `evaluate`).

Метрики считаются ОДНИМ кодом для всех вариантов — отдельных расчётов
«под вариант» не существует (п. 11). Поэтому здесь нет ветвлений по имени
варианта: на вход приходит набор площадок, дальше всё одинаково.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..costs.timer import CostLog
from ..metrics import coverage as cov
from ..metrics import sfca
from ..scoring import weights as weights_mod
from ..sites import feasible as sites_mod
from ..variants import solve as solver


@dataclass
class VariantResult:
    """Решение варианта и его статус."""
    name: str
    sites: tuple[int, ...]
    status: str = "ok"            # ok | skipped_duplicate | failed | missing_external
    objective: float | None = None
    tie_count: int = 0
    note: str = ""
    distribution: np.ndarray | None = None   # только для A

    def as_dict(self) -> dict:
        return {"variant": self.name, "sites": list(self.sites),
                "status": self.status, "objective": self.objective,
                "tie_count": self.tie_count, "note": self.note}


@dataclass
class Evaluation:
    """Метрики всех вариантов города."""
    table: pd.DataFrame
    differences: pd.DataFrame
    variants: list[VariantResult] = field(default_factory=list)


def _metrics_row(name: str, times: np.ndarray, base: np.ndarray,
                 population: np.ndarray, cfg: dict,
                 facility_times: np.ndarray, floor: float) -> dict:
    """Все метрики одного варианта. Общий код — в этом весь смысл (п. 11)."""
    row: dict = {"variant": name}
    for threshold in sorted({cfg["T_minutes"], *cfg["T_sensitivity"]}):
        row[f"Cov_{threshold}"] = cov.coverage(times, population, threshold)
        row[f"dCov_{threshold}"] = cov.delta_coverage(times, base, population,
                                                      threshold)
        row[f"dN_{threshold}"] = cov.delta_people(times, base, population,
                                                  threshold)
    row["worst_decile_min"] = cov.worst_decile_time(times, population)
    row["mean_time_min"] = cov.mean_time(times, population)
    row["unreachable_people"] = cov.unreachable_people(times, population,
                                                       cfg["t_max_minutes"])
    row.update(sfca.summary(facility_times, population, cfg["T_minutes"], floor))
    return row


def run_variants(data, scores: dict[str, np.ndarray], cfg: dict,
                 costs: CostLog | None = None, logger=None) -> Evaluation:
    """Решить A, B, M, C, D и посчитать метрики. scores — скоры по вариантам."""
    costs = costs or CostLog()
    say = logger.info if logger else (lambda *a, **k: None)

    base = data.base_times
    population = data.population
    site_times = data.site_times
    allowed = data.allowed_pairs
    k = min(cfg["k"], sites_mod.max_feasible_k(allowed, cfg["k"]))
    t_max = cfg["t_max_minutes"]

    results: list[VariantResult] = []
    base_weights = weights_mod.demand(population)

    # --- B: p-median по населению ---
    with costs.stage("solve", "variant:B"):
        solution = solver.solve_pmedian(site_times, base, base_weights, allowed, k, "B")
    results.append(VariantResult("B", solution.sites, objective=solution.objective,
                                 tie_count=solution.tie_count))
    say("вариант B", sites=list(solution.sites), ties=solution.tie_count)

    # --- M: максимальное покрытие ---
    with costs.stage("solve", "variant:M"):
        solution = solver.solve_max_coverage(site_times, base, population, allowed,
                                             k, cfg["T_minutes"])
    results.append(VariantResult("M", solution.sites, objective=solution.objective,
                                 tie_count=solution.tie_count))
    say("вариант M", sites=list(solution.sites), ties=solution.tie_count)

    # --- C и D: те же веса плюс скор ---
    for name, key in (("C", "text"), ("D", "image")):
        if key not in scores:
            results.append(VariantResult(name, (), status="missing_external",
                                         note=f"нет скоров режима {key}"))
            continue
        with costs.stage("solve", f"variant:{name}"):
            weights = weights_mod.demand(population, scores[key], cfg["alpha"])
            solution = solver.solve_pmedian(site_times, base, weights, allowed, k, name)
        comparison = weights_mod.compare(base_weights, weights)
        # п. 9: если решение совпало с B, дубликат не считается
        same_as_b = solution.sites == results[0].sites
        results.append(VariantResult(
            name, solution.sites,
            status="skipped_duplicate" if same_as_b else "ok",
            objective=solution.objective, tie_count=solution.tie_count,
            note=(f"решение совпало с B; корреляция весов "
                  f"{comparison['correlation']:.4f}" if same_as_b else
                  f"корреляция весов с B {comparison['correlation']:.4f}")))
        say(f"вариант {name}", sites=list(solution.sites), same_as_b=same_as_b)

    # --- A: случайный выбор ---
    with costs.stage("solve", "variant:A"):
        sets = solver.solve_random(allowed, k, cfg["random_draws"], cfg["seed"])
        deltas = np.array([
            cov.delta_coverage(cov.times_with(base, site_times[list(s)], t_max),
                               base, population, cfg["T_minutes"]) for s in sets])
    results.append(VariantResult("A", (), note=f"{len(sets)} наборов",
                                 distribution=deltas))
    say("вариант A", sets=len(sets), median=float(np.median(deltas)),
        p95=float(np.percentile(deltas, 95)))

    # --- метрики ---
    floor = sfca.floor_value(len(data.clinics), population)
    clinic_times = _facility_times(data, None)
    rows = [_metrics_row("исходная сеть", base, base, population, cfg,
                         clinic_times, floor)]
    for result in results:
        if not result.sites:
            continue
        times = cov.times_with(base, site_times[list(result.sites)], t_max)
        rows.append(_metrics_row(result.name, times, base, population, cfg,
                                 _facility_times(data, result.sites), floor))

    table = pd.DataFrame(rows)
    return Evaluation(table=table,
                      differences=_differences(table, results, cfg),
                      variants=results)


def _facility_times(data, sites) -> np.ndarray:
    """Времена [учреждения x клетки] для 2SFCA: существующие плюс новые."""
    existing = data.clinic_times if hasattr(data, "clinic_times") else None
    if existing is None:
        existing = data.base_times[None, :]
    if not sites:
        return existing
    return np.vstack([existing, data.site_times[list(sites)]])


def _differences(table: pd.DataFrame, results: list[VariantResult],
                 cfg: dict) -> pd.DataFrame:
    """Разницы между вариантами в п.п. и жителях (п. 11.4)."""
    threshold = cfg["T_minutes"]
    indexed = table.set_index("variant")
    available = [r.name for r in results if r.sites and r.name in indexed.index]
    reference = "D" if "D" in available else ("C" if "C" in available else None)

    rows = []
    for left in available:
        for right in available:
            if left >= right:
                continue
            rows.append({
                "pair": f"{left} − {right}",
                f"dCov_{threshold}_pp": (indexed.loc[left, f"dCov_{threshold}"]
                                         - indexed.loc[right, f"dCov_{threshold}"]),
                f"dN_{threshold}_people": (indexed.loc[left, f"dN_{threshold}"]
                                           - indexed.loc[right, f"dN_{threshold}"]),
            })

    random_result = next((r for r in results if r.name == "A"), None)
    if random_result is not None and random_result.distribution is not None:
        p95 = float(np.percentile(random_result.distribution, 95))
        for name in available:
            value = indexed.loc[name, f"dCov_{threshold}"]
            rows.append({
                "pair": f"{name} − A (медиана)",
                f"dCov_{threshold}_pp": value - float(np.median(random_result.distribution)),
                f"dN_{threshold}_people": np.nan,
                "above_A_p95": bool(value > p95),
                "percentile_in_A": float(100 * (random_result.distribution < value).mean()),
            })
    if reference is None:
        rows.append({"pair": "D − …", f"dCov_{threshold}_pp": np.nan,
                     f"dN_{threshold}_people": np.nan,
                     "note": "вариант D не считался: нет снимков"})
    return pd.DataFrame(rows)
