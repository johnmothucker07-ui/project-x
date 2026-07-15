"""
ЭТАП 4 · Доступность — ГЛАВНАЯ метрика (не зависит от реестра).

Считает среднее время до ближайшей клиники ДО и ПОСЛЕ добавления
рекомендованных точек, и сравнивает с baseline (случайно / по плотности).
Это основное доказательство "локация выбрана грамотно".
"""
from __future__ import annotations

import pandas as pd

from ..optimize.routing import nearest_clinic_time


def mean_access_time(cells, clinics, osrm_url: str | None = None) -> float:
    """Среднее по населению время до ближайшей клиники (минуты)."""
    times = nearest_clinic_time(cells, clinics, osrm_url)
    weights = cells["population"].to_numpy()
    if weights.sum() == 0:
        return float(times.mean())
    # взвешиваем по населению: важно время для людей, а не для пустых клеток
    return float((times.to_numpy() * weights).sum() / weights.sum())


def before_after(cells, existing_clinics, recommended, osrm_url: str | None = None) -> dict:
    """Время до/после добавления рекомендованных точек. Вернуть {before, after, improvement}."""
    before = mean_access_time(cells, existing_clinics, osrm_url)

    # рекомендованные точки добавляем как новые клиники (нужны колонки lon/lat)
    new_points = pd.DataFrame({
        "lon": recommended["center_lon"].to_numpy(),
        "lat": recommended["center_lat"].to_numpy(),
    })
    combined = pd.concat([existing_clinics[["lon", "lat"]], new_points], ignore_index=True)
    after = mean_access_time(cells, combined, osrm_url)

    return {"before": before, "after": after, "improvement": before - after}


def baseline_placement(cells, n: int, method: str = "population_density"):
    """Baseline-расстановка для сравнения (случайно или по плотности населения).
    Возвращает таблицу точек с center_lon/center_lat (как рекомендации)."""
    if method == "random":
        chosen = cells.sample(n=n, random_state=0)
    else:  # population_density: наивно ставим там, где больше всего людей
        chosen = cells.nlargest(n, "population")
    return chosen[["cell_id", "center_lon", "center_lat"]].reset_index(drop=True)
