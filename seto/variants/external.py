"""Интерфейс для внешнего решения GIS-AHP, вариант E (п. 10.5).

Сам метод AHP здесь НЕ реализуется — его делает отдельная команда. Наша часть:
отдать им те же данные, на которых работают остальные варианты, и принять
готовое решение, проверив его ТЕМ ЖЕ валидатором.

Импортированное решение не «чинится»: нарушение — отказ с причиной (п. 22).
Иначе разница D − E мерила бы нашу правку, а не метод AHP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..sites.feasible import ConstraintViolation, validate


class ExternalSolutionError(ValueError):
    """Решение E не принято."""


def export_package(data, cfg: dict, city: str, data_manifest_hash: str,
                   out_dir: Path) -> dict[str, Path]:
    """Пакет данных для команды AHP: тот же снимок, та же сетка, те же площадки."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "grid": out_dir / "grid.parquet",
        "cell_features": out_dir / "cell_features.parquet",
        "clinics": out_dir / "clinics.csv",
        "sites": out_dir / "feasible_sites.parquet",
        "interface": out_dir / "AHP_INTERFACE.json",
    }
    data.grid.to_parquet(paths["grid"])
    data.features.assign(population=data.population).to_parquet(
        paths["cell_features"])
    data.clinics.drop(columns="geometry", errors="ignore").to_csv(
        paths["clinics"], index=False, encoding="utf-8")
    sites = data.grid.iloc[data.site_index][["cell_id", "geometry", "x", "y"]]
    sites = sites.assign(site_position=np.arange(len(data.site_index)))
    sites.to_parquet(paths["sites"])

    interface = {
        "city": city,
        "data_manifest_sha256": data_manifest_hash,
        "k": cfg["k"],
        "T_minutes": cfg["T_minutes"],
        "constraints": {
            "min_dist_to_existing_m": cfg["min_dist_to_existing_m"],
            "min_dist_between_new_m": cfg["min_dist_between_new_m"],
            "measured_along": "пешеходную сеть OSM, не по прямой",
            "note": ("собственного разнесения у варианта E нет: правило одно "
                     "на все варианты (п. 8)"),
        },
        "answer_format": {
            "city": "<имя города>",
            "data_manifest_sha256": "<тот же хеш, что здесь>",
            "k": cfg["k"],
            "sites": ["<site_position из feasible_sites.parquet>", "..."],
            "method_version": "<версия вашего метода>",
            "weight_sensitivity_solutions": [["<site_position>", "..."], "..."],
        },
        "fields": {
            "feasible_sites.parquet": "site_position — то, что нужно вернуть; "
                                      "cell_id и координаты для справки",
            "cell_features.parquet": "признаки клетки по её полигону плюс население",
            "clinics.csv": "существующая сеть, использованная всеми вариантами",
        },
    }
    paths["interface"].write_text(
        json.dumps(interface, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def import_solution(path: Path, allowed: np.ndarray, cfg: dict, city: str,
                    data_manifest_hash: str) -> dict:
    """Принять решение E. Нарушение — отказ с причиной, а не исправление."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if payload.get("city") != city:
        raise ExternalSolutionError(
            f"решение для города {payload.get('city')!r}, а не {city!r}")
    if payload.get("data_manifest_sha256") != data_manifest_hash:
        raise ExternalSolutionError(
            "решение посчитано на других данных: хеш манифеста не совпадает")

    sites = payload.get("sites")
    if not isinstance(sites, list):
        raise ExternalSolutionError("в решении нет списка sites")
    try:
        selection = tuple(int(s) for s in sites)
    except (TypeError, ValueError) as error:
        raise ExternalSolutionError(f"площадки не целые числа: {sites}") from error

    try:
        validate(selection, allowed, expected_k=cfg["k"])
    except ConstraintViolation as error:
        raise ExternalSolutionError(f"решение нарушает ограничения: {error}") from error

    extra = []
    for variant in payload.get("weight_sensitivity_solutions") or []:
        candidate = tuple(int(s) for s in variant)
        try:
            validate(candidate, allowed, expected_k=cfg["k"])
            extra.append(candidate)
        except ConstraintViolation:
            continue      # недопустимые варианты устойчивости просто не берём

    return {"sites": selection, "method_version": payload.get("method_version"),
            "sensitivity_solutions": extra}


def import_costs(path: Path) -> pd.DataFrame:
    """Журнал затрат команды AHP в формате human_log.csv (п. 10.5)."""
    frame = pd.read_csv(path)
    required = {"city", "variant", "mode", "minutes", "activity"}
    missing = required - set(frame.columns)
    if missing:
        raise ExternalSolutionError(
            f"в журнале затрат нет колонок: {', '.join(sorted(missing))}")
    return frame
