"""Подэксперимент: как разрешение снимка влияет на суждение модели.

Протокол — docs/RESOLUTION_STUDY.md. Один город, одни клетки, одна модель и
промпт; меняется только разрешение входа.

Два решения, которые видно в коде и легко потерять при правке:

1. Все наборы приводятся к ОДНОМУ размеру (вход кодировщика модели). Иначе
   менялись бы сразу две вещи — детализация и размер входа, — и разницу
   нельзя было бы приписать разрешению. Замер 4.0 показал, что кодировщик
   сжимает любую картинку к 256 токенам, поэтому размер берётся его.
2. Огрубление — усреднением по площади (BOX), как ведёт себя сенсор с
   крупным пикселем, а не выбором ближайшего пикселя.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..imagery import source as source_mod


@dataclass
class BuildReport:
    """Что собралось, что отсеялось и почему."""
    cells_total: int = 0
    cells_built: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    bytes_written: int = 0

    def as_dict(self) -> dict:
        return {"cells_total": self.cells_total, "cells_built": self.cells_built,
                "skipped": self.skipped,
                "megabytes": round(self.bytes_written / 1e6, 1)}


def degrade(cell: np.ndarray, cell_m: float, target_gsd: float,
            encoder_px: int) -> bytes:
    """Огрубить клетку до target_gsd и вернуть к размеру кодировщика.

    Возврат к общему размеру обязателен: см. решение 1 в шапке модуля."""
    side = max(1, int(round(cell_m / target_gsd)))
    image = Image.fromarray(cell)
    coarse = image.resize((side, side), Image.BOX)        # усреднение по площади
    back = coarse.resize((encoder_px, encoder_px), Image.NEAREST)
    buffer = io.BytesIO()
    back.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def build_sets(grid, tiles: list, levels: list[float], cell_m: float,
               encoder_px: int, out_dir: Path, source_gsd: float = 0.5,
               logger=None) -> tuple[pd.DataFrame, BuildReport]:
    """Нарезать клетки и собрать по одному набору на каждое разрешение.

    Клетка читается один раз в исходном разрешении, из неё делаются все
    уровни: так все наборы гарантированно про одну и ту же территорию."""
    say = logger.info if logger else (lambda *a, **k: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport(cells_total=len(grid))
    source_px = int(round(cell_m / source_gsd))
    rows = []

    for n, cell in enumerate(grid.itertuples(), 1):
        try:
            pixels = source_mod.read_cell(tiles, tuple(cell.geometry.bounds),
                                          source_px)
        except source_mod.CoverageError as error:
            reason = str(error).split(":")[0] if ":" in str(error) else "нет покрытия"
            report.skipped[reason] = report.skipped.get(reason, 0) + 1
            continue

        for level in levels:
            payload = degrade(pixels, cell_m, level, encoder_px)
            name = f"{cell.cell_id}__{level:g}m.png"
            (out_dir / name).write_bytes(payload)
            report.bytes_written += len(payload)
            rows.append({"cell_id": cell.cell_id, "gsd_m": level, "file": name})

        report.cells_built += 1
        if n % 25 == 0:
            say("нарезка", done=n, total=len(grid), built=report.cells_built)

    return pd.DataFrame(rows), report


def score_sets(catalogue: pd.DataFrame, out_dir: Path, cfg: dict,
               cache_dir: Path, logger=None) -> pd.DataFrame:
    """Прогнать модель по всем наборам. Кэш общий, ключ включает хеш входа,
    поэтому наборы разных разрешений не перепутаются."""
    from ..scoring.client import score_cells

    say = logger.info if logger else (lambda *a, **k: None)
    frames = []
    for level in sorted(catalogue["gsd_m"].unique()):
        part = catalogue[catalogue["gsd_m"] == level]
        payloads = [(out_dir / name).read_bytes() for name in part["file"]]
        scores, report = score_cells(part["cell_id"].tolist(), payloads,
                                     "image", cfg, cache_dir)
        say("скоринг", gsd=level, **report.as_dict())
        frames.append(scores.assign(gsd_m=level))
    return pd.concat(frames, ignore_index=True)


def agreement(scores: pd.DataFrame, reference_gsd: float) -> pd.DataFrame:
    """Согласие каждого разрешения с эталоном: ранговая корреляция и сдвиг.

    Корреляция ловит, разъехался ли порядок клеток; доля крупных сдвигов —
    насколько это заметно в баллах. Первое чувствительнее, второе понятнее."""
    from scipy.stats import spearmanr

    wide = scores.pivot(index="cell_id", columns="gsd_m", values="score")
    rows = []
    for level in sorted(wide.columns):
        # выбирать [эталон, уровень] нельзя: на самом эталоне это дубль
        # колонки, и spearmanr получает матрицу вместо двух рядов
        both = wide[[reference_gsd]].join(
            wide[level].rename("value"), how="inner").dropna()
        if len(both) < 3:
            rows.append({"gsd_m": level, "cells": len(both)})
            continue
        rho = spearmanr(both[reference_gsd], both["value"]).statistic
        diff = (both["value"] - both[reference_gsd]).abs()
        rows.append({
            "gsd_m": level, "cells": len(both),
            "spearman": round(float(rho), 4),
            "mean_abs_shift": round(float(diff.mean()), 3),
            "share_shift_over_2": round(float((diff > 2).mean()), 4),
            "max_shift": int(diff.max()),
            "mean_score": round(float(both["value"].mean()), 3),
        })
    return pd.DataFrame(rows)


def solve_by_level(scores: pd.DataFrame, grid, population: np.ndarray,
                   base_times: np.ndarray, site_times: np.ndarray,
                   allowed: np.ndarray, clinic_times: np.ndarray,
                   cfg: dict) -> pd.DataFrame:
    """Решить задачу размещения отдельно по скорам каждого разрешения.

    Меры 2 и 3 протокола: меняется ли выбор площадок и во что это обходится
    в покрытии. Веса и метрики считаются ТЕМ ЖЕ кодом, что в основном
    эксперименте, — иначе сравнивали бы реализации, а не разрешения."""
    from ..metrics import sfca
    from ..scoring import weights as weights_mod
    from ..sites import feasible as sites_mod
    from ..variants import solve as solver
    from .evaluate import _metrics_row

    order = {cell: n for n, cell in enumerate(grid["cell_id"])}
    k = min(cfg["k"], sites_mod.max_feasible_k(allowed, cfg["k"]))
    floor = sfca.floor_value(len(clinic_times), population)
    rows = []

    for level in sorted(scores["gsd_m"].unique()):
        part = scores[scores["gsd_m"] == level]
        values = np.full(len(grid), np.nan)
        for cell, score in zip(part["cell_id"], part["score"]):
            if cell in order:
                values[order[cell]] = score

        normalized, _ = weights_mod.normalize(values)
        demand = weights_mod.demand(population, normalized, cfg["alpha"])
        solution = solver.solve_pmedian(site_times, base_times, demand,
                                        allowed, k, str(level))
        chosen = list(solution.sites)
        times = np.minimum(base_times, site_times[chosen].min(axis=0))
        facility = np.vstack([clinic_times, site_times[chosen]])

        row = _metrics_row(str(level), times, base_times, population, cfg,
                           facility, floor)
        row["gsd_m"] = level
        row["sites"] = tuple(chosen)
        row["tie_count"] = solution.tie_count
        rows.append(row)

    return pd.DataFrame(rows)


def noise_floor(scores: pd.DataFrame) -> dict:
    """Во сколько баллов обходится сам факт пересъёмки той же клетки.

    Нужен, чтобы не принять шум за эффект: если сдвиг от огрубления того же
    порядка, что разброс между соседними уровнями, разрешение ни при чём.
    Считается по самым близким уровням, где физической разницы почти нет."""
    wide = scores.pivot(index="cell_id", columns="gsd_m", values="score")
    levels = sorted(wide.columns)
    if len(levels) < 2:
        return {}
    close = wide[[levels[0], levels[1]]].dropna()
    diff = (close[levels[1]] - close[levels[0]]).abs()
    return {"pair": f"{levels[0]} vs {levels[1]}",
            "mean_abs_shift": round(float(diff.mean()), 3),
            "share_over_2": round(float((diff > 2).mean()), 4)}
