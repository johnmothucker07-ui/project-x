"""Отчёт по городу (п. 18.1).

Главная таблица — вариант × метрики при T = 15 и k = 2; устойчивость по T
и k уходит в приложение. Рядом обязательно идут данные о качестве входа:
даты источников, полнота OSM, непривязанные жители, доля imputed. Без них
числа нельзя читать — непонятно, чему они стоят.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAIN_COLUMNS = ("variant", "Cov_15", "dCov_15", "dN_15", "worst_decile_min",
                "mean_time_min", "share_below_floor_pct", "p10", "p25")


def main_table(evaluation, cfg: dict) -> pd.DataFrame:
    """Таблица при главных T и k."""
    threshold = cfg["T_minutes"]
    columns = ["variant", f"Cov_{threshold}", f"dCov_{threshold}",
               f"dN_{threshold}", "worst_decile_min", "mean_time_min",
               "share_below_floor_pct", "p10", "p25"]
    present = [c for c in columns if c in evaluation.table.columns]
    return evaluation.table[present].copy()


def appendix_table(evaluation, cfg: dict) -> pd.DataFrame:
    """Устойчивость по порогу T (п. 12)."""
    thresholds = sorted({cfg["T_minutes"], *cfg["T_sensitivity"]})
    columns = ["variant"] + [f"{prefix}_{t}" for t in thresholds
                             for prefix in ("Cov", "dCov", "dN")]
    present = [c for c in columns if c in evaluation.table.columns]
    return evaluation.table[present].copy()


def pilot_criteria(evaluation, cfg: dict, costs: dict | None = None) -> dict:
    """Критерии пилота (п. 18.3).

    Качество: ΔCov_T варианта D выше 95-го процентиля случайного выбора A.
    Если D не считался, критерий помечается «нет данных», а не подменяется
    другим вариантом — подмена исказила бы вывод."""
    threshold = cfg["T_minutes"]
    indexed = evaluation.table.set_index("variant")
    random_result = next((v for v in evaluation.variants if v.name == "A"), None)

    quality: dict = {"metric": f"dCov_{threshold}"}
    if random_result is None or random_result.distribution is None:
        quality["status"] = "нет данных: вариант A не считался"
    else:
        p95 = float(np.percentile(random_result.distribution, 95))
        quality["A_p95"] = p95
        for name in ("D", "C", "B", "M"):
            if name in indexed.index:
                quality[f"{name}_dCov"] = float(indexed.loc[name, f"dCov_{threshold}"])
        if "D" in indexed.index:
            quality["status"] = ("выполнен"
                                 if quality.get("D_dCov", -np.inf) > p95
                                 else "не выполнен")
        else:
            quality["status"] = ("нет данных: вариант D не считался "
                                 "(нет снимков, Л6)")

    return {
        "quality": quality,
        "costs": (costs or {"status": "нет данных: журнал затрат не передан"}),
        "reliability": {
            "statuses": {v.name: v.status for v in evaluation.variants},
            "manual_fixes": 0,
            "status": ("выполнен"
                       if all(v.status in ("ok", "skipped_duplicate")
                              for v in evaluation.variants if v.name != "D")
                       else "не выполнен"),
        },
    }


def build(evaluation, city_report: dict, cfg: dict, out_dir: Path,
          costs: dict | None = None, scoring: dict | None = None) -> dict:
    """Собрать отчёт: таблицы в CSV, сводка в JSON и читаемый Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    main = main_table(evaluation, cfg)
    appendix = appendix_table(evaluation, cfg)
    main.to_csv(out_dir / "metrics_main.csv", index=False, encoding="utf-8")
    appendix.to_csv(out_dir / "metrics_thresholds.csv", index=False,
                    encoding="utf-8")
    evaluation.differences.to_csv(out_dir / "differences.csv", index=False,
                                  encoding="utf-8")
    evaluation.table.to_parquet(out_dir / "metrics.parquet")
    paths["metrics_main"] = out_dir / "metrics_main.csv"

    summary = {
        "city": city_report,
        "variants": [v.as_dict() for v in evaluation.variants],
        "pilot_criteria": pilot_criteria(evaluation, cfg, costs),
        "scoring": scoring or {},
        "costs": costs or {},
    }
    (out_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    paths["report_json"] = out_dir / "report.json"

    (out_dir / "report.md").write_text(
        _markdown(main, appendix, evaluation, city_report, cfg, summary),
        encoding="utf-8")
    paths["report_md"] = out_dir / "report.md"
    return paths


def _markdown(main: pd.DataFrame, appendix: pd.DataFrame, evaluation,
              city: dict, cfg: dict, summary: dict) -> str:
    threshold = cfg["T_minutes"]
    lines = [
        f"# Отчёт по городу: {city.get('name', '—')}",
        "",
        "## Данные",
        "",
        f"- граница: {city['boundary']['area_km2']} км², источник "
        f"`{city['boundary']['source']}`",
        f"- клеток: {city['grid']['cells']} по {cfg['grid']['size_m']} м",
        f"- население: {city['population']['transferred_total']:,.0f}, "
        f"расхождение при переносе {100 * city['population']['loss_share']:.3f} %",
        f"- поликлиник в базовой сети: {city['clinics']['state_proxy']} "
        f"({city['clinics']['selection']})",
        f"- граф: {city['graph']['nodes']:,} узлов, "
        f"выброшено {city['graph']['dropped_nodes']:,}",
        f"- непривязанных клеток: {city['accessibility']['unattached_cells']}, "
        f"недостижимых: {city['accessibility']['unreachable_cells']}",
        f"- допустимых площадок: {city['sites']['feasible']} "
        f"из {city['sites']['total_cells']}",
        f"- доля зданий с этажностью: "
        f"{100 * city['osm']['buildings_levels_share']:.1f} %",
        "",
    ]
    if city.get("k_reduced"):
        lines += [f"> **k снижен до {city['max_feasible_k']}**: набора нужного "
                  "размера при ограничениях не существует, и все варианты "
                  "решают задачу с ним.", ""]

    lines += [f"## Главная таблица (T = {threshold}, k = {cfg['k']})", "",
              main.to_markdown(index=False, floatfmt=".2f"), "",
              "## Разницы между вариантами", "",
              evaluation.differences.to_markdown(index=False, floatfmt=".2f"), "",
              "## Устойчивость по порогу", "",
              appendix.to_markdown(index=False, floatfmt=".2f"), "",
              "## Критерии пилота", "",
              "```json",
              json.dumps(summary["pilot_criteria"], ensure_ascii=False, indent=2),
              "```", ""]

    statuses = {v.name: (v.status, v.note) for v in evaluation.variants}
    lines += ["## Статусы вариантов", ""]
    for name, (status, note) in statuses.items():
        lines.append(f"- **{name}**: {status}" + (f" — {note}" if note else ""))
    return "\n".join(lines) + "\n"
