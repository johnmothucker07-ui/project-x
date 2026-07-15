"""Точка входа: валидация — доступность до/после добавления точек + сравнение с baseline.

Главная метрика проекта: среднее по населению время до ближайшей клиники.
(Проверка типа через реестр — задачи 4.2-4.3 — отложена, см. TODO.)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from clinic_siting.utils.io import load_config, load_cells, export_csv
from clinic_siting.validate.accessibility import before_after, baseline_placement


def main():
    cfg = load_config()
    cells = load_cells(f"{cfg['output']['processed_dir']}/cells_estimated.parquet")
    clinics = load_cells("data/interim/clinics.parquet")
    recommended = pd.read_csv(f"{cfg['output']['tables_dir']}/recommendations.csv")

    # доступность: модель против наивного baseline
    model = before_after(cells, clinics, recommended)
    baseline_method = cfg["validate"]["baseline"]
    base_points = baseline_placement(cells, len(recommended), baseline_method)
    baseline = before_after(cells, clinics, base_points)

    report = pd.DataFrame([
        {"scenario": "model", **model},
        {"scenario": f"baseline_{baseline_method}", **baseline},
    ])
    out_csv = f"{cfg['output']['tables_dir']}/validation.csv"
    export_csv(report, out_csv)

    print("Валидация доступности (среднее по населению время до ближайшей клиники, мин):")
    print(report.to_string(index=False))
    print(f"\nafter < before (стало ближе): {model['after'] < model['before']}")
    print(f"Модель лучше baseline: {model['improvement'] >= baseline['improvement']}")
    print(f"Сохранено: {out_csv}")


if __name__ == "__main__":
    main()
