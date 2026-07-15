"""Точка входа: прогнать клетки через VLM (Estimate) → сохранить оценки района.

Необязательный аргумент — лимит числа клеток для пробного прогона:
    python scripts/03_estimate.py 5     # только первые 5 клеток
    python scripts/03_estimate.py       # все клетки
Благодаря кэшу повторный запуск досчитывает только новые клетки (не тратит токены).
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinic_siting.estimate.estimate import estimate_cells
from clinic_siting.utils.io import load_config, load_cells, save_cells, export_csv


def main():
    cfg = load_config()
    cells = load_cells(f"{cfg['output']['processed_dir']}/cells.parquet")

    # опциональный лимит числа клеток (проба); без аргумента — все
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(cells)
    subset = cells.head(limit)
    print(f"Прогон Estimate: {len(subset)} из {len(cells)} клеток (модель {cfg['estimate']['model']})")

    context_by_cell = dict(zip(cells["cell_id"], cells["context"]))
    estimates = estimate_cells(subset, context_by_cell, cfg)

    # мёржим оценки в таблицу клеток по cell_id
    result = cells.merge(estimates, on="cell_id", how="left")

    out_parquet = f"{cfg['output']['processed_dir']}/cells_estimated.parquet"
    out_csv = f"{cfg['output']['tables_dir']}/cells_estimated.csv"
    save_cells(result, out_parquet)
    export_csv(result, out_csv)

    filled = int(result["wealth_score"].notna().sum())
    print(f"Готово: {out_parquet} и {out_csv}")
    print(f"  оценено клеток: {filled}/{len(result)}")


if __name__ == "__main__":
    main()
