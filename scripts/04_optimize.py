"""Точка входа: посчитать скоры, решить размещение (точки + тип), сохранить рекомендации."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinic_siting.optimize.optimizer import filter_candidates, solve
from clinic_siting.utils.io import load_config, load_cells, export_csv


def main():
    cfg = load_config()
    cells = load_cells(f"{cfg['output']['processed_dir']}/cells_estimated.parquet")

    candidates = filter_candidates(cells)
    print(f"Кандидатов после фильтра area_character: {len(candidates)} из {len(cells)}")

    # time_matrix пока не нужен (расстояния для 1000 м считаются внутри solve по haversine)
    selected = solve(candidates, None, cfg)

    # selected — обычная таблица (точки заданы center_lon/lat), сохраняем в CSV
    out_csv = f"{cfg['output']['tables_dir']}/recommendations.csv"
    export_csv(selected, out_csv)

    print(f"Рекомендовано точек: {len(selected)} (бюджет {cfg['optimize']['budget_clinics']})")
    print(selected.to_string(index=False))
    print(f"Сохранено: {out_csv}")


if __name__ == "__main__":
    main()
