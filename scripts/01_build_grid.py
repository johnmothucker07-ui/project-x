"""Точка входа: построить сетку по району из config → сохранить в data/interim."""
import sys
from pathlib import Path

# консоль Windows по умолчанию не UTF-8 — иначе русский текст и символы вроде → роняют print
sys.stdout.reconfigure(encoding="utf-8")

# делаем пакет clinic_siting импортируемым при запуске скрипта (src на пути)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinic_siting.data.grid import build_grid
from clinic_siting.utils.io import load_config, save_cells


def main():
    cfg = load_config()
    grid = build_grid(
        cfg["area"]["bbox"],
        cfg["grid"]["cell_size_m"],
        cfg["grid"]["id_prefix"],
    )
    out_path = "data/interim/grid.parquet"
    save_cells(grid, out_path)
    print(f"Сетка построена: {len(grid)} клеток → {out_path}")


if __name__ == "__main__":
    main()
