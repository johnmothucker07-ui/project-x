"""Точка входа: собрать OSM (конкуренты + выжимка) и население Kontur по клеткам."""
import sys
import time
from pathlib import Path

# консоль Windows по умолчанию не UTF-8 — иначе русский вывод роняет print
sys.stdout.reconfigure(encoding="utf-8")

# делаем пакет clinic_siting импортируемым при запуске скрипта (src на пути)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinic_siting.data.grid import build_grid
from clinic_siting.data.osm import fetch_pois, count_competitors, build_context_text
from clinic_siting.data.population import load_kontur, population_per_cell
from clinic_siting.utils.io import load_config, save_cells, export_csv


def fetch_clinics_with_retry(osm_cfg: dict, bbox: list[float], retries: int = 4, pause_s: float = 8.0):
    """Запросить клиники из Overpass с ретраями — публичный сервер иногда отвечает 504."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fetch_pois(
                bbox,
                osm_cfg["clinic_tags"],
                osm_cfg["overpass_url"],
                osm_cfg.get("exclude_healthcare"),
                osm_cfg.get("exclude_name_keywords"),
            )
        except Exception as error:  # сетевые/HTTP-ошибки Overpass
            last_error = error
            print(f"Overpass попытка {attempt}/{retries} не удалась: {type(error).__name__}. Повтор через {pause_s:.0f}с...")
            time.sleep(pause_s)
    raise RuntimeError(f"Overpass не ответил после {retries} попыток") from last_error


def main():
    cfg = load_config()
    bbox = cfg["area"]["bbox"]

    # 1. сетка клеток
    cells = build_grid(bbox, cfg["grid"]["cell_size_m"], cfg["grid"]["id_prefix"])
    print(f"Сетка: {len(cells)} клеток")

    # 2. существующие клиники из OSM
    clinics = fetch_clinics_with_retry(cfg["data"]["osm"], bbox)
    print(f"OSM: {len(clinics)} клиник-конкурентов")

    # 3. число конкурентов на клетку + текстовая выжимка окружения
    cells = count_competitors(cells, clinics)
    cells["context"] = cells["cell_id"].map(build_context_text(cells, clinics))

    # 4. население Kontur на клетку
    kontur = load_kontur(cfg["data"]["population"]["file"], bbox=bbox)
    cells = population_per_cell(cells, kontur)

    # 5. сохранить рабочую таблицу (GeoParquet) и человекочитаемую выгрузку (CSV)
    cells = cells[["cell_id", "geometry", "center_lon", "center_lat",
                   "population", "existing_clinics", "context"]]
    processed_path = f"{cfg['output']['processed_dir']}/cells.parquet"
    csv_path = f"{cfg['output']['tables_dir']}/cells.csv"
    save_cells(cells, processed_path)
    export_csv(cells, csv_path)

    print(f"Готово: {processed_path} и {csv_path}")
    print(f"  население района: {int(cells['population'].sum())}, "
          f"клиник всего: {int(cells['existing_clinics'].sum())}")


if __name__ == "__main__":
    main()
