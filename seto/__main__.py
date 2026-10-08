"""Единая точка входа: python -m seto <команда>.

Список команд — по п. 15 списка работ. Нереализованные честно говорят об этом
и называют шаг, на котором появятся, вместо того чтобы молча ничего не сделать.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# консоль Windows по умолчанию не UTF-8 — иначе русский вывод роняет print
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from . import settings as config_mod  # noqa: E402
from .geo import utm_epsg
from .log import RunLogger
from .run import Run, git_state

# команды из п. 15, которые ещё не написаны: имя -> шаг плана
PLANNED = {
    "profile-cities": "п. 4.1",
    "acquire-imagery": "Ш7 (п. 5)",
    "prompt-dev": "п. 7.4",
    "sensitivity": "п. 12",
    "freeze-protocol": "п. 13",
    "run-test": "п. 13",
    "package-results": "п. 18.5",
    "export-for-ahp": "п. 10.5",
    "import-solution": "п. 10.5",
    "log-human": "п. 14",
}


def cmd_cities(_args: argparse.Namespace) -> int:
    """Показать города, для которых есть конфиг."""
    for city in config_mod.available_cities():
        cfg = config_mod.load_city(city)
        print(f"{city:<12} {cfg.get('role', '?'):<8} {cfg.get('title', '')}")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    """Напечатать эффективный конфиг (глобальный + городской)."""
    import yaml
    cfg = config_mod.load(args.city)
    yaml.safe_dump(cfg, sys.stdout, allow_unicode=True, sort_keys=False)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Проверить окружение: конфиги, git, зависимости, ключ модели."""
    import os
    from pathlib import Path

    print("--- конфиги ---")
    cfg = config_mod.load(args.city)
    print(f"  global.yaml       OK (k={cfg['k']}, T={cfg['T_minutes']}, "
          f"сетка {cfg['grid']['size_m']} м)")
    if args.city:
        city = cfg["city"]
        print(f"  {args.city:<17} OK (роль {city['role']}, "
              f"relation {city['boundary']['relation_id']})")

    print("--- git ---")
    state = git_state()
    print(f"  коммит {state['commit']}  ветка {state['branch']}  "
          f"грязное дерево: {state['dirty']}")

    print("--- где будут данные ---")
    root = config_mod.data_root(cfg).resolve()
    print(f"  data_root: {root}")
    # сырые PBF и растры — это гигабайты; синхронизация их в облако тормозит запись
    synced = [part for part in ("OneDrive", "Dropbox", "Яндекс.Диск", "Google Drive")
              if part.lower() in str(root).lower()]
    if synced:
        print(f"  ВНИМАНИЕ: путь внутри {synced[0]} — задай SETO_DATA_ROOT в .env")

    print("--- ключ модели ---")
    env = Path(".env")
    key_name = cfg["vlm"]["api_key_env"]
    has_key = bool(os.environ.get(key_name)) or (
        env.is_file() and key_name in env.read_text(encoding="utf-8"))
    print(f"  {key_name}: {'найден' if has_key else 'НЕ НАЙДЕН'}")

    print("--- зависимости ---")
    # find_spec вместо import: нужно знать, установлен ли пакет, а не выполнять его.
    # Импорт стоит дорого при первом обращении — компилируется байт-код всего
    # пакета (networkx без кэша .pyc импортировался 158 с, с кэшем 1,3 с).
    import importlib.util
    # pyrosm намеренно не нужен: на Windows не собирается (зависимость cykhash
    # требует компилятор), PBF читаем драйвером OSM из GDAL через pyogrio.
    needed = ("geopandas", "shapely", "pyproj", "pyogrio", "numpy", "pandas",
              "yaml", "osmnx", "networkx", "rasterio", "scipy")
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    print(f"  установлено: {len(needed) - len(missing)}/{len(needed)}")
    if missing:
        print(f"  НЕ ХВАТАЕТ: {', '.join(missing)}")
    return 1 if missing else 0


def cmd_new_run(args: argparse.Namespace) -> int:
    """Создать папку прогона с манифестом. Проверка каркаса."""
    cfg = config_mod.load(args.city)
    with Run.start(args.city, args.label, cfg) as run:
        logger = RunLogger(run.run_id, run.dir / "log.jsonl")
        logger.info("прогон создан", city=args.city, dir=str(run.dir))
        boundary = cfg["city"]["boundary"]
        logger.info("город", relation_id=boundary["relation_id"],
                    utm_epsg=cfg["city"].get("utm_epsg") or "определится по центроиду")
        logger.close()
    print(f"создано: {run.dir}")
    print(f"манифест: {run.dir / 'manifest.json'}")
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    """Скачать сырые данные города в raw/<city>/<snapshot_date>/ и зафиксировать."""
    from datetime import date

    from .acquire.download import write_data_manifest
    from .acquire.osm import describe_layers, download_pbf

    cfg = config_mod.load(args.city)
    snapshot = args.snapshot_date or date.today().isoformat()
    raw = config_mod.raw_dir(cfg, args.city, snapshot)
    sources_cfg = cfg["city"].get("sources") or {}
    if "osm_pbf" not in sources_cfg:
        print("в конфиге города не задан sources.osm_pbf", file=sys.stderr)
        return 1

    with Run.start(args.city, "acquire", cfg) as run:
        logger = RunLogger(run.run_id, run.dir / "log.jsonl")
        logger.info("загрузка OSM", snapshot_date=snapshot, target=str(raw))

        osm_cfg = sources_cfg["osm_pbf"]
        source = download_pbf(osm_cfg["url"], raw, osm_cfg.get("license"),
                              overwrite=args.overwrite)
        logger.info("файл получен", size_mb=round(source.size_bytes / (1 << 20), 1),
                    sha256=source.sha256[:16], last_modified=source.last_modified)
        run.record_input(source.path)

        # сначала смотрим, что в файле, и только потом что-то парсим
        layers = describe_layers(Path(source.path))
        for name, info in layers.items():
            logger.info("слой", layer=name, **{k: v for k, v in info.items()
                                               if k != "fields"})

        manifest = write_data_manifest([source], raw / "data_manifest.json",
                                       snapshot, notes={"layers": layers})
        run.record_output(manifest)
        logger.close()

    print(f"\nсырые данные: {raw}")
    print(f"манифест данных: {manifest}")
    print(f"манифест прогона: {run.dir / 'manifest.json'}")
    return 0


def cmd_prepare_city(args: argparse.Namespace) -> int:
    """Подготовить все общие данные города: сетка, население, граф, площадки."""
    import json

    from .costs.timer import CostLog
    from .experiment.prepare import prepare_city

    cfg = config_mod.load(args.city)
    if args.snapshot_date:
        cfg["snapshot_date"] = args.snapshot_date
    if not cfg.get("snapshot_date"):
        print("не задан snapshot_date: укажи --snapshot-date или впиши в config",
              file=sys.stderr)
        return 1

    costs = CostLog(mode=args.mode)
    with Run.start(args.city, "prepare-city", cfg) as run:
        logger = RunLogger(run.run_id, run.dir / "log.jsonl")
        data = prepare_city(args.city, cfg, run.dir, logger, costs)
        for path in data.save(run.dir).values():
            run.record_output(path)
        report = run.dir / "city_report.json"
        report.write_text(json.dumps(data.report, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        run.record_output(report)
        run.record_output(costs.save(run.dir / "costs.json"))
        logger.info("подготовка завершена", seconds=round(costs.total(), 1))
        logger.close()

    print()
    print(f"готово за {costs.total():.0f} с: {run.dir}")
    if data.report["k_reduced"]:
        print(f"ВНИМАНИЕ: k снижен до {data.report['max_feasible_k']} — "
              "все варианты будут решать задачу с ним")
    return 0


def cmd_run_city(args: argparse.Namespace) -> int:
    """Сквозной прогон города: подготовка, скоринг, варианты, метрики, отчёт."""
    from pathlib import Path as _Path

    from .experiment.runner import ProtocolRequired, run_city

    cfg = config_mod.load(args.city)
    if args.snapshot_date:
        cfg["snapshot_date"] = args.snapshot_date
    protocol = _Path(args.protocol) if args.protocol else None
    try:
        out = run_city(args.city, cfg, protocol, args.with_image, args.mode)
    except ProtocolRequired as error:
        print(f"запуск запрещён: {error}", file=sys.stderr)
        return 3
    print()
    print(f"прогон: {out}")
    print(f"отчёт:  {out / 'report.md'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Показать готовый отчёт прогона."""
    path = Path(args.run_dir) / "report.md"
    if not path.is_file():
        print(f"нет отчёта: {path}", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_utm(args: argparse.Namespace) -> int:
    """Показать зону UTM для координат."""
    print(utm_epsg(args.lon, args.lat))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seto", description="Стенд сравнения вариантов размещения поликлиник")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cities", help="список городов из конфигов").set_defaults(
        func=cmd_cities)

    p = sub.add_parser("config-show", help="эффективный конфиг")
    p.add_argument("city", nargs="?", default=None)
    p.set_defaults(func=cmd_config_show)

    p = sub.add_parser("doctor", help="проверка окружения")
    p.add_argument("city", nargs="?", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("new-run", help="создать папку прогона с манифестом")
    p.add_argument("city")
    p.add_argument("--label", default="manual")
    p.set_defaults(func=cmd_new_run)

    p = sub.add_parser("acquire", help="скачать сырые данные города в raw/")
    p.add_argument("city")
    p.add_argument("--snapshot-date", default=None,
                   help="дата среза, по умолчанию сегодня")
    p.add_argument("--overwrite", action="store_true",
                   help="перекачать, даже если файл уже есть")
    p.set_defaults(func=cmd_acquire)

    p = sub.add_parser("prepare-city", help="сетка, население, граф, площадки")
    p.add_argument("city")
    p.add_argument("--snapshot-date", default=None)
    p.add_argument("--mode", default="onboard_city",
                   choices=("dev_build", "onboard_city", "recompute"),
                   help="режим учёта затрат (п. 14)")
    p.set_defaults(func=cmd_prepare_city)

    p = sub.add_parser("run-city", help="сквозной прогон города до отчёта")
    p.add_argument("city")
    p.add_argument("--snapshot-date", default=None)
    p.add_argument("--protocol", default=None,
                   help="путь к protocol.json (обязателен для городов test)")
    p.add_argument("--with-image", action="store_true",
                   help="считать вариант D (нужны загруженные снимки)")
    p.add_argument("--mode", default="onboard_city",
                   choices=("dev_build", "onboard_city", "recompute"))
    p.set_defaults(func=cmd_run_city)

    p = sub.add_parser("report", help="показать отчёт прогона")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("utm", help="зона UTM по координатам")
    p.add_argument("lon", type=float)
    p.add_argument("lat", type=float)
    p.set_defaults(func=cmd_utm)

    for name, step in PLANNED.items():
        sp = sub.add_parser(name, help=f"[не реализовано, {step}]")
        sp.add_argument("rest", nargs="*")
        sp.set_defaults(func=_not_implemented, _name=name, _step=step)

    return parser


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"команда '{args._name}' ещё не реализована — это {args._step}",
          file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except config_mod.ConfigError as error:
        print(f"ошибка конфигурации: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
