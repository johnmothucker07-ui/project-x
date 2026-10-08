"""Подготовка города: от снимка до допустимых площадок (п. 15, `prepare-city`).

Всё, что считается ОДИН раз и общее для всех вариантов: граница, сетка,
население, признаки, поликлиники, пешеходный граф, матрица времён, площадки.
Затраты на это идут тегом shared_prep и в отчёте показываются отдельно —
приписывать общую подготовку одному варианту было бы нечестно (п. 14).

Результаты складываются в папку прогона и больше не перезаписываются.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import settings
from ..acquire.osm import describe_layers
from ..costs.timer import SHARED, CostLog
from ..data import boundary as boundary_mod
from ..data import features as features_mod
from ..data import grid as grid_mod
from ..data import osm_layers as layers_mod
from ..data import population as population_mod
from ..routing import graph as graph_mod
from ..routing import matrix as matrix_mod
from ..sites import feasible as sites_mod


@dataclass
class CityData:
    """Готовые данные города, на которых дальше считаются варианты."""
    grid: gpd.GeoDataFrame
    features: pd.DataFrame
    population: np.ndarray
    clinics: gpd.GeoDataFrame
    base_times: np.ndarray            # до ближайшей существующей поликлиники
    clinic_times: np.ndarray          # [поликлиники x клетки], нужна для 2SFCA
    site_times: np.ndarray            # [площадки x клетки]
    site_index: np.ndarray            # позиции площадок в сетке
    allowed_pairs: np.ndarray
    report: dict

    def save(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "grid": out_dir / "grid.parquet",
            "cell_features": out_dir / "cell_features.parquet",
            "clinics": out_dir / "clinics.csv",
            "times": out_dir / "times.npz",
            "sites": out_dir / "sites.parquet",
        }
        self.grid.to_parquet(paths["grid"])
        self.features.assign(population=self.population).to_parquet(
            paths["cell_features"])
        self.clinics.drop(columns="geometry").assign(
            lon=self.clinics.to_crs(4326).geometry.x,
            lat=self.clinics.to_crs(4326).geometry.y).to_csv(
                paths["clinics"], index=False, encoding="utf-8")
        np.savez_compressed(paths["times"], base=self.base_times,
                            clinics=self.clinic_times,
                            sites=self.site_times, site_index=self.site_index,
                            allowed=self.allowed_pairs)
        self.grid.iloc[self.site_index][["cell_id", "geometry"]].to_parquet(
            paths["sites"])
        return paths


def _state_clinics(medical: gpd.GeoDataFrame, cfg: dict) -> gpd.GeoDataFrame:
    """Сеть государственных поликлиник.

    ВРЕМЕННО по названию: тег operator заполнен у 6 % объектов, а
    operator:type у 0,2 % — по тегам OSM гос и частные не разделить.
    Настоящее выделение требует реестра (п. 4.5, Л7), и до него это
    ограничение идёт в протокол."""
    pattern = cfg["city"].get("state_clinic_name_pattern", "поликлин")
    names = medical["name"].fillna("").str.lower()
    return medical[names.str.contains(pattern)].reset_index(drop=True)


def prepare_city(city: str, cfg: dict, out_dir: Path, logger=None,
                 costs: CostLog | None = None) -> CityData:
    """Собрать все общие данные города."""
    costs = costs or CostLog()
    say = logger.info if logger else (lambda *a, **k: None)

    raw = settings.raw_dir(cfg, city, cfg["snapshot_date"])
    pbf = raw / "osm.pbf"
    if not pbf.is_file():
        raise FileNotFoundError(
            f"нет снимка OSM: {pbf}. Сначала `python -m seto acquire {city}`")

    with costs.stage("boundary", SHARED):
        bounds_cfg = cfg["city"]["boundary"]
        area = boundary_mod.load(pbf, bounds_cfg.get("relation_id"),
                                 bounds_cfg.get("admin_level"))
        say("граница", **area.as_dict())

    with costs.stage("grid", SHARED) as extra:
        grid = grid_mod.build_grid(area.frame, cfg["grid"]["size_m"],
                                   cfg["grid"]["id_format"])
        extra.update(grid_mod.grid_summary(grid))
        say("сетка", cells=len(grid))

    bbox = tuple(area.frame.to_crs(4326).total_bounds)
    crs = area.frame.crs

    with costs.stage("osm_layers", SHARED) as extra:
        layers = {k: v.to_crs(crs)
                  for k, v in layers_mod.extract_polygons(pbf, bbox).items()}
        roads = layers_mod.extract_roads(pbf, bbox).to_crs(crs)
        points = layers_mod.extract_points(pbf, bbox).to_crs(crs)
        extra.update(layers_mod.completeness_report(
            layers["buildings"], roads, points))
        say("слои OSM", buildings=len(layers["buildings"]), roads=len(roads))

    with costs.stage("population", SHARED) as extra:
        raster = _population_raster(raw)
        transfer = population_mod.to_cells(
            raster, grid, cfg["population"]["mass_tolerance"])
        extra.update(transfer.as_dict())
        population = transfer.per_cell.to_numpy()
        say("население", total=int(population.sum()),
            loss=transfer.loss_share)

    with costs.stage("features", SHARED):
        features = features_mod.build(grid, layers, roads, points)
        say("признаки", columns=len(features.columns))

    with costs.stage("clinics", SHARED) as extra:
        medical = layers_mod.medical_facilities(
            points, layers["poi_polygons"], cfg["osm"]["clinic_tags"]).to_crs(crs)
        clinics = _state_clinics(medical, cfg)
        extra.update({"candidates": len(medical), "state_proxy": len(clinics)})
        say("поликлиники", candidates=len(medical), state=len(clinics))

    with costs.stage("graph", SHARED) as extra:
        graph = graph_mod.build(roads, cfg["walk_speed_m_per_min"],
                                cfg["osm"]["walk_network"])
        extra.update({"nodes": graph.node_count, "edges": graph.edge_count,
                      "dropped_nodes": graph.dropped_nodes})
        say("граф", nodes=graph.node_count, edges=graph.edge_count)

    speed = cfg["walk_speed_m_per_min"]
    snap_max = cfg["snap_max_m"]
    t_max = cfg["t_max_minutes"]
    cell_xy = grid[["x", "y"]].to_numpy()

    with costs.stage("times_existing", SHARED) as extra:
        cells_snapped = matrix_mod.snap(graph, cell_xy, snap_max, speed)
        clinic_snapped = matrix_mod.snap(
            graph, np.c_[clinics.geometry.x, clinics.geometry.y], snap_max, speed)
        clinic_times = matrix_mod.travel_times(graph, clinic_snapped,
                                               cells_snapped, t_max)
        base_times = matrix_mod.nearest_time(clinic_times, t_max)
        unreachable = int((base_times >= t_max).sum())
        extra.update({"unattached_cells": cells_snapped.unattached_count,
                      "unreachable_cells": unreachable,
                      "unreachable_people": float(population[base_times >= t_max].sum())})
        say("времена до существующей сети",
            median=float(np.median(base_times)), unreachable=unreachable)

    with costs.stage("sites", SHARED) as extra:
        to_existing = matrix_mod.network_distance(
            graph, cells_snapped, clinic_snapped, speed,
            limit_m=cfg["min_dist_to_existing_m"]).min(axis=1)
        to_existing = np.where(np.isfinite(to_existing), to_existing, np.inf)
        chosen = sites_mod.select(features, cfg["feasible_site"],
                                  cells_snapped.attached, to_existing,
                                  cfg["min_dist_to_existing_m"])
        extra.update(chosen.as_dict())
        say("допустимые площадки", **chosen.as_dict())

    with costs.stage("times_sites", SHARED) as extra:
        site_snapped = matrix_mod.snap(graph, cell_xy[chosen.index], snap_max, speed)
        site_times = matrix_mod.travel_times(graph, site_snapped,
                                             cells_snapped, t_max)
        between = matrix_mod.network_distance(graph, site_snapped, site_snapped,
                                              speed, cfg["min_dist_between_new_m"])
        allowed = sites_mod.pair_allowed(
            np.where(np.isfinite(between), between, np.inf),
            cfg["min_dist_between_new_m"])
        feasible_k = sites_mod.max_feasible_k(allowed, cfg["k"])
        extra.update({"matrix": list(site_times.shape),
                      "max_feasible_k": feasible_k})
        say("матрица площадок", shape=list(site_times.shape),
            max_feasible_k=feasible_k)

    report = {
        "boundary": area.as_dict(),
        "grid": grid_mod.grid_summary(grid),
        "population": transfer.as_dict(),
        "osm": layers_mod.completeness_report(layers["buildings"], roads, points),
        "clinics": {"candidates": len(medical), "state_proxy": len(clinics),
                    "selection": "по названию — реестра нет (Л7)"},
        "graph": {"nodes": graph.node_count, "edges": graph.edge_count,
                  "dropped_nodes": graph.dropped_nodes},
        "accessibility": {"unattached_cells": cells_snapped.unattached_count,
                          "unreachable_cells": unreachable},
        "sites": chosen.as_dict(),
        "max_feasible_k": feasible_k,
        "k_reduced": feasible_k < cfg["k"],
        "layers": describe_layers(pbf),
    }
    data = CityData(grid=grid, features=features, population=population,
                    clinics=clinics, base_times=base_times,
                    clinic_times=clinic_times,
                    site_times=site_times, site_index=chosen.index,
                    allowed_pairs=allowed, report=report)
    data.save(out_dir)
    return data


def _population_raster(raw: Path) -> Path:
    """Найти растр населения в папке снимка."""
    candidates = sorted(raw.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"в {raw} нет растра населения (*.tif). Скачай его командой acquire")
    return candidates[0]
