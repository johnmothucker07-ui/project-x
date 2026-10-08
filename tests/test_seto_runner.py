"""Тесты оркестрации, отчёта и чувствительности (п. 11.4, 12, 13, 18)."""
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.experiment import sensitivity as sens  # noqa: E402
from seto.experiment.evaluate import run_variants  # noqa: E402
from seto.experiment.runner import ProtocolRequired, guard_test_city  # noqa: E402
from seto.report import city as report_mod  # noqa: E402

UTM = "EPSG:32637"


@dataclass
class FakeCity:
    """Маленький город с известным ответом: три площадки, четыре клетки."""
    grid: gpd.GeoDataFrame
    features: pd.DataFrame
    population: np.ndarray
    clinics: gpd.GeoDataFrame
    base_times: np.ndarray
    clinic_times: np.ndarray
    site_times: np.ndarray
    site_index: np.ndarray
    allowed_pairs: np.ndarray
    report: dict


@pytest.fixture
def city():
    cells = [box(0, 0, 500, 500), box(500, 0, 1000, 500),
             box(0, 500, 500, 1000), box(500, 500, 1000, 1000)]
    grid = gpd.GeoDataFrame({"cell_id": [f"cell_{i:06d}" for i in range(4)]},
                            geometry=cells, crs=UTM)
    population = np.array([1000.0, 800.0, 50.0, 10.0])
    base = np.array([40.0, 40.0, 40.0, 40.0])          # всё непокрыто
    site_times = np.array([
        [5.0, 40.0, 40.0, 40.0],     # площадка 0 закрывает клетку 0
        [40.0, 5.0, 40.0, 40.0],     # площадка 1 — клетку 1
        [40.0, 40.0, 5.0, 5.0],      # площадка 2 — мелкие клетки 2 и 3
    ])
    allowed = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(allowed, False)
    return FakeCity(
        grid=grid,
        features=pd.DataFrame({"cell_id": grid["cell_id"]}),
        population=population,
        clinics=gpd.GeoDataFrame(geometry=[Point(5000, 5000)], crs=UTM),
        base_times=base, clinic_times=base[None, :],
        site_times=site_times, site_index=np.array([0, 1, 2]),
        allowed_pairs=allowed, report={})


@pytest.fixture
def cfg():
    return {"k": 2, "k_sensitivity": [1], "T_minutes": 15,
            "T_sensitivity": [10, 20], "t_max_minutes": 60, "alpha": 1.0,
            "alpha_sensitivity": [0.5, 2.0], "random_draws": 3, "seed": 0,
            "grid": {"size_m": 500}, "city": {"role": "dev", "name": "test"}}


# --- расчёт вариантов ---

def test_variants_pick_the_populated_cells(city, cfg):
    """B и M должны взять площадки 0 и 1: там живут люди."""
    result = run_variants(city, {}, cfg)
    by_name = {v.name: v for v in result.variants}
    assert by_name["B"].sites == (0, 1)
    assert by_name["M"].sites == (0, 1)


def test_metrics_table_has_a_row_per_variant(city, cfg):
    result = run_variants(city, {}, cfg)
    assert "исходная сеть" in result.table["variant"].tolist()
    assert {"B", "M"} <= set(result.table["variant"])


def test_missing_scores_mark_variant_not_computed(city, cfg):
    """Без снимков D не считается и помечается, а не подменяется другим."""
    result = run_variants(city, {}, cfg)
    by_name = {v.name: v for v in result.variants}
    assert by_name["D"].status == "missing_external"
    assert by_name["C"].status == "missing_external"


def test_scores_change_the_choice(city, cfg):
    """Скор входит через веса — это единственный его канал (п. 9)."""
    scores = np.array([0.0, 0.0, 10.0, 10.0])      # модель тянет к мелким клеткам
    result = run_variants(city, {"text": scores}, cfg)
    by_name = {v.name: v for v in result.variants}
    assert by_name["C"].status in ("ok", "skipped_duplicate")
    assert by_name["C"].sites


def test_identical_solution_marked_as_duplicate(city, cfg):
    """Правило «C ≈ B»: совпало с B — дубликат не считается (п. 9)."""
    scores = np.zeros(4)                            # скор ничего не меняет
    result = run_variants(city, {"text": scores}, cfg)
    by_name = {v.name: v for v in result.variants}
    assert by_name["C"].status == "skipped_duplicate"


def test_random_variant_has_a_distribution(city, cfg):
    result = run_variants(city, {}, cfg)
    random_result = next(v for v in result.variants if v.name == "A")
    assert random_result.distribution is not None
    assert len(random_result.distribution) >= 1


def test_differences_include_comparison_with_random(city, cfg):
    result = run_variants(city, {}, cfg)
    pairs = result.differences["pair"].tolist()
    assert any("B − M" in p for p in pairs)
    assert any("− A (медиана)" in p for p in pairs)


# --- отчёт ---

def test_pilot_quality_is_no_data_without_D(city, cfg):
    """Без варианта D критерий не подменяется другим — это исказило бы вывод."""
    result = run_variants(city, {}, cfg)
    criteria = report_mod.pilot_criteria(result, cfg)
    assert "нет данных" in criteria["quality"]["status"]


def test_report_files_are_written(city, cfg, tmp_path):
    result = run_variants(city, {}, cfg)
    city_report = {"name": "test", "boundary": {"area_km2": 1.0, "source": "x"},
                   "grid": {"cells": 4}, "population": {"transferred_total": 1860,
                                                        "loss_share": 0.0},
                   "clinics": {"state_proxy": 1, "selection": "по названию"},
                   "graph": {"nodes": 10, "dropped_nodes": 0},
                   "accessibility": {"unattached_cells": 0, "unreachable_cells": 0},
                   "sites": {"feasible": 3, "total_cells": 4},
                   "osm": {"buildings_levels_share": 0.5}}
    paths = report_mod.build(result, city_report, cfg, tmp_path)
    assert paths["report_md"].is_file()
    text = paths["report_md"].read_text(encoding="utf-8")
    assert "Главная таблица" in text and "Критерии пилота" in text


# --- защита протокола ---

def test_dev_city_runs_without_protocol():
    guard_test_city({"city": {"role": "dev", "name": "moscow"}}, None)


def test_test_city_requires_frozen_protocol():
    """Узнать об этом после прогона поздно: результат пришлось бы выбросить."""
    with pytest.raises(ProtocolRequired, match="протокол не заморожен"):
        guard_test_city({"city": {"role": "test", "name": "kazan"}}, None)


def test_test_city_runs_with_protocol(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    guard_test_city({"city": {"role": "test", "name": "kazan"}}, protocol)


# --- чувствительность ---

def test_max_coverage_is_resolved_per_threshold(city, cfg):
    """M зависит от порога, в отличие от p-median, и пересчитывается заново."""
    table = sens.over_thresholds(city, {"B": (0, 1)}, cfg)
    assert set(table["T"]) == {10, 15, 20}
    assert "M" in set(table["variant"])


def test_sensitivity_over_k_covers_both_values(city, cfg):
    table = sens.over_k(city, cfg, {})
    assert set(table["k"]) == {1, 2}


def test_alpha_is_skipped_outside_dev_cities(city, cfg):
    """Подбирать α на городах проверки значило бы настраивать систему
    на данных, по которым её оценивают (п. 7.4)."""
    table = sens.over_alpha(city, cfg, {"text": np.zeros(4)}, is_dev_city=False)
    assert table.iloc[0]["status"] == "пропущено"


def test_verdict_reports_unstable_ranking():
    thresholds = pd.DataFrame([
        {"T": 10, "variant": "B", "dCov": 1.0},
        {"T": 10, "variant": "M", "dCov": 2.0},
        {"T": 15, "variant": "B", "dCov": 3.0},
        {"T": 15, "variant": "M", "dCov": 1.0}])
    result = sens.verdict(thresholds, pd.DataFrame())
    assert not result["stable_over_thresholds"]
    assert "МЕНЯЕТСЯ" in result["note"]
