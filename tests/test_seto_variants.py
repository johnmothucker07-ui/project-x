"""Тесты Ш5: допустимые площадки, валидатор ограничений, варианты A, B, M."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.sites import feasible as FS  # noqa: E402
from seto.variants import solve as SV  # noqa: E402

RULE = {"min_built_area_share": 0.05}


def make_features(n: int, **overrides) -> pd.DataFrame:
    base = {
        "residential_share": np.full(n, 0.2), "public_share": np.zeros(n),
        "water_share": np.zeros(n), "green_share": np.zeros(n),
        "landuse_industrial_share": np.zeros(n), "landuse_railway_share": np.zeros(n),
        "landuse_cemetery_share": np.zeros(n), "landuse_military_share": np.zeros(n),
    }
    base.update(overrides)
    return pd.DataFrame(base)


# --- отбор площадок ---

def test_selects_cells_passing_all_rules():
    sites = FS.select(make_features(3), RULE, np.ones(3, bool),
                      np.array([900.0, 900.0, 900.0]), 500)
    assert len(sites) == 3


def test_low_built_share_is_excluded():
    features = make_features(3, residential_share=np.array([0.2, 0.01, 0.2]))
    sites = FS.select(features, RULE, np.ones(3, bool), np.full(3, 900.0), 500)
    assert sites.index.tolist() == [0, 2]
    assert sites.reasons["мало жилой и общественной застройки"] == 1


def test_public_share_counts_towards_built():
    """«Жилая ИЛИ общественная застройка» — суммируются обе."""
    features = make_features(1, residential_share=np.array([0.03]),
                             public_share=np.array([0.04]))
    assert len(FS.select(features, RULE, np.ones(1, bool),
                         np.full(1, 900.0), 500)) == 1


def test_barriers_exclude_cells():
    features = make_features(4, water_share=np.array([0.9, 0, 0, 0]),
                             green_share=np.array([0, 0.9, 0, 0]),
                             landuse_industrial_share=np.array([0, 0, 0.9, 0]))
    sites = FS.select(features, RULE, np.ones(4, bool), np.full(4, 900.0), 500)
    assert sites.index.tolist() == [3]
    assert sites.reasons["вода"] == 1
    assert sites.reasons["парк или зелень"] == 1
    assert sites.reasons["промзона"] == 1


def test_unattached_cell_cannot_host_a_clinic():
    sites = FS.select(make_features(2), RULE, np.array([True, False]),
                      np.full(2, 900.0), 500)
    assert sites.index.tolist() == [0]
    assert sites.reasons["не привязана к графу"] == 1


def test_too_close_to_existing_clinic_is_excluded():
    sites = FS.select(make_features(3), RULE, np.ones(3, bool),
                      np.array([499.0, 500.0, 900.0]), 500)
    assert sites.index.tolist() == [1, 2]      # ровно 500 м допустимо


def test_reasons_sum_to_excluded_count():
    features = make_features(5, residential_share=np.array([0.01, 0.2, 0.2, 0.2, 0.2]),
                             water_share=np.array([0, 0.9, 0, 0, 0]))
    sites = FS.select(features, RULE, np.array([1, 1, 0, 1, 1], bool),
                      np.array([900, 900, 900, 100, 900.0]), 500)
    assert len(sites) + sum(sites.reasons.values()) == 5


# --- разнесение и валидатор ---

def _allowed(distances: np.ndarray, minimum: float = 500) -> np.ndarray:
    return FS.pair_allowed(distances, minimum)


def test_pair_allowed_excludes_self_and_close_pairs():
    distances = np.array([[0.0, 400.0, 900.0],
                          [400.0, 0.0, 900.0],
                          [900.0, 900.0, 0.0]])
    allowed = _allowed(distances)
    assert not allowed[0, 0]
    assert not allowed[0, 1]
    assert allowed[0, 2]


def test_validator_rejects_wrong_size():
    allowed = _allowed(np.array([[0.0, 900.0], [900.0, 0.0]]))
    with pytest.raises(FS.ConstraintViolation, match="вместо"):
        FS.validate([0], allowed, expected_k=2)


def test_validator_rejects_duplicates():
    allowed = _allowed(np.array([[0.0, 900.0], [900.0, 0.0]]))
    with pytest.raises(FS.ConstraintViolation, match="повторяется"):
        FS.validate([0, 0], allowed, expected_k=2)


def test_validator_rejects_too_close_pair():
    allowed = _allowed(np.array([[0.0, 400.0], [400.0, 0.0]]))
    with pytest.raises(FS.ConstraintViolation, match="меньше минимума"):
        FS.validate([0, 1], allowed, expected_k=2)


def test_validator_rejects_site_outside_list():
    """Так отсеивается импортированное решение E с чужими площадками."""
    allowed = _allowed(np.array([[0.0, 900.0], [900.0, 0.0]]))
    with pytest.raises(FS.ConstraintViolation, match="вне списка"):
        FS.validate([0, 5], allowed, expected_k=2)


def test_validator_accepts_valid_set():
    allowed = _allowed(np.array([[0.0, 900.0], [900.0, 0.0]]))
    FS.validate([0, 1], allowed, expected_k=2)      # не бросает


def test_max_feasible_k_drops_when_no_pair_fits():
    """Если пары не существует, все варианты решают с k = 1 (п. 8)."""
    allowed = _allowed(np.array([[0.0, 100.0], [100.0, 0.0]]))
    assert FS.max_feasible_k(allowed, k=2) == 1
    assert FS.max_feasible_k(_allowed(np.array([[0.0, 900.0], [900.0, 0.0]])), 2) == 2


# --- варианты ---

def test_pmedian_picks_the_pair_that_cuts_time_most():
    #                      клетки:  0     1     2
    base = np.array([30.0, 30.0, 30.0])
    site_times = np.array([[5.0, 30.0, 30.0],      # площадка 0 помогает клетке 0
                           [30.0, 5.0, 30.0],      # площадка 1 — клетке 1
                           [30.0, 30.0, 29.0]])    # площадка 2 почти бесполезна
    weights = np.ones(3)
    allowed = np.ones((3, 3), bool); np.fill_diagonal(allowed, False)
    solution = SV.solve_pmedian(site_times, base, weights, allowed, k=2)
    assert solution.sites == (0, 1)


def test_pmedian_follows_the_weights():
    """Та же геометрия, но вес смещён — выбор меняется. Это и есть канал VLM."""
    base = np.array([30.0, 30.0])
    site_times = np.array([[5.0, 30.0], [30.0, 5.0], [20.0, 20.0]])
    allowed = np.ones((3, 3), bool); np.fill_diagonal(allowed, False)
    even = SV.solve_pmedian(site_times, base, np.array([1.0, 1.0]), allowed, 1)
    skewed = SV.solve_pmedian(site_times, base, np.array([1.0, 10.0]), allowed, 1)
    assert even.sites == (0,)
    assert skewed.sites == (1,)


def test_max_coverage_counts_union_not_sum():
    """Две площадки, покрывающие одних и тех же жителей, вместе не лучше одной."""
    base = np.array([60.0, 60.0, 60.0])
    population = np.array([100.0, 100.0, 1.0])
    site_times = np.array([[5.0, 5.0, 60.0],     # 0 и 1 покрывают одно и то же
                           [5.0, 5.0, 60.0],
                           [60.0, 60.0, 5.0]])   # 2 добавляет новую клетку
    allowed = np.ones((3, 3), bool); np.fill_diagonal(allowed, False)
    solution = SV.solve_max_coverage(site_times, base, population, allowed,
                                     k=2, threshold=15)
    assert solution.objective == pytest.approx(201.0)
    assert 2 in solution.sites


def test_max_coverage_ignores_cells_already_covered():
    base = np.array([5.0, 60.0])                  # клетка 0 уже покрыта
    population = np.array([1000.0, 1.0])
    site_times = np.array([[5.0, 60.0], [60.0, 5.0]])
    allowed = np.ones((2, 2), bool); np.fill_diagonal(allowed, False)
    solution = SV.solve_max_coverage(site_times, base, population, allowed,
                                     k=1, threshold=15)
    assert solution.sites == (1,)


def test_ties_are_resolved_deterministically():
    """Ничьи реальны: на сухом прогоне B и M дали разные точки при равной метрике."""
    base = np.array([30.0, 30.0])
    site_times = np.array([[5.0, 30.0], [5.0, 30.0], [5.0, 30.0]])
    allowed = np.ones((3, 3), bool); np.fill_diagonal(allowed, False)
    first = SV.solve_pmedian(site_times, base, np.ones(2), allowed, k=1)
    second = SV.solve_pmedian(site_times, base, np.ones(2), allowed, k=1)
    assert first.sites == second.sites == (0,)
    assert first.tie_count == 2


def test_random_sets_are_all_valid_and_unique():
    distances = np.full((6, 6), 900.0)
    distances[0, 1] = distances[1, 0] = 100.0     # эта пара запрещена
    allowed = _allowed(distances)
    sets = SV.solve_random(allowed, k=2, draws=10, seed=0)
    assert len(sets) == len(set(sets))
    assert (0, 1) not in sets
    for selection in sets:
        FS.validate(selection, allowed, expected_k=2)


def test_random_is_reproducible_with_seed():
    allowed = _allowed(np.full((8, 8), 900.0))
    assert SV.random_sets(allowed, 2, 5, seed=1) == SV.random_sets(allowed, 2, 5, seed=1)


def test_random_enumerates_when_rejection_sampling_is_not_enough():
    """Допустимых пар всего три — вернуть должны ровно их, а не меньше."""
    distances = np.full((3, 3), 900.0)
    sets = SV.random_sets(_allowed(distances), k=2, draws=100, seed=0)
    assert sorted(sets) == [(0, 1), (0, 2), (1, 2)]


def test_estimate_pairs_reports_volume_before_running():
    distances = np.full((4, 4), 900.0)
    distances[0, 1] = distances[1, 0] = 100.0
    estimate = SV.estimate_pairs(4, _allowed(distances))
    assert estimate["pairs_total"] == 6 and estimate["pairs_allowed"] == 5


# --- ILP при k > 2 (п. 10.3) ---

def _toy():
    base = np.array([60.0, 60.0, 60.0, 60.0])
    site_times = np.array([[5.0, 60, 60, 60], [60, 5.0, 60, 60],
                           [60, 60, 5.0, 60], [60, 60, 60, 5.0]])
    population = np.array([100.0, 90.0, 80.0, 5.0])
    allowed = np.ones((4, 4), bool)
    np.fill_diagonal(allowed, False)
    return site_times, base, population, allowed


def test_ilp_matches_exhaustive_at_k2():
    """Главная проверка решателя: на малой задаче ILP и перебор совпадают."""
    site_times, base, population, allowed = _toy()
    ilp = SV.solve_ilp(site_times, base, population, allowed, k=2, threshold=15)
    exhaustive = SV.solve_max_coverage(site_times, base, population, allowed,
                                       2, 15)
    assert set(ilp.sites) == set(exhaustive.sites)


def test_ilp_handles_k_above_two():
    """Полный перебор при k > 2 неподъёмен: на 1497 площадках полмиллиарда троек."""
    site_times, base, population, allowed = _toy()
    solution = SV.solve_ilp(site_times, base, population, allowed, k=3,
                            threshold=15)
    assert len(solution.sites) == 3
    assert 3 not in solution.sites          # самая малонаселённая клетка лишняя


def test_ilp_respects_separation():
    site_times, base, population, allowed = _toy()
    allowed[0, 1] = allowed[1, 0] = False   # эту пару нельзя вместе
    solution = SV.solve_ilp(site_times, base, population, allowed, k=2,
                            threshold=15)
    assert not {0, 1} <= set(solution.sites)


def test_ilp_pmedian_prefers_high_weight_cells():
    site_times, base, population, allowed = _toy()
    solution = SV.solve_ilp(site_times, base, population, allowed, k=2)
    assert set(solution.sites) == {0, 1}    # там наибольший вес спроса
