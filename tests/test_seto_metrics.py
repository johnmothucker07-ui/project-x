"""Тесты Ш4: покрытие и 2SFCA на ручных примерах с известным ответом."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.metrics import coverage as C, sfca as S  # noqa: E402


# --- покрытие ---

def test_coverage_is_in_percent():
    """×100 обязателен: метрика в процентах, разницы в п.п. (п. 11.1)."""
    times = np.array([5.0, 20.0])
    population = np.array([300.0, 100.0])
    assert C.coverage(times, population, 15) == pytest.approx(75.0)


def test_coverage_counts_population_not_cells():
    times = np.array([5.0, 5.0, 20.0])
    population = np.array([10.0, 10.0, 80.0])
    assert C.coverage(times, population, 15) == pytest.approx(20.0)


def test_threshold_is_inclusive():
    assert C.coverage(np.array([15.0]), np.array([1.0]), 15) == 100.0
    assert C.coverage(np.array([15.001]), np.array([1.0]), 15) == 0.0


def test_coverage_monotone_in_threshold():
    times = np.array([5.0, 12.0, 18.0, 40.0])
    population = np.array([1.0, 2.0, 3.0, 4.0])
    values = [C.coverage(times, population, t) for t in (10, 15, 20, 60)]
    assert values == sorted(values)


def test_unreachable_stays_in_denominator():
    """Жители недостижимых клеток остаются в знаменателе (п. 11.1).

    Иначе покрытие вырастет само собой, без единой новой поликлиники."""
    times = np.array([5.0, 60.0])
    population = np.array([50.0, 50.0])
    assert C.coverage(times, population, 15) == pytest.approx(50.0)


def test_delta_people_counts_newly_covered():
    before = np.array([20.0, 5.0, 30.0])
    after = np.array([10.0, 5.0, 30.0])
    population = np.array([1000.0, 500.0, 200.0])
    assert C.delta_people(after, before, population, 15) == pytest.approx(1000.0)


def test_delta_is_zero_for_empty_set():
    times = np.array([5.0, 20.0])
    population = np.array([1.0, 1.0])
    assert C.delta_coverage(times, times, population, 15) == 0.0
    assert C.delta_people(times, times, population, 15) == 0.0


def test_times_with_takes_minimum_and_caps():
    existing = np.array([30.0, 80.0])
    added = np.array([[12.0, 50.0], [40.0, 9.0]])
    result = C.times_with(existing, added, t_max=60)
    assert result.tolist() == [12.0, 9.0]
    assert C.times_with(existing, None, t_max=60).tolist() == [30.0, 60.0]


def test_overlap_is_less_than_sum_of_parts():
    """ΔCov — функция НАБОРА, а не сумма по точкам.

    Две близкие площадки накрывают во многом одних и тех же жителей."""
    population = np.array([100.0, 100.0, 100.0])
    existing = np.array([60.0, 60.0, 60.0])
    a = np.array([[5.0, 5.0, 60.0]])
    b = np.array([[5.0, 5.0, 60.0]])

    only_a = C.delta_coverage(C.times_with(existing, a, 60), existing, population, 15)
    only_b = C.delta_coverage(C.times_with(existing, b, 60), existing, population, 15)
    both = C.delta_coverage(C.times_with(existing, np.vstack([a, b]), 60),
                            existing, population, 15)
    assert both < only_a + only_b
    assert both == pytest.approx(only_a)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        C.coverage(np.array([1.0, 2.0]), np.array([1.0]), 15)


# --- худшие 10 % ---

def test_worst_decile_takes_the_tail():
    times = np.array([1.0, 1.0, 50.0])
    population = np.array([45.0, 45.0, 10.0])
    assert C.worst_decile_time(times, population) == pytest.approx(50.0)


def test_worst_decile_splits_boundary_cell():
    """Клетка на границе отсечки учитывается долей, а не целиком."""
    # верхние 10 % = 10 человек, все из тяжёлой клетки
    assert C.worst_decile_time(np.array([100.0, 0.0]),
                               np.array([20.0, 80.0])) == pytest.approx(100.0)
    # а если в тяжёлой клетке всего 5 человек — добираем из лёгкой
    assert C.worst_decile_time(np.array([100.0, 0.0]),
                               np.array([5.0, 95.0])) == pytest.approx(50.0)


# --- среднее время ---

def test_mean_time_is_population_weighted():
    times = np.array([10.0, 20.0])
    population = np.array([3.0, 1.0])
    assert C.mean_time(times, population) == pytest.approx(12.5)


# --- 2SFCA ---

def test_decay_is_one_at_zero_and_zero_at_threshold():
    assert S.decay(np.array([0.0]), 15)[0] == pytest.approx(1.0)
    assert S.decay(np.array([15.0]), 15)[0] == pytest.approx(0.0, abs=1e-12)
    assert S.decay(np.array([15.001]), 15)[0] == 0.0


def test_decay_is_monotone():
    values = S.decay(np.array([0.0, 5.0, 10.0, 14.0]), 15)
    assert np.all(np.diff(values) < 0)


def test_sfca_invariant_mean_equals_supply_over_population():
    """Инвариант п. 11.3: взвешенное среднее A_i = Σ S_j / Σ P_i.

    Именно поэтому среднее не выводится как метрика — оно не зависит
    от размещения. Проверка с точностью 1e-9."""
    rng = np.random.default_rng(0)
    times = rng.uniform(0, 14, size=(4, 25))
    population = rng.uniform(10, 500, size=25)
    access, ratio = S.accessibility(times, population, threshold=15)

    mean = (access * population).sum() / population.sum()
    expected = (ratio > 0).sum() / population.sum()
    assert mean == pytest.approx(expected, abs=1e-9)


def test_sfca_zero_outside_zone():
    times = np.array([[20.0, 5.0]])
    population = np.array([100.0, 100.0])
    access, _ = S.accessibility(times, population, threshold=15)
    assert access[0] == 0.0 and access[1] > 0


def test_sfca_ratio_zero_when_no_population_in_zone():
    times = np.array([[40.0, 40.0]])
    population = np.array([100.0, 100.0])
    _, ratio = S.accessibility(times, population, threshold=15)
    assert ratio[0] == 0.0


def test_sfca_competition_lowers_access():
    """Больше людей на то же учреждение — меньше достаётся каждому."""
    times = np.array([[5.0, 5.0]])
    few = S.accessibility(times, np.array([10.0, 10.0]), 15)[0]
    many = S.accessibility(times, np.array([1000.0, 1000.0]), 15)[0]
    assert few[0] > many[0]


def test_floor_is_positive_even_with_zero_median():
    """A_0 задаётся по исходной сети и не равен нулю (п. 11.3)."""
    population = np.array([100.0, 100.0])
    assert S.floor_value(3, population) == pytest.approx(0.0075)
    assert S.floor_value(1, population) > 0


def test_summary_has_no_mean():
    """Среднее A_i выводить запрещено (п. 22)."""
    times = np.array([[5.0, 10.0, 20.0]])
    population = np.array([100.0, 100.0, 100.0])
    result = S.summary(times, population, 15, S.floor_value(1, population))
    assert set(result) == {"share_below_floor_pct", "p10", "p25"}


def test_weighted_percentile_respects_population():
    values = np.array([1.0, 10.0])
    population = np.array([99.0, 1.0])
    assert S.weighted_percentile(values, population, 0.5) == pytest.approx(1.0)
