"""Тесты частного сценария (п. 17): модель Хаффа и риск зависимости."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seto.experiment import private  # noqa: E402


def test_scenario_refuses_without_wealth_data():
    """Нет источника wealth_score — сценарий не проводится (Л19)."""
    with pytest.raises(private.WealthDataMissing, match="Л19"):
        private.require_wealth({}, None)


def test_probabilities_sum_to_one_per_cell():
    times = np.array([[5.0, 10.0], [10.0, 5.0]])
    probabilities = private.huff_probabilities(times, beta=2.0)
    assert np.allclose(probabilities.sum(axis=0), 1.0)


def test_closer_clinic_gets_more_demand():
    times = np.array([[5.0], [20.0]])
    probabilities = private.huff_probabilities(times, beta=2.0)
    assert probabilities[0, 0] > probabilities[1, 0]


def test_unreachable_cell_gives_demand_to_nobody():
    """Клетка без единой достижимой клиники не создаёт спроса."""
    times = np.array([[np.inf, 5.0]])
    probabilities = private.huff_probabilities(times, beta=2.0)
    assert probabilities[0, 0] == 0.0
    assert probabilities[0, 1] == pytest.approx(1.0)


def test_zero_time_does_not_divide_by_zero():
    """Клиника ровно в центроиде клетки — артефакт сетки, не реальность."""
    probabilities = private.huff_probabilities(np.array([[0.0, 5.0]]), beta=2.0)
    assert np.isfinite(probabilities).all()


def test_beta_must_be_positive():
    with pytest.raises(ValueError):
        private.huff_probabilities(np.array([[1.0]]), beta=0.0)


def test_demand_scales_with_population_and_wealth():
    times = np.array([[5.0, 5.0]])
    population = np.array([100.0, 100.0])
    poor = private.expected_demand(times, population, np.array([0.0, 0.0]), 2.0)
    mixed = private.expected_demand(times, population, np.array([0.0, 10.0]), 2.0)
    assert mixed[0] > poor[0]        # платежеспособная клетка добавляет спрос


def test_stability_detects_stable_ranking():
    times = np.array([[5.0, 5.0], [30.0, 30.0]])
    frame = private.stability_over_beta(times, np.array([100.0, 100.0]),
                                        np.array([5.0, 5.0]))
    assert frame.attrs["stable"]
    assert set(frame["beta"]) == {1.5, 1.6, 1.7, 1.8, 1.9, 2.0}


def test_method_dependence_flags_high_correlation():
    """При ρ ≥ 0,7 вывод ограничивается: обе величины мерят благополучие."""
    scores = np.arange(20, dtype=float)
    assert private.method_dependence(scores, scores)["limited_conclusion"]
    rng = np.random.default_rng(0)
    independent = private.method_dependence(scores, rng.permutation(scores))
    assert not independent["limited_conclusion"]
