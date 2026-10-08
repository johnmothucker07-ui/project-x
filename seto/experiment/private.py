"""Частный сценарий: модель Хаффа (п. 17).

Запускается ПОСЛЕ основного результата и только при подтверждённом источнике
wealth_score (Л19). Без него сценарий не проводится, и это фиксируется
в протоколе — пропозал прямо это разрешает.

Чем эта метрика НЕ является: wealth_score (медианная цена 1 м² жилья) не равен
числу пациентов или выручке, а данных о реальных потоках для калибровки нет.
E(D) — модельная оценка потенциала спроса, а не прогноз выручки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scoring import weights as weights_mod


class WealthDataMissing(RuntimeError):
    """Нет источника wealth_score — сценарий не проводится (Л19)."""


def require_wealth(cfg: dict, wealth: np.ndarray | None) -> np.ndarray:
    """Проверка, что данные о платежеспособности есть."""
    if wealth is None:
        raise WealthDataMissing(
            "частный сценарий требует wealth_score — медианной цены 1 м² "
            "жилья по клеткам. Источник не подтверждён (Л19), поэтому "
            "сценарий не проводится, и это идёт в протокол.")
    return np.asarray(wealth, dtype=float)


def huff_probabilities(times: np.ndarray, beta: float,
                       attractiveness: np.ndarray | None = None) -> np.ndarray:
    """Prob_ij = A_j·d_ij^(−β) / Σ_k A_k·d_ik^(−β).

    Привлекательность у всех равна 1 — данных о мощностях нет (допущение
    пропозала). Нулевое время заменяется малым: делить на ноль нельзя,
    а клиника ровно в центроиде клетки — артефакт сетки, не реальность."""
    if beta <= 0:
        raise ValueError("beta должна быть больше нуля")
    distances = np.maximum(np.asarray(times, dtype=float), 1e-6)
    supply = (np.ones(distances.shape[0]) if attractiveness is None
              else np.asarray(attractiveness, dtype=float))

    pull = supply[:, None] * distances ** (-beta)
    total = pull.sum(axis=0)
    # клетка, до которой не дотянулась ни одна клиника, не даёт спроса никому
    return np.divide(pull, total, out=np.zeros_like(pull), where=total > 0)


def expected_demand(times: np.ndarray, population: np.ndarray,
                    wealth: np.ndarray, beta: float,
                    attractiveness: np.ndarray | None = None) -> np.ndarray:
    """E(D_j) = Σ_i D_i · Prob_ij, где D_i = P_i · ŵ_i."""
    normalized, _ = weights_mod.normalize(wealth)
    demand = np.asarray(population, dtype=float) * normalized
    return huff_probabilities(times, beta, attractiveness) @ demand


def stability_over_beta(times: np.ndarray, population: np.ndarray,
                        wealth: np.ndarray, betas=(1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
                        ) -> pd.DataFrame:
    """Устойчивость ранжирования по β (п. 17).

    Вывод считается устойчивым, если порядок не меняется при всех β."""
    rows = []
    for beta in betas:
        values = expected_demand(times, population, wealth, beta)
        order = tuple(np.argsort(-values).tolist())
        rows.append({"beta": beta, "best": int(order[0]),
                     "ranking": order, "E_D_best": float(values[order[0]])})
    frame = pd.DataFrame(rows)
    frame.attrs["stable"] = frame["ranking"].nunique() == 1
    return frame


def method_dependence(vlm_scores: np.ndarray, wealth: np.ndarray) -> dict:
    """Риск: не мерят ли VLM-скор и wealth_score одно и то же (п. 17).

    При ρ ≥ 0,7 вывод по частному сценарию ограничивается: у MVP было бы
    встроенное преимущество, потому что обе величины считали бы благополучие
    района."""
    from scipy.stats import spearmanr

    rho = float(spearmanr(np.asarray(vlm_scores, dtype=float),
                          np.asarray(wealth, dtype=float)).statistic)
    limited = abs(rho) >= 0.7
    return {"spearman": rho, "limited_conclusion": limited,
            "note": ("вывод ограничен: скор и wealth_score мерят близкое"
                     if limited else "скор и wealth_score различаются")}
