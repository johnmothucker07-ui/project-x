"""
ЭТАП 4 · Проверка осмысленности wealth_score.

Корреляция Спирмена между wealth_score (от модели) и реальной ценой м²
(ЦИАН/Дом.РФ) на выборке клеток. Единственная объективная проверка
Estimate-этапа — сравнение с ВНЕШНИМ фактом, а не с мнением.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def spearman_vs_price(cells, price_reference_path: str) -> dict:
    """Корреляция Спирмена wealth_score с ценой м². Вернуть {rho, p_value, n}.
    price_reference_path — CSV с колонками cell_id, price_per_m2 (выборка клеток с ценами)."""
    prices = pd.read_csv(price_reference_path)
    merged = cells[["cell_id", "wealth_score"]].merge(prices, on="cell_id", how="inner")
    merged = merged.dropna(subset=["wealth_score", "price_per_m2"])

    rho, p_value = spearmanr(merged["wealth_score"], merged["price_per_m2"])
    return {"rho": float(rho), "p_value": float(p_value), "n": len(merged)}
