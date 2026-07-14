"""
ЭТАП 4 · Проверка осмысленности wealth_score.

Корреляция Спирмена между wealth_score (от модели) и реальной ценой м²
(ЦИАН/Дом.РФ) на выборке клеток. Единственная объективная проверка
Estimate-этапа — сравнение с ВНЕШНИМ фактом, а не с мнением.
"""
from __future__ import annotations


def spearman_vs_price(cells, price_reference_path: str) -> dict:
    """Корреляция Спирмена wealth_score с ценой м². Вернуть {rho, p_value, n}."""
    raise NotImplementedError
