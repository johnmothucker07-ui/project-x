"""
ЭТАП 3 · Два рейтинга. СЧИТАЕТ КОД, не модель.

EquityScore  = f(население, нет клиник рядом, далеко до ближайшей)  → государственная
ProfitScore  = f(население, wealth_score, мало конкурентов)         → частная

Обе шкалы НОРМИРУЮТСЯ (0-1) перед сравнением — иначе argmax нечестный
(население и население×wealth разного порядка).
"""
from __future__ import annotations

import pandas as pd


def normalize(series):
    """Нормировать шкалу в 0-1 (min-max) перед сравнением скоров."""
    s = pd.Series(series, dtype="float64")
    low, high = s.min(), s.max()
    if high == low:
        # все значения одинаковы — вклад фактора нулевой (иначе деление на ноль)
        return pd.Series(0.0, index=s.index)
    return (s - low) / (high - low)


def _scarcity(cells):
    """Фактор «мало клиник рядом»: выше там, где конкурентов меньше.
    Пока это прокси компонента «далеко до ближайшей» (точную дистанцию даст routing, 3.2)."""
    return normalize(1.0 / (1.0 + cells["existing_clinics"]))


def equity_score(cells):
    """Скор непокрытого спроса (гос-логика). Только числа из данных.
    Высок там, где много людей и мало/нет клиник."""
    population = normalize(cells["population"])
    return normalize(population * _scarcity(cells))


def profit_score(cells):
    """Скор платёжеспособного спроса (частная логика). Использует wealth_score от модели.
    Высок там, где людно, богато и мало конкурентов."""
    population = normalize(cells["population"])
    wealth = normalize(cells["wealth_score"])
    return normalize(population * wealth * _scarcity(cells))
