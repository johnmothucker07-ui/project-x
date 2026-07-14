"""
ЭТАП 3 · Два рейтинга. СЧИТАЕТ КОД, не модель.

EquityScore  = f(население, нет клиник рядом, далеко до ближайшей)  → государственная
ProfitScore  = f(население, wealth_score, мало конкурентов)         → частная

Обе шкалы НОРМИРУЮТСЯ (0-1) перед сравнением — иначе argmax нечестный
(население и население×wealth разного порядка).
"""
from __future__ import annotations


def equity_score(cells):
    """Скор непокрытого спроса (гос-логика). Только числа из данных."""
    raise NotImplementedError


def profit_score(cells):
    """Скор платёжеспособного спроса (частная логика). Использует wealth_score от модели."""
    raise NotImplementedError


def normalize(series):
    """Нормировать шкалу в 0-1 (min-max) перед сравнением скоров."""
    raise NotImplementedError
