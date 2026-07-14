"""
ЭТАП 2 · Прогон клеток через модель.

Для каждой клетки: выжимка → промпт → вызов модели → JSON → в таблицу.
Инкрементально сохраняет результат (кэш + запись на диск), чтобы обрыв
на N-й клетке не терял предыдущие. Пауза между вызовами (рейт-лимит).
"""
from __future__ import annotations


def estimate_cells(cells, context_by_cell: dict, cfg: dict):
    """Прогнать все клетки через VLM. Вернуть таблицу [cell_id, wealth_score, area_character, confidence].
    С кэшированием, ретраями и инкрементальным сохранением."""
    raise NotImplementedError
