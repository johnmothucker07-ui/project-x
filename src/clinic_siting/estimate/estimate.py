"""
ЭТАП 2 · Прогон клеток через модель.

Для каждой клетки: выжимка → промпт → вызов модели → JSON → в таблицу.
Инкрементально сохраняет результат (кэш + запись на диск), чтобы обрыв
на N-й клетке не терял предыдущие. Пауза между вызовами (рейт-лимит).
"""
from __future__ import annotations

import hashlib
import time

import pandas as pd

from .llm_client import make_client, call_vlm
from .prompt import build_prompt, parse_response
from ..utils.io import cache_get, cache_set


def _cache_key(model: str, prompt: str) -> str:
    """Ключ кэша = хеш модели и промпта. Меняешь промпт/модель → кэш инвалидируется."""
    raw = f"{model}\n{prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _call_with_retries(client, model: str, prompt: str, temperature: float, max_retries: int) -> dict:
    """Вызвать модель и распарсить; при кривом JSON — повторить до max_retries раз."""
    last_error = None
    for _ in range(max_retries + 1):
        raw = call_vlm(client, model, prompt, temperature=temperature)
        try:
            return parse_response(raw)
        except (ValueError, TypeError) as error:
            last_error = error
    raise RuntimeError(f"Модель не вернула валидный JSON после {max_retries + 1} попыток: {last_error}")


def estimate_cells(cells, context_by_cell: dict, cfg: dict):
    """Прогнать все клетки через VLM. Вернуть таблицу [cell_id, wealth_score, area_character, confidence].
    С кэшированием (повтор не тратит токены), ретраями и паузой между вызовами."""
    est = cfg["estimate"]
    client = make_client(cfg)
    model = est["model"]
    cache_dir = est["cache_dir"]
    max_retries = est.get("max_retries", 2)
    pause_s = est.get("request_pause_s", 0.5)

    rows = []
    for cell_id in cells["cell_id"]:
        prompt = build_prompt(context_by_cell[cell_id])
        key = _cache_key(model, prompt)

        cached = cache_get(key, cache_dir)
        if cached is not None:
            parsed = cached
        else:
            parsed = _call_with_retries(client, model, prompt, est["temperature"], max_retries)
            cache_set(key, parsed, cache_dir)
            time.sleep(pause_s)  # рейт-лимит: пауза только после реального вызова к API

        rows.append({"cell_id": cell_id, **parsed})

    return pd.DataFrame(rows)
