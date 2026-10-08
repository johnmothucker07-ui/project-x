"""Скоринг клеток моделью (п. 7).

Одна функция на оба варианта: различается ТОЛЬКО подготовка входа. Модель,
промпт, шкала, температура и seed общие — иначе разница D − C мерила бы
не изображение, а разные условия вызова.

Кэш обязателен: ключ включает вход, поэтому прогон со снимком не может
молча подхватить текстовый ответ (в MVP ключ входа не учитывал, и это была
мина). Он же даёт возобновление после обрыва — готовые клетки пропускаются.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .prompt import build, prompt_hash


class ScoreSchemaError(ValueError):
    """Ответ модели не соответствует схеме."""


@dataclass
class ScoringReport:
    """Итог прогона: сколько вызвано, сколько из кэша, чего стоило."""
    cells: int = 0
    from_cache: int = 0
    api_calls: int = 0
    retries: int = 0
    imputed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def imputed_share(self) -> float:
        return self.imputed / self.cells if self.cells else 0.0

    def as_dict(self) -> dict:
        return {"cells": self.cells, "from_cache": self.from_cache,
                "api_calls": self.api_calls, "retries": self.retries,
                "imputed": self.imputed, "imputed_share": self.imputed_share,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "seconds": round(self.seconds, 1),
                "failures": self.failures[:20]}


def parse_response(raw: str | None) -> dict:
    """Разобрать ответ и проверить по схеме {score: 0-10, rationale: str}.

    Пустой content бывает у reasoning-моделей, когда рассуждение съело весь
    лимит токенов, — это ошибка разбора, а не «ноль»."""
    if not raw or not raw.strip():
        raise ScoreSchemaError("пустой ответ модели")

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ScoreSchemaError(f"в ответе нет JSON-объекта: {raw[:120]!r}")

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ScoreSchemaError(f"невалидный JSON: {error}") from error

    if "score" not in data:
        raise ScoreSchemaError(f"в ответе нет поля score: {data}")
    try:
        score = int(data["score"])
    except (TypeError, ValueError) as error:
        raise ScoreSchemaError(f"score не целое: {data['score']!r}") from error
    if not 0 <= score <= 10:
        raise ScoreSchemaError(f"score вне шкалы 0-10: {score}")

    return {"score": score, "rationale": str(data.get("rationale", ""))}


def _input_hash(mode: str, payload: str | bytes) -> str:
    """Хеш входа: текст описания или байты изображения."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(mode.encode() + b"|" + data).hexdigest()


def cache_key(cell_id: str, model: str, prompt: str,
              mode: str, payload: str | bytes) -> str:
    """Ключ кэша по п. 7.1: клетка, модель, хеш промпта, хеш входа."""
    parts = f"{cell_id}|{model}|{prompt_hash(prompt)}|{_input_hash(mode, payload)}"
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def _make_client(cfg: dict):
    from openai import OpenAI

    base_url = os.environ.get(cfg["vlm"]["base_url_env"])
    api_key = os.environ.get(cfg["vlm"]["api_key_env"])
    if not base_url or not api_key:
        raise RuntimeError(
            f"не заданы {cfg['vlm']['base_url_env']} и {cfg['vlm']['api_key_env']} "
            "в .env")
    return OpenAI(base_url=base_url, api_key=api_key,
                  timeout=cfg["vlm"].get("timeout_s", 180))


def _message(prompt: str, mode: str, payload) -> list:
    if mode == "text":
        return [{"role": "user", "content": prompt}]
    url = "data:image/png;base64," + base64.b64encode(payload).decode()
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}}]}]


def score_cells(cell_ids, inputs, mode: str, cfg: dict,
                cache_dir: Path, raw_dir: Path | None = None):
    """Скоры по всем клеткам. inputs — тексты (mode='text') или байты (mode='image').

    Возвращает (DataFrame[cell_id, score, rationale, imputed], ScoringReport)."""
    if mode not in ("text", "image"):
        raise ValueError(f"режим должен быть 'text' или 'image', а не {mode!r}")
    if len(cell_ids) != len(inputs):
        raise ValueError("число клеток и входов не совпадает")

    vlm = cfg["vlm"]
    size_m = cfg["grid"]["size_m"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)

    report = ScoringReport(cells=len(cell_ids))
    client = None
    started = time.time()

    def one(item):
        nonlocal client
        cell_id, payload = item
        prompt = build(mode, size_m, payload if mode == "text" else None)
        path = cache_dir / f"{cache_key(cell_id, vlm['model'], prompt, mode, payload)}.json"
        if path.is_file():
            return cell_id, json.loads(path.read_text(encoding="utf-8")), True, None

        if client is None:
            client = _make_client(cfg)
        last_error = None
        # один повтор, как требует п. 7.1; дальше клетка идёт в imputed
        for attempt in range(1 + vlm.get("retries", 1)):
            try:
                response = client.chat.completions.create(
                    model=vlm["model"], messages=_message(prompt, mode, payload),
                    temperature=vlm.get("temperature", 0),
                    max_tokens=vlm.get("max_tokens", 300),
                    seed=vlm.get("seed"))
                parsed = parse_response(response.choices[0].message.content)
                usage = response.usage
                parsed["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                parsed["completion_tokens"] = getattr(usage, "completion_tokens", 0)
                parsed["attempt"] = attempt
                path.write_text(json.dumps(parsed, ensure_ascii=False),
                                encoding="utf-8")
                if raw_dir is not None:
                    (raw_dir / f"{cell_id}.json").write_text(
                        json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
                return cell_id, parsed, False, None
            except Exception as error:          # сетевые и схемные — одинаково
                last_error = f"{cell_id}: {type(error).__name__}: {error}"
        return cell_id, None, False, last_error

    rows = []
    with ThreadPoolExecutor(max_workers=vlm.get("max_parallel", 4)) as pool:
        for cell_id, parsed, cached, error in pool.map(one, zip(cell_ids, inputs)):
            if cached:
                report.from_cache += 1
            elif parsed is not None:
                report.api_calls += 1
                report.retries += parsed.get("attempt", 0)
                report.prompt_tokens += parsed.get("prompt_tokens", 0)
                report.completion_tokens += parsed.get("completion_tokens", 0)
            if parsed is None:
                report.failures.append(error)
                rows.append({"cell_id": cell_id, "score": np.nan,
                             "rationale": "", "imputed": True})
            else:
                rows.append({"cell_id": cell_id, "score": parsed["score"],
                             "rationale": parsed["rationale"], "imputed": False})

    scores = pd.DataFrame(rows)
    report.imputed = int(scores["imputed"].sum())
    report.seconds = time.time() - started

    if report.imputed:
        # п. 7.1: сбойной клетке ставится медиана скоров города
        median = scores["score"].median()
        if pd.isna(median):
            raise RuntimeError("ни одна клетка не получила скор — считать нечего")
        scores.loc[scores["imputed"], "score"] = median

    scores["score"] = scores["score"].astype(float)
    return scores, report


def check_imputed(report: ScoringReport, cfg: dict) -> None:
    """Если доля imputed выше порога, вариант в городе неуспешен (п. 7.1, 13)."""
    limit = cfg.get("imputation_max_share", 0.05)
    if report.imputed_share > limit:
        raise RuntimeError(
            f"доля клеток без ответа модели {100 * report.imputed_share:.1f} % "
            f"выше порога {100 * limit:.0f} % — вариант в этом городе неуспешен")
