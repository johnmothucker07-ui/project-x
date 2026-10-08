"""Промпт и текстовое описание клетки (п. 7.1, 7.3).

Один шаблон на оба варианта: C и D отличаются РОВНО одной фразой — тем, что
подаётся на вход. Иначе разница D − C перестала бы означать вклад изображения
и начала бы мерить разницу формулировок.

Промпт новый, а не унаследованный от MVP (решение Д12): MVP спрашивал про
премиальность района и тип застройки, а пропозал требует спрашивать ОДНО —
насколько клетка нуждается в первичной медицинской помощи.
"""
from __future__ import annotations

import hashlib

import pandas as pd

# Общая часть. {input_clause} — единственное, чем отличаются C и D.
TEMPLATE = """Ты оцениваешь городскую клетку {size} м по {input_clause}

Вопрос один: насколько жителям этой клетки нужна рядом взрослая поликлиника \
первичной медико-санитарной помощи?

Не вычисляй ничего и не оценивай численность людей — суди качественно, \
как если бы просто осмотрелся вокруг.

Верни СТРОГО один JSON-объект без markdown:
{{"score": целое число 0-10, "rationale": "одно короткое предложение"}}"""

IMAGE_CLAUSE = "спутниковому снимку."
TEXT_CLAUSE = "описанию ниже.\n\nОписание клетки:\n{text}"


def build(mode: str, size_m: int, text: str | None = None) -> str:
    """Промпт для варианта D (mode='image') или C (mode='text')."""
    size = f"{size_m}x{size_m}"
    if mode == "image":
        return TEMPLATE.format(size=size, input_clause=IMAGE_CLAUSE)
    if mode == "text":
        if not text:
            raise ValueError("для режима 'text' нужно описание клетки")
        return TEMPLATE.format(size=size,
                               input_clause=TEXT_CLAUSE.format(text=text))
    raise ValueError(f"режим должен быть 'image' или 'text', а не {mode!r}")


def prompt_hash(prompt: str) -> str:
    """Хеш промпта для манифеста и протокола (п. 7.1)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def diff() -> dict:
    """Чем именно отличаются промпты C и D — идёт в протокол (п. 7.1)."""
    return {"common_template": TEMPLATE,
            "image_clause": IMAGE_CLAUSE,
            "text_clause": TEXT_CLAUSE,
            "differs_only_in": "input_clause"}


# --- текст варианта C ---

TEXT_TEMPLATE_VERSION = "1.0"
_MISSING = "нет данных"


def _number(value, digits: int = 0) -> str:
    """Пропуск — явным словом, а не нулём (п. 7.3): ноль школ и неизвестное
    число школ это разные вещи, и модель не должна их путать."""
    if value is None or pd.isna(value):
        return _MISSING
    return f"{float(value):.{digits}f}" if digits else f"{int(value)}"


def _percent(value) -> str:
    if value is None or pd.isna(value):
        return _MISSING
    return f"{100 * float(value):.0f} %"


def describe(row) -> str:
    """Детерминированное описание клетки из признаков OSM (п. 7.3).

    Населения в тексте НЕТ: оно уже входит в вес w_i = P_i·(1 + α·ŝ_i),
    а модель не должна считать числа."""
    levels = (_MISSING if pd.isna(row.get("mean_levels"))
              else f"{float(row['mean_levels']):.1f}")
    return (
        f"Застройка занимает {_percent(row.get('built_share'))} площади клетки: "
        f"жилая {_percent(row.get('residential_share'))}, "
        f"общественная и деловая {_percent(row.get('public_share'))}. "
        f"Зданий: {_number(row.get('buildings_count'))}, "
        f"средняя этажность {levels} "
        f"(этажность известна у {_percent(row.get('levels_known_share'))} зданий). "
        f"Школы: {_number(row.get('schools'))}, "
        f"детские сады: {_number(row.get('kindergartens'))}, "
        f"аптеки: {_number(row.get('pharmacies'))}, "
        f"магазины: {_number(row.get('shops'))}, "
        f"остановки транспорта: {_number(row.get('transport_stops'))}. "
        f"Парки и зелень: {_percent(row.get('green_share'))}, "
        f"вода: {_percent(row.get('water_share'))}, "
        f"промзона: {_percent(row.get('landuse_industrial_share'))}. "
        f"Плотность улиц: "
        f"{_number(row.get('street_density_m_per_km2'))} м на кв. км."
    )


def describe_all(features: pd.DataFrame) -> pd.Series:
    """Описания по всем клеткам, в порядке таблицы признаков."""
    return features.apply(describe, axis=1)
