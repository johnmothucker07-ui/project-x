"""
ЭТАП 2 · Промпт и JSON-схема.

Модель возвращает СУЖДЕНИЯ о районе, НЕ считает числа.
Строгий JSON: wealth_score (0-10), area_character (категория), confidence.

Важно: в промпте явный запрет считать то, что требует расчёта
(население, спрос, расстояния) — это приходит из данных, не от модели.
"""
from __future__ import annotations

import json
import re

# Ожидаемая схема ответа (для валидации распарсенного JSON)
RESPONSE_SCHEMA = {
    "wealth_score": int,        # 0-10, премиальность на вид
    "area_character": str,      # residential | commercial | industrial | mixed
    "confidence": float,        # 0-1
}

# допустимые значения area_character
_ALLOWED_AREA = {"residential", "commercial", "industrial", "mixed"}


def build_prompt(context_text: str) -> str:
    """Собрать промпт из текстовой выжимки OSM. Требует вернуть только JSON по схеме."""
    return f"""Ты оцениваешь городской район по краткому описанию его окружения.
Дай КАЧЕСТВЕННЫЕ суждения — так, как если бы ты просто осмотрелся вокруг, без калькулятора.

ВАЖНО: ничего не вычисляй. Не оценивай население, спрос, расстояния или количество объектов —
эти числа приходят из данных, а не от тебя. Оцени только «на глаз».

Описание окружения района:
"{context_text}"

Верни СТРОГО один JSON-объект, без markdown и без пояснений, ровно с такими полями:
{{
  "wealth_score": целое число 0-10 — насколько богато/премиально выглядит район,
  "area_character": одно из "residential", "commercial", "industrial", "mixed" — тип района,
  "confidence": число от 0 до 1 — насколько ты уверен в оценке
}}"""


def parse_response(raw: str) -> dict:
    """Распарсить ответ модели в dict, почистив markdown-обёртки. Проверить по схеме."""
    text = raw.strip()
    # снять markdown-обёртку ```json ... ``` если есть
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # выдернуть первый JSON-объект на случай лишнего текста вокруг
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    data = json.loads(text)

    # валидация по схеме: наличие полей + приведение типов
    result = {}
    for key, expected_type in RESPONSE_SCHEMA.items():
        if key not in data:
            raise ValueError(f"В ответе модели нет поля '{key}': {data}")
        result[key] = expected_type(data[key])

    # проверка диапазонов и допустимых категорий
    if not 0 <= result["wealth_score"] <= 10:
        raise ValueError(f"wealth_score вне диапазона 0-10: {result['wealth_score']}")
    if not 0 <= result["confidence"] <= 1:
        raise ValueError(f"confidence вне диапазона 0-1: {result['confidence']}")
    area = result["area_character"].strip().lower()
    if area not in _ALLOWED_AREA:
        raise ValueError(f"area_character не из {_ALLOWED_AREA}: {area!r}")
    result["area_character"] = area

    return result
