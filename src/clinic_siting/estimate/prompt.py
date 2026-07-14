"""
ЭТАП 2 · Промпт и JSON-схема.

Модель возвращает СУЖДЕНИЯ о районе, НЕ считает числа.
Строгий JSON: wealth_score (0-10), area_character (категория), confidence.

Важно: в промпте явный запрет считать то, что требует расчёта
(население, спрос, расстояния) — это приходит из данных, не от модели.
"""
from __future__ import annotations

# Ожидаемая схема ответа (для валидации распарсенного JSON)
RESPONSE_SCHEMA = {
    "wealth_score": int,        # 0-10, премиальность на вид
    "area_character": str,      # residential | commercial | industrial | mixed
    "confidence": float,        # 0-1
}


def build_prompt(context_text: str) -> str:
    """Собрать промпт из текстовой выжимки OSM. Требует вернуть только JSON по схеме."""
    raise NotImplementedError


def parse_response(raw: str) -> dict:
    """Распарсить ответ модели в dict, почистив markdown-обёртки. Проверить по схеме."""
    raise NotImplementedError
