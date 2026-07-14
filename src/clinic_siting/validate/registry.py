"""
ЭТАП 4 · Валидация ТИПА через "скрытый реестр" (backtesting).

Прячем часть реальных клиник — ОДНОВРЕМЕННО из реестра И из OSM
(фикс утечки: иначе модель подсмотрит их в OSM-конкурентах).
Запускаем пайплайн, сверяем предсказанный тип с реальным статусом.
"""
from __future__ import annotations


def load_registry(path: str):
    """Загрузить реестр лицензий. Геокодировать адреса в координаты. Статус: public/private."""
    raise NotImplementedError


def hide_clinics(registry, osm_clinics, fraction: float):
    """Спрятать долю реальных клиник из ОБОИХ источников (реестр + OSM). Вернуть held-out набор."""
    raise NotImplementedError


def score_type_prediction(predicted, held_out) -> dict:
    """Сверить предсказанный тип со скрытым реестром. Вернуть accuracy/F1 по public vs private."""
    raise NotImplementedError
