"""Единая настройка логирования для всех этапов пайплайна."""
from __future__ import annotations
import logging


def get_logger(name: str) -> logging.Logger:
    """Вернуть настроенный логгер (формат с временем, уровнем, именем модуля)."""
    raise NotImplementedError
